#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from forkcert.config import load_config
from forkcert.io import read_jsonl
from forkcert.logprob_runner import PathConfig, attention_backend_context, cleanup_memory, configure_determinism, load_hf_path, precision_context


COMPONENT_MATCHERS = {
    "attention": lambda name, module: "attention" in module.__class__.__name__.lower(),
    "mlp": lambda name, module: "mlp" in module.__class__.__name__.lower(),
    "rmsnorm": lambda name, module: "rmsnorm" in module.__class__.__name__.lower(),
    "lm_head": lambda name, module: name.removeprefix("_orig_mod.") == "lm_head",
}


def config_from_dict(item: dict[str, Any], *, compile_model: bool | None = None) -> PathConfig:
    return PathConfig(
        name=item["name"], model_name_or_path=item["model_name_or_path"], dtype=item.get("dtype", "bf16"),
        autocast_dtype=item.get("autocast_dtype"), device=item.get("device", "cuda"),
        compile_model=item.get("compile_model", False) if compile_model is None else compile_model,
        attn_implementation=item.get("attn_implementation"), attention_backend=item.get("attention_backend"),
        logits_upcast_fp32=item.get("logits_upcast_fp32", True), model_training_mode=item.get("model_training_mode", False),
        gradient_checkpointing=item.get("gradient_checkpointing", False),
    )


def target_batch(samples: list[dict[str, Any]], cert: dict[str, Any]) -> list[dict[str, Any]]:
    by_case = {str(row["case_id"]): row for row in samples}
    target = by_case[str(cert["case_id"])]
    rollout = int(target["metadata"]["rollout_batch"])
    return [row for row in samples if int(row["metadata"]["rollout_batch"]) == rollout]


def make_batch(tokenizer: Any, samples: list[dict[str, Any]], device: str) -> tuple[Any, Any, int]:
    import torch
    prompts = [[int(v) for v in row["prompt_ids"]] for row in samples]
    responses = [[int(v) for v in row["response_ids"]] for row in samples]
    max_prompt, max_response = max(map(len, prompts)), max(map(len, responses))
    ids, masks = [], []
    for prompt, response in zip(prompts, responses, strict=True):
        left, right = max_prompt - len(prompt), max_response - len(response)
        ids.append([tokenizer.pad_token_id] * left + prompt + response + [tokenizer.pad_token_id] * right)
        masks.append([0] * left + [1] * (len(prompt) + len(response)) + [0] * right)
    return torch.tensor(ids, device=device), torch.tensor(masks, device=device), max_prompt


def clip_active(logp: float, cert: dict[str, Any]) -> bool:
    ratio_log = logp - float(cert["old_logp"])
    return ratio_log > math.log1p(float(cert["eps"])) if int(cert["advantage_sign"]) > 0 else ratio_log < math.log1p(-float(cert["eps"]))


def output_tensor(output: Any) -> Any:
    return output[0] if isinstance(output, (tuple, list)) else output


def replace_output(original: Any, tensor: Any) -> Any:
    if isinstance(original, tuple):
        return (tensor, *original[1:])
    if isinstance(original, list):
        return [tensor, *original[1:]]
    return tensor


def decoder_layers(model: Any) -> list[tuple[str, Any]]:
    rows = []
    for name, module in model.named_modules():
        normalized = name.removeprefix("_orig_mod.")
        if ".layers." not in normalized:
            continue
        tail = normalized.rsplit(".layers.", 1)[1]
        if tail.isdigit():
            rows.append((normalized, module))
    return sorted(rows, key=lambda row: int(row[0].rsplit(".", 1)[1]))


def disable_component_before_compile(model: Any, component: str) -> list[str]:
    import torch
    matcher = COMPONENT_MATCHERS[component]
    changed = []
    for name, module in model.named_modules():
        if name and matcher(name, module):
            module.forward = torch.compiler.disable(module.forward)
            changed.append(name)
    if not changed:
        raise RuntimeError(f"component matcher selected no modules: {component}")
    return changed


def run_target(model: Any, tokenizer: Any, cfg: PathConfig, samples: list[dict[str, Any]], cert: dict[str, Any], *, grad: bool = False) -> tuple[Any, float | None]:
    import torch
    ids, mask, max_prompt = make_batch(tokenizer, samples, cfg.device)
    target_index = next(i for i, row in enumerate(samples) if str(row["case_id"]) == str(cert["case_id"]))
    position = max_prompt + int(cert["token_index"])
    grad_context = nullcontext() if grad else torch.no_grad()
    with grad_context, attention_backend_context(cfg), precision_context(cfg):
        logits = model(input_ids=ids, attention_mask=mask).logits
        if cfg.logits_upcast_fp32:
            logits = logits.float()
        value = torch.nn.functional.log_softmax(logits[target_index, position - 1], dim=-1)[ids[target_index, position]]
    if not grad:
        return value.detach(), None
    model.zero_grad(set_to_none=True)
    old = torch.tensor(float(cert["old_logp"]), dtype=value.dtype, device=value.device)
    advantage = torch.tensor(float(cert["metadata"]["rollout_advantage"]), dtype=value.dtype, device=value.device)
    ratio = torch.exp(value - old)
    clipped = torch.clamp(ratio, 1.0 - float(cert["eps"]), 1.0 + float(cert["eps"]))
    loss = -torch.minimum(ratio * advantage, clipped * advantage)
    loss.backward()
    square = torch.zeros((), dtype=torch.float32, device=value.device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            square += parameter.grad.detach().float().square().sum()
    return value.detach(), float(torch.sqrt(square).item())


def capture_reference_layers(model: Any, tokenizer: Any, cfg: PathConfig, samples: list[dict[str, Any]], cert: dict[str, Any]) -> tuple[dict[int, Any], dict[str, Any], float]:
    captured: dict[int, Any] = {}
    submodules: dict[str, Any] = {}
    handles = []
    for name, module in decoder_layers(model):
        index = int(name.rsplit(".", 1)[1])
        handles.append(module.register_forward_hook(lambda _m, _i, out, index=index: captured.__setitem__(index, output_tensor(out).detach())))
    for name, module in model.named_modules():
        normalized = name.removeprefix("_orig_mod.")
        if not normalized or ".layers.0." not in normalized:
            continue
        if not any(part in normalized.lower() for part in ["self_attn", "mlp", "input_layernorm", "post_attention_layernorm"]):
            continue
        if any(module.children()) and not (normalized.endswith("self_attn") or normalized.endswith("mlp")):
            continue
        handles.append(module.register_forward_hook(lambda _m, _i, out, normalized=normalized: submodules.__setitem__(normalized, output_tensor(out).detach())))
    try:
        value, _ = run_target(model, tokenizer, cfg, samples, cert)
        return captured, submodules, float(value.item())
    finally:
        for handle in handles:
            handle.remove()


def canary(model: Any, tokenizer: Any, cfg: PathConfig, samples: list[dict[str, Any]], cert: dict[str, Any], component: str, baseline: float) -> dict[str, Any]:
    matcher = COMPONENT_MATCHERS[component]
    selected = next(((name, module) for name, module in model.named_modules() if name and matcher(name, module)), None)
    if selected is None:
        return {"passed": False, "reason": "no matching module"}
    name, module = selected
    def perturb(_module: Any, _inputs: Any, output: Any) -> Any:
        import torch
        tensor = output_tensor(output)
        # A uniform logit shift is exactly cancelled by log_softmax. Use a
        # bounded non-uniform perturbation so the same canary is observable for
        # hidden states and lm_head outputs.
        delta = torch.tanh(tensor.float()).to(tensor.dtype) * tensor.new_tensor(1e-3)
        return replace_output(output, tensor + delta)
    handle = module.register_forward_hook(perturb)
    try:
        perturbed, _ = run_target(model, tokenizer, cfg, samples, cert)
    finally:
        handle.remove()
    delta = float(perturbed.item()) - baseline
    return {"module": name, "injected_perturbation": 1e-3, "observed_signed_logp_delta": delta, "passed": abs(delta) > 0.0}


def measurement(name: str, logp: float, cert: dict[str, Any], **extra: Any) -> dict[str, Any]:
    boundary_state = logp - float(cert["old_logp"]) - float(cert["clip_boundary"])
    return {
        "intervention": name, "logp": logp, "signed_delta_vs_ref": logp - float(cert["logp_ref"]),
        "signed_margin": boundary_state, "clip_active": clip_active(logp, cert),
        "fork_vs_reference": clip_active(logp, cert) != bool(cert["clip_ref"]), **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-fork component repair and residual-stream splice attribution.")
    parser.add_argument("--config", default="configs/hf_compile_sdpa_math_step5.yaml")
    parser.add_argument("--certificates", default="results/phase4_certificates.jsonl")
    parser.add_argument("--samples", default="data/phase6_step5_replay_samples.jsonl")
    parser.add_argument("--case-id", default="grpo_000001_2817771126c0")
    parser.add_argument("--token-index", type=int, default=80)
    parser.add_argument("--out", default="results/attribution/clip-step5-grpo_000001_2817771126c0-t80.json")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    configure_determinism(0)
    cfg_data = load_config(args.config)
    cert = next(row for row in read_jsonl(args.certificates) if row.get("actual_fork") and str(row["case_id"]) == args.case_id and int(row["token_index"]) == args.token_index)
    samples = target_batch(read_jsonl(args.samples), cert)
    ref_cfg = config_from_dict(cfg_data["path_ref"], compile_model=False)
    alt_cfg = config_from_dict(cfg_data["path_alt"], compile_model=False)
    rows = []

    tokenizer, ref_model = load_hf_path(ref_cfg)
    ref_layers, ref_submodules, ref_logp = capture_reference_layers(ref_model, tokenizer, ref_cfg, samples, cert)
    ref_value, ref_grad = run_target(ref_model, tokenizer, ref_cfg, samples, cert, grad=True)
    rows.append(measurement("reference_eager", float(ref_value.item()), cert, grad_norm=ref_grad))
    del ref_model
    cleanup_memory()

    for component in [None, "attention", "mlp", "rmsnorm", "lm_head"]:
        tokenizer, model = load_hf_path(alt_cfg)
        disabled = []
        if component:
            disabled = disable_component_before_compile(model, component)
        import torch
        model = torch.compile(model)
        run_target(model, tokenizer, alt_cfg, samples, cert)  # warm compile graph
        value, _ = run_target(model, tokenizer, alt_cfg, samples, cert)
        logp = float(value.item())
        row = measurement("compile_baseline" if component is None else f"compile_repair_{component}", logp, cert, disabled_modules=disabled)
        if component:
            row["canary"] = canary(model, tokenizer, alt_cfg, samples, cert, component, logp)
            row["valid_intervention"] = bool(row["canary"]["passed"])
            row["attribution_scope"] = "compile_region_barrier"
            row["unique_component_attribution_valid"] = False
        rows.append(row)
        del model
        del tokenizer
        cleanup_memory()

    # Compile once and splice one reference decoder-block output at a time. The
    # sparse pass brackets the first useful splice without an unbounded search.
    tokenizer, model = load_hf_path(alt_cfg)
    import torch
    model = torch.compile(model)
    run_target(model, tokenizer, alt_cfg, samples, cert)
    layers = decoder_layers(model)
    probe_indices = sorted({0, len(layers) // 4, len(layers) // 2, 3 * len(layers) // 4, len(layers) - 1})
    for index in probe_indices:
        _, module = layers[index]
        replacement = ref_layers[index]
        identity_handle = module.register_forward_hook(lambda _m, _i, out: out)
        try:
            identity_value, _ = run_target(model, tokenizer, alt_cfg, samples, cert)
        finally:
            identity_handle.remove()
        handle = module.register_forward_hook(lambda _m, _i, out, replacement=replacement: replace_output(out, replacement.to(output_tensor(out).device, dtype=output_tensor(out).dtype)))
        try:
            value, _ = run_target(model, tokenizer, alt_cfg, samples, cert)
        finally:
            handle.remove()
        splice_logp = float(value.item())
        identity_logp = float(identity_value.item())
        rows.append(measurement(
            f"splice_decoder_layer_{index}", splice_logp, cert,
            identity_hook_logp=identity_logp,
            splice_effect_vs_identity=splice_logp - identity_logp,
            valid_intervention=abs(splice_logp - identity_logp) > 0.0,
            splice_canary={"replacement_max_abs": float(replacement.abs().max().item()), "passed": abs(splice_logp - identity_logp) > 0.0},
        ))

    compiled_modules = {name.removeprefix("_orig_mod."): module for name, module in model.named_modules()}
    splice_targets = (
        "input_layernorm", "self_attn.q_proj", "self_attn", "post_attention_layernorm", "mlp.down_proj", "mlp"
    )
    for name, replacement in sorted(ref_submodules.items()):
        if not any(name.endswith(target) for target in splice_targets):
            continue
        module = compiled_modules.get(name)
        if module is None:
            rows.append({"intervention": f"splice_{name}", "valid_intervention": False, "reason": "compiled module name missing"})
            continue
        identity_handle = module.register_forward_hook(lambda _m, _i, out: out)
        try:
            identity_value, _ = run_target(model, tokenizer, alt_cfg, samples, cert)
        finally:
            identity_handle.remove()
        handle = module.register_forward_hook(
            lambda _m, _i, out, replacement=replacement: replace_output(
                out, replacement.to(output_tensor(out).device, dtype=output_tensor(out).dtype)
            )
        )
        try:
            value, _ = run_target(model, tokenizer, alt_cfg, samples, cert)
        finally:
            handle.remove()
        splice_logp = float(value.item())
        identity_logp = float(identity_value.item())
        effect = splice_logp - identity_logp
        rows.append(measurement(
            f"splice_{name}", splice_logp, cert,
            identity_hook_logp=identity_logp, splice_effect_vs_identity=effect,
            valid_intervention=abs(effect) > 0.0,
            splice_canary={"reference_tensor_shape": list(replacement.shape), "passed": abs(effect) > 0.0},
            attribution_scope="layer0_submodule_output",
        ))

    optimizer_step = int(cert["metadata"]["phase1_metadata"]["online_state"]["optimizer_step"])
    payload = {
        "schema_version": "forkcert.attribution.v1", "fork_id": f"clip-step{optimizer_step}-{args.case_id}-t{args.token_index}",
        "contract": "Same checkpoint, token IDs, batch, mask, position semantics, old_logp, advantage and FP16 autocast; only compile repair/splice changes.",
        "certificate": cert, "sample_case_ids": [row["case_id"] for row in samples], "measurements": rows,
        "limitations": [
            "torch.compiler.disable introduces graph barriers; exact repair is compile-region evidence, not unique component attribution.",
            "Sparse layer splices localize residual-stream sufficiency, not necessarily a unique kernel root cause.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "measurements": len(rows), "ref_replay_error": ref_logp - float(cert["logp_ref"])}, indent=2))


if __name__ == "__main__":
    main()
