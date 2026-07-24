#!/usr/bin/env python
"""Run one fresh-process split-candidate or final-RMSNorm-repair transition."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import qwen3_grpo_natural_transition_v0_2 as natural


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--arm", choices=["split_candidate", "final_norm_repair"], required=True)
    parser.add_argument("--repeat", type=int, choices=[1, 2], required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    snapshot_dir = Path(manifest["snapshot_dir"]).resolve()
    contract = natural.load_realization_contract(
        Path(manifest["realization_contract"]), snapshot_dir, 236
    )
    baseline_path = Path(manifest["baseline_records"][f"compiled_{args.repeat}"])
    baseline = json.loads(baseline_path.read_text())
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from accelerate import Accelerator
    from safetensors.torch import save_file
    from torch import nn
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, get_scheduler
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from trl.trainer.grpo_trainer import selective_log_softmax

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.recompile_limit = 64

    metadata = json.loads((snapshot_dir / "forkcert_transition_snapshot.json").read_text())
    target_path = Path(metadata["target_minibatch_path"])
    if not target_path.is_file():
        target_path = snapshot_dir / "compiler_history" / target_path.name
    inputs = natural.move_tree(
        torch.load(target_path, map_location="cpu", weights_only=False), "cuda"
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    accelerator = Accelerator(mixed_precision="fp16")
    wrapped = accelerator.prepare_model(model)
    raw = accelerator.unwrap_model(wrapped)
    named_parameters = [
        (name, parameter) for name, parameter in wrapped.named_parameters() if parameter.requires_grad
    ]
    named_buffers = list(wrapped.named_buffers())

    optimizer_snapshot = torch.load(
        snapshot_dir / "optimizer.pt", map_location="cpu", weights_only=False
    )
    optimizer = natural.make_optimizer(torch, wrapped, optimizer_snapshot)
    del optimizer_snapshot
    training_horizon = int(metadata["training_horizon_optimizer_steps"])
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=training_horizon
    )
    scheduler.load_state_dict(
        torch.load(snapshot_dir / "scheduler.pt", map_location="cpu", weights_only=False)
    )
    scaler = accelerator.scaler
    if scaler is None:
        raise RuntimeError("missing native FP16 GradScaler")
    scaler.load_state_dict(
        torch.load(snapshot_dir / "scaler.pt", map_location="cpu", weights_only=False)
    )
    saved_rng = torch.load(snapshot_dir / "rng_state.pth", map_location="cpu", weights_only=False)

    class PreNormDecoder(nn.Module):
        def __init__(self, body: Any):
            super().__init__()
            self.embed_tokens = body.embed_tokens
            self.layers = body.layers
            self.rotary_emb = body.rotary_emb
            self.config = body.config
            self.has_sliding_layers = body.has_sliding_layers

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            hidden_states = self.embed_tokens(input_ids)
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
            kwargs = dict(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
            masks = {"full_attention": create_causal_mask(**kwargs)}
            if self.has_sliding_layers:
                masks["sliding_attention"] = create_sliding_window_causal_mask(**kwargs)
            positions = self.rotary_emb(hidden_states, position_ids)
            for index, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                hidden_states = layer(
                    hidden_states,
                    attention_mask=masks[self.config.layer_types[index]],
                    position_embeddings=positions,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                )
            return hidden_states

    compile_audit: dict[str, dict[str, Any]] = {}

    def tracked_compile(module: Any, region_id: str) -> Any:
        from torch._dynamo.backends.registry import lookup_backend

        inductor = lookup_backend("inductor")
        audit = {"backend_compiles": 0, "runtime_invocations": 0, "graphs": []}
        compile_audit[region_id] = audit

        def backend(graph_module: Any, example_inputs: list[Any]):
            graph_hash = hashlib.sha256(graph_module.code.encode()).hexdigest()
            audit["backend_compiles"] += 1
            audit["graphs"].append(
                {
                    "sha256": graph_hash,
                    "node_count": sum(1 for _ in graph_module.graph.nodes),
                }
            )
            compiled = inductor(graph_module, example_inputs)

            def counted(*values: Any):
                audit["runtime_invocations"] += 1
                return compiled(*values)

            return counted

        return torch.compile(module, backend=backend)

    prefix = PreNormDecoder(raw.model)
    compiled_prefix = tracked_compile(prefix, "decoder_prefix")
    compiled_norm = tracked_compile(raw.model.norm, "final_rmsnorm")
    compiled_head = tracked_compile(raw.lm_head, "lm_head")

    def score(value: dict[str, Any], repair: bool) -> Any:
        completion = value["completion_ids"]
        input_ids = torch.cat([value["prompt_ids"], completion], dim=1)
        attention_mask = torch.cat([value["prompt_mask"], value["completion_mask"]], dim=1)
        with sdpa_kernel(SDPBackend.MATH), accelerator.autocast():
            hidden = compiled_prefix(input_ids, attention_mask)
            hidden = raw.model.norm(hidden) if repair else compiled_norm(hidden)
            logits = compiled_head(hidden[:, -(completion.size(1) + 1) :, :])
        logits = logits.float()[:, :-1, :]
        return selective_log_softmax(logits[:, -completion.size(1) :, :], completion)

    # Compile/warm only through the partitioned candidate, never the repair.
    natural.restore_rng(torch, saved_rng)
    for record in metadata["compiler_history"]:
        history_path = Path(record["path"])
        if not history_path.is_file():
            history_path = snapshot_dir / "compiler_history" / history_path.name
        history = natural.move_tree(
            torch.load(history_path, map_location="cpu", weights_only=False), "cuda"
        )
        value = score(history, repair=False)
        del history, value
        gc.collect()

    natural.restore_rng(torch, saved_rng)
    with torch.no_grad():
        candidate_control = score(inputs, repair=False).detach().float().cpu()
    candidate_control_hash = natural.tensor_sha256(candidate_control)
    candidate_anchor_exact = candidate_control_hash == contract["candidate_scorer_sha256"]
    compile_before_transition = json.loads(json.dumps(compile_audit))

    pre_parameter_hashes, pre_parameter_digest = natural.named_tensor_hashes(named_parameters)
    pre_buffer_hashes, pre_buffer_digest = natural.named_tensor_hashes(named_buffers)
    pre_optimizer = natural.optimizer_tensor_hashes(optimizer, named_parameters)
    pre_scheduler = scheduler.state_dict()
    pre_scaler = scaler.state_dict()
    natural.restore_rng(torch, saved_rng)
    rng_before = natural.rng_fingerprint(torch)
    optimizer.zero_grad(set_to_none=True)

    logps = score(inputs, repair=args.arm == "final_norm_repair")
    scorer_hash = natural.tensor_sha256(logps)
    decisions = natural.clip_decisions(torch, logps.detach(), inputs, epsilon=0.2)
    loss = natural.grpo_loss(torch, logps, inputs, epsilon=0.2)
    loss_value = float(loss.detach().item())
    scale_before = float(scaler.get_scale())
    scaler.scale(loss).backward()
    scaled = natural.tensor_collection_summary(
        torch, [(name, parameter.grad) for name, parameter in named_parameters]
    )
    scaler.unscale_(optimizer)
    unscaled = natural.tensor_collection_summary(
        torch, [(name, parameter.grad) for name, parameter in named_parameters]
    )
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in named_parameters], max_norm=1.0
    )
    clipped_pairs = [(name, parameter.grad) for name, parameter in named_parameters]
    clipped = natural.tensor_collection_summary(torch, clipped_pairs)
    before_parameters = {
        name: parameter.detach().cpu().clone() for name, parameter in named_parameters
    }
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    skipped = scale_after < scale_before
    if not skipped:
        scheduler.step()
    updates = {
        name: (parameter.detach().cpu() - before_parameters[name]).contiguous()
        for name, parameter in named_parameters
    }
    update_summary = natural.tensor_collection_summary(torch, list(updates.items()))
    clipped_cpu = {
        name: gradient.detach().cpu().contiguous()
        for name, gradient in clipped_pairs
        if gradient is not None
    }
    save_file(clipped_cpu, out_dir / "clipped_gradients.safetensors")
    save_file(updates, out_dir / "parameter_updates.safetensors")
    post_parameter_hashes, post_parameter_digest = natural.named_tensor_hashes(named_parameters)
    post_buffer_hashes, post_buffer_digest = natural.named_tensor_hashes(named_buffers)
    post_optimizer = natural.optimizer_tensor_hashes(optimizer, named_parameters)
    compile_after_transition = json.loads(json.dumps(compile_audit))

    expected_split_identity = args.arm == "split_candidate"
    split_update_exact = (
        natural.sha256_file(out_dir / "parameter_updates.safetensors")
        == baseline["vector_artifacts"]["parameter_updates"]["sha256"]
    )
    split_gradient_exact = (
        natural.sha256_file(out_dir / "clipped_gradients.safetensors")
        == baseline["vector_artifacts"]["clipped_gradients"]["sha256"]
    )
    split_scorer_exact = scorer_hash == contract["candidate_scorer_sha256"]
    candidate_transition_exact = split_scorer_exact and split_update_exact and split_gradient_exact
    valid = all(
        [
            candidate_anchor_exact,
            scaled["finite"],
            unscaled["finite"],
            clipped["finite"],
            update_summary["finite"],
            candidate_transition_exact if expected_split_identity else True,
        ]
    )
    result = {
        "schema_version": "forkcert.qwen3-step236-transition-intervention-arm.v0.1",
        "valid": valid,
        "verdict": "VALID" if valid else "INVALID",
        "arm": args.arm,
        "repeat": args.repeat,
        "case_id": manifest["case_id"],
        "state_id": manifest["state_id"],
        "anchors": {
            "candidate_control_sha256": candidate_control_hash,
            "candidate_control_anchor_exact": candidate_anchor_exact,
            "observed_scorer_sha256": scorer_hash,
            "split_candidate_scorer_exact": split_scorer_exact if expected_split_identity else None,
            "split_candidate_gradient_artifact_exact": split_gradient_exact if expected_split_identity else None,
            "split_candidate_update_artifact_exact": split_update_exact if expected_split_identity else None,
            "split_candidate_transition_exact": candidate_transition_exact if expected_split_identity else None,
        },
        "compile_context": {
            "before_transition": compile_before_transition,
            "after_transition": compile_after_transition,
            "autotuning": {"status": "UNOBSERVED"},
        },
        "pre_state": {
            "parameter_hashes_sha256": natural.json_sha256(pre_parameter_hashes),
            "parameter_digest": pre_parameter_digest,
            "buffer_hashes_sha256": natural.json_sha256(pre_buffer_hashes),
            "buffer_digest": pre_buffer_digest,
            "optimizer_digest": pre_optimizer["sha256"],
            "scheduler_digest": natural.json_sha256(pre_scheduler),
            "scaler_digest": natural.json_sha256(pre_scaler),
            "rng": rng_before,
        },
        "continuous": {
            "scorer_logps": logps.detach().float().cpu().tolist(),
            "loss": loss_value,
            "scaled_gradient": scaled,
            "unscaled_gradient": unscaled,
            "clipped_gradient": clipped,
            "pre_clip_gradient_norm": float(grad_norm.item()),
            "parameter_update": update_summary,
        },
        "semantic": {
            "clip_decisions": decisions.detach().cpu().tolist(),
            "clip_count": int(decisions.sum().item()),
            "gradient_clip_triggered": bool(float(grad_norm.item()) > 1.0),
            "amp_scale_before": scale_before,
            "amp_scale_after": scale_after,
            "optimizer_step_skipped": skipped,
        },
        "post_state": {
            "parameter_hashes_sha256": natural.json_sha256(post_parameter_hashes),
            "parameter_digest": post_parameter_digest,
            "buffer_hashes_sha256": natural.json_sha256(post_buffer_hashes),
            "buffer_digest": post_buffer_digest,
            "optimizer_digest": post_optimizer["sha256"],
            "scheduler_digest": natural.json_sha256(scheduler.state_dict()),
            "scaler_digest": natural.json_sha256(scaler.state_dict()),
            "rng": natural.rng_fingerprint(torch),
        },
        "vector_artifacts": {
            "clipped_gradients": {
                "path": str((out_dir / "clipped_gradients.safetensors").resolve()),
                "sha256": natural.sha256_file(out_dir / "clipped_gradients.safetensors"),
            },
            "parameter_updates": {
                "path": str((out_dir / "parameter_updates.safetensors").resolve()),
                "sha256": natural.sha256_file(out_dir / "parameter_updates.safetensors"),
            },
        },
        "baseline_compiled_record": {
            "path": str(baseline_path),
            "sha256": natural.sha256_file(baseline_path),
        },
        "limitations": [
            "split candidate identity must include scorer, clipped-gradient and update artifacts",
            "repair remains intervention-dependent unless non-target graph/kernel context is exact",
            "selected state only; no population or correctness claim",
        ],
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "valid": valid,
                "arm": args.arm,
                "repeat": args.repeat,
                "candidate_anchor_exact": candidate_anchor_exact,
                "split_candidate_transition_exact": result["anchors"]["split_candidate_transition_exact"],
                "loss": loss_value,
                "clip_count": result["semantic"]["clip_count"],
                "update_l2": update_summary["l2"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if valid else 2)


if __name__ == "__main__":
    main()
