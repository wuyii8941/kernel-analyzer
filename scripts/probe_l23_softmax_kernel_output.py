#!/usr/bin/env python3
"""Probe the generated layer-23 softmax-backward output in place.

This is a narrow causal test.  It keeps the compiled graph, operands, and all
downstream calls unchanged, then replaces only the output buffer written by
the selected generated softmax-backward kernel with the eager reference
tensor.  It is deliberately separate from the broader attention decomposition
so that a kernel-level source claim cannot be inferred from a semantic tensor
replacement alone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_SRC = ROOT / "archive" / "round1_code" / "src"
for path in (OLD_SRC, ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.long_horizon_trigger import build_model, load_eval_states, load_milestone, under_root


PARAMETER = "model.layers.23.self_attn.q_proj.weight"
ROWS = slice(1152, 1280)
COLUMNS = slice(1664, 1792)
TARGET_NODE_MARKER = "bmm_75]"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, default=Path("results/final/long_horizon_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--direction", type=Path, default=Path("results/final/l23_qproj_tile_direction.pt"))
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--state-index", type=int, action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/property/root_cause_closure_v1/l23_softmax_kernel_output_probe_v1.json"),
    )
    parser.add_argument(
        "--intervention",
        choices=("output", "input", "scores", "statistics", "carrier_right"),
        default="output",
        help="replace the generated softmax-backward output or its upstream gradient input",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def projection(value, direction, torch) -> float:
    return float(torch.dot(value.reshape(-1).float(), direction.reshape(-1).float()) / direction.float().norm())


def main() -> None:
    cfg = args()
    bank_path = under_root(cfg.bank, "bank")
    model_path = under_root(cfg.model, "model")
    direction_path = under_root(cfg.direction, "direction")
    output_path = under_root(cfg.output, "output")
    state_indices = cfg.state_index or list(range(8, 16))
    if len(set(state_indices)) != len(state_indices) or min(state_indices) < 8:
        raise ValueError("state indices must be unique and held out (>=8)")

    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/kernel_analyzer/l23_softmax_output_probe")

    import torch
    import torch.nn.functional as F
    from torch._dynamo.backends.registry import lookup_backend
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.select_algorithm import extern_kernels
    from transformers import AutoTokenizer
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    device = torch.device(cfg.device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    bank = json.loads(bank_path.read_text())
    milestone = next(row for row in bank["milestones"] if int(row["step"]) == cfg.step)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    all_states, evaluation = load_eval_states(tokenizer, 1024, max(state_indices) + 1, device)
    model = build_model(model_path, device)
    load_milestone(model, milestone, model_path)
    direction = torch.load(direction_path, map_location="cpu", weights_only=False)["direction"].float().to(device)

    class LossStep(torch.nn.Module):
        def __init__(self, subject):
            super().__init__()
            self.subject = subject

        def forward(self, input_ids, labels):
            return self.subject(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]

    module_start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend=lookup_backend("inductor"), fullgraph=True, dynamic=False)
    model.zero_grad(set_to_none=True)
    warm_loss = candidate(*all_states[state_indices[0]])
    warm_loss.backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[module_start:])
    source_matches = []
    for module in modules:
        source_path = Path(module.__file__)
        source = source_path.read_text()
        if TARGET_NODE_MARKER in source and "mm_267" in source:
            source_matches.append((module, source_path, source))
    if len(source_matches) != 1:
        raise RuntimeError(f"expected one exact layer-23 backward source, got {len(source_matches)}")
    source_module, source_path, source = source_matches[0]
    marker_position = source.index(TARGET_NODE_MARKER)
    call_start = source.rfind("def call(", 0, marker_position)
    target_ordinal = source[call_start:marker_position].count("extern_kernels.bmm(")
    softmax_calls = list(re.finditer(
        r"([A-Za-z0-9_]*softmax[A-Za-z0-9_]*backward[A-Za-z0-9_]*)\.run\(",
        source[call_start:marker_position],
    ))
    if not softmax_calls:
        raise RuntimeError("target softmax backward kernel was not found")
    target_softmax_symbol = softmax_calls[-1].group(1)
    target_softmax_position = call_start + softmax_calls[-1].start()
    target_softmax_ordinal = source[call_start:target_softmax_position].count(
        f"{target_softmax_symbol}.run("
    )
    target_softmax_kernel = getattr(source_module, target_softmax_symbol)

    original_attention = modeling_qwen3.eager_attention_forward
    original_bmm = extern_kernels.bmm
    q_proj_parameter = dict(model.named_parameters())[PARAMETER]
    eager_capture: dict[str, torch.Tensor] = {}

    def captured_eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = modeling_qwen3.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_qwen3.repeat_kv(value, module.num_key_value_groups)
        raw_scores = torch.matmul(query, key_states.transpose(2, 3))
        if module.layer_idx == 23:
            eager_capture["K"] = key_states.detach().reshape(16, 1024, 128).clone()
            def raw_hook(gradient):
                eager_capture["S"] = gradient.detach().reshape(16, 1024, 1024).clone()
                return gradient
            raw_scores.register_hook(raw_hook)
        attn_weights = raw_scores * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        probability = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        if module.layer_idx == 23:
            eager_capture["L"] = attn_weights.detach().reshape(16, 1024, 1024).clone()
        attn_weights = probability.to(query.dtype)
        if module.layer_idx == 23:
            eager_capture["P"] = attn_weights.detach().reshape(16, 1024, 1024).clone()
            score32 = (raw_scores * scaling).float()
            if attention_mask is not None:
                score32 = score32 + attention_mask[:, :, :, : key_states.shape[-2]].float()
            amax = score32.amax(dim=-1, keepdim=True)
            normalizer = torch.exp(score32 - amax).sum(dim=-1, keepdim=True)
            eager_capture["Amax"] = amax.detach().reshape(16, 1024, 1).clone()
            eager_capture["Sum"] = normalizer.detach().reshape(16, 1024, 1).clone()
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        if module.layer_idx == 23:
            def probability_hook(gradient):
                eager_capture["U"] = gradient.detach().reshape(16, 1024, 1024).clone()
                return gradient
            attn_weights.register_hook(probability_hook)
        return torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous(), attn_weights

    def run_candidate(inputs, replacement=None):
        model.zero_grad(set_to_none=True)
        loss = candidate(*inputs)
        bmm_count = {"value": 0}
        softmax_count = {"value": 0}
        observed: dict[str, object] = {}
        original_softmax_run = target_softmax_kernel.run

        def wrapped_softmax_run(*values, **kwargs):
            ordinal = softmax_count["value"]
            softmax_count["value"] += 1
            if ordinal != target_softmax_ordinal:
                return original_softmax_run(*values, **kwargs)
            if len(values) < 4 or tuple(values[1].shape) != (16, 1024, 1024):
                raise RuntimeError("unexpected target softmax operands")
            call_values = list(values)
            output = call_values[0]
            observed["score_buffer_before"] = call_values[0].detach().clone()
            observed["input_before"] = call_values[1].detach().clone()
            observed["amax_before"] = call_values[2].detach().clone()
            observed["sum_before"] = call_values[3].detach().clone()
            if replacement is not None and cfg.intervention == "input":
                if call_values[1].numel() != replacement.numel():
                    raise RuntimeError("softmax input/reference element-count mismatch")
                call_values[1] = replacement.reshape_as(call_values[1]).to(call_values[1].dtype)
                observed["input_replaced"] = True
            elif replacement is not None and cfg.intervention == "scores":
                if call_values[0].numel() != replacement.numel():
                    raise RuntimeError("softmax score/reference element-count mismatch")
                call_values[0].copy_(replacement.reshape_as(call_values[0]).to(call_values[0].dtype))
                observed["scores_replaced"] = True
            elif replacement is not None and cfg.intervention == "statistics":
                if len(replacement) != 2:
                    raise RuntimeError("statistics replacement must contain max and normalizer")
                if call_values[2].numel() != replacement[0].numel() or call_values[3].numel() != replacement[1].numel():
                    raise RuntimeError("softmax statistics/reference shape mismatch")
                call_values[2] = replacement[0].reshape_as(call_values[2]).to(call_values[2].dtype)
                call_values[3] = replacement[1].reshape_as(call_values[3]).to(call_values[3].dtype)
                observed["statistics_replaced"] = True
            result = original_softmax_run(*call_values, **kwargs)
            observed["target"] = True
            observed["output_shape"] = list(output.shape)
            observed["softmax_output_ptr"] = int(output.data_ptr())
            observed["kernel_output"] = output.detach().clone()
            if replacement is not None and cfg.intervention == "output":
                if output.numel() != replacement.numel():
                    raise RuntimeError("softmax output/reference element-count mismatch")
                output.copy_(replacement.reshape_as(output))
                observed["replaced"] = True
            return result

        def wrapped_bmm(*values, **kwargs):
            ordinal = bmm_count["value"]
            bmm_count["value"] += 1
            call_values = list(values)
            if ordinal == target_ordinal:
                observed["bmm_left"] = values[0].detach().clone()
                observed["bmm_right"] = values[1].detach().clone()
                observed["bmm_left_ptr"] = int(values[0].data_ptr())
                observed["bmm_right_ptr"] = int(values[1].data_ptr())
                if cfg.intervention == "carrier_right" and replacement is not None:
                    if call_values[1].numel() != replacement.numel():
                        raise RuntimeError("carrier right/reference element-count mismatch")
                    call_values[1] = replacement.reshape_as(call_values[1]).to(call_values[1].dtype)
                    observed["carrier_right_replaced"] = True
            result = original_bmm(*call_values, **kwargs)
            if ordinal == target_ordinal:
                out = kwargs.get("out")
                if out is not None:
                    observed["bmm_output"] = out.detach().clone()
            return result

        target_softmax_kernel.run = wrapped_softmax_run
        extern_kernels.bmm = wrapped_bmm
        try:
            loss.backward()
            torch.cuda.synchronize(device)
        finally:
            target_softmax_kernel.run = original_softmax_run
            extern_kernels.bmm = original_bmm
        if not observed.get("target"):
            raise RuntimeError("target softmax kernel was not observed")
        observed["qproj_full_grad"] = q_proj_parameter.grad.detach().clone()
        return (
            float(loss.detach().float().cpu()),
            q_proj_parameter.grad.detach()[ROWS, COLUMNS].clone(),
            observed,
        )

    rows = []
    try:
        for state_index in state_indices:
            inputs = all_states[state_index]
            eager_capture.clear()
            model.zero_grad(set_to_none=True)
            modeling_qwen3.eager_attention_forward = captured_eager_attention
            try:
                eager_loss = model(input_ids=inputs[0], labels=inputs[1], use_cache=False, return_dict=False)[0]
                eager_loss.backward()
            finally:
                modeling_qwen3.eager_attention_forward = original_attention
            torch.cuda.synchronize(device)
            required_eager = {"S"}
            if cfg.intervention == "input":
                required_eager.add("U")
            elif cfg.intervention == "scores":
                required_eager.add("L")
            elif cfg.intervention == "statistics":
                required_eager.update(("Amax", "Sum"))
            elif cfg.intervention == "carrier_right":
                required_eager.add("K")
            if not required_eager.issubset(eager_capture):
                raise RuntimeError(f"eager capture was incomplete: {sorted(eager_capture)}")
            eager_tile = q_proj_parameter.grad.detach()[ROWS, COLUMNS].clone()

            candidate_loss, candidate_tile, baseline = run_candidate(inputs)
            replacement = {
                "output": eager_capture["S"],
                "input": eager_capture["U"],
                "scores": eager_capture["L"],
                "statistics": (eager_capture["Amax"], eager_capture["Sum"]),
                "carrier_right": eager_capture["K"],
            }[cfg.intervention]
            repaired_loss, repaired_tile, repaired = run_candidate(inputs, replacement=replacement)
            baseline_s = baseline["kernel_output"].reshape_as(eager_capture["S"])
            repaired_s = repaired["kernel_output"].reshape_as(eager_capture["S"])
            carrier_side = "left" if baseline["bmm_left"].numel() == eager_capture["S"].numel() else "right"
            baseline_bmm_carrier = baseline[f"bmm_{carrier_side}"].reshape_as(eager_capture["S"])
            baseline_bmm_output = baseline.get("bmm_output")
            repaired_bmm_output = repaired.get("bmm_output")
            total = candidate_tile.float() - eager_tile.float()
            residual = repaired_tile.float() - eager_tile.float()
            if cfg.intervention == "output":
                intervention_l2 = float(
                    (baseline["kernel_output"].float() - repaired["kernel_output"].float()).norm()
                )
            elif cfg.intervention == "input":
                intervention_l2 = float(
                    (baseline["input_before"].float() - replacement.float()).norm()
                )
            elif cfg.intervention == "scores":
                intervention_l2 = float(
                    (baseline["score_buffer_before"].float() - eager_capture["L"].float()).norm()
                )
            elif cfg.intervention == "carrier_right":
                intervention_l2 = float(
                    (baseline["bmm_right"].float() - eager_capture["K"].float()).norm()
                )
            else:
                intervention_l2 = float(
                    torch.cat((
                        (baseline["amax_before"].float() - eager_capture["Amax"].float()).reshape(-1),
                        (baseline["sum_before"].float() - eager_capture["Sum"].float()).reshape(-1),
                    )).norm()
                )
            rows.append({
                "state_index": state_index,
                "offset": evaluation["offsets"][state_index],
                "eager_loss": float(eager_loss.detach().float().cpu()),
                "candidate_loss": candidate_loss,
                "repaired_loss": repaired_loss,
                "softmax_carrier_side": carrier_side,
                "intervention": cfg.intervention,
                "candidate_kernel_output_equals_bmm_input": bool(torch.equal(baseline_s, baseline_bmm_carrier)),
                "candidate_kernel_output_to_bmm_input_max_abs": float((baseline_s.float() - baseline_bmm_carrier.float()).abs().max()),
                "candidate_kernel_output_ptr_equals_bmm_input": bool(
                    baseline.get("softmax_output_ptr") == baseline.get(f"bmm_{carrier_side}_ptr")
                ),
                "target_bmm_output_repair_max_abs": float(
                    (baseline_bmm_output.float() - repaired_bmm_output.float()).abs().max()
                ) if baseline_bmm_output is not None and repaired_bmm_output is not None else None,
                "target_bmm_output_repair_l2": float(
                    (baseline_bmm_output.float() - repaired_bmm_output.float()).norm()
                ) if baseline_bmm_output is not None and repaired_bmm_output is not None else None,
                "candidate_kernel_output_vs_eager_s_max_abs": float((baseline_s.float() - eager_capture["S"].float()).abs().max()),
                "candidate_softmax_input_vs_eager_u_max_abs": float(
                    (baseline.get("input_before", baseline["bmm_left"]).float() - eager_capture["U"].float()).abs().max()
                ) if "U" in eager_capture else None,
                "candidate_scores_vs_eager_max_abs": float(
                    (baseline["score_buffer_before"].float() - eager_capture["L"].float()).abs().max()
                ) if "L" in eager_capture else None,
                "intervention_operand_repair_l2": intervention_l2,
                "kernel_output_repair_residual_projection": projection(residual, direction, torch),
                "kernel_output_repair_removal_projection": projection(total - residual, direction, torch),
                "total_projection": projection(total, direction, torch),
                "kernel_output_repair_residual_l2": float(residual.norm()),
                "full_qproj_grad_repair_l2": float(
                    (baseline.get("qproj_full_grad") - repaired.get("qproj_full_grad")).float().norm()
                ) if baseline.get("qproj_full_grad") is not None and repaired.get("qproj_full_grad") is not None else None,
                "candidate_kernel_output_l2": float((baseline_s.float() - eager_capture["S"].float()).norm()),
                "repaired_kernel_output_vs_reference_max_abs": float((repaired_s.float() - eager_capture["S"].float()).abs().max()),
                "target_kernel_observed": bool(baseline.get("target") and repaired.get("target")),
                "target_kernel_replaced": bool(repaired.get("replaced")),
                "target_input_replaced": bool(repaired.get("input_replaced")),
                "target_scores_replaced": bool(repaired.get("scores_replaced")),
                "target_statistics_replaced": bool(repaired.get("statistics_replaced")),
                "target_carrier_right_replaced": bool(repaired.get("carrier_right_replaced")),
            })
            print(json.dumps({"state": state_index, "total": rows[-1]["total_projection"], "repaired": rows[-1]["kernel_output_repair_residual_projection"]}), flush=True)
    finally:
        modeling_qwen3.eager_attention_forward = original_attention

    payload = {
        "schema": "kernel-analyzer-l23-softmax-kernel-output-probe-v1",
        "status": "COMPLETE",
        "model": str(model_path),
        "checkpoint_step": cfg.step,
        "state_indices": state_indices,
        "parameter": PARAMETER,
        "tile": {"rows": [ROWS.start, ROWS.stop], "columns": [COLUMNS.start, COLUMNS.stop]},
        "target": {
            "backward_node": "bmm_75",
            "softmax_kernel_symbol": target_softmax_symbol,
            "softmax_kernel_ordinal": target_softmax_ordinal,
            "intervention": (
                "replace only the target generated softmax-backward output buffer after its normal execution"
                if cfg.intervention == "output" else
                "replace only the upstream gradient input of the target generated softmax-backward call"
                if cfg.intervention == "input" else
                "replace only the pre-softmax score buffer consumed by the target generated softmax-backward call"
                if cfg.intervention == "scores" else
                "replace only the max and normalizer statistics consumed by the target generated softmax-backward call"
                if cfg.intervention == "statistics" else
                "replace only the right key carrier of the target bmm_75 VJP call"
            ),
        },
        "rows": rows,
        "summary": {
            "total_mean": sum(row["total_projection"] for row in rows) / len(rows),
            "residual_mean": sum(row["kernel_output_repair_residual_projection"] for row in rows) / len(rows),
            "residual_positive_count": sum(row["kernel_output_repair_residual_projection"] > 0 for row in rows),
            "residual_max_abs_output_error": max(row["repaired_kernel_output_vs_reference_max_abs"] for row in rows),
            "all_replacements_observed": all(row["target_kernel_replaced"] for row in rows),
            "all_input_replacements_observed": all(row["target_input_replaced"] for row in rows),
            "all_score_replacements_observed": all(row["target_scores_replaced"] for row in rows),
            "all_statistics_replacements_observed": all(row["target_statistics_replaced"] for row in rows),
            "all_carrier_right_replacements_observed": all(row["target_carrier_right_replaced"] for row in rows),
            "all_repaired_outputs_equal_reference": all(row["repaired_kernel_output_vs_reference_max_abs"] == 0.0 for row in rows),
        },
        "claim_boundary": "This is a same-input, path-preserving generated-kernel intervention at the declared endpoint. It can isolate the selected endpoint from the other retained operands, but it does not establish a population mean or training-quality consequence.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output_path.relative_to(ROOT)), "states": len(rows)}))


if __name__ == "__main__":
    main()
