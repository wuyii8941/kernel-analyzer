#!/usr/bin/env python
"""Replay an original Inductor final-RMSNorm kernel group on captured launch inputs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from forkcert.operator_evidence import production_mediation_interpretation, tensor_fingerprint


def tensor_sha256(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def metrics(torch: Any, left: Any, right: Any) -> dict[str, Any]:
    delta = right.double() - left.double()
    return {
        "max_abs": float(delta.abs().max().item()),
        "l2": float(torch.linalg.vector_norm(delta).item()),
        "mean_signed": float(delta.mean().item()),
        "nonzero": int(torch.count_nonzero(delta).item()),
    }


def load_generated_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("forkcert_original_inductor_output", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generated output code: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grouped_dump_paths(result: dict[str, Any], kernel: str) -> list[list[Path]]:
    rows = [
        row
        for row in result.get("observability", {}).get("kernel_input_dumps", [])
        if kernel in row.get("kernel_run_dir", "")
    ]
    groups: dict[str, list[Path]] = {}
    for row in rows:
        groups.setdefault(row["kernel_run_dir"], []).append(Path(row["path"]))
    return [
        sorted(paths, key=lambda path: int(path.stem.split("_")[-1]))
        for _, paths in sorted(
            groups.items(),
            key=lambda item: max(path.stat().st_mtime_ns for path in item[1]),
            reverse=True,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dump-result", required=True)
    parser.add_argument("--observability-gate", required=True)
    parser.add_argument("--dump-equivalence-gate", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    dump_result = json.loads(Path(args.dump_result).read_text())
    gate = json.loads(Path(args.observability_gate).read_text())
    dump_gate = json.loads(Path(args.dump_equivalence_gate).read_text())
    inventory = json.loads(Path(args.inventory).read_text())
    if gate.get("forward_kernel_inventory_eligible") is not True:
        raise RuntimeError("whole-model forward observability gate did not pass")
    if dump_gate.get("forward_observability_equivalent") is not True:
        raise RuntimeError("kernel-input dump changed the forward treatment")
    matching = [
        row
        for row in inventory["kernels"]
        if row["generated_symbol"] == manifest["pointwise_kernel"]
    ]
    if len(matching) != 1:
        raise RuntimeError(f"expected one original pointwise kernel, found {len(matching)}")
    kernel_row = matching[0]
    output_code = Path(kernel_row["output_code_path"])

    import torch
    from safetensors import safe_open
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from trl.trainer.grpo_trainer import selective_log_softmax

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    module = load_generated_module(output_code)
    reduction_kernel = getattr(module, manifest["reduction_kernel"])
    kernel = getattr(module, manifest["pointwise_kernel"])
    dump_groups = grouped_dump_paths(dump_result, manifest["pointwise_kernel"])
    reduction_dump_groups = grouped_dump_paths(dump_result, manifest["reduction_kernel"])
    if not dump_groups:
        raise RuntimeError("no captured pointwise-kernel launch inputs")
    if not reduction_dump_groups:
        raise RuntimeError("no captured reduction-kernel launch inputs")

    snapshot = Path(manifest["snapshot_dir"])
    metadata = json.loads((snapshot / "forkcert_transition_snapshot.json").read_text())
    target_path = Path(metadata["target_minibatch_path"])
    if not target_path.is_file():
        target_path = snapshot / "compiler_history" / target_path.name
    target = torch.load(target_path, map_location="cpu", weights_only=False)
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as handle:
        head_weight_cpu = handle.get_tensor("model.embed_tokens.weight")
    # Preserve the generated wrapper's column-major (1, hidden_size) layout.
    head_weight = head_weight_cpu.to(device="cuda", dtype=torch.float16).t()
    del head_weight_cpu

    contract = json.loads(Path(manifest["realization_contract"]).read_text())
    expected_scorer = contract["candidate_scorer_sha256"]

    def fixed_suffix(boundary: Any) -> Any:
        flat = boundary.reshape(-1, manifest["hidden_size"])
        mm_out = torch.empty(
            (flat.shape[0], head_weight.shape[1]),
            device="cuda",
            dtype=torch.float16,
        )
        module.extern_kernels.mm(flat, head_weight, out=mm_out)
        logits = mm_out.reshape(
            target["completion_ids"].shape[0],
            manifest["completion_plus_one"],
            head_weight.shape[1],
        ).float()
        values = selective_log_softmax(
            logits[:, :-1, :], target["completion_ids"].to("cuda")
        )
        return values

    candidates = []
    chosen = None
    for group in dump_groups:
        tensors = [torch.load(path, map_location="cuda", weights_only=True) for path in group]
        if len(tensors) != 6:
            candidates.append({"paths": [str(path) for path in group], "valid": False})
            continue
        component_hashes = [tensor_sha256(tensors[index]) for index in (1, 2, 3)]
        paired_reduction = None
        for reduction_group in reduction_dump_groups:
            reduction_tensors = [
                torch.load(path, map_location="cuda", weights_only=True)
                for path in reduction_group
            ]
            if len(reduction_tensors) == 4 and [
                tensor_sha256(reduction_tensors[index]) for index in (1, 2, 3)
            ] == component_hashes:
                paired_reduction = (reduction_group, reduction_tensors)
                break
            del reduction_tensors
        if paired_reduction is None:
            candidates.append({
                "paths": [str(path) for path in group],
                "valid": False,
                "reason": "no reduction launch with identical component inputs",
            })
            del tensors
            continue
        reduction_group, reduction_tensors = paired_reduction
        output_template = tensors[5]
        compiled_outputs = []
        reciprocal_rms_outputs = []
        for _ in range(2):
            reciprocal_rms = torch.empty_strided(
                reduction_tensors[0].shape,
                reduction_tensors[0].stride(),
                dtype=reduction_tensors[0].dtype,
                device=reduction_tensors[0].device,
            )
            reduction_kernel.run(
                reciprocal_rms,
                reduction_tensors[1],
                reduction_tensors[2],
                reduction_tensors[3],
                manifest["reduction_xnumel"],
                manifest["reduction_rnumel"],
                stream=module.get_raw_stream(0),
            )
            out = torch.empty_strided(
                output_template.shape,
                output_template.stride(),
                dtype=output_template.dtype,
                device=output_template.device,
            )
            kernel.run(
                tensors[0], tensors[1], tensors[2], tensors[3], reciprocal_rms, out,
                manifest["dynamic_sequence_length"], manifest["pointwise_xnumel"],
                stream=module.get_raw_stream(0)
            )
            torch.cuda.synchronize()
            reciprocal_rms_outputs.append(reciprocal_rms)
            compiled_outputs.append(out)
        scorer = fixed_suffix(compiled_outputs[0])
        scorer_hash = tensor_sha256(scorer)
        row = {
            "paths": [str(path) for path in group],
            "valid": True,
            "compiled_repeat_exact": tensor_sha256(compiled_outputs[0])
            == tensor_sha256(compiled_outputs[1]),
            "reduction_repeat_exact": tensor_sha256(reciprocal_rms_outputs[0])
            == tensor_sha256(reciprocal_rms_outputs[1]),
            "reduction_replay_matches_captured": tensor_sha256(reciprocal_rms_outputs[0])
            == tensor_sha256(tensors[4]),
            "reduction_paths": [str(path) for path in reduction_group],
            "scorer_sha256": scorer_hash,
            "candidate_scorer_anchor_exact": scorer_hash == expected_scorer,
        }
        row["valid"] = all(
            [
                row["compiled_repeat_exact"],
                row["reduction_repeat_exact"],
                row["reduction_replay_matches_captured"],
                row["candidate_scorer_anchor_exact"],
            ]
        )
        candidates.append(row)
        if row["valid"] and chosen is None:
            chosen = (
                tensors,
                reduction_tensors,
                reciprocal_rms_outputs,
                compiled_outputs,
                scorer,
                row,
            )
        else:
            del tensors, reduction_tensors, reciprocal_rms_outputs, compiled_outputs, scorer
        gc.collect()
    if chosen is None:
        raise RuntimeError("no captured kernel invocation reproduced the candidate scorer anchor")
    (
        tensors,
        reduction_tensors,
        reciprocal_rms_outputs,
        compiled_outputs,
        compiled_scorer,
        chosen_row,
    ) = chosen

    hidden = tensors[1].float() + tensors[2].float()
    hidden = hidden + tensors[3].float()
    reciprocal_rms = torch.rsqrt(
        hidden.square().mean(dim=-1, keepdim=True) + float(manifest["epsilon"])
    )
    reference_output = (
        tensors[0].float() * hidden * reciprocal_rms
    )[:, -int(manifest["completion_plus_one"]):, :].to(torch.float16).contiguous()
    reference_scorer = fixed_suffix(reference_output)

    old_logps = target["old_per_token_logps"].to("cuda")
    advantages = target["advantages"].to("cuda").unsqueeze(1)

    def decisions(logps: Any) -> Any:
        ratio = torch.exp(logps - old_logps)
        return ((ratio < 0.8) & (advantages < 0)) | ((ratio > 1.2) & (advantages > 0))

    compiled_decisions = decisions(compiled_scorer)
    reference_decisions = decisions(reference_scorer)
    upward = (~compiled_decisions) & reference_decisions
    downward = compiled_decisions & (~reference_decisions)
    count = compiled_decisions.numel()
    local_metrics = metrics(torch, reference_output, compiled_outputs[0])
    semantic_disagreement = float((upward.sum() + downward.sum()).item() / count)
    result = {
        "schema_version": "forkcert.original-kernel-group-replay.v0.1",
        "valid": True,
        "case_id": manifest["case_id"],
        "state_id": manifest["state_id"],
        "kernel_group": {
            "reduction_kernel": manifest["reduction_kernel"],
            "pointwise_kernel": manifest["pointwise_kernel"],
            "original_kernel_id": kernel_row["kernel_id"],
            "provenance": kernel_row,
        },
        "same_input_production": {
            "captured_launch_inputs": [tensor_fingerprint(tensor) for tensor in tensors[:-1]],
            "compiled_repeat_exact": chosen_row["compiled_repeat_exact"],
            "reduction_repeat_exact": chosen_row["reduction_repeat_exact"],
            "reduction_replay_matches_captured": chosen_row[
                "reduction_replay_matches_captured"
            ],
            "compiled_candidate_scorer_anchor_exact": chosen_row[
                "candidate_scorer_anchor_exact"
            ],
            "reference_expression": manifest["reference_expression"],
            "output_discrepancy": local_metrics,
            "interpretation": production_mediation_interpretation(
                local_metrics["nonzero"] > 0, None
            ),
        },
        "fixed_suffix_mediation": {
            "suffix": manifest["fixed_suffix"],
            "compiled_scorer_sha256": tensor_sha256(compiled_scorer),
            "reference_fragment_scorer_sha256": tensor_sha256(reference_scorer),
            "continuous_scorer_discrepancy": metrics(
                torch, compiled_scorer, reference_scorer
            ),
            "compiled_clip_count": int(compiled_decisions.sum().item()),
            "reference_fragment_clip_count": int(reference_decisions.sum().item()),
            "off_to_on": int(upward.sum().item()),
            "on_to_off": int(downward.sum().item()),
            "semantic_disagreement": semantic_disagreement,
            "production_and_semantic_mediation": production_mediation_interpretation(
                local_metrics["nonzero"] > 0, semantic_disagreement > 0
            ),
        },
        "capture_candidates": candidates,
        "claim_scope": "original generated final-RMSNorm kernel-group forward production and fixed-suffix mediation at one matched state",
        "limitations": [
            "ATen reference expression is implementation-relative, not mathematical truth",
            "forward replay does not establish backward or update mediation",
            "one selected state does not establish population prevalence",
            "kernel-group evidence is not a compiler-pass root cause",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "valid": True,
        "output_discrepancy": local_metrics,
        "semantic_disagreement": semantic_disagreement,
        "compiled_candidate_scorer_anchor_exact": chosen_row["candidate_scorer_anchor_exact"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
