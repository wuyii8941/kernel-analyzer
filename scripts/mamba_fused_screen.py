#!/usr/bin/env python3
"""Candidate-blind screen of official fused Mamba F+B versus recurrence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import MambaForCausalLM
from transformers.models.mamba import modeling_mamba


ROOT = Path(__file__).resolve().parents[1]
FAST_GLOBALS = {
    "selective_scan_fn": modeling_mamba.selective_scan_fn,
    "mamba_inner_fn": modeling_mamba.mamba_inner_fn,
    "selective_state_update": modeling_mamba.selective_state_update,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/state-spaces/mamba-130m-hf"))
    parser.add_argument("--input-bank", type=Path, default=ROOT / "results/mamba_scan/input_bank.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/mamba_scan/fused_screen.json")
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--calibration-states", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def set_fused(enabled: bool) -> None:
    for name, value in FAST_GLOBALS.items():
        setattr(modeling_mamba, name, value if enabled else None)


def targets(model: MambaForCausalLM) -> dict[str, torch.nn.Parameter]:
    suffixes = (
        ".mixer.A_log",
        ".mixer.D",
        ".mixer.x_proj.weight",
        ".mixer.dt_proj.weight",
        ".mixer.dt_proj.bias",
    )
    return {name: parameter for name, parameter in model.named_parameters() if name.endswith(suffixes)}


def run(
    model: MambaForCausalLM,
    selected: dict[str, torch.nn.Parameter],
    input_ids: torch.Tensor,
    fused: bool,
) -> tuple[float, dict[str, torch.Tensor]]:
    set_fused(fused)
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    loss.backward()
    gradients = {}
    for name, parameter in selected.items():
        if parameter.grad is None:
            raise RuntimeError(f"missing actual gradient: {name}")
        gradients[name] = parameter.grad.detach().float().cpu().clone()
    return float(loss.detach().cpu()), gradients


def main() -> None:
    args = parse_args()
    if any(value is None for value in FAST_GLOBALS.values()):
        raise RuntimeError("official fused Mamba globals are unavailable")
    bank = json.loads(args.input_bank.read_text())
    if args.states > len(bank["states"]):
        raise ValueError("input bank too small")
    model = MambaForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    ).to(args.device).train()
    model.config.use_cache = False
    selected = targets(model)
    if len(selected) != len(model.backbone.layers) * 5:
        raise RuntimeError(f"unexpected target denominator: {len(selected)}")

    state_rows = []
    deltas = []
    runtime_deltas = []
    repeat_gates = {}
    for state_id in range(args.states):
        bank_row = bank["states"][state_id]
        cpu_ids = torch.tensor(bank_row["token_ids"], dtype=torch.long)
        if hashlib.sha256(cpu_ids.numpy().tobytes()).hexdigest() != bank_row["token_sha256"]:
            raise RuntimeError(f"token digest mismatch: {state_id}")
        input_ids = cpu_ids.unsqueeze(0).to(args.device)
        reference_loss, reference = run(model, selected, input_ids, fused=False)
        candidate_loss_1, candidate_1 = run(model, selected, input_ids, fused=True)
        candidate_loss_2, candidate_2 = run(model, selected, input_ids, fused=True)
        candidate_mean = {
            name: (candidate_1[name].double() + candidate_2[name].double()) / 2
            for name in selected
        }
        delta = {name: candidate_mean[name] - reference[name].double() for name in selected}
        runtime_delta = {
            name: candidate_2[name].double() - candidate_1[name].double()
            for name in selected
        }
        deltas.append(delta)
        runtime_deltas.append(runtime_delta)
        if state_id == 0:
            rr_loss, rr = run(model, selected, input_ids, fused=False)
            repeat_gates = {
                "reference_loss_exact": rr_loss == reference_loss,
                "candidate_loss_exact": candidate_loss_1 == candidate_loss_2,
                "reference_gradients_bitwise_exact": all(torch.equal(reference[name], rr[name]) for name in selected),
                "candidate_gradients_bitwise_exact": all(torch.equal(candidate_1[name], candidate_2[name]) for name in selected),
            }
        signal_l2 = torch.sqrt(sum(torch.sum(value ** 2) for value in delta.values()))
        runtime_l2 = torch.sqrt(sum(torch.sum(value ** 2) for value in runtime_delta.values()))
        state_rows.append({
            "state_id": state_id,
            "split": "CALIBRATION" if state_id < args.calibration_states else "HELDOUT",
            "token_sha256": bank_row["token_sha256"],
            "reference_loss": reference_loss,
            "candidate_loss_repeats": [candidate_loss_1, candidate_loss_2],
            "candidate_mean_loss_delta": (candidate_loss_1 + candidate_loss_2) / 2 - reference_loss,
            "all_finite": all(torch.isfinite(value).all().item() for value in delta.values()),
            "changed_parameters": sum(bool(torch.count_nonzero(value)) for value in delta.values()),
            "global_scan_parameter_delta_l2": float(signal_l2),
            "candidate_runtime_delta_l2": float(runtime_l2),
            "runtime_over_signal_l2": float(runtime_l2 / signal_l2) if signal_l2 > 0 else None,
        })
        del input_ids, reference, candidate_1, candidate_2, candidate_mean
        torch.cuda.empty_cache()

    parameter_rows = []
    for name in selected:
        direction_raw = sum(deltas[index][name] for index in range(args.calibration_states))
        norm = torch.linalg.vector_norm(direction_raw.double())
        if norm == 0:
            continue
        direction = direction_raw.double() / norm
        projections = [float(torch.sum(delta[name].double() * direction)) for delta in deltas]
        runtime_projections = [
            float(torch.sum(delta[name].double() * direction)) for delta in runtime_deltas
        ]
        heldout = projections[args.calibration_states :]
        parameter_rows.append({
            "parameter": name,
            "numel": direction.numel(),
            "calibration_projections": projections[: args.calibration_states],
            "heldout_projections": heldout,
            "heldout_positive": sum(value > 0 for value in heldout),
            "heldout_negative": sum(value < 0 for value in heldout),
            "heldout_mean": sum(heldout) / len(heldout),
            "heldout_min": min(heldout),
            "heldout_max": max(heldout),
            "max_abs_runtime_projection": max(abs(value) for value in runtime_projections),
            "runtime_projection_over_abs_heldout_mean": (
                max(abs(value) for value in runtime_projections) / abs(sum(heldout) / len(heldout))
                if sum(heldout) != 0 else None
            ),
            "persistent_positive": all(value > 0 for value in heldout),
            "persistent_negative": all(value < 0 for value in heldout),
        })
    parameter_rows.sort(key=lambda row: abs(row["heldout_mean"]), reverse=True)
    output = {
        "schema": "kernel-analyzer-mamba-official-fused-screen-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "dtype": "bfloat16",
        "seq_len": bank["seq_len"],
        "states": args.states,
        "calibration_states": list(range(args.calibration_states)),
        "heldout_states": list(range(args.calibration_states, args.states)),
        "reference": "Transformers explicit sequential selective-state recurrence",
        "candidate": "official mamba_ssm mamba_inner_fn fused CUDA F+B plus causal_conv1d",
        "candidate_blind_direction_freeze": True,
        "target_parameter_denominator": len(selected),
        "repeat_gates": repeat_gates,
        "state_rows": state_rows,
        "parameter_rows": parameter_rows,
        "persistent_positive_count": sum(row["persistent_positive"] for row in parameter_rows),
        "persistent_negative_count": sum(row["persistent_negative"] for row in parameter_rows),
        "tensor_values_saved": False,
        "claim_boundary": "SCREEN_ONLY_REQUIRES_INDEPENDENT_CONFIRMATION_LOCAL_FB_INTERVENTION_AND_LIVE_WEIGHT",
    }
    output["result_sha256"] = canonical_hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "persistent_positive": output["persistent_positive_count"],
        "persistent_negative": output["persistent_negative_count"],
        "top": parameter_rows[:5],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
