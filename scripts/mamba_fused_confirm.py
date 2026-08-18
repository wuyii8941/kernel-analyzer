#!/usr/bin/env python3
"""Independently confirm frozen-direction Mamba fused F+B screen hits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import MambaForCausalLM
from transformers.models.mamba import modeling_mamba

from mamba_fused_screen import FAST_GLOBALS, canonical_hash, run


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/state-spaces/mamba-130m-hf"))
    parser.add_argument("--input-bank", type=Path, default=ROOT / "results/mamba_scan/input_bank_seq256.json")
    parser.add_argument("--screen", type=Path, default=ROOT / "results/mamba_scan/fused_screen_seq256.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/mamba_scan/fused_confirm_seq256.json")
    parser.add_argument("--confirmation-start", type=int, default=8)
    parser.add_argument("--confirmation-states", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(value is None for value in FAST_GLOBALS.values()):
        raise RuntimeError("official fused Mamba globals are unavailable")
    screen = json.loads(args.screen.read_text())
    bank = json.loads(args.input_bank.read_text())
    selected_rows = [row for row in screen["parameter_rows"] if row["persistent_positive"]]
    if not selected_rows:
        raise RuntimeError("screen contains no persistent-positive hit")
    selected_names = {row["parameter"] for row in selected_rows}
    calibration_states = screen["calibration_states"]
    confirmation_states = list(range(args.confirmation_start, args.confirmation_start + args.confirmation_states))
    used_by_screen = set(calibration_states) | set(screen["heldout_states"])
    if used_by_screen.intersection(confirmation_states):
        raise RuntimeError("confirmation states overlap screen states")
    if max(confirmation_states) >= len(bank["states"]):
        raise RuntimeError("input bank too small")

    model = MambaForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    ).to(args.device).train()
    model.config.use_cache = False
    named = dict(model.named_parameters())
    selected = {name: named[name] for name in selected_names}

    state_ids = calibration_states + confirmation_states
    mean_deltas: dict[int, dict[str, torch.Tensor]] = {}
    runtime_deltas: dict[int, dict[str, torch.Tensor]] = {}
    state_rows = []
    for state_id in state_ids:
        bank_row = bank["states"][state_id]
        cpu_ids = torch.tensor(bank_row["token_ids"], dtype=torch.long)
        if hashlib.sha256(cpu_ids.numpy().tobytes()).hexdigest() != bank_row["token_sha256"]:
            raise RuntimeError(f"token digest mismatch: {state_id}")
        ids = cpu_ids.unsqueeze(0).to(args.device)
        reference_loss, reference = run(model, selected, ids, fused=False)
        candidate_loss_1, candidate_1 = run(model, selected, ids, fused=True)
        candidate_loss_2, candidate_2 = run(model, selected, ids, fused=True)
        mean_deltas[state_id] = {
            name: (candidate_1[name].double() + candidate_2[name].double()) / 2 - reference[name].double()
            for name in selected
        }
        runtime_deltas[state_id] = {
            name: candidate_2[name].double() - candidate_1[name].double()
            for name in selected
        }
        state_rows.append({
            "state_id": state_id,
            "split": "DIRECTION_CALIBRATION" if state_id in calibration_states else "INDEPENDENT_CONFIRMATION",
            "token_sha256": bank_row["token_sha256"],
            "reference_loss": reference_loss,
            "candidate_loss_repeats": [candidate_loss_1, candidate_loss_2],
            "candidate_loss_repeat_exact": candidate_loss_1 == candidate_loss_2,
            "all_finite": all(torch.isfinite(value).all().item() for value in mean_deltas[state_id].values()),
        })
        del ids, reference, candidate_1, candidate_2
        torch.cuda.empty_cache()

    rows = []
    for name in sorted(selected_names):
        raw = sum(mean_deltas[state_id][name] for state_id in calibration_states)
        norm = torch.linalg.vector_norm(raw)
        if norm == 0:
            raise RuntimeError(f"zero calibration direction: {name}")
        direction = raw / norm
        calibration = [float(torch.sum(mean_deltas[state_id][name] * direction)) for state_id in calibration_states]
        confirmation = [float(torch.sum(mean_deltas[state_id][name] * direction)) for state_id in confirmation_states]
        runtime = [float(torch.sum(runtime_deltas[state_id][name] * direction)) for state_id in confirmation_states]
        confirmed = all(value > 0 for value in confirmation)
        sign_p = 2.0 ** (-len(confirmation)) if confirmed else None
        mean = sum(confirmation) / len(confirmation)
        max_runtime = max(abs(value) for value in runtime)
        rows.append({
            "parameter": name,
            "calibration_projections": calibration,
            "confirmation_projections": confirmation,
            "confirmation_positive": sum(value > 0 for value in confirmation),
            "confirmation_negative": sum(value < 0 for value in confirmation),
            "confirmation_mean": mean,
            "confirmation_min": min(confirmation),
            "confirmation_max": max(confirmation),
            "max_abs_confirmation_runtime_projection": max_runtime,
            "runtime_projection_over_abs_confirmation_mean": max_runtime / abs(mean) if mean else None,
            "directional_pass": confirmed,
            "one_sided_sign_p_if_pass": sign_p,
            "holm_pass_original_family": confirmed and sign_p < 0.05 / screen["target_parameter_denominator"],
        })

    output = {
        "schema": "kernel-analyzer-mamba-official-fused-independent-confirmation-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "dtype": "bfloat16",
        "seq_len": bank["seq_len"],
        "screen_result_sha256": screen["result_sha256"],
        "screen_family_size": screen["target_parameter_denominator"],
        "screen_selected_count": len(rows),
        "direction_calibration_states": calibration_states,
        "screen_discovery_states_not_reused": screen["heldout_states"],
        "independent_confirmation_states": confirmation_states,
        "state_rows": state_rows,
        "rows": rows,
        "confirmed_count": sum(row["holm_pass_original_family"] for row in rows),
        "natural_bias_case_added": False,
        "tensor_values_saved": False,
        "claim_boundary": "CONFIRMED_DIRECTION_REQUIRES_LOCAL_ACTUAL_FB_INTERVENTION_AND_LIVE_WEIGHT_BEFORE_CASE",
    }
    output["result_sha256"] = canonical_hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "confirmed": output["confirmed_count"], "rows": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
