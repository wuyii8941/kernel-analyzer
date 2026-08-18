#!/usr/bin/env python3
"""Independent confirmation of every persistent Mamba scan screen direction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, MambaForCausalLM

from mamba_scan_screen import canonical_hash, natural_windows, run_arm, target_parameters


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/state-spaces/mamba-130m-hf"))
    parser.add_argument(
        "--validation-arrow",
        type=Path,
        default=Path(
            "/data1/tzh/cache/huggingface/datasets/Salesforce___wikitext/"
            "wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/"
            "wikitext-validation.arrow"
        ),
    )
    parser.add_argument("--screen", type=Path, default=ROOT / "results/mamba_scan/screen.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/mamba_scan/family_confirmation.json")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--confirmation-states", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    screen = json.loads(args.screen.read_text())
    selected = [
        row for row in screen["parameter_rows"]
        if row["persistent_positive"] or row["persistent_negative"]
    ]
    selected_names = {row["parameter"] for row in selected}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    total = 8 + args.confirmation_states
    windows = natural_windows(tokenizer, args.validation_arrow, args.seq_len, total)
    for state_id in range(8):
        digest = hashlib.sha256(windows[state_id].numpy().tobytes()).hexdigest()
        if digest != screen["state_rows"][state_id]["token_sha256"]:
            raise RuntimeError(f"screen token provenance mismatch at {state_id}")

    model = MambaForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, local_files_only=True
    ).to(args.device).train()
    model.config.use_cache = False
    all_targets = target_parameters(model)
    targets = {name: all_targets[name] for name in selected_names}

    deltas: dict[int, dict[str, torch.Tensor]] = {}
    run_states = list(screen["calibration_states"]) + list(range(8, total))
    repeat_exact = None
    for state_id in run_states:
        ids = windows[state_id].unsqueeze(0).to(args.device)
        reference_loss, reference = run_arm(model, targets, ids, parallel=False)
        candidate_loss, candidate = run_arm(model, targets, ids, parallel=True)
        deltas[state_id] = {name: candidate[name] - reference[name] for name in targets}
        if state_id == 8:
            repeat_loss, repeat = run_arm(model, targets, ids, parallel=True)
            repeat_exact = candidate_loss == repeat_loss and all(
                torch.equal(candidate[name], repeat[name]) for name in targets
            )
        del ids
        torch.cuda.empty_cache()

    rows = []
    for screen_row in selected:
        name = screen_row["parameter"]
        direction_raw = sum(deltas[index][name] for index in screen["calibration_states"])
        norm = torch.linalg.vector_norm(direction_raw.double())
        direction = direction_raw.double() / norm
        calibration = [
            float(torch.sum(deltas[index][name].double() * direction))
            for index in screen["calibration_states"]
        ]
        if max(abs(left - right) for left, right in zip(calibration, screen_row["calibration_projections"])) > 1e-15:
            raise RuntimeError(f"direction reconstruction mismatch: {name}")
        confirmation = [
            float(torch.sum(deltas[index][name].double() * direction))
            for index in range(8, total)
        ]
        expected_positive = screen_row["persistent_positive"]
        directional_pass = (
            all(value > 0 for value in confirmation)
            if expected_positive else all(value < 0 for value in confirmation)
        )
        rows.append({
            "parameter": name,
            "screen_direction": "POSITIVE" if expected_positive else "NEGATIVE",
            "confirmation_projections": confirmation,
            "confirmation_positive": sum(value > 0 for value in confirmation),
            "confirmation_negative": sum(value < 0 for value in confirmation),
            "confirmation_mean": sum(confirmation) / len(confirmation),
            "directional_pass": directional_pass,
            "one_sided_sign_p_if_pass": 2.0 ** (-len(confirmation)) if directional_pass else None,
            "holm_pass_original_family": directional_pass and 2.0 ** (-len(confirmation)) < 0.05 / screen["target_parameter_denominator"],
        })

    output = {
        "schema": "kernel-analyzer-mamba-scan-family-confirmation-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "dtype": "float32",
        "seq_len": args.seq_len,
        "screen_family_size": screen["target_parameter_denominator"],
        "screen_persistent_candidates": len(selected),
        "calibration_states": screen["calibration_states"],
        "discovery_states": screen["heldout_states"],
        "independent_confirmation_states": list(range(8, total)),
        "candidate_repeat_exact": repeat_exact,
        "rows": rows,
        "confirmed_count": sum(row["holm_pass_original_family"] for row in rows),
        "natural_bias_case_added": False,
        "tensor_values_saved": False,
        "claim_boundary": "A_DIRECTION_MUST_PASS_INDEPENDENT_CONFIRMATION_BEFORE_LOCAL_FB_AND_LIVE_WEIGHT_TESTS",
    }
    output["result_sha256"] = canonical_hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "confirmed": output["confirmed_count"], "rows": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
