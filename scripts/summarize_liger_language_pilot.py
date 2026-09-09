#!/usr/bin/env python3
"""Check reference repeat and summarize the development pilot without inference."""
import json
import math
import torch
from scripts.run_liger_language_pilot import PILOT
from scripts.run_training_numerical_v2 import save_new


def tensor_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            tensor_state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            tensor_state_equal(x, y) for x, y in zip(left, right))
    return left == right


def main():
    conditions = ("candidate", "reference", "reference_repeat")
    reports = {c: json.loads((PILOT / c / "status.json").read_text()) for c in conditions}
    if not all(r["status"] == "COMPLETE_DEVELOPMENT_PILOT" for r in reports.values()):
        raise RuntimeError("Incomplete training pilot")
    saved = {c: torch.load(PILOT / c / "final.pt", map_location="cpu", weights_only=True) for c in conditions}
    same_initial = len({r["initial_parameters_sha256"] for r in reports.values()}) == 1
    repeat_state = tensor_state_equal(saved["reference"], saved["reference_repeat"])
    repeat_loss = reports["reference"]["evaluations"] == reports["reference_repeat"]["evaluations"]
    same_steps = len({x["steps"] for x in saved.values()}) == 1
    energy = reference_energy = 0.0
    for name, value in saved["reference"]["master"].items():
        effect = saved["candidate"]["master"][name].double() - value.double()
        energy += float(effect.square().sum())
        reference_energy += float(value.double().square().sum())
    gaps = []
    for c, r in zip(reports["candidate"]["evaluations"], reports["reference"]["evaluations"]):
        if c["step"] != r["step"]:
            raise RuntimeError("Unmatched evaluation checkpoints")
        gaps.append({"step": c["step"], "candidate_minus_reference_loss":
                     c["shared_evaluation_loss"] - r["shared_evaluation_loss"]})
    save_new(PILOT / "summary.json", {
        "status": "VALID_DEVELOPMENT_PILOT" if all((same_initial, repeat_state, repeat_loss, same_steps)) else "INVALID_CONTROL",
        "same_initial_parameters": same_initial, "same_step_count": same_steps,
        "reference_repeat_exact_parameters_and_optimizer": repeat_state,
        "reference_repeat_exact_evaluation": repeat_loss,
        "paired_loss_gaps": gaps,
        "parameter_relative_l2": math.sqrt(energy / reference_energy),
        "training_outcome": "SHORT_DEVELOPMENT_TRAJECTORY_ONLY",
        "independent_run_inference": "NOT_ASSESSED",
        "persistent_bias": "NOT_ESTABLISHED_BY_THIS_PILOT",
        "cost": {c: {k: reports[c][k] for k in (
            "elapsed_seconds_including_evaluation", "peak_allocated_bytes")} for c in conditions},
        "next_requirement": "Freeze independent paired-run design after development variance and mechanism checks; do not relabel this pilot confirmation.",
    })


if __name__ == "__main__":
    main()
