#!/usr/bin/env python3
"""Reuse the validated language runner for a fixed four-pair development set.

Initializations vary between pairs; data order is held fixed. These results
estimate development variability, not a powered quality-benefit confirmation.
"""
import argparse
import json
from pathlib import Path
import torch

from scripts import run_liger_language_pilot as pilot
from scripts.run_training_numerical_v2 import BASE, save_new
from scripts.run_liger_single_boundary_collapse import file_sha256

OUT = BASE / "language_training_followup"


def freeze():
    control = json.loads((pilot.PILOT / "summary.json").read_text())
    if control["status"] != "VALID_DEVELOPMENT_PILOT":
        raise SystemExit("The execution control has not passed")
    original = json.loads((pilot.PILOT / "protocol.json").read_text())
    seeds = [20261001, 20261002, 20261003, 20261004]
    save_new(OUT / "plan.json", {
        "status": "FROZEN_BEFORE_FOLLOWUP_RESULTS", "steps": 1024,
        "pairs": len(seeds), "initialization_seeds": seeds,
        "data_use": "DEVELOPMENT_VARIANCE_NOT_POWERED_CONFIRMATION",
        "selection": "All four fixed initializations retained, independent of sign or size of loss gap",
        "primary_endpoint": "Common FP32 evaluation loss at step 1024",
        "population_scope": "FIXED_PAIRED_RUN_SET_WITH_VARYING_INITIALIZATION_AND_FIXED_DATA_ORDER",
        "prediction": "Changing dW accumulation can change parameters and loss; no worsening or benefit direction is asserted",
        "no_early_stopping_on_loss_gap": True,
        "no_injection_or_repeated_within_batch_samples": True,
        "runner_sha256": file_sha256(Path(__file__)),
        "pilot_protocol_sha256": file_sha256(pilot.PILOT / "protocol.json"),
    })
    for pair, seed in enumerate(seeds):
        directory = OUT / f"pair{pair}"
        protocol = json.loads(json.dumps(original))
        protocol.update(steps=1024, eval_every=256,
                        purpose="Fixed independent-initialization paired development runs",
                        quality_claim="NOT_ASSESSED_NO_POWERED_CONFIRMATION")
        protocol["model"]["initialization_seed"] = seed
        save_new(directory / "protocol.json", protocol)
        for split in ("train", "validation"):
            (directory / f"{split}.npy").symlink_to((pilot.PILOT / f"{split}.npy").resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "train"))
    parser.add_argument("--pair", type=int, choices=range(4))
    parser.add_argument("--condition", choices=("candidate", "reference"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
        return
    if args.pair is None or args.condition is None:
        parser.error("train requires pair and condition")
    plan = json.loads((OUT / "plan.json").read_text())
    if plan["runner_sha256"] != file_sha256(Path(__file__)):
        raise SystemExit("Frozen followup driver changed")
    pilot.PILOT = OUT / f"pair{args.pair}"
    pilot.train(args.condition, torch.device(args.device))


if __name__ == "__main__":
    main()
