#!/usr/bin/env python3
"""Summarize all frozen pairs; incomplete pairs cannot silently disappear."""
import json
import math
import statistics
import torch
from scripts.run_liger_language_followup import OUT
from scripts.run_training_numerical_v2 import save_new


def main():
    plan = json.loads((OUT / "plan.json").read_text())
    rows = []
    for pair in range(plan["pairs"]):
        root = OUT / f"pair{pair}"
        statuses = [root / condition / "status.json" for condition in ("candidate", "reference")]
        if not all(p.exists() for p in statuses):
            raise SystemExit(f"Pair {pair} is incomplete; no complete-set summary emitted")
        c, r = [json.loads(p.read_text()) for p in statuses]
        if any(p["status"] != "COMPLETE_DEVELOPMENT_PILOT" for p in (c, r)):
            raise SystemExit(f"Pair {pair} did not complete normally; inspect retained failure")
        if c["initial_parameters_sha256"] != r["initial_parameters_sha256"]:
            raise RuntimeError("Pair does not share initial parameters")
        lines = [[json.loads(s) for s in (root / condition / "steps.jsonl").read_text().splitlines()]
                 for condition in ("candidate", "reference")]
        if any(len(l) != plan["steps"] for l in lines):
            raise RuntimeError("Training horizon differs from frozen plan")
        if any(x["step"] != y["step"] or x["offsets"] != y["offsets"] for x, y in zip(*lines)):
            raise RuntimeError("Pair data or step order differs")
        saved = [torch.load(root / condition / "final.pt", map_location="cpu", weights_only=True)
                 for condition in ("candidate", "reference")]
        effect = base = 0.0
        for name, value in saved[1]["master"].items():
            effect += float((saved[0]["master"][name].double() - value.double()).square().sum())
            base += float(value.double().square().sum())
        checkpoints = []
        for ec, er in zip(c["evaluations"], r["evaluations"]):
            if ec["step"] != er["step"]:
                raise RuntimeError("Evaluation steps differ")
            checkpoints.append({"step": ec["step"], "candidate_minus_reference_loss":
                                ec["shared_evaluation_loss"] - er["shared_evaluation_loss"]})
        if checkpoints[-1]["step"] != plan["steps"]:
            raise RuntimeError("Missing primary evaluation endpoint")
        rows.append({"pair": pair, "initialization_seed": plan["initialization_seeds"][pair],
                     "primary_loss_gap": checkpoints[-1]["candidate_minus_reference_loss"],
                     "evaluation_gaps": checkpoints,
                     "parameter_relative_l2": math.sqrt(effect / base),
                     "training_loss_differing_steps": sum(x["loss"] != y["loss"] for x, y in zip(*lines)),
                     "candidate_seconds": c["elapsed_seconds_including_evaluation"],
                     "reference_seconds": r["elapsed_seconds_including_evaluation"]})
        del saved
    gaps = [row["primary_loss_gap"] for row in rows]
    save_new(OUT / "summary.json", {
        "status": "COMPLETE_FIXED_PAIRED_DEVELOPMENT_RUN_SET", "rows": rows,
        "mean_primary_loss_gap": statistics.mean(gaps),
        "sample_sd_primary_loss_gap": statistics.stdev(gaps),
        "persistent_bias": "NOT_DETERMINED_BY_LOSS_GAPS",
        "quality_benefit": "NOT_ESTABLISHED_BY_A_POWERED_CONFIRMATION",
        "scope": plan["population_scope"],
        "claim_boundary": "Small four-layer language model using WikiText and real tokenizer; varying initialization with fixed data order, not pretrained full-size LLM fine-tuning or a universal quality claim.",
    })


if __name__ == "__main__":
    main()
