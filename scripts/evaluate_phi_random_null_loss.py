#!/usr/bin/env python3
"""Evaluate one repeated-random-null arm on the common unseen FP32 loss bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bank_loss(model, states, device):
    values = []
    with torch.no_grad():
        for row in states:
            ids = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
            values.append(float(model(input_ids=ids, labels=ids, use_cache=False).loss.float().cpu()))
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-distribution", type=Path, required=True)
    parser.add_argument("--carrier", default="model.norm.weight")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--eval-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host GPU required")
    distribution = json.loads(args.random_distribution.read_text())
    row = next(row for row in distribution["rows"] if row["carrier"] == args.carrier)
    random = next(item for item in row["random_nulls"] if int(item["seed"]) == args.seed)
    if "final_master" not in random or row.get("final_masters") is None:
        raise RuntimeError("random and natural repair final masters were not retained")
    random_meta = random["final_master"]
    repair_meta = row["final_masters"]["repair_path"]
    repair_path = Path(repair_meta)
    if digest(repair_path) != row["final_masters"]["repair_sha256"]:
        raise RuntimeError("repair master digest mismatch")
    random_path = Path(random_meta["path"])
    if digest(random_path) != random_meta["sha256"]:
        raise RuntimeError("random master digest mismatch")
    bank = json.loads(args.eval_bank.read_text())
    states = bank.get("states", bank.get("records"))
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager", local_files_only=True,
    ).to(device).eval()
    parameter = dict(model.named_parameters())[args.carrier]
    reference_loss = bank_loss(model, states, device)
    values = {}
    for name, path in (("random_null", random_path), ("repair", repair_path)):
        master = torch.load(path, map_location="cpu", weights_only=True).float()
        with torch.no_grad():
            parameter.copy_(master.to(device, dtype=parameter.dtype))
        values[name] = bank_loss(model, states, device)
    payload = {
        "schema": "kernel-analyzer-phi-random-null-loss-v1",
        "status": "COMPLETE_UNSEEN_FP32_EVALUATION",
        "carrier": args.carrier,
        "seed": args.seed,
        "evaluation_bank": str(args.eval_bank),
        "evaluation_bank_sha256": digest(args.eval_bank),
        "reference_checkpoint_loss": reference_loss,
        "random_null_loss": values["random_null"],
        "repair_loss": values["repair"],
        "loss_gap_random_minus_repair": values["random_null"] - values["repair"],
        "absolute_loss_gap_random_minus_repair": abs(values["random_null"] - values["repair"]),
        "random_null_A": float(random["coherence_amplification"]),
        "claim_boundary": "One RMS/support-matched repeated-random null arm on the same final-norm carrier and unseen FP32 loss bank; it is a downstream consequence check, not a formation label.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "random_null_A": payload["random_null_A"], "loss_gap": payload["loss_gap_random_minus_repair"]}))


if __name__ == "__main__":
    main()
