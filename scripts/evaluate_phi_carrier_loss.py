#!/usr/bin/env python3
"""Evaluate saved Phi carrier masters on an unseen fixed input bank.

The candidate and repair vectors come from the same 32-step one-carrier
trajectory.  This script only measures downstream loss; it never changes the
coherence verdict or promotes a carrier based on loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loss_on_bank(model, parameter, states, device: torch.device) -> float:
    values = []
    with torch.no_grad():
        for row in states:
            ids = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
            out = model(input_ids=ids, labels=ids, use_cache=False)
            values.append(float(out.loss.detach().float().cpu()))
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", type=Path, required=True)
    parser.add_argument("--eval-bank", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("loss evaluation requires the host GPU")
    distribution = json.loads(args.distribution.read_text())
    bank = json.loads(args.eval_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if len(states) < 2:
        raise RuntimeError("unseen evaluation bank is incomplete")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).to(device).eval()
    parameters = dict(model.named_parameters())
    rows = []
    # Original FP32 checkpoint is a reference only; candidate/repair deltas are
    # always compared to each other under exactly the same evaluation path.
    baseline = loss_on_bank(model, parameters[distribution["rows"][0]["carrier"]], states, device)
    for item in distribution["rows"]:
        name = item["carrier"]
        parameter = parameters[name]
        masters = item.get("final_masters")
        if not masters:
            raise RuntimeError(f"missing saved final masters for {name}")
        candidate_path = Path(masters["candidate_path"])
        repair_path = Path(masters["repair_path"])
        if digest(candidate_path) != masters["candidate_sha256"] or digest(repair_path) != masters["repair_sha256"]:
            raise RuntimeError(f"final master digest mismatch for {name}")
        candidate = torch.load(candidate_path, map_location="cpu", weights_only=True).float()
        repair = torch.load(repair_path, map_location="cpu", weights_only=True).float()
        with torch.no_grad():
            parameter.copy_(candidate.to(device, dtype=parameter.dtype))
        candidate_loss = loss_on_bank(model, parameter, states, device)
        with torch.no_grad():
            parameter.copy_(repair.to(device, dtype=parameter.dtype))
        repair_loss = loss_on_bank(model, parameter, states, device)
        rows.append({
            **{key: item[key] for key in ("index", "carrier", "stratum", "layer_index")},
            "candidate_loss": candidate_loss,
            "repair_loss": repair_loss,
            "candidate_minus_repair": candidate_loss - repair_loss,
            "absolute_loss_gap": abs(candidate_loss - repair_loss),
            "coherence_amplification": item["measurement"]["coherence_amplification"],
            "candidate_master_sha256": masters["candidate_sha256"],
            "repair_master_sha256": masters["repair_sha256"],
        })
    payload = {
        "schema": "kernel-analyzer-phi-carrier-loss-v1",
        "status": "COMPLETE_UNSEEN_FP32_EVALUATION",
        "model": str(args.model),
        "evaluation_bank": str(args.eval_bank),
        "evaluation_state_count": len(states),
        "evaluation_bank_sha256": digest(args.eval_bank),
        "evaluation_dtype": "float32",
        "reference_checkpoint_loss": baseline,
        "rows": rows,
        "claim_boundary": (
            "Loss is a downstream consequence measured on an unseen fixed bank. "
            "It is not used to define or relabel persistence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "reference_loss": baseline}))


if __name__ == "__main__":
    main()
