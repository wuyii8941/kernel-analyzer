#!/usr/bin/env python3
"""Evaluate the four Phi perturbation arms through one common FP32 loss path."""

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
    vals = []
    with torch.no_grad():
        for row in states:
            ids = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
            vals.append(float(model(input_ids=ids, labels=ids, use_cache=False).loss.float().cpu()))
    return sum(vals) / len(vals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--eval-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct"))
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("host GPU required")
    device = torch.device(args.device)
    arms = json.loads(args.arms.read_text()); bank = json.loads(args.eval_bank.read_text()); states = bank.get("states", bank.get("records"))
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager", local_files_only=True).to(device).eval()
    parameter = dict(model.named_parameters())[arms["updated_parameter"]]
    reference = bank_loss(model, states, device)
    pairs = {"A_operator": ("a_candidate", "a_repair"), "B_rng": ("b_seed0", "b_seed1"), "C_data_order": ("c_order0", "c_order1"), "D_precision": ("d_bf16", "d_fp32")}
    rows = []
    for arm, names in pairs.items():
        losses = {}
        for name in names:
            meta = arms["final_masters"][name]; path = Path(meta["path"])
            if digest(path) != meta["sha256"]: raise RuntimeError(f"master digest mismatch: {name}")
            value = torch.load(path, map_location="cpu", weights_only=True).float()
            with torch.no_grad(): parameter.copy_(value.to(device, dtype=parameter.dtype))
            losses[name] = bank_loss(model, states, device)
        rows.append({"arm": arm, "left": names[0], "right": names[1], "left_loss": losses[names[0]], "right_loss": losses[names[1]], "loss_gap": losses[names[0]] - losses[names[1]], "absolute_loss_gap": abs(losses[names[0]] - losses[names[1]]), "coherence_amplification": arms["arms"][arm]["coherence_amplification"]})
    payload = {"schema":"kernel-analyzer-phi-four-scale-loss-v1","status":"COMPLETE_UNSEEN_FP32_EVALUATION","evaluation_bank":str(args.eval_bank),"evaluation_bank_sha256":digest(args.eval_bank),"evaluation_state_count":len(states),"reference_checkpoint_loss":reference,"rows":rows,"claim_boundary":"Loss is a downstream consequence measured on a common unseen FP32 bank; it does not define coherence or relabel arms."}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print(json.dumps({"status":payload["status"],"rows":len(rows),"reference_loss":reference}))


if __name__ == "__main__": main()
