#!/usr/bin/env python3
"""Test Phi lm-head dX error in the same-state final-norm gradient frame."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch._inductor.codecache import PyCodeCache


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT), str(ROOT / "src"), str(ROOT / "archive/round1_code/src")
]

from kernel_analyzer.reference_relative_oracle import (  # noqa: E402
    ReferenceRelativeObservation,
    certify_reference_relative,
)
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import load_model  # noqa: E402
from scripts.run_phi_mm_transport_intervention import run_branch  # noqa: E402


CARRIER = "model.norm.weight"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/property/bias_oracle_recovery/phi_reference_relative.json",
    )
    args = parser.parse_args()
    if args.states < 4 or args.states > 16:
        raise ValueError("Phi reference-relative screen supports 4--16 states")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    bank = json.loads((
        ROOT / "results/coverage/phi4_seq64_input_bank.json"
    ).read_text(encoding="utf-8"))
    states = bank["states"][16:16 + args.states]
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(
        "phi", Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct"), device
    )
    start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend="inductor", fullgraph=False, dynamic=False
    )
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    if not modules:
        raise RuntimeError("compiled Phi step produced no executable modules")
    observations = []
    rows = []
    empty_transport = np.empty((0, 0), dtype=np.float32)
    for index, state in enumerate(states):
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        seed = 64000 + index
        standard = run_branch(
            model, candidate, values, modules, seed, None, None, empty_transport
        )
        repair = run_branch(
            model, candidate, values, modules, seed,
            "REPAIR_FP32_CAST_BF16", None, empty_transport,
        )
        if standard["loss_digest"] != repair["loss_digest"]:
            raise RuntimeError("Phi repair changed the loss")
        sham_exact = None
        if index == 0:
            sham = run_branch(
                model, candidate, values, modules, seed,
                "SHAM", None, empty_transport,
            )
            sham_exact = bool(
                sham["loss_digest"] == standard["loss_digest"]
                and np.array_equal(sham["gradient"], standard["gradient"])
            )
            if not sham_exact:
                raise RuntimeError("Phi same-implementation sham changed the carrier")
        error = standard["gradient"].astype(np.float64) - repair["gradient"].astype(np.float64)
        reference = repair["gradient"].astype(np.float64)
        observation = ReferenceRelativeObservation(
            condition_id=str(state.get("state_id", state.get("sequence_id", index + 16))),
            error_reference_dot=float(np.dot(error, reference)),
            error_energy=float(np.dot(error, error)),
            reference_energy=float(np.dot(reference, reference)),
        )
        observations.append(observation)
        rows.append({**observation.as_dict(), "sham_exact": sham_exact})
        print(json.dumps({
            "event": "STATE_COMPLETE", "index": index,
            "coefficient": observation.coefficient,
            "cosine": observation.cosine,
        }, sort_keys=True), flush=True)
        del values, standard, repair, error, reference
        torch.cuda.empty_cache()
    certificate = certify_reference_relative(observations)
    payload = {
        "schema": "kernel-analyzer-phi-reference-relative-screen-v1",
        "case_id": "phi4_seq64_lmhead_dx",
        "reference_frame": (
            "same-state model.norm.weight gradient under the FP32-MM/BF16-ABI repair"
        ),
        "binding": {
            "same_process_candidate_and_repair": True,
            "target_identity": "unique backward MM with left operand shape [64,200064]",
            "old_frozen_wrapper_byte_identity_required": False,
        },
        "rows": rows,
        "certificate": certificate.as_dict(),
        "status": certificate.status,
        "claim_boundary": (
            "development recovery measurement; this tests a reference-relative "
            "gradient component and does not close the previously missing analytic "
            "token-level transport factors"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output), "status": payload["status"],
        "certificate": certificate.as_dict(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
