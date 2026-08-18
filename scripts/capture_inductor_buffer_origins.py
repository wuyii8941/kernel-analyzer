#!/usr/bin/env python3
"""Recompile one frozen cell and retain exact IR-buffer origin identities."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch._inductor.codecache import PyCodeCache


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))

from scripts.inductor_buffer_origins import InductorBufferOriginRecorder  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import load_model  # noqa: E402
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture", choices=("qwen", "mamba", "phi", "deepseek8"), required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-graph-breaks", action="store_true")
    args = parser.parse_args()

    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if not states:
        raise RuntimeError("input bank is empty")
    token_ids = states[0].get("token_ids", states[0].get("input_ids"))
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend="inductor",
        fullgraph=not args.allow_graph_breaks, dynamic=False,
    )
    values = torch.tensor([token_ids], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    with InductorBufferOriginRecorder() as recorder:
        loss = candidate(values)
        loss.backward()
    torch.cuda.synchronize(device)
    modules = wrapper_modules(list(PyCodeCache.modules[start:]))
    capture = json.loads((args.release_dir / "capture.json").read_text())
    validate_release(modules, capture)
    certificate = recorder.certificate()
    payload = {
        "schema": "kernel-analyzer-frozen-cell-buffer-origin-certificate-v1",
        "status": certificate["status"],
        "architecture": args.architecture,
        "sequence_length": len(token_ids),
        "release_capture_sha256": capture["result_sha256"],
        "input_bank_sha256": hashlib.sha256(args.input_bank.read_bytes()).hexdigest(),
        "compiler_capture": certificate,
        "gates": {
            "frozen_wrapper_bytes_exact": True,
            "scheduler_provenance_observed": (
                certificate["denominator"]["scheduler_records"] > 0
            ),
            "all_materialized_buffers_have_exact_origin": (
                certificate["status"] == "COMPLETE_EXACT_IR_BUFFER_ORIGIN_CAPTURE"
            ),
        },
        "claim_boundary": (
            "Exact compiler-carried materialized-buffer origin identities for the frozen "
            "candidate. Numerical same-dtype comparison is a separate gate."
        ),
    }
    payload["result_sha256"] = digest(payload)
    write(args.output, payload)
    print(json.dumps({
        "output": str(args.output), "status": payload["status"],
        **certificate["denominator"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
