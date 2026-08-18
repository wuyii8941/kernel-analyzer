#!/usr/bin/env python3
"""Audit whether frozen Triton FP32-storage replays respected compiled pointer ABIs."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POINTER = re.compile(r"'[^']+': '\*([^']+)'")


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    cells = []
    totals = {"campaign_rows": 0, "invalid_campaign_rows": 0, "positive_rows": 0, "invalid_positive_rows": 0}
    for model in ("qwen", "mamba", "phi4", "deepseek8b"):
        for seq_len in (64, 128, 256):
            release = ROOT / f"results/coverage/runtime_releases/{model}_seq{seq_len}_r1"
            campaign = load(release / "campaign.json.gz")
            oracle = load(release / "triton_oracle.json.gz")
            sources = "\n".join(
                path.read_text()
                for path in sorted((release / "trace").glob("*/output_code.py"))
            )
            symbol_invalid: dict[str, bool] = {}
            unresolved = []
            for row in campaign["rows"]:
                symbol = row["symbol"]
                if symbol in symbol_invalid:
                    continue
                marker = f"{symbol} = async_compile.triton"
                start = sources.find(marker)
                if start < 0:
                    unresolved.append(symbol)
                    continue
                block = sources[start:start + 12000]
                pointer_types = POINTER.findall(block)
                if not pointer_types:
                    unresolved.append(symbol)
                    continue
                symbol_invalid[symbol] = any(dtype in {"bf16", "fp16"} for dtype in pointer_types)
            invalid_campaign = sum(symbol_invalid.get(row["symbol"], False) for row in campaign["rows"])
            positives = [row for row in oracle["rows"] if row["verdict"] == "DIRECTIONAL_BIAS_SCREEN_POSITIVE"]
            invalid_positives = sum(symbol_invalid.get(row["function"], False) for row in positives)
            cell = {
                "model": model,
                "sequence_length": seq_len,
                "campaign_rows": len(campaign["rows"]),
                "invalid_reference_abi_campaign_rows": invalid_campaign,
                "directional_screen_positive_rows": len(positives),
                "invalid_reference_abi_positive_rows": invalid_positives,
                "unresolved_symbol_count": len(set(unresolved)),
            }
            cells.append(cell)
            totals["campaign_rows"] += cell["campaign_rows"]
            totals["invalid_campaign_rows"] += invalid_campaign
            totals["positive_rows"] += len(positives)
            totals["invalid_positive_rows"] += invalid_positives
    output = {
        "schema": "kernel-analyzer-triton-reference-abi-audit-v1",
        "status": "INVALID_REFERENCE_ABI",
        "finding": (
            "The frozen observer passed FP32 allocations to already compiled Triton programs "
            "whose triton_meta pointer signatures remained *bf16/*fp16. This is byte "
            "reinterpretation, not an FP32 implementation of the generated program."
        ),
        "decisive_counterexample": {
            "candidate": "mamba_seq128_backward_10610_out_ptr0",
            "program": "out[x,y] = float32(in[y,x]) followed by store",
            "compiled_signature": {"in_ptr0": "*bf16", "out_ptr0": "*bf16"},
            "mathematical_expectation": (
                "A pure reorder/copy has zero same-value FP32-to-BF16 cast difference."
            ),
            "observed_invalid_repair": (
                "196552-196573 of 196608 coordinates changed and a parameter gradient changed; "
                "therefore the replay interpreted memory under the wrong ABI."
            ),
            "invalid_artifact": "results/coverage/mamba_seq128_backward10610_causal.json",
        },
        "totals": totals,
        "cells": cells,
        "required_action": (
            "Do not use frozen Triton FP32 screen verdicts or interventions as case evidence. "
            "Regenerate a semantically equivalent Triton program with FP32 pointer signatures, "
            "or use an independent analytic/reference implementation at a typed boundary."
        ),
        "claim_boundary": (
            "This invalidates the FP32-storage Triton reference, not the static invocation "
            "denominator or the non-Triton typed-operation screens."
        ),
    }
    output["result_sha256"] = canonical_hash(output)
    path = ROOT / "results/coverage/triton_reference_abi_audit.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "totals": totals}))


if __name__ == "__main__":
    main()
