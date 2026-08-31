#!/usr/bin/env python3
"""Build frozen 32-state banks for the three mechanism-diversity profiles."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/three_mechanism_profiles_v1/input_banks"

CASES = {
    "gemma4_text128_scan_0037": (
        ROOT / "results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128.json",
        ROOT / "results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128_trajectory4096.json",
    ),
    "llama32_text128_scan_0000": (
        ROOT / "results/property/tcmp_allop_v1/input_banks/llama32_3b_text128.json",
        ROOT / "results/property/tcmp_allop_v1/input_banks/llama32_3b_text128_trajectory4096.json",
    ),
    "llama32_text512_scan_extern_dc4ef40f35eb": (
        ROOT / "results/property/tcmp_allop_v1/input_banks/llama32_3b_text512.json",
        ROOT / "results/property/three_mechanism_profiles_v1/input_banks/llama32_3b_text512_natural32.json",
    ),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for case_id, (base_path, trajectory_path) in CASES.items():
        base = json.loads(base_path.read_text())
        trajectory = json.loads(trajectory_path.read_text())
        engineering = [dict(row) for row in base["states"] if row.get("role") == "ENGINEERING"][:2]
        selected = []
        seen = set()
        for row in trajectory["states"]:
            digest = row.get("token_sha256")
            if digest in seen:
                continue
            seen.add(digest)
            item = dict(row)
            item["role"] = "CONFIRMATION"
            item["order_within_role"] = len(selected)
            selected.append(item)
            if len(selected) == 32:
                break
        if len(engineering) < 1 or len(selected) != 32:
            raise RuntimeError(f"{case_id}: cannot freeze 32 distinct states")
        payload = {
            "schema": "kernel-analyzer-three-mechanism-profile-bank-v1",
            "case_id": case_id,
            "split": {"calibration": [x["state_id"] for x in selected[:16]],
                      "confirmation": [x["state_id"] for x in selected[16:]]},
            "source_base_bank": str(base_path.relative_to(ROOT)),
            "source_trajectory_bank": str(trajectory_path.relative_to(ROOT)),
            "selection_rule": "first 32 distinct token_sha256 values in frozen trajectory order",
            "states": engineering + selected,
        }
        (OUT / f"{case_id}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
