#!/usr/bin/env python3
"""Validate the v2.1 formation detector on synthetic controls only."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.bias_formation_v21 import FormationPolicy, FormationStatus, summarize_state_vectors  # noqa: E402

OUT = ROOT / "results/property/bias_formation/synthetic_controls.json"


def independent(seed: int, n: int = 16, d: int = 16) -> list[list[float]]:
    rng = random.Random(seed)
    return [[-1.0 if rng.randrange(2) else 1.0 for _ in range(d)] for _ in range(n)]


def run() -> dict:
    policy = FormationPolicy(min_states=16, bootstrap_samples=2000)
    cases = {
        "PURE_VARIANCE": independent(20260818),
        "DIRECTIONAL_BIAS": [[1.0] * 16 for _ in range(16)],
        "CANCELLATION": [[1.0] * 16 if index % 2 == 0 else [-1.0] * 16 for index in range(16)],
    }
    expected = {
        "PURE_VARIANCE": {FormationStatus.CENTERED.value},
        "DIRECTIONAL_BIAS": {FormationStatus.BIASED.value},
        "CANCELLATION": {
            FormationStatus.CENTERED.value,
            FormationStatus.CANCELING_STRUCTURE.value,
            FormationStatus.UNRESOLVED_INSUFFICIENT_STATES.value,
        },
    }
    rows = []
    for name, vectors in cases.items():
        cert = summarize_state_vectors(vectors, state_ids=[f"{name.lower()}_{i}" for i in range(16)], layer="synthetic", partition="control", policy=policy)
        rows.append({
            "control": name,
            "status": cert.status,
            "expected_status_set": sorted(expected[name]),
            "cross_state_u_statistic": cert.cross_state_u_statistic,
            "cross_state_ratio": cert.cross_state_ratio,
            "bootstrap_lower": cert.bootstrap_lower,
            "bootstrap_upper": cert.bootstrap_upper,
            "pass": cert.status in expected[name],
        })
    return {
        "schema": "kernel-analyzer-bias-formation-synthetic-controls-v1",
        "status": "PASS" if all(row["pass"] for row in rows) else "FAIL",
        "scientific_natural_case_evidence": False,
        "policy": policy.as_dict(),
        "controls": rows,
    }


if __name__ == "__main__":
    result = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUT.with_name("." + OUT.name + ".tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(OUT)
    print(json.dumps({"output": str(OUT), "status": result["status"], "scientific_natural_case_evidence": False}, sort_keys=True))
