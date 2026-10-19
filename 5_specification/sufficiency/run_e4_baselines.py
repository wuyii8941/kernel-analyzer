#!/usr/bin/env python3
"""Exploratory addition to E4 (protocol deviation D4, decided after seeing the E4 results): measure the UNMODIFIED
base bindings under exactly the E4 harness (scratch registration, development 1 + confirmation 2 units) so that a
case's G_pi - f exclusions can be compared with its base's. A base whose own program already differs from the task
(e.g. a rounded constant) makes "detected" uninformative for every case on that base.

    python 5_specification/sufficiency/run_e4_baselines.py --spec-src FROZEN_5_SPEC --root NEW_DIR --bases B [B ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_e4  # noqa: E402


def residuals(report):
    out = {}
    for level in (report or {}).get("levels", []):
        for name, o in (level.get("outputs") or {}).items():
            sem = o.get("semantic") or {}
            out[name] = {"excludes_zero": sem.get("reference_difference_excludes_zero"), "ok_elements": sem.get("ok_elements"),
                         "max_gap_relative": sem.get("max_gap_relative")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-src", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--bases", nargs="+", required=True)
    ap.add_argument("--gpu", default="3")
    a = ap.parse_args()
    cases = [{"id": f"BASE-{b}", "mechanism": "base", "kind": "base", "source": "base", "base": b, "edits": [],
              "expected": "n/a", "rationale": "unmodified base binding"} for b in a.bases]
    run_e4_cases = sys.modules.get("e4_cases")
    import e4_cases
    e4_cases.CASES[:] = cases            # the harness reads CASES from this module
    sys.argv = [sys.argv[0], "--spec-src", str(a.spec_src), "--root", str(a.root), "--gpu", a.gpu]
    run_e4.main()
    rows = [json.loads(l) for l in open(a.root / "results.jsonl")]
    for r in rows:
        rp = a.root / "reports" / f"{r['id']}.json"
        r["outputs"] = residuals(json.loads(rp.read_text())) if rp.exists() else None
    (a.root / "baselines.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
