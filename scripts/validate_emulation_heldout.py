#!/usr/bin/env python3
"""Held-out validation of the bitwise emulation: rules and lowering choices frozen, new inputs.

The lowering rules of emulate.py were developed on the capture set of
inkernel_localization.json ("development"); the output-determined choices
(only torchao needed any) are taken from there per kernel and frozen.  The
held-out packages come from new seeds (and, for torchao, one more optimizer
step); nothing is adjusted on them.

    python scripts/validate_emulation_heldout.py --dev results/reference_eval/inkernel_localization.json \
        --heldout .cache/heldout/own .cache/heldout/liger_kernels .cache/heldout/torchao \
        --out results/reference_eval/emulation_heldout.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval.capture import load_launch  # noqa: E402
from kernel_analyzer.reference_eval.emulate import HardwareOracle, verify  # noqa: E402


def dev_choices(dev: Path) -> dict:
    """kernel name (first 80 characters) -> frozen (uncontracted, swapped), union over development packages."""

    out = collections.defaultdict(lambda: (set(), set()))
    for row in json.loads(dev.read_text())["coverage"]:
        name = row["package"]  # "<group>__<kernel name[:80]>__<suffix>"; kernel names may contain "__"
        kernel = name[name.index("__") + 2:name.rindex("__")]
        lc = row["lowering_choices"]
        out[kernel][0].update(lc.get("uncontracted_from_output", []) + lc.get("uncontracted", []))
        out[kernel][1].update(lc.get("swapped_from_output", []) + lc.get("swapped", []))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    choices = dev_choices(args.dev)
    oracle = HardwareOracle(options={"num_warps": 4, "enable_fp_fusion": True})
    rows = []
    for root in args.heldout:
        for pkg in sorted(p for p in root.iterdir() if (p / "launch.json").exists()):
            launch = load_launch(pkg)
            ung, swp = choices[launch.kernel_name[:80]] if launch.kernel_name[:80] in choices else (set(), set())
            r = verify(launch, oracle=oracle, ungrouped=sorted(ung), swapped=sorted(swp))
            r.update({"package": f"{root.name}/{pkg.name}", "kernel": launch.kernel_name,
                      "frozen_choices_from_development": bool(ung or swp)})
            rows.append(r)
            print(r["package"][:70], r["status"], {k: (v["bit_identical"], v["emulated"], v["written"])
                                                   for k, v in r["buffers"].items()}, flush=True)
    summary = collections.defaultdict(collections.Counter)
    for r in rows:
        group = r["package"].split("/")[0]
        summary[group]["packages"] += 1
        summary[group]["status:" + r["status"]] += 1
        summary[group]["with_frozen_choices"] += int(r["frozen_choices_from_development"])
        for b in r["buffers"].values():
            for k in ("written", "emulated", "bit_identical", "nan_both", "mismatch"):
                summary[group][k] += b[k]
    report = {"schema": "kernel-analyzer-emulation-heldout-v1", "development": str(args.dev),
              "rule": "rules and output-determined choices frozen from the development captures; no adjustment",
              "summary": {g: dict(c) for g, c in summary.items()}, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
