#!/usr/bin/env python3
"""Pre-evaluation coverage check for blind_test_v1 (allowed by the protocol, section 8).

Every program is compiled and run once (seed 0) under the launch recorder; the TTIR of every launch is
checked against the mapping table and the automatic reference is evaluated once.  Same procedure for all
programs; nothing here depends on the answer key.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator  # noqa: E402
from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch

    sys.path.insert(0, str(args.package / "programs"))
    inputs = importlib.import_module("inputs")
    manifest = json.loads((args.package / "manifest.json").read_text())
    rows = []
    for entry in manifest["programs"]:
        row = {"program": entry["id"], "family": entry["family"]}
        try:
            mod = importlib.import_module(entry["id"])
            inp = inputs.make_inputs(mod.FAMILY, 0)
            rec = TritonLaunchRecorder()
            with rec:
                mod.launch(inp)
                torch.cuda.synchronize()
            row["launches"] = []
            for launch in rec.launches:
                module = parse_ttir(launch.asm["ttir"])
                cov = kernel_coverage(module)
                info = {"kernel": launch.kernel_name, "grid": list(launch.grid), "coverage_complete": cov["complete"],
                        "rejected": [r["rejected"] for r in cov["rejected"]][:5]}
                if cov["complete"]:
                    res = KernelReferenceEvaluator(module).evaluate(launch, programs=[(0, 0, 0)])
                    info["aborted"] = sorted(set(res.aborted.values()))[:2]
                row["launches"].append(info)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
            row["traceback"] = traceback.format_exc()[-1500:]
        rows.append(row)
        print(row["program"], row["family"], row.get("error", "") or [(l["kernel"], l["coverage_complete"], l.get("rejected"), l.get("aborted")) for l in row["launches"]], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
