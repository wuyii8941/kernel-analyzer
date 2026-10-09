#!/usr/bin/env python3
"""Unified entry CLI: python scripts/measure.py --declaration D.json --out R.json"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kernel_analyzer import measure  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--declaration", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
try:
    rep = measure.run(a.declaration, a.out)
except measure.MissingDeclaration as exc:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"status": "declaration incomplete", "missing": exc.items}, indent=1) + "\n")
    print("declaration incomplete:", *exc.items, sep="\n  ")
    sys.exit(2)
for lv in rep["levels"]:
    print(json.dumps({"level": lv["level"], "status": lv["status"], "seconds": lv.get("seconds"),
                      "outputs": {k: {"status": v.get("status"), "complete_rate": (v.get("reference") or {}).get("complete_rate"),
                                      "resolved": (v.get("reference") or {}).get("resolved_fraction"),
                                      "fixed_mean": ((v.get("statistics") or {}).get("fixed_mean") or {}).get("summary"),
                                      "aligned": ((v.get("statistics") or {}).get("aligned") or {}).get("summary"),
                                      "failures": v.get("failure_classes") or v.get("failure_class")}
                                  for k, v in (lv.get("outputs") or {}).items()},
                      "reason": lv.get("reason")}, default=str))
print("failure counts", rep["failure_counts"])
