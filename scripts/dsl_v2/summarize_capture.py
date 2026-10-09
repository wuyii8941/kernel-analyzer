#!/usr/bin/env python3
"""Summary of capture result files written by main_capture_plugin.py / tutorial_capture.py (DSL v2 W3).

    python scripts/dsl_v2/summarize_capture.py RUN.jsonl [--baseline OLD.jsonl] [--out SUMMARY.json]

Counts launches per status (complete counts set targets separately), the non-complete launches by first reason and by
test family, and, with a baseline, the status change per test family.  Special values (NaN, +-inf) are established
reference values; files written before the increment-5 plugin fix counted them as not complete (status "partial"
without a reason), which this summary reports as "partial (no reason: special values, old plugin)".
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def family(row) -> str:
    test = row.get("test") or row.get("tutorial", "")
    file = test.split("::")[0].split("/")[-1]
    name = test.split("::")[-1].split("[")[0]
    return f"{file}::{name}" if "::" in test else test


def status(row) -> str:
    s = row["status"]
    if s == "complete" and row.get("set_reasons"):
        return "complete (set target)"
    if s == "partial" and not (row.get("reasons") or []):
        return "partial (no reason: special values, old plugin)"
    return s


def first_reason(row) -> str:
    x = row.get("reasons") or row.get("aborted") or [row.get("reason", "")]
    return re.sub(r"@.*", "", (x[0] if x else ""))[:120]


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summarize(rows):
    by_status = collections.Counter(status(r) for r in rows)
    reasons = collections.Counter((status(r), first_reason(r)) for r in rows if status(r) not in
                                  ("complete", "complete (set target)", "nothing written"))
    fam = collections.defaultdict(collections.Counter)
    for r in rows:
        fam[family(r)][status(r)] += 1
    return {"launches": len(rows), "by_status": dict(by_status.most_common()),
            "not_complete_by_reason": [{"status": s, "reason": k, "launches": v} for (s, k), v in reasons.most_common()],
            "by_family": {k: dict(v) for k, v in sorted(fam.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    doc = {"run": str(a.run), **summarize(load(a.run))}
    if a.baseline:
        base = summarize(load(a.baseline))
        doc["baseline"] = {"file": str(a.baseline), "launches": base["launches"], "by_status": base["by_status"]}
        changes = []
        for f in sorted(set(doc["by_family"]) | set(base["by_family"])):
            new, old = doc["by_family"].get(f, {}), base["by_family"].get(f, {})
            if new != old:
                changes.append({"family": f, "before": old, "after": new})
        doc["family_changes"] = changes
    text = json.dumps(doc, indent=1)
    if a.out:
        if a.out.exists():
            raise SystemExit(f"refusing to overwrite {a.out}")
        a.out.write_text(text + "\n")
    print(json.dumps({k: doc[k] for k in ("launches", "by_status")}, indent=1))
    if a.baseline:
        print("baseline", json.dumps(doc["baseline"]["by_status"]))
    for x in doc["not_complete_by_reason"][:25]:
        print(x["launches"], x["status"], x["reason"])


if __name__ == "__main__":
    main()
