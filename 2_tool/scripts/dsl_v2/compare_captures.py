#!/usr/bin/env python3
"""DSL v2 increment 14 evaluation: per launch, the status in a baseline capture file vs the increment-14 rerun of the
same official tests (launches matched by test id after '::' and their order within the test).

    python inc14_compare.py BASELINE.jsonl RUN.jsonl [--select SUBSTRING] --out OUT.json
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_capture import read_lines, status  # noqa: E402

TARGET = "set-valued integer atomic return value (L_E) is not representable as a point"


def keyed(path, select):
    out, seen = {}, collections.Counter()
    for line in read_lines(path):
        if not line.strip():
            continue
        r = json.loads(line)
        test = r.get("test", "").split("::", 1)[-1]
        if select and select not in test:
            continue
        k = (test, seen[test])
        seen[test] += 1
        out[k] = r
    return out


def reasons(r):
    return sorted({re.sub(r"@.*", "", x).replace("not_established:", "") for x in (r.get("reasons") or [])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("run")
    ap.add_argument("--select", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    old, new = keyed(a.baseline, a.select), keyed(a.run, a.select)
    common = sorted(set(old) & set(new))
    trans = collections.Counter((status(old[k]), status(new[k])) for k in common)
    only_target = [k for k in common if reasons(old[k]) == [TARGET]]
    became = collections.Counter(status(new[k]) for k in only_target)
    left = collections.Counter(tuple(reasons(new[k])) for k in only_target if status(new[k]) != "complete (set target)")
    regressions = [{"launch": list(k), "before": status(old[k]), "after": status(new[k]), "reasons": reasons(new[k])}
                   for k in common if status(old[k]) in ("complete", "complete (set target)")
                   and status(new[k]) != status(old[k])]
    doc = {"baseline": a.baseline, "run": a.run, "select": a.select, "matched_launches": len(common),
           "baseline_only": len(set(old) - set(new)), "run_only": len(set(new) - set(old)),
           "transitions": {f"{x} -> {y}": n for (x, y), n in sorted(trans.items())},
           "baseline_only_reason_was_integer_set": len(only_target),
           "their_status_now": dict(became), "their_remaining_reasons": {" | ".join(k): v for k, v in left.items()},
           "previously_complete_changed": regressions,
           "run_status": dict(collections.Counter(status(r) for r in new.values()))}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k != "previously_complete_changed"}, indent=1))
    print("previously complete changed:", len(regressions))


if __name__ == "__main__":
    main()
