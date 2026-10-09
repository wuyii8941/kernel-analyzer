#!/usr/bin/env python3
"""Instruction coverage of the TTIR the official main build produced (DSL v2 W0/W3): every dumped ``*.ttir`` under a
TRITON_DUMP_DIR, deduplicated by content, parsed with the tool's parser and checked op by op against the mapping table
(``ttir_mapping.kernel_coverage``).  Names, not semantics: a covered kernel can still fail at evaluation.

    python scripts/dsl_v2/main_ttir_coverage.py --dump .cache/dsl_v2/w0_dump [--before ISO-TIME] --out OUT.json
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--before", default=None, help="only files modified before this local time (reproduces a stage)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir
    limit = dt.datetime.fromisoformat(a.before).timestamp() if a.before else None
    texts = {}
    for path in sorted(a.dump.glob("*/*.ttir")):
        if limit is not None and path.stat().st_mtime >= limit:
            continue
        text = path.read_text(errors="replace")
        texts.setdefault(hashlib.sha256(text.encode()).hexdigest(), (path, text))
    parse_fail, covered, blocked = [], 0, collections.Counter()
    blocked_kernels = collections.defaultdict(list)
    for digest, (path, text) in texts.items():
        try:
            report = kernel_coverage(parse_ttir(text))
        except Exception as exc:  # noqa: BLE001 -- a parser failure is a finding
            parse_fail.append({"file": str(path.relative_to(a.dump)), "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        if report["complete"]:
            covered += 1
            continue
        names = sorted({r["op"] + ": " + r["rejected"][:90]
                        for r in report["rejected"]})
        for n in names:
            blocked[n] += 1
            blocked_kernels[n].append(path.parent.name[:16])
    doc = {"dump": str(a.dump), "before": a.before, "unique_ttir": len(texts), "parse_failures": parse_fail,
           "fully_covered_by_name": covered, "blocking_names": dict(blocked.most_common()),
           "examples": {k: v[:5] for k, v in blocked_kernels.items()},
           "note": "operation names (and extern symbols / inline asm) against the mapping table; not an evaluation"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("unique_ttir", "fully_covered_by_name")}), len(parse_fail), "parse failures")
    for k, v in blocked.most_common(15):
        print(v, k)


if __name__ == "__main__":
    main()
