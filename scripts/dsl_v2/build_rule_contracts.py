#!/usr/bin/env python3
"""W1 (reference DSL v2 rc3 04): one rule contract per registry entry, in the rc3 schema, with the evidence that exists.

    python scripts/dsl_v2/build_rule_contracts.py --trace '.cache/dsl_v2/w1/trig.*' --test-log .cache/dsl_v2/w1/full_tests.log

Writes results/dsl_v2/contracts/contracts.json (all records) and results/dsl_v2/contracts/coverage.json (the W1 completion
measure: every supported signature with the three test categories and trigger evidence).  The contracts state what the
code does today, not what the design wants: interval endpoints are float64, so no rule is "refinable" to any requested
width (rc3 02 proposition C); test categories that do not exist stay empty; trigger evidence counts only tests that
passed in the given log; implementation_status is never "validated-scope" (that needs the reviewer's independent check).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kernel_analyzer.reference_eval.rule_registry import build  # noqa: E402

OUT = ROOT / "results/dsl_v2/contracts"
OFFICIAL = ROOT / "results/dsl_v2/w0/source_ops_e50b186e8bd2.json"
FORBIDDEN = ["no captured intermediate value is filled in as a complete reference",
             "no undeclared change of the target and no removal of an official signature"]

EFFECTS = {"H": ["memory read/write through the reference memory", "program id / grid"],
           "A": ["control flow follows the reference predicate"]}
OBLIGATIONS = {
    "A": [("integer widths, overflow flags and division by zero handled per the source op", "static", "source-domain",
           "ttir_eval integer ops")],
    "B": [("operands finite and in the declared domain on the reference path", "per_sample", "reference-reachable",
           "ttir_eval status propagation")],
    "C": [("a comparison decided on the reference interval, else both outcomes joined", "per_sample",
           "reference-reachable", "ttir_eval comparisons / select")],
    "D": [("input interval inside the function's domain", "per_sample", "reference-reachable", "ttir_eval elementary")],
    "E": [("combination order trusted (layout of the captured TTGIR) or the combiner order-free", "per_sample",
           "reference-reachable", "ttir_eval _op_reduce / _tree_reduce")],
    "F": [("conversion of a point value exact; an interval conversion encloses the lost units", "static",
           "source-domain", "ttir_eval conversions")],
    "G": [("bit reinterpretation only of point values", "per_sample", "reference-reachable", "ttir_eval bitcast")],
    "H": [("no conflicting access without happens-before (cross-program, same-program cross-thread)", "per_sample",
           "reference-reachable", "ttir_eval _check_read_then_write / _same_program_write_read"),
          ("addresses inside the captured storages", "per_sample", "reference-reachable", "ttir_eval _addresses")],
    "I": [("the symbol or assembly text has a registered declaration", "static", "source-domain",
           "ttir_mapping LIBDEVICE / INLINE_ASM tables")],
}


def passed_tests(log: Path) -> set:
    """pytest -rfE summary lists failures and errors; every other collected test id seen in the trace passed."""
    failed = set()
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            m = re.match(r"^(FAILED|ERROR) (\S+)", line)
            if m:
                failed.add(m.group(2).split(" - ")[0])
    return failed


def signatures(entry) -> list:
    op, attrs, sub, internal = entry["key"]["op"], entry["key"]["attrs"], entry["key"]["subregion"], entry["internal"]
    if op == "tt.reduce" and sub != "*":
        return [f"tt.reduce/{'welford' if sub.startswith('welford') else sub}"]
    if op == "tt.scan" and sub != "*":
        return [f"tt.scan/{sub}"]
    if op == "tt.extern_elementwise" and attrs.startswith("symbol="):
        return [f"tt.extern_elementwise/{attrs}"]
    if op == "tt.elementwise_inline_asm" and attrs.startswith("asm~"):
        return ["re:tt.elementwise_inline_asm/asm=" + attrs[4:]]
    if op == "tt.atomic_rmw" and attrs.startswith("return"):
        return [f"tt.atomic_rmw/{attrs}"]
    return [f"{op}:{internal}"] if internal else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--test-log", type=Path, required=True)
    a = ap.parse_args()
    failed = passed_tests(a.test_log)
    hits = defaultdict(set)
    for f in glob.glob(a.trace):
        for line in Path(f).read_text().splitlines():
            test, sig = line.split("\t", 1)
            if test and test not in failed:
                hits[sig].add(test)
    official = {r["id"]: r for r in json.loads(OFFICIAL.read_text())["records"]} if OFFICIAL.exists() else {}
    reg = build()
    records, rows = [], []
    for e in reg["entries"]:
        op, status, cat = e["key"]["op"], e["status"], e["category"]
        trig = set()
        for sig in signatures(e):
            if sig.startswith("re:"):
                pat = re.compile(sig[3:])
                for s, tests in hits.items():
                    if pat.fullmatch(s):
                        trig |= tests
            else:
                trig |= hits.get(sig, set())
        rule_id = op + "".join(f"|{k}={e['key'][k]}" for k in ("attrs", "subregion") if e["key"][k] != "*")
        supported = status in ("SUPPORTED", "DECLARED_PREMISE")
        # per-signature tests carry their category in the test id (tests/test_signatures_*.py); any other test that
        # executed the rule counts as a positive case
        by_cat = {"boundary": sorted(t for t in trig if "-boundary]" in t),
                  "premise": sorted(t for t in trig if "-premise_violation]" in t)}
        tests = {"positive": sorted(t for t in trig if not ("-boundary]" in t or "-premise_violation]" in t))[:20],
                 "boundary": by_cat["boundary"][:20],
                 # a registry counterexample test counts for this rule only if it actually executed the rule
                 "premise_violation": (by_cat["premise"] + ([e["counterexample_test"]] if e.get("counterexample_test")
                                                            and supported and e["counterexample_test"] in trig
                                                            else []))[:20]}
        rec = {
            "schema": "reference-dsl-v2-rule-rc3", "rule_id": rule_id,
            "language_status": "specified" if supported else "needs-specification",
            "implementation_status": "implemented-unvalidated" if supported else "planned",
            "source": {"op_pattern": op,
                       "official_source_ids": [f"e50b186e:{op}" if op in official else f"triton-3.6.0-registry:{op}"],
                       "source_profile": "triton-3.6.0 regression profile (sm_86); official e50b186e listed where the op "
                                         "still exists",
                       "target_guard": "NVIDIA sm_86, Triton 3.6.0 lowering",
                       "type_guard": e["key"]["types"] if e["key"]["types"] != "*" else "as admitted by the 3.6.0 verifier",
                       "attrs_guard": [] if e["key"]["attrs"] == "*" else [e["key"]["attrs"]],
                       "region_signature": e["key"]["subregion"] if e["key"]["subregion"] != "*" else "none",
                       "effects": EFFECTS.get(cat, [])},
            "semantics": e.get("containment_argument") or e.get("reason") or "(none)",
            "policy_contracts": [{
                "mode": "numerical_difference", "when": "as admitted (see admission obligations)",
                "target_kind": "single", "precision": "enclosure-only" if supported else "not-established",
                "preserved_rounding": [], "intrinsic_width": "none",
                "evaluator_obligation": "float64 directed-rounding interval (not refinable to an arbitrary width)"
                                        if supported else "not implemented"}],
            "admission_obligations": [{"obligation": o, "evidence_kind": k, "covers": c, "checker": ch}
                                      for o, k, c, ch in OBLIGATIONS.get(cat, [("semantics missing", "declared",
                                                                                "source-domain", "none")])],
            "output_guarantee": ("enclosure of the reference value on the evaluated sample"
                                 if supported else "none: rejected or not established"),
            "proof_status": "argument" if e.get("containment_argument") else "none",
            "proof_evidence": ["results/dsl_v2/rule_registry.json"] if e.get("containment_argument") else [],
            "tests": tests,
            "trigger_evidence": sorted(trig)[:50],
            "device_validation": [{"target_profile": "nvidia-sm_86 (RTX A6000), Triton 3.6.0",
                                   "status": "device-tested" if trig else "not-run",
                                   "evidence": sorted(trig)[:5]}],
            "forbidden_fallback": FORBIDDEN}
        records.append(rec)
        rows.append({"rule_id": rule_id, "status": status, "category": cat, "supported": supported,
                     "positive": bool(tests["positive"]), "boundary": bool(tests["boundary"]),
                     "premise_violation": bool(tests["premise_violation"]), "triggered": bool(trig)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contracts.json").write_text(json.dumps(records, indent=1, ensure_ascii=False) + "\n")
    sup = [r for r in rows if r["supported"]]
    gpu_dialect = [r for r in rows if r["rule_id"].startswith("gpu.")]
    cov = {"note": "W1 completion measure (rc3 04): a supported signature is complete when it has the three test categories "
                   "and trigger evidence.  Boundary tests are not yet classified per rule, so that column is empty "
                   "everywhere; this is the honest starting point, not a result.",
           "entries": len(rows), "supported_or_premise": len(sup),
           "supported_with_trigger": sum(r["triggered"] for r in sup),
           "supported_with_positive": sum(r["positive"] for r in sup),
           "supported_with_premise_violation_test": sum(r["premise_violation"] for r in sup),
           "supported_with_all_three_and_trigger": sum(r["positive"] and r["boundary"] and r["premise_violation"]
                                                       and r["triggered"] for r in sup),
           "rejected_or_not_established": len(rows) - len(sup), "gpu_dialect_entries": len(gpu_dialect),
           "untriggered_supported": sorted(r["rule_id"] for r in sup if not r["triggered"]),
           "rows": rows}
    (OUT / "coverage.json").write_text(json.dumps(cov, indent=1, ensure_ascii=False) + "\n")
    print({k: v for k, v in cov.items() if isinstance(v, int)})


if __name__ == "__main__":
    main()
