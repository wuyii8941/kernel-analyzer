#!/usr/bin/env python3
"""Contract checker: structure, examples, diagnostic readings and mutation requirements of every contract.

    python tools/check_contracts.py [--json OUT] [--record-checks]

Exit status: 0 when every example and reading is consistent and every mutation requirement is rejected by one of
its witnesses; 1 on any contradiction, structure error, implementation error or surviving mutation; 2 when the only
problems are results that are not established. A consistent example means "consistent with the reference on this
example"; it does not review the contract, and nothing here changes a review status.

--record-checks records only the date and number of development checks that passed
in contracts/DEVELOPMENT_CHECKS.json. It does not bind file content and is never
used for runtime admission. Independent implementation review retains the exact
reviewed files separately.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernel_spec import compare as C  # noqa: E402
from kernel_spec import registry, schema  # noqa: E402
from kernel_spec.num import evaluate  # noqa: E402
from kernel_spec.contracts_io import load_contract  # noqa: E402


def safe_compare(res, expected, outputs):
    """compare_result that turns any exception of the comparison itself into a structure error for this example."""
    try:
        return C.compare_result(res, expected, outputs)
    except Exception as e:  # noqa: BLE001
        return C.STRUCTURE, [("", C.STRUCTURE, f"comparison failed: {type(e).__name__}: {str(e)[:120]}")]


def killed(mres, expected, outputs):
    # Failure to evaluate a reading supplies no evidence about its mathematical
    # result, including when the witness expects an invalid/undefined input.
    if mres.outcome in {"implementation_error", "not_established", "unsupported"}:
        return False, f"not applicable ({mres.outcome}): {mres.note[:80]}"
    if expected["outcome"] != "value":
        return mres.outcome != expected["outcome"], f"outcome {mres.outcome}"
    if mres.outcome != "value":
        return True, f"outcome {mres.outcome}"
    # A different mathematical formula can need an enclosure even when the
    # target is rational. Disjoint rigorous values prove a disagreement; an
    # enclosure merely violating the target's exact representation does not.
    witness_outputs = {k: dict(info, guarantee="rigorous_enclosure") if info["guarantee"] == "exact" else info
                       for k, info in outputs.items()}
    v, checks = safe_compare(mres, expected, witness_outputs)
    # a structure difference rejects a mutant only when it is a real difference of shape, keys, type or outcome;
    # 'cannot compare' (an inexact allowed-set element, a failed comparison) is not evidence
    def real(vv, d):
        return vv == C.CONTRADICTION or (vv == C.STRUCTURE and not any(
            k in d for k in ("not exact", "is an enclosure", "comparison failed", "expected value:",
                            "reference value of type", "reference guarantee:")))
    bad = [f"{p}: {d}" for p, vv, d in checks if real(vv, d)]
    return bool(bad), (bad[0] if bad else v)[:120]


def check_contract(path: Path, sources: dict, adapters: Path) -> dict:
    try:
        data = load_contract(path)
        if not isinstance(data, dict):
            raise ValueError("not a mapping")
    except (yaml.YAMLError, ValueError) as e:
        return {"id": path.stem, "path": str(path), "structure": [f"unreadable YAML: {str(e)[:200]}"], "examples": [],
                "readings": [], "mutations": [], "clauses": {}, "verdict": C.STRUCTURE, "review_status": None}
    rep = {"id": data.get("id", path.stem), "path": str(path), "structure": schema.validate(data, sources),
           "examples": [], "readings": [], "mutations": [], "clauses": {}, "verdict": None,
           "review_status": (data.get("review") or {}).get("status")}
    if data.get("id") != path.stem:
        rep["structure"].append(f"id {data.get('id')!r} does not match the file name")
    if rep["structure"]:
        rep["verdict"] = C.STRUCTURE
        return rep
    try:
        apath, A = registry.import_adapter(data["id"], adapters)
    except Exception as e:  # noqa: BLE001
        rep["structure"].append(f"adapter: {type(e).__name__}: {e}")
        rep["verdict"] = C.STRUCTURE
        return rep
    readings_impl = getattr(A, "READINGS", {})
    mutants = getattr(A, "MUTANTS", {})
    by_id = {e["id"]: e for e in data["examples"]}
    for e in data["examples"]:
        outputs = schema.profile_outputs(data, e['profile'])
        res = evaluate(A.reference, e["profile"], e["input"])
        v, checks = safe_compare(res, e["expected"], outputs)
        rep["examples"].append({"id": e["id"], "verdict": v, "outcome": res.outcome,
                                "detail": [f"{p}: {d}" for p, vv, d in checks if vv != C.CONSISTENT][:3]
                                or [checks[0][2] if checks else ""]})
        for rid, block in (e.get("readings") or {}).items():
            fn = readings_impl.get(rid)
            if fn is None:
                rep["readings"].append({"id": f"{e['id']}[{rid}]", "verdict": C.STRUCTURE, "detail": ["not implemented"]})
                continue
            rres = evaluate(fn, e["profile"], e["input"])
            rv, rchecks = safe_compare(rres, block, outputs)
            rep["readings"].append({"id": f"{e['id']}[{rid}]", "verdict": rv,
                                    "detail": [f"{p}: {d}" for p, vv, d in rchecks if vv != C.CONSISTENT][:3]})
    for m in data["mutation_requirements"]:
        fn = mutants.get(m["id"])
        if fn is None:
            rep["mutations"].append({"id": m["id"], "rejected": False, "by": [], "detail": "no implementation"})
            continue
        by, notes = [], []
        for w in m["witnesses"]:
            outputs = schema.profile_outputs(data, by_id[w]['profile'])
            ok, note = killed(evaluate(fn, by_id[w]["profile"], by_id[w]["input"]), by_id[w]["expected"], outputs)
            notes.append(f"{w}: {note}")
            if ok:
                by.append(w)
        rep["mutations"].append({"id": m["id"], "rejected": bool(by), "by": by, "detail": "; ".join(notes)[:240]})
    ex = {r["id"]: r["verdict"] for r in rep["examples"]}
    for cl in data["clauses"]:
        vs = [ex[e["id"]] for e in data["examples"] if cl["id"] in e["clauses"]]
        rep["clauses"][cl["id"]] = C.aggregate(vs)
    verdicts = ([r["verdict"] for r in rep["examples"]] + [r["verdict"] for r in rep["readings"]]
                + [C.CONTRADICTION for m in rep["mutations"] if not m["rejected"]])
    rep["verdict"] = C.aggregate(verdicts)
    return rep


def source_status(data, sources):
    used = {b.split(".")[0] for node in [*data.get("clauses", []), *data.get("decisions", [])]
            for b in node.get("basis", []) if b.split(".")[0] in sources}
    status = {}
    for sid in sorted(used):
        source = sources[sid]
        if not registry.frozen_source(source):
            status[sid] = "no snapshot"
        elif source.get("snapshot"):
            status[sid] = 'retained local source'
            if source.get('version_review', {}).get('status') == 'unverified':
                status[sid] += '; claimed version unverified'
        else:
            status[sid] = "standard, " + source["edition"]
    return status


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contracts", type=Path, default=registry.CONTRACTS)
    ap.add_argument("--adapters", type=Path, default=registry.ADAPTERS)
    ap.add_argument("--sources", type=Path, default=registry.SOURCES)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--record-checks", action="store_true")
    ap.add_argument("--only", nargs="+", help="check only these contract IDs")
    a = ap.parse_args(argv)
    try:
        sources = json.loads(a.sources.read_text())
    except (OSError, ValueError) as e:
        print(f"cannot read sources: {e}")
        return 1
    paths = sorted(a.contracts.glob("*.yaml"))
    if a.only is not None:
        missing = set(a.only) - {p.stem for p in paths}
        if missing:
            print(f"requested contracts not found: {', '.join(sorted(missing))}")
            return 1
        paths = [p for p in paths if p.stem in a.only]
    if not paths:
        print(f"no contracts in {a.contracts}: an empty run is not a success")
        return 1
    reps = [check_contract(p, sources, a.adapters) for p in paths]
    print("| contract | review | examples | readings | mutations rejected | verdict |")
    print("|---|---|---|---|---|---|")
    for r in reps:
        ne = sum(x["verdict"] == C.CONSISTENT for x in r["examples"])
        nr = sum(x["verdict"] == C.CONSISTENT for x in r["readings"])
        nm = sum(m["rejected"] for m in r["mutations"])
        print(f"| {r['id']} | {r['review_status']} | {ne}/{len(r['examples'])} | {nr}/{len(r['readings'])} | "
              f"{nm}/{len(r['mutations'])} | {r['verdict']} |")
    problems = []
    for r in reps:
        problems += [f"{r['id']}: structure: {s}" for s in r["structure"]]
        problems += [f"{r['id']} {x['id']}: {x['verdict']}: {'; '.join(x['detail'])}"
                     for x in r["examples"] + r["readings"] if x["verdict"] != C.CONSISTENT]
        problems += [f"{r['id']} {m['id']}: mutation not rejected ({m['detail']})" for m in r["mutations"]
                     if not m["rejected"]]
    print("\nProblems:" if problems else "\nNo problems.")
    for p in problems:
        print(f"- {p}")
    data = {r["id"]: load_contract(Path(r["path"])) for r in reps if not r["structure"]}
    print("\nSources used by each contract:")
    for r in reps:
        if not r["structure"]:
            print(f"- {r['id']}: " + ", ".join(f"{k} ({v})" for k, v in source_status(data[r['id']], sources).items()))
    if a.json:
        a.json.write_text(json.dumps(reps, indent=1, ensure_ascii=False, default=str) + "\n")
    if a.record_checks:
        record_path = a.contracts / "DEVELOPMENT_CHECKS.json"
        try:
            records = json.loads(record_path.read_text()) if record_path.is_file() else {}
        except (OSError, ValueError):
            records = {}
        for r in reps:
            if r["verdict"] == C.CONSISTENT:
                records[r["id"]] = {"checked": datetime.date.today().isoformat(),
                                     "examples": len(r["examples"]), "readings": len(r["readings"]),
                                     "mutations": len(r["mutations"]), "verdict": C.CONSISTENT}
            else:
                records.pop(r["id"], None)
        record_path.write_text(json.dumps(records, indent=1, sort_keys=True) + "\n")
    worst = C.aggregate([r["verdict"] for r in reps])
    print(f"\nTOTAL: {len(reps)} contracts; verdict {worst}")
    return {C.CONSISTENT: 0, C.NOT_ESTABLISHED: 2}.get(worst, 1)


if __name__ == "__main__":
    sys.exit(main())
