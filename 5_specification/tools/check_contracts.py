#!/usr/bin/env python3
"""Contract checker: structure, examples, diagnostic readings and mutation requirements of every contract.

    python tools/check_contracts.py [--json OUT] [--lock]

Exit status: 0 when every example and reading is consistent and every mutation requirement is rejected by one of
its witnesses; 1 on any contradiction, structure error, implementation error or surviving mutation; 2 when the only
problems are results that are not established. A consistent example means "consistent with the reference on this
example"; it does not review the contract, and nothing here changes a review status.

--lock records the contract and adapter hashes of the contracts that passed completely in contracts/LOCK.json
(implementation validated at these hashes); contracts that did not pass are removed from it.
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


def killed(mres, expected, outputs):
    if mres.outcome == "implementation_error":
        return False, f"not applicable: {mres.note[:80]}"
    if expected["outcome"] != "value":
        return mres.outcome != expected["outcome"], f"outcome {mres.outcome}"
    if mres.outcome != "value":
        return True, f"outcome {mres.outcome}"
    v, checks = C.compare_result(mres, expected, outputs)
    bad = [f"{p}: {d}" for p, vv, d in checks if vv in (C.CONTRADICTION, C.STRUCTURE)]
    return v in (C.CONTRADICTION, C.STRUCTURE), (bad[0] if bad else v)[:120]


def check_contract(path: Path, sources: dict, adapters: Path) -> dict:
    data = yaml.safe_load(path.read_text())
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
    outputs = data["outputs"]
    readings_impl = getattr(A, "READINGS", {})
    mutants = getattr(A, "MUTANTS", {})
    by_id = {e["id"]: e for e in data["examples"]}
    for e in data["examples"]:
        res = evaluate(A.reference, e["profile"], e["input"])
        v, checks = C.compare_result(res, e["expected"], outputs)
        rep["examples"].append({"id": e["id"], "verdict": v, "outcome": res.outcome,
                                "detail": [f"{p}: {d}" for p, vv, d in checks if vv != C.CONSISTENT][:3]
                                or [checks[0][2] if checks else ""]})
        for rid, block in (e.get("readings") or {}).items():
            fn = readings_impl.get(rid)
            if fn is None:
                rep["readings"].append({"id": f"{e['id']}[{rid}]", "verdict": C.STRUCTURE, "detail": ["not implemented"]})
                continue
            rres = evaluate(fn, e["profile"], e["input"])
            rv, rchecks = C.compare_result(rres, block, outputs)
            rep["readings"].append({"id": f"{e['id']}[{rid}]", "verdict": rv,
                                    "detail": [f"{p}: {d}" for p, vv, d in rchecks if vv != C.CONSISTENT][:3]})
    for m in data["mutation_requirements"]:
        fn = mutants.get(m["id"])
        if fn is None:
            rep["mutations"].append({"id": m["id"], "rejected": False, "by": [], "detail": "no implementation"})
            continue
        by, notes = [], []
        for w in m["witnesses"]:
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
    return {s: ("snapshot " + sources[s]["snapshot_sha256"][:12]) if sources[s].get("snapshot_sha256")
            else "no snapshot" for s in sorted(used)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contracts", type=Path, default=registry.CONTRACTS)
    ap.add_argument("--adapters", type=Path, default=registry.ADAPTERS)
    ap.add_argument("--sources", type=Path, default=registry.SOURCES)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--lock", action="store_true")
    a = ap.parse_args(argv)
    try:
        sources = json.loads(a.sources.read_text())
    except (OSError, ValueError) as e:
        print(f"cannot read sources: {e}")
        return 1
    paths = sorted(a.contracts.glob("*.yaml"))
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
    data = {r["id"]: yaml.safe_load(Path(r["path"]).read_text()) for r in reps}
    print("\nSources used by each contract:")
    for r in reps:
        if not r["structure"]:
            print(f"- {r['id']}: " + ", ".join(f"{k} ({v})" for k, v in source_status(data[r['id']], sources).items()))
    if a.json:
        a.json.write_text(json.dumps(reps, indent=1, ensure_ascii=False, default=str) + "\n")
    if a.lock:
        lock = registry.load_lock() if a.contracts == registry.CONTRACTS else {}
        for r in reps:
            if r["verdict"] == C.CONSISTENT:
                c = registry.load(r["id"], a.contracts, a.adapters)
                lock[r["id"]] = {"contract_sha256": c.contract_sha256, "adapter_sha256": c.adapter_sha256,
                                 "examples": len(r["examples"]), "readings": len(r["readings"]),
                                 "mutations": len(r["mutations"]), "checked": datetime.date.today().isoformat()}
            else:
                lock.pop(r["id"], None)
        (a.contracts / "LOCK.json").write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
    worst = C.aggregate([r["verdict"] for r in reps])
    print(f"\nTOTAL: {len(reps)} contracts; verdict {worst}")
    return {C.CONSISTENT: 0, C.NOT_ESTABLISHED: 2}.get(worst, 1)


if __name__ == "__main__":
    sys.exit(main())
