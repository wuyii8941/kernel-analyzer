#!/usr/bin/env python3
"""Item A1 (docs/protocol_essential_bugs_phase2_20261007.md): targeted re-capture, with detector 2.2, of the phase-1 FR
records that the address-based binding of detector 2.1 affected (an IndexError, or an output the tool itself flagged as
modified after the last Triton write).  Phase-1 records stay frozen; this writes results/essential/phase2a/a1_recapture.json.

    python scripts/essential/a1_recapture.py
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify as Q  # noqa: E402
import common  # noqa: E402
import fr_stage as FS  # noqa: E402
import run_phase1 as R  # noqa: E402

ROOT = common.ROOT
OUT = ROOT / "results/essential/phase2a"


def main():
    affected = json.loads((ROOT / ".cache/essential/a1_affected.json").read_text())
    old = {fam: pickle.loads((R.CACHE / "fr" / fam / "inductor_cuda32.pkl").read_bytes()) for fam in affected}
    rows = []
    for fam, ids in affected.items():
        conds = {c["id"]: c for c in R.FAMILIES[fam][0]()}
        for cid in ids:
            cond = conds[cid]
            setup, tensors, launch = FS._case(fam, cond, "inductor_cuda32")
            shapes = [None]

            def make_inputs(seed, _t=tensors):
                d = _t(seed)
                d["_seed"] = seed
                return d

            def launch_rec(t, _l=launch):
                outs = _l(t)
                shapes[0] = {k: tuple(v.shape) for k, v in outs.items()}
                return outs

            def spec(t, _c=cond, _f=fam):
                return FS.spec_for(_f, _c, t["_seed"], "float32", shapes[0])

            try:
                rep, keep = common.fr_run(f"{fam}/{cid}", setup, make_inputs, launch_rec, spec)
                new = {"status": "ok",
                       "keep": {k: [{kk: p[kk] for kk in ("r_lo", "r_hi", "k", "ok", "shape")} for p in v] for k, v in keep.items()},
                       "notes": {k: rep.get(k) for k in ("tool_version", "outputs_not_written_by_triton",
                                                         "outputs_binding_not_established",
                                                         "outputs_at_address_of_another_recorded_storage",
                                                         "outputs_modified_after_last_triton_write", "ttir_coverage_complete")},
                       "mixed": {k: v.get("depends_on_non_triton_intermediates") for k, v in rep.get("outputs", {}).items()}}
            except Exception as exc:  # noqa: BLE001
                new = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:300]}
            specs32 = {s: R.load_spec(fam, cid, s, "base", "float32", "f64") for s in R.C.SEEDS}
            o = old[fam].get(cid, {})
            row = {"family": fam, "condition": cid,
                   "detector_2_1": {"status": o.get("status"), "reason": o.get("reason"),
                                    "modified_after": (o.get("notes") or {}).get("outputs_modified_after_last_triton_write")},
                   "detector_2_2": {k: v for k, v in new.items() if k not in ("keep",)},
                   "FR_2_2": Q.fr_assess(fam, cond, "inductor_cuda32", new, specs32) if new["status"] == "ok" else None}
            if row["FR_2_2"]:
                row["FR_2_2"] = {"outputs": {k: {kk: v.get(kk) for kk in ("semantic_vs_main", "not_established", "ok_elements")}
                                             for k, v in row["FR_2_2"]["outputs"].items()}}
            rows.append(row)
            print(fam, cid, "2.1:", row["detector_2_1"]["status"], row["detector_2_1"]["modified_after"],
                  "| 2.2:", new.get("status"), (new.get("notes") or {}).get("outputs_not_written_by_triton"),
                  (new.get("notes") or {}).get("outputs_binding_not_established"),
                  "| FR:", (row["FR_2_2"] or {}).get("outputs"), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a1_recapture.json").write_text(json.dumps(rows, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
