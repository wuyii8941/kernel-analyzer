#!/usr/bin/env python3
"""Batch 1 acceptance runner (protocol.md).  Runs every hold-out item of items.json through the unified entry
(``kernel_analyzer.measure.run``, one process per item, GPUs round-robin), compares with the frozen answers
(items.json, predictions.json) and writes per item the report, and a results.json with one of three states per
item: 成立 (holds) / 不成立 (does not hold) / 无法判断 (cannot be judged: error, item not evaluated, statistics that
cannot judge).  Nothing is retried silently; every error and manual intervention stays in the results.

    python run_acceptance.py --out DIR [--items H-P1a,H-R1] [--gpus 0,1,2,3]
    python run_acceptance.py --out DIR --evaluate-only          # recompute the states from the saved reports
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = "/data1/tzh/envs/ka_main/bin/python"
HOLDS, FAILS, UNJUDGED = "成立", "不成立", "无法判断"

ONE = r"""
import json, sys, traceback
from kernel_analyzer import measure
decl = json.loads(sys.argv[1]); out = sys.argv[2]
decl["_base_dir"] = sys.argv[3]
try:
    measure.run(decl, out=out)
except measure.MissingDeclaration as exc:
    json.dump({"status": "declaration incomplete", "missing": exc.items}, open(out, "w"), indent=1)
except BaseException as exc:
    json.dump({"status": "runner error", "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc()[-4000:]}, open(out, "w"), indent=1)
"""


def env(gpu):
    e = dict(os.environ)
    c = str(ROOT / ".cache")
    e.update(HOME=c, XDG_CACHE_HOME=c, TRITON_CACHE_DIR=c + "/triton", TMPDIR=c + "/tmp", PYTHONDONTWRITEBYTECODE="1",
             PYTHONPATH=str(ROOT / "2_tool" / "src"), CUDA_VISIBLE_DEVICES=str(gpu))
    return e


def run_item(it, out_dir, gpu):
    rep = out_dir / f"{it['id']}.json"
    t0 = time.time()
    if it["group"] in ("provenance", "precision", "bias"):
        p = subprocess.run([PY, "-c", ONE, json.dumps(it["declaration"]), str(rep), str(HERE)], env=env(gpu),
                           capture_output=True, text=True, timeout=7200)
        (out_dir / f"{it['id']}.log").write_text(p.stdout[-20000:] + "\n--- stderr ---\n" + p.stderr[-20000:])
        if not rep.exists():
            rep.write_text(json.dumps({"status": "runner error", "error": f"exit {p.returncode}, no report"}))
    elif it["group"] == "packaging":
        p = subprocess.run(["bash", str(ROOT / "2_tool/scripts/acceptance/clean_wheel_install.sh"), "HEAD",
                            str(ROOT / ".cache/batch1/wheel")], env=env(gpu), capture_output=True, text=True,
                           timeout=7200)
        (out_dir / f"{it['id']}.log").write_text(p.stdout[-40000:] + "\n--- stderr ---\n" + p.stderr[-20000:])
        rep.write_text(json.dumps({"status": "ok" if p.returncode == 0 else "error", "returncode": p.returncode,
                                   "log_tail": p.stdout[-3000:]}, indent=1))
    else:
        rep.write_text(json.dumps({"status": "run separately", "note": "regression_v11.py; evaluated by "
                                                                        "compare_regression.py"}))
    return it["id"], round(time.time() - t0, 1)


def _scope_kind(scope):
    if scope is None:
        return None
    if scope == "call-level":
        return "call-level"
    if scope.startswith("call-level (upstream"):
        return "call-level (producer record)"
    if scope.startswith("kernel-level"):
        return "kernel-level"
    return scope


def _bias_value(y, key):
    cls, rule, axis = key.split(".")
    st = (y.get("statistics") or {}).get(cls) or {}
    axes = st.get("axes") or {}
    if axis == "approximate":
        return (axes.get("nonzero") or {}).get("approximate", {}).get("rules", {}).get(rule)
    if axis == "bounded":
        b = (axes.get("nonzero") or {}).get("bounded") or {}
        return (b.get("rules") or {}).get(rule, {}).get("family_verdict") if b.get("available") else "unavailable"
    if axis == "equivalence":
        return ((axes.get("equivalence") or {}).get("rules") or {}).get(rule)
    return None


def evaluate(it, rep, predictions):
    """(state, observed, note) for one item."""
    if rep.get("status") in ("runner error",):
        return UNJUDGED, None, rep.get("error")
    g = it["group"]
    if g == "packaging":
        ok = it["expected"]["log_line"] in rep.get("log_tail", "")
        return (HOLDS if ok else FAILS), rep.get("returncode"), None
    if g == "regression":
        return UNJUDGED, None, "evaluated by compare_regression.py"
    if it["id"] == "H-Q3":
        miss = rep.get("missing") or []
        ok = rep.get("status") == "declaration incomplete" and any(
            m.startswith(it["expected"]["missing_prefix"]) for m in miss)
        return (HOLDS if ok else FAILS), miss or rep.get("status"), None
    lv = (rep.get("levels") or [{}])[0]
    if lv.get("status") != "ok":
        return UNJUDGED, lv.get("status"), lv.get("reason")
    y = (lv.get("outputs") or {}).get("y") or {}
    if g == "provenance":
        if y.get("status") != "evaluated":
            return UNJUDGED, y.get("status"), y.get("reason")
        kind = _scope_kind(y["reference"].get("reference_scope"))
        unsound = it["truth"] == "not clean" and kind.startswith("call-level")
        note = "UNSOUND: call-level although the bytes read are not the call's own values" if unsound else None
        if kind == it["expected"]["reference_scope"] and it["truth"] == "clean" and kind == "kernel-level":
            note = "conservative (truth clean, as pre-registered)"
        return (HOLDS if kind == it["expected"]["reference_scope"] else FAILS), kind, note
    if g == "precision":
        ref = lv.get("refinement") or {}
        exp = predictions.get(it["id"]) if it["expected"].get("from_predictions") else it["expected"]
        obs = {"category": ref.get("category"), "final_level": ref.get("final_level"), "outcome": ref.get("outcome")}
        ok = obs["category"] == exp["category"]
        lvl = exp.get("level", exp.get("final_level"))
        if ok and lvl is not None:
            ok = obs["final_level"] == lvl or (exp.get("uncertain") and abs(obs["final_level"] - lvl) <= 1)
        if ok and exp.get("outcome_prefix"):
            ok = str(obs["outcome"]).startswith(exp["outcome_prefix"])
        return (HOLDS if ok else FAILS), obs, None
    if g == "bias":
        if y.get("status") != "evaluated":
            return UNJUDGED, y.get("status"), y.get("reason")
        obs = {k: _bias_value(y, k) for k in it["expected"]}
        bad = [k for k, v in obs.items() if v != it["expected"][k]]
        cannot = [k for k in bad if str(obs[k]).startswith(("cannot judge", "not established", "unavailable"))]
        if not bad:
            return HOLDS, obs, None
        if len(cannot) == len(bad):
            return UNJUDGED, obs, "cannot judge: " + ", ".join(cannot)
        return FAILS, obs, "mismatch: " + ", ".join(bad)
    return UNJUDGED, None, "unknown group"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--items")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads((HERE / "items.json").read_text())["items"]
    if a.items:
        keep = set(a.items.split(","))
        items = [it for it in items if it["id"] in keep]
    predictions = json.loads((HERE / "predictions.json").read_text())
    seconds = {}
    if not a.evaluate_only:
        gpus = a.gpus.split(",")
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futs = [pool.submit(run_item, it, out, gpus[i % len(gpus)]) for i, it in enumerate(items)]
            for f in futs:
                k, s = f.result()
                seconds[k] = s
                print("done", k, s, flush=True)
    rows = []
    for it in items:
        p = out / f"{it['id']}.json"
        rep = json.loads(p.read_text()) if p.exists() else {"status": "runner error", "error": "no report"}
        state, obs, note = evaluate(it, rep, predictions)
        rows.append({"id": it["id"], "group": it["group"], "lineage": it.get("lineage"), "structure": it["structure"],
                     "truth": it.get("truth"), "expected": it["expected"] if not it["expected"].get("from_predictions")
                     else predictions.get(it["id"]), "observed": obs, "state": state, "note": note,
                     "seconds": seconds.get(it["id"])})
        print(it["id"], state, obs, note or "", flush=True)
    counts = {s: sum(r["state"] == s for r in rows) for s in (HOLDS, FAILS, UNJUDGED)}
    (out / "results.json").write_text(json.dumps({"items": rows, "counts": counts}, indent=1, ensure_ascii=False,
                                                 default=str) + "\n")
    print(counts)


if __name__ == "__main__":
    main()
