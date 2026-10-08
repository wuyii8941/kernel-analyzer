#!/usr/bin/env python3
"""Section 5 of the general-round task book: run the frozen tool on the reviewer's unseen structures.

Package layout (delivered by the reviewer; the answers stay sealed until the run is complete):
    PKG/cases/<case>/declaration.json      unified-entry declaration (call + samplers + compare + budget)
    PKG/cases/<case>/*.py                   the binding and samplers the declaration names
    PKG/cases/<case>/ordinary_reference.py  optional: function(inputs) -> {output: tensor} for the baseline
    PKG/answers.sha256                      hash of the sealed answer file

    python scripts/general/acceptance_run.py --package PKG --out OUT          # run (answers unopened)
    python scripts/general/acceptance_run.py --package PKG --out OUT --score ANSWERS.json   # after unsealing

Per case: structure covered (reference established for the measured outputs), complete reference rate, decidable
rate (rule classes not "cannot judge"), the verdicts per class; the baseline "ordinary reference + same statistics"
(K - ordinary_reference(inputs in float64), point residuals through the same assess_units and class Holm); human cost
(declaration lines) and run cost (seconds).  Cases that fail with "semantics missing" are listed apart (they need a
new general rule).  The tool version and source-tree hash are recorded; the run refuses to start if the working tree
of src/ differs from the frozen tag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kernel_analyzer import check, measure  # noqa: E402
from kernel_analyzer.reference_eval.analysis import assess_units  # noqa: E402

FROZEN_TAG = "general-v3.0"


def tree_hash():
    return subprocess.run(["git", "rev-parse", "HEAD:src"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def frozen_check():
    r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{FROZEN_TAG}:src"], cwd=ROOT, capture_output=True, text=True)
    tag = r.stdout.strip() if r.returncode == 0 else ""
    dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"frozen_tag": FROZEN_TAG, "tag_src_tree": tag, "head_src_tree": tree_hash(), "src_dirty": bool(dirty),
            "ok": bool(tag) and tag == tree_hash() and not dirty}


def baseline(decl, case_dir):
    """ordinary reference + same statistics: residual K - ordinary(inputs as float64), point intervals."""
    ref_path = case_dir / "ordinary_reference.py"
    if not ref_path.exists():
        return {"status": "not available", "reason": "no ordinary_reference.py in the case"}
    exp = measure.expand(decl)
    call = measure.resolve(exp["call"], str(case_dir))
    ref = measure.resolve(f"{ref_path}:reference", str(case_dir))
    u = exp["units"]
    seeds = list(range(u["seed_offset"], u["seed_offset"] + min(u["development"] + u["confirmation"], exp["budget"]["max_units"])))
    n_dev = min(u["development"], max(1, len(seeds) // 3))
    out = {}
    for level in exp["factor_levels"]:
        rows = {k: ([], []) for k in exp["compare"]["measure"]}
        for s in seeds:
            inp = measure.make_inputs(exp, level, s)
            inp.pop("_seed")
            k = call(inp)
            r = ref({n: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for n, v in inp.items()})
            for name in rows:
                kk = k[name].detach().double().cpu().numpy().ravel()
                rr = r[name].detach().double().cpu().numpy().ravel()
                rows[name][0].append(kk - rr)
                rows[name][1].append(rr)
        res = {}
        for name, (e, rr) in rows.items():
            e, rr = np.stack(e), np.stack(rr)
            rec, _ = assess_units(f"baseline:{name}", e, e, rr, np.isfinite(e) & np.isfinite(rr), n_dev,
                                  ["R1", "R2", "R3", "R5"], alignment_reference=rr, run_detector=False)
            res[name] = measure.class_statistics(rec, exp["rule_classes"], exp["alpha"])
        out[str(level)] = res
    return {"status": "ok", "levels": out}


def run_case(case_dir: Path):
    decl = measure.load(case_dir / "declaration.json")
    lines = len((case_dir / "declaration.json").read_text().splitlines())
    t0 = time.time()
    try:
        rep = measure.run(decl)
    except measure.MissingDeclaration as exc:
        return {"status": "declaration incomplete", "missing": exc.items, "declaration_lines": lines}
    seconds = time.time() - t0
    levels = []
    for lv in rep["levels"]:
        outs = lv.get("outputs") or {}
        covered = lv.get("status") == "ok" and all(o.get("status") == "evaluated" for o in outs.values())
        complete = [o["reference"]["complete_rate"] for o in outs.values() if o.get("reference")]
        decidable = [not v["summary"].startswith("cannot judge") for o in outs.values() for v in (o.get("statistics") or {}).values()]
        levels.append({"level": lv["level"], "status": lv["status"], "structure_covered": covered,
                       "complete_rate": float(np.mean(complete)) if complete else None,
                       "decidable_rate": float(np.mean(decidable)) if decidable else None,
                       "verdicts": {k: {c: v["summary"] for c, v in (o.get("statistics") or {}).items()} for k, o in outs.items()},
                       "failure_classes": lv.get("failure_classes") or ([lv["failure_class"]] if "failure_class" in lv else []),
                       "needs_new_general_rule": "semantics missing" in (lv.get("failure_classes") or [lv.get("failure_class")])})
    try:
        base = baseline(decl, case_dir)
    except Exception as exc:  # noqa: BLE001
        base = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:300]}
    return {"status": "ok", "declaration_lines": lines, "seconds": round(seconds, 1), "levels": levels, "baseline": base,
            "tool_report": rep}


def score(results, answers):
    """answers: {case: {"output": name, "class": "fixed_mean"|"aligned", "effect": "none"|"positive"|"negative"}}."""
    rows = []
    for case, ans in answers.items():
        r = results["cases"].get(case, {})
        verdicts = [lv["verdicts"].get(ans["output"], {}).get(ans["class"], "") for lv in r.get("levels", [])]
        detected = any(v.startswith("average effect nonzero") for v in verdicts)
        rows.append({"case": case, "truth": ans["effect"], "detected": detected,
                     "outcome": ("true positive" if detected and ans["effect"] != "none" else
                                 "false positive" if detected else
                                 "miss" if ans["effect"] != "none" else "true negative")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score")
    ap.add_argument("--allow-unfrozen", action="store_true", help="smoke tests only; never for the acceptance run")
    a = ap.parse_args()
    out = Path(a.out)
    if a.score:
        results = json.loads((out / "results.json").read_text())
        rows = score(results, json.loads(Path(a.score).read_text()))
        (out / "score.json").write_text(json.dumps(rows, indent=1) + "\n")
        print(json.dumps(rows, indent=1))
        return
    fz = frozen_check()
    if not fz["ok"] and not a.allow_unfrozen:
        sys.exit(f"tool not frozen at {FROZEN_TAG}: {fz}")
    pkg = Path(a.package)
    answers_hash = (pkg / "answers.sha256").read_text().strip() if (pkg / "answers.sha256").exists() else None
    results = {"frozen": fz, "tool_version": check.TOOL_VERSION, "answers_sha256_at_start": answers_hash, "cases": {}}
    for case_dir in sorted((pkg / "cases").iterdir()):
        if case_dir.is_dir():
            results["cases"][case_dir.name] = run_case(case_dir)
            print(case_dir.name, json.dumps({k: results["cases"][case_dir.name].get(k) for k in ("status", "seconds")}),
                  [(lv["structure_covered"], lv["complete_rate"], lv["decidable_rate"], lv["verdicts"])
                   for lv in results["cases"][case_dir.name].get("levels", [])], flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
