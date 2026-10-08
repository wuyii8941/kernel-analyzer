#!/usr/bin/env python3
"""Structure acceptance set v1.1-rc1: fill RUN_FREEZE before measurement, then run every job in fresh processes.

    python scripts/acceptance/structure_v11_queue.py freeze --run-id ID     # writes RUN_FREEZE.json + manifests
    git add ... && git commit                                               # the freeze is committed before measuring
    python scripts/acceptance/structure_v11_queue.py run --run-id ID --workers 16

`run` refuses to start unless RUN_FREEZE.json is committed, the tool commit it names is an ancestor of HEAD with
src/, the frozen scripts and scripts/acceptance unchanged since, the working tree of those paths is clean, and
the package hashes still match.  Jobs whose result file exists are skipped (resume).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import structure_v11 as S  # noqa: E402

PY = "/data1/tzh/envs/ka_main/bin/python"
CACHE = ROOT / ".cache"
WATCHED = ["src", "scripts/essential/contract_v3.py", "scripts/essential/common.py", "scripts/general/acceptance_run.py",
           "scripts/acceptance"]
BUDGET = {"regular": 3600, "atomic": 14400}
MODES_ROUNDS = [("B", "main"), ("A", "main"), ("B", "replication")]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def detector_manifest():
    files = []
    for p in sorted((ROOT / "src").rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            files.append(p)
    files += [ROOT / "scripts/essential/contract_v3.py", ROOT / "scripts/essential/common.py",
              ROOT / "scripts/general/acceptance_run.py"]
    files += sorted((ROOT / "scripts/acceptance").glob("*.py"))
    return [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in files]


def families(man):
    """registered Holm families: round x program x output x comparison (mode B) x rule class"""
    out = []
    for rnd in ("main", "replication"):
        for pid, e in sorted(man.items()):
            for name in e["output_names"]:
                for comp in S.COMPARISONS["B"]:
                    for cls, members in S.RULE_CLASSES.items():
                        out.append({"family": f"{rnd}/{pid}/{name}/{comp}/{cls}", "members": members,
                                    "lane": e["execution_lane"],
                                    "status_if_not_run": "retained as not established (no shrinkage)"})
    return out


def jobs(man):
    rows = []
    for pid, e in sorted(man.items()):
        if e["execution_lane"] != "measurement":
            continue
        atomic = int(e.get("execution_repeats", 1)) > 1
        for mode, rnd in MODES_ROUNDS:
            units = (len(S.ROUNDS[rnd][0]) + len(S.ROUNDS[rnd][1])) * int(e.get("execution_repeats", 1))
            rows.append({"program": pid, "mode": mode, "round": rnd, "atomic": atomic, "units": units,
                         "timeout": BUDGET["atomic" if atomic else "regular"]})
    # longest first
    rows.sort(key=lambda r: (-r["units"], r["mode"] != "B", r["program"]))
    return rows


def freeze(run_id):
    import torch
    import triton
    man = S.manifest()
    out = S.RESULTS / run_id
    out.mkdir(parents=True, exist_ok=True)
    tmpl = json.loads((S.PKG / "RUN_FREEZE.template.json").read_text())
    pc = S.package_check()
    if not pc["ok"]:
        sys.exit(f"package changed: {pc}")
    dm = detector_manifest()
    (out / "detector_files_manifest.json").write_text(json.dumps(dm, indent=1) + "\n")
    fam = families(man)
    (out / "families.json").write_text(json.dumps(fam, indent=1) + "\n")
    js = jobs(man)
    (out / "jobs.json").write_text(json.dumps(js, indent=1) + "\n")
    from kernel_analyzer.reference_eval.analysis import RULES as RULE_DEFS, rule_base
    rules = {}
    for cls, members in S.RULE_CLASSES.items():
        for r in members:
            b = rule_base(r)
            rules[r] = {"class": cls, "frozen_name": b, "definition": RULE_DEFS[b]["definition"],
                        "positive": RULE_DEFS[b]["positive"], "negative": RULE_DEFS[b]["negative"],
                        "coordinates": "valid on every development unit (assess_units)",
                        "development_source": "seeds 0-31 (both rounds)" if b == "fixed_direction" else
                                              ("per-unit reference midpoint (alignment)" if b in ("toward_zero", "scale_down")
                                               else "constant"),
                        "weights_saved": "per-unit normalization scalars in each record; learned vectors (R5) in the "
                                         "job's arrays file (sha256 in the job)"}
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *WATCHED)
    freeze_doc = dict(tmpl)
    freeze_doc.update({
        "status": "FILLED_BEFORE_MEASUREMENT",
        "run_id": run_id,
        "filled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool_commit": head,
        "tool_tag": "general-v3.1 (src tree and frozen paths identical; HEAD adds repository tidying and this adapter)",
        "tool_src_tree": git("rev-parse", "HEAD:src"),
        "tool_src_tree_at_tag": git("rev-parse", "general-v3.1:src"),
        "tool_dirty": bool(dirty),
        "tool_dirty_paths": dirty.splitlines(),
        "tool_version": S.check.TOOL_VERSION,
        "semantic_registry_sha256": sha(ROOT / "results/general/rule_registry.json"),
        "semantic_registry_files": {p: sha(ROOT / p) for p in ["results/general/rule_registry.json",
                                                               "results/reference_eval/ttir_op_registry.json"]},
        "detector_files_manifest_sha256": sha(out / "detector_files_manifest.json"),
        "rules_resolved": {"rule_classes": S.RULE_CLASSES, "rules_run": S.RULES, "rules": rules,
                           "code_sha256": {p: sha(ROOT / p) for p in ["src/kernel_analyzer/reference_eval/analysis.py",
                                                                    "src/kernel_analyzer/measure.py",
                                                                    "src/kernel_analyzer/check.py",
                                                                    "scripts/essential/contract_v3.py"]},
                           "source": "frozen production defaults (measure.DEFAULT_RULE_CLASSES, check.RULES)"},
        "package_sha256sums_sha256": pc["sha256sums_sha256"],
        "protocol_sha256": sha(S.PKG / "PROTOCOL.md"),
        "package_path": str(S.PKG.relative_to(ROOT)),
        "bindings": {"generator": "package tools/make_bindings.py", "dir": str(S.BINDINGS.relative_to(ROOT)),
                     "sha256": {p.name: sha(p) for p in sorted(S.BINDINGS.glob("*.py"))}},
        "seed_blocks": {"development": [0, 31], "confirmation": [32, 95], "replication": [96, 191],
                        "replication_development": "the development block 0-31 fixes the coordinate set and the "
                                                   "learned direction in the replication round too; confirmation units "
                                                   "96-191 (96 units)"},
        "prior_confirmation_used_to_develop": False,
        "alpha": S.ALPHA,
        "multiplicity": {"method": "Holm",
                         "families_resolved": {"file": "families.json", "sha256": sha(out / "families.json"),
                                               "count": len(fam),
                                               "definition": "one family per round x program x output x comparison "
                                                             "(FR_e_num, FR_e_sem, F_total, baseline) x rule class "
                                                             "(fixed_mean: R1, R5; aligned: R2, R3)"},
                         "missing_cells": "retain_as_unestablished; no post-hoc family shrinkage: a registered member "
                                          "without a p-value enters Holm as p = 1; the frozen class_statistics (Holm over "
                                          "tested members) is reported next to it and is never the final column",
                         "rounds": "confirmation and replication corrected separately; never pooled",
                         "mode_A": "consistency run of FR_e_num only; not a scoring family"},
        "statistics_method_id": "general-v3.1 frozen decision layer: analysis.assess_units (strict projection bounds, "
                                "endpoint-conservative t per rule) + measure.class_statistics (contract_v3 cannot-judge "
                                "rules S0 = 2, N0 = 64, n_min = 16, BOOTSTRAP_OK_N = 64)",
        "statistics_assumptions": ["statistical unit: the input seed (atomic programs: the mean of 8 executions)",
                                   "units independent and identically distributed under the package input generator",
                                   "the t approximation for the mean projection; not trusted when |skewness| > 2 with "
                                   "n <= 64 (cannot judge (distribution)) or n < 16 (cannot judge (sample))",
                                   "zero sample variance / identical endpoints: cannot judge (degenerate), never p = 0",
                                   "approximate route only: no validated magnitude bound, so no finite-sample "
                                   "distribution-free guarantee is claimed"],
        "bound_M_source": "none (bounded route not applied by the entry; no validated per-unit magnitude bound)",
        "equivalence_delta": None,
        "equivalence_delta_basis": "not run: no delta with an independent basis was declared before measurement",
        "atomic_repetitions": 8,
        "atomic_independent_unit": tmpl["atomic_independent_unit"],
        "atomic_protocol": "unit id = seed * 8 + execution; each execution regenerates the inputs from the seed "
                           "(digests checked equal) and the program allocates a fresh zeroed output; per-execution "
                           "residual intervals are averaged within the input (exact sums one ulp outward, exact "
                           "division by 8); order metadata (execution index, launch order, launch time, stream) saved",
        "budget": {"gpu_seconds_per_program": {"per job (program x mode x round), regular": BUDGET["regular"],
                                               "per job, atomic (8 executions per input)": BUDGET["atomic"]},
                   "reference_seconds_per_program": "included in the per-job budget (capture + reference + "
                                                    "specification + statistics); over budget is a result",
                   "localization_seconds": 0},
        "localization": "not attempted in this run",
        "isolated_boundary": "prog_26 not run: isolated opt-in not approved; counted in the 33-item denominator as not "
                             "established and listed apart",
        "comparison_objects": {"FR_e_num": "K - G (check.run record)", "FR_e_sem": "G - f (check.run record)",
                               "F_total": "K - f (check.run_black_box definition)",
                               "baseline": "K - ref_fp64, point residual, same rules, no detector "
                                           "(scripts/general/acceptance_run.py baseline definition)",
                               "E": "K vs ref_fp64 element-wise, descriptive",
                               "P": "family property relation, raw residuals, not_scored"},
        "reference_scope": "kernel-level only; call-level not established (PROTOCOL section 1)",
        "jobs": {"file": "jobs.json", "sha256": sha(out / "jobs.json"), "count": len(js)},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "triton": triton.__version__,
                        "gpu_arch": f"{torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}",
                        "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                                 capture_output=True, text=True).stdout.split()[0],
                        "compile_options": "Triton 3.6.0 defaults plus the launch options in each program (num_warps); "
                                           "no TRITON_* environment overrides; check.run sets allow_tf32 = False and "
                                           "float32 matmul precision 'highest'; TRITON_CACHE_DIR under the repository",
                        "triton_env": {k: v for k, v in os.environ.items() if k.startswith("TRITON_")}},
        "exposure_log": [
            {"item": "v1 package, v1 answers, history/", "status": "never opened (history/ seen only as a file listing)"},
            {"item": "v1.1 package documents (README, PROTOCOL, CHANGELOG, docs/, manifest, seal_status, template)",
             "status": "read for integration on 2026-10-08"},
            {"item": "v1.1 program sources", "status": "inspected during integration (the programs ship in the package): "
                                                       "at least prog_01, prog_08, prog_11 in full, T7 programs by search"},
            {"item": "package CPU validation and GPU preflight (seed 0, all 32 measurement programs)",
             "status": "run 2026-10-08; outputs: shapes, dtypes, digests, non-finite counts only"},
            {"item": "adapter pilot / smoke", "status": "prog_01, prog_03, prog_07 on units 0-2; prog_01, prog_08, "
                                                        "prog_11 on seeds 0-7 (development block only); seen: reference "
                                                        "completeness, element-wise E/F, failure reasons"},
            {"item": "confirmation seeds 32-95 and replication seeds 96-191", "status": "not run before this freeze"},
            {"item": "tool development", "status": "the executor is the tool developer; the tool was frozen at tag "
                                                   "general-v3.1 (2026-10-08 14:34 +0800) before the package was "
                                                   "received; no tool change after receipt"},
            {"item": "reviewer opinions", "status": "seen: repository-tidy proposal and earlier round audits; none "
                                                    "about the v1.1 labels"}],
        "answer_commitment": {"status": "ISSUER_RESEAL_REQUIRED", "sha256": None},
        "scoring": "no official blind sensitivity / specificity score until the issuer reseals (seal_status.json)",
    })
    (out / "RUN_FREEZE.json").write_text(json.dumps(freeze_doc, indent=1, default=str) + "\n")
    print(f"wrote {out / 'RUN_FREEZE.json'}: {len(fam)} families, {len(js)} jobs, dirty={bool(dirty)}")


def check_freeze(run_id):
    out = S.RESULTS / run_id
    rel = str((out / "RUN_FREEZE.json").relative_to(ROOT))
    if git("status", "--porcelain", "--", rel):
        sys.exit("RUN_FREEZE.json is not committed")
    fz = json.loads((out / "RUN_FREEZE.json").read_text())
    tc = fz["tool_commit"]
    if subprocess.run(["git", "merge-base", "--is-ancestor", tc, "HEAD"], cwd=ROOT).returncode:
        sys.exit("tool commit is not an ancestor of HEAD")
    changed = git("diff", "--name-only", tc, "HEAD", "--", *WATCHED)
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *WATCHED)
    dirty = "\n".join(l for l in dirty.splitlines() if "__pycache__" not in l)
    if changed or dirty or fz["tool_dirty"]:
        sys.exit(f"watched paths changed since the freeze: {changed} {dirty}")
    if S.package_check()["sha256sums_sha256"] != fz["package_sha256sums_sha256"] or not S.package_check()["ok"]:
        sys.exit("package changed since the freeze")
    if sha(out / "jobs.json") != fz["jobs"]["sha256"]:
        sys.exit("jobs.json changed")
    return fz


def run(run_id, workers, gpus):
    check_freeze(run_id)
    out = S.RESULTS / run_id
    js = json.loads((out / "jobs.json").read_text())
    logs = S.RUNS_CACHE / run_id / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HOME=str(CACHE), XDG_CACHE_HOME=str(CACHE), TRITON_CACHE_DIR=str(CACHE / "triton"),
               TMPDIR=str(CACHE / "tmp"), PYTHONPATH="src", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    pending = [j for j in js if not (out / "jobs" / f"{j['round']}_{j['mode']}_{j['program']}.json").exists()]
    running = {}
    slot = 0
    print(f"{len(pending)} of {len(js)} jobs pending", flush=True)
    while pending or running:
        while pending and len(running) < workers:
            j = pending.pop(0)
            tag = f"{j['round']}_{j['mode']}_{j['program']}"
            e = dict(env, CUDA_VISIBLE_DEVICES=str(gpus[slot % len(gpus)]))
            slot += 1
            log = open(logs / f"{tag}.log", "w")
            p = subprocess.Popen([PY, "scripts/acceptance/structure_v11.py", "--run-id", run_id, "--program",
                                  j["program"], "--mode", j["mode"], "--round", j["round"], "--timeout",
                                  str(j["timeout"])], cwd=ROOT, env=e, stdout=log, stderr=subprocess.STDOUT)
            running[p.pid] = (p, tag, time.time(), log)
        time.sleep(5)
        for pid, (p, tag, t0, log) in list(running.items()):
            if p.poll() is not None:
                log.close()
                print(f"{time.strftime('%H:%M:%S')} {tag} exit={p.returncode} {time.time() - t0:.0f}s "
                      f"({len(pending)} pending, {len(running) - 1} running)", flush=True)
                del running[pid]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--run-id", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run-id", required=True)
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--gpus", default="0,1,2,3")
    a = ap.parse_args()
    if a.cmd == "freeze":
        freeze(a.run_id)
    else:
        run(a.run_id, a.workers, [int(x) for x in a.gpus.split(",")])


if __name__ == "__main__":
    main()
