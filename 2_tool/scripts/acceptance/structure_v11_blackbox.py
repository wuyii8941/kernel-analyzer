#!/usr/bin/env python3
"""Supplementary jobs (deviation D1 of the run): the comparison objects that do not need K_R -- F_total (K - f)
and baseline (K - ref_fp64) -- for outputs whose automatic reference was not established in the main jobs.
structure_v11.py computed them only for evaluated outputs; PROTOCOL section 5 asks that an unsupported method not
block the others.

    python scripts/acceptance/structure_v11_blackbox.py --run-id ID --program prog_08 --round main --timeout 3600

F_total is the frozen check.run_black_box record (same decision layer); baseline is the frozen harness's definition
on the same K (the case keeps K per unit); E is descriptive.  Same Holm families (missing members p = 1).
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import structure_v11 as S  # noqa: E402
from structure_v11 import check, measure, acceptance_run  # noqa: E402


class BlackBoxCase(S.AcceptanceCase):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.k = {}

    def launch(self, inp):
        out = super().launch(inp)
        self.k[self.unit] = {n: v.detach().double().cpu().numpy().reshape(-1) for n, v in out.items()}
        return out


def run_job(pid, rnd, run_id, timeout):
    e = S.manifest()[pid]
    if int(e.get("execution_repeats", 1)) != 1:
        raise SystemExit("atomic programs are not part of this supplement")
    dev_s, conf_s = S.ROUNDS[rnd]
    mod, bpath = S.load_binding(pid, "B")
    case = BlackBoxCase(mod, pid, 1, dev_s[0])
    job = {"schema": "structure-v1.1-rc1-supplement-v1", "deviation": "D1", "run_id": run_id, "program": pid,
           "family": e["family"], "mode": "B", "round": rnd, "comparisons_run": ["F_total", "baseline"],
           "seeds": {"development": [dev_s[0], dev_s[-1]], "confirmation": [conf_s[0], conf_s[-1]]},
           "binding": {"path": str(bpath.relative_to(S.ROOT)), "sha256": S.sha256_file(bpath)},
           "frozen": acceptance_run.frozen_check(), "package": S.package_check(), "versions": measure._versions(),
           "tool_version": check.TOOL_VERSION, "timeout_seconds": timeout}
    if not job["frozen"]["ok"] or not job["package"]["ok"]:
        job.update(status="refused", reason="tool not frozen or package changed")
        return job, {}
    t0 = time.time()
    old = measure._alarm(timeout)
    try:
        rep = check.run_black_box(case, dev=dev_s, conf=conf_s)
    except measure._Timeout as exc:
        job.update(status="over budget", reason=str(exc), failure_class="over budget", seconds=round(time.time() - t0, 1))
        return job, {}
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"[:400]
        job.update(status="error", reason=msg, failure_class=measure.classify_failure(msg),
                   traceback=traceback.format_exc()[-3000:], seconds=round(time.time() - t0, 1))
        return job, {}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    arrays, outputs = {}, {}
    seeds = dev_s + conf_s
    for name in e["output_names"]:
        o = rep["outputs"][name]
        rec = o["total_black_box"]
        frozen = measure.class_statistics(rec, S.RULE_CLASSES, S.ALPHA)
        k = np.stack([case.k[s][name] for s in seeds])
        ref = np.stack([case.ref[s][name] for s in seeds])
        f_lo = np.stack([case.f[s][name][0].reshape(-1) for s in seeds])
        f_hi = np.stack([case.f[s][name][1].reshape(-1) for s in seeds])
        eb = k - ref
        eb_ok = np.isfinite(eb) & np.isfinite(ref)
        t_ok = np.isfinite(f_lo) & np.isfinite(f_hi) & np.isfinite(k)
        outputs[name] = {
            "status": "black-box comparisons only (automatic reference not established in the main job)",
            "k_sha256_per_unit": [__import__("hashlib").sha256(np.asarray(r, np.float32).tobytes()).hexdigest() for r in k],
            "comparisons": {
                "F_total": {"record": rec, "class_statistics_frozen": frozen,
                            "class_statistics_protocol": S.protocol_holm(rec, frozen),
                            "source": "check.run_black_box (frozen)"},
                "baseline": S.statistics(f"baseline:{name}", np.where(eb_ok, eb, 0.0), np.where(eb_ok, eb, 0.0), ref,
                                         eb_ok, len(dev_s), seeds, False, arrays, f"{name}::baseline")},
            "elementwise": {"E_K_vs_ref64": S.elementwise_point(k, ref, np.isfinite(k) & np.isfinite(ref)),
                            "F_K_minus_f": S.elementwise_interval(k, f_lo, f_hi, t_ok)},
            "special_values": o.get("special_values")}
    job.update(status="ok", outputs=outputs, timing_seconds=rep.get("timing_seconds"), executions=case.executions,
               seconds=round(time.time() - t0, 1))
    return job, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--program", required=True)
    ap.add_argument("--round", choices=sorted(S.ROUNDS), required=True)
    ap.add_argument("--timeout", type=float, required=True)
    a = ap.parse_args()
    out_dir = S.RESULTS / a.run_id / "supplement_D1"
    arr_dir = S.RUNS_CACHE / a.run_id / "arrays"
    out_dir.mkdir(parents=True, exist_ok=True)
    arr_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.round}_B_{a.program}"
    job, arrays = run_job(a.program, a.round, a.run_id, a.timeout)
    if arrays:
        p = arr_dir / f"D1_{tag}.npz"
        np.savez_compressed(p, **{k.replace("::", "__").replace(" ", "_").replace(":", "_"): v for k, v in arrays.items()})
        job["arrays"] = {"path": str(p.relative_to(S.ROOT)), "sha256": S.sha256_file(p), "keys": sorted(arrays)}
    (out_dir / f"{tag}.json").write_text(json.dumps(job, indent=1, default=str) + "\n")
    print(tag, job.get("status"), job.get("seconds"), flush=True)


if __name__ == "__main__":
    main()
