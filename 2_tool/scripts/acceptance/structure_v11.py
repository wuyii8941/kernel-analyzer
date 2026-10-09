#!/usr/bin/env python3
"""Structure acceptance set v1.1-rc1 under the frozen tool (tag general-v3.1): one program x mode x round per process.

    python scripts/acceptance/structure_v11.py --run-id ID --program prog_01 --mode B --round main --timeout 3600

The package exposes programs, inputs, an independent task specification f (exact rationals, outward float64) and a
plain FP64 baseline through its ``binding_support``; its ``tools/make_bindings.py`` generated one check.BindingCase
binding per program and mode.  This adapter adds no detector or statistics logic.  Capture, K_R, residual intervals
and the decision layer are the frozen ``check.run`` / ``analysis.assess_units`` / ``measure.reference_quality`` /
``measure.class_statistics``.  The adapter only arranges the comparison objects of the package protocol
(PROTOCOL.md section 2) and its unit rules (sections 3-4), each with an existing frozen definition:

  FR_e_num   K - G       check.run's record (same call recomputed from the kept arrays for the learned directions,
                         and checked equal)
  FR_e_sem   G - f       idem (mode B)
  F_total    K - f       check.run_black_box's definition (residual_interval(K, f), f and K finite, alignment f mid)
  baseline   K - ref64   scripts/general/acceptance_run.py baseline's definition (point residual in float64,
                         R1 / R2 / R3 / R5, no detector), on the same K as FR
  E          K vs ref64 element-wise, descriptive        P  the family's property relation, raw, not_scored

Atomic programs (manifest execution_repeats > 1): every input runs `repeats` times (the program allocates a fresh
output buffer per launch); per-execution residual intervals are averaged within the input with outward rounding
(exact sums one ulp outward, exact division by the power of two `repeats`), and the input is the unit.  check.run's
own statistics for these programs treat executions as units and are dropped from the saved report.

Multiplicity (RUN_FREEZE): one Holm family per round x program x output x comparison x rule class with its
registered members; a member without a p-value enters as p = 1, so a family is never shrunk.  The frozen
class_statistics (Holm over the tested members) is saved next to it; the protocol column can only be stricter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "general"))
from kernel_analyzer import check, measure  # noqa: E402
from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval.analysis import assess_units, residual_interval  # noqa: E402
import acceptance_run  # noqa: E402  (frozen: frozen_check)

ACC = ROOT.parent / ".cache" / "acceptance"
PKG = ACC / "structure_acceptance_v1_1_rc1"
BINDINGS = ACC / "acceptance_v11_bindings"
RESULTS = ROOT.parent / ".cache" / "acceptance" / "structure_v1_1_rc1"  # extract the frozen run archive here
RUNS_CACHE = ACC / "runs"
ROUNDS = {"main": (list(range(0, 32)), list(range(32, 96))),
          "replication": (list(range(0, 32)), list(range(96, 192)))}
RULE_CLASSES = measure.DEFAULT_RULE_CLASSES
RULES = check.RULES
ALPHA = 0.05
ULP_FRACTION = measure.DEFAULT_RESOLUTION["ulp_fraction"]
COMPARISONS = {"A": ["FR_e_num"], "B": ["FR_e_num", "FR_e_sem", "F_total", "baseline"]}


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def manifest() -> dict:
    return {e["id"]: e for e in json.loads((PKG / "manifest.json").read_text())["programs"]}


def package_check() -> dict:
    """every file listed in the package's SHA256SUMS still has its hash"""
    bad = []
    for line in (PKG / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        h, rel = line.split(None, 1)
        rel = rel.strip().lstrip("*")
        p = PKG / rel
        if not p.exists() or sha256_file(p) != h:
            bad.append(rel)
    return {"sha256sums_sha256": sha256_file(PKG / "SHA256SUMS"), "mismatched": bad, "ok": not bad}


def load_binding(pid: str, mode: str):
    path = BINDINGS / f"{pid}_mode{mode}.py"
    spec = importlib.util.spec_from_file_location(f"acc_v11_{pid}_{mode}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


class AcceptanceCase(check.Case):
    """unit u = seed * repeats + execution; inputs depend on the seed only"""

    def __init__(self, mod, pid, repeats, warm_unit):
        self.mod, self.pid, self.repeats, self.warm_unit = mod, pid, repeats, warm_unit
        self.name = getattr(mod, "NAME", pid)
        self.implementation = getattr(mod, "IMPLEMENTATION", "")
        self.specification = getattr(mod, "SPECIFICATION", "none (mode A)")
        self.spec_bound = getattr(mod, "SPEC_BOUND", "rigorous float64 enclosure")
        self.f, self.ref, self.digests, self.executions, self.dtypes = {}, {}, {}, [], {}
        self.unit = None

    def seed_of(self, unit):
        return unit // self.repeats

    def setup(self):  # as measure.run_level: empty Dynamo cache, one warm-up launch outside the recorder
        torch._dynamo.reset()
        self.mod.run(self.mod.make_inputs(self.seed_of(self.warm_unit)))
        torch.cuda.synchronize()

    def inputs(self, unit):
        self.unit = unit
        seed = self.seed_of(unit)
        inp = self.mod.make_inputs(seed)
        d = sorted(check.input_digests(inp))
        if self.digests.setdefault(seed, d) != d:
            raise RuntimeError(f"inputs of seed {seed} differ between executions")
        if seed not in self.ref:
            r = self.mod.bs.ordinary_reference(self.pid, inp)
            self.ref[seed] = {k: v.detach().double().cpu().numpy().reshape(-1) for k, v in r.items()}
        return inp

    def launch(self, inp):
        t = time.time_ns()
        out = self.mod.run(inp)
        self.executions.append({"unit": self.unit, "seed": self.seed_of(self.unit),
                                "execution": self.unit % self.repeats, "order": len(self.executions),
                                "launch_start_ns": t, "stream": int(torch.cuda.current_stream().cuda_stream),
                                "contiguous": all(v.is_contiguous() for v in out.values())})
        self.dtypes = {k: str(v.dtype).replace("torch.", "") for k, v in out.items()}
        return out

    def spec(self, inp):
        if not hasattr(self.mod, "spec"):
            return None
        seed = self.seed_of(self.unit)
        if seed not in self.f:
            f = self.mod.spec(inp)
            self.f[seed] = {k: (np.asarray(lo, np.float64), np.asarray(hi, np.float64)) for k, (lo, hi) in f.items()}
        return self.f[seed]


# ------------------------------------------------------------------------------------------------ helpers

def within_input_mean(lo, hi, ok, repeats):
    """(units = seeds x repeats) -> seeds: outward enclosure of the mean over the executions; ok in every one"""
    if repeats == 1:
        return lo, hi, ok
    s, c = lo.shape[0] // repeats, lo.shape[1]
    okr = ok.reshape(s, repeats, c)
    lo3 = np.where(okr, lo.reshape(s, repeats, c), 0.0)
    hi3 = np.where(okr, hi.reshape(s, repeats, c), 0.0)
    s_lo, s_hi = iv.fsum_bounds(lo3, hi3, axis=1)
    return iv.div_bounds(s_lo, float(repeats))[0], iv.div_bounds(s_hi, float(repeats))[1], okr.all(axis=1)


def per_input_point(a, repeats):
    """a quantity that must not depend on the execution (K_R midpoint, f, ref64): the value and its spread"""
    if repeats == 1:
        return a, 0.0
    s = a.shape[0] // repeats
    a3 = a.reshape(s, repeats, -1)
    with np.errstate(invalid="ignore"):
        spread = float(np.nanmax(np.abs(a3 - a3[:, :1]))) if a3.size else 0.0
    return a3[:, 0], spread


def _summary(per):
    js = [v["judgment"] for v in per.values()]
    if any(x.startswith("nonzero") for x in js):
        return "average effect nonzero: " + ", ".join(f"{n} {v['judgment'][9:-1]}" for n, v in per.items()
                                                      if v["judgment"].startswith("nonzero"))
    if all(x.startswith(("cannot judge", "not established")) for x in js):
        return "cannot judge / not established (no statement about bias)"
    return "not confirmed on these projections (not a statement that the vector has no bias)"


def protocol_holm(rec, frozen):
    """Holm over the registered members of each class; a member without a p-value counts as p = 1"""
    rules = {r.get("rule"): r for r in (rec or {}).get("rules", []) if isinstance(r, dict)}
    out = {}
    for cls, names in RULE_CLASSES.items():
        p = [rules.get(n, {}).get("p_value_two_sided_conservative") for n in names]
        adj = measure._holm([1.0 if x is None else float(x) for x in p])
        per = {}
        for n, pn, a in zip(names, p, adj):
            j = dict(frozen[cls]["rules"][n])
            if j["judgment"].startswith("nonzero") and a > ALPHA:
                j.update(judgment="not confirmed",
                         basis=f"Holm over the registered family {names} (missing p = 1): adjusted p = {a:.3g} > {ALPHA}")
            j["holm_adjusted_p_registered_family"] = a
            j["p_missing"] = pn is None
            per[n] = j
        out[cls] = {"family": list(names), "rules": per, "summary": _summary(per)}
    return out


def statistics(label, lo, hi, align, ok, n_dev, unit_ids, detector, arrays, key):
    rec, arr = assess_units(label, lo, hi, align, ok, n_dev, RULES, alignment_reference=align, unit_ids=unit_ids,
                            run_detector=detector)
    for k, v in arr.items():
        arrays[f"{key}::{k}"] = np.asarray(v)
    frozen = measure.class_statistics(rec, RULE_CLASSES, ALPHA)
    return {"record": rec, "class_statistics_frozen": frozen, "class_statistics_protocol": protocol_holm(rec, frozen)}


def _canon(x):
    return json.dumps(x, sort_keys=True, default=str)


def same_record(a, b):
    a = {k: v for k, v in (a or {}).items() if k != "scale"}
    b = {k: v for k, v in (b or {}).items() if k != "scale"}
    if _canon(a) == _canon(b):
        return {"equal": True}
    return {"equal": False, "differing_keys": sorted(k for k in set(a) | set(b) if _canon(a.get(k)) != _canon(b.get(k)))}


def ulp32(x):
    return measure.ulp_of(np.asarray(x, np.float64), "float32")


def elementwise_point(k, ref, ok):
    """E: K against a point reference (descriptive)"""
    if not ok.any():
        return {"elements": 0}
    d, r = (k - ref)[ok], ref[ok]
    u = ulp32(r)
    return {"elements": int(ok.sum()), "max_abs": float(np.abs(d).max()),
            "relative_rms": float(np.sqrt((d ** 2).mean() / max((r ** 2).mean(), 1e-300))),
            "max_over_ulp": float((np.abs(d) / u).max()), "mean_signed_over_ulp": float((d / u).mean()),
            "fraction_over_half_ulp": float((np.abs(d) / u > 0.5).mean()),
            "fraction_bitwise_equal_after_rounding_ref": float((k[ok] == r.astype(np.float32).astype(np.float64)).mean())}


def elementwise_interval(k, f_lo, f_hi, ok):
    """F: K - f element-wise; an interval that excludes 0 is sample-difference evidence, not a contract violation"""
    if not ok.any():
        return {"elements": 0}
    lo, hi = residual_interval(k, f_lo, f_hi)
    lo, hi, mid = lo[ok], hi[ok], 0.5 * (f_lo + f_hi)[ok]
    u = ulp32(mid)
    return {"elements": int(ok.sum()), "interval_excludes_zero_fraction": float(((lo > 0) | (hi < 0)).mean()),
            "positive_fraction": float((lo > 0).mean()), "negative_fraction": float((hi < 0).mean()),
            "max_abs_over_ulp": float((np.maximum(np.abs(lo), np.abs(hi)) / u).max()),
            "mean_mid_over_ulp": float((0.5 * (lo + hi) / u).mean()),
            "f_width_over_ulp_max": float(((f_hi - f_lo)[ok] / u).max())}


def _fsum_rows(a):
    return np.array([math.fsum(r) for r in np.asarray(a, np.float64).reshape(a.shape[0], -1).tolist()])


def property_raw(family, inp, ys):
    """P (section 5): raw relation residuals per row, float64; None when the family has no listed property"""
    x = inp["x"].detach().double().cpu().numpy()
    if family == "T2":
        y = ys["output"].reshape(x.shape)
        d = np.diff(y, axis=1, prepend=0.0) - x * float(inp["s"])
        return {"relation": "y[j] - y[j-1] - x[j] * s (exact target 0; float differences need a budget)",
                "per_row_max_abs": np.abs(d).max(1), "per_row_mean": d.mean(1),
                "scale": np.abs(x * float(inp["s"])).max(1)}
    if family == "T5":
        y = ys["output"].reshape(x.shape)
        return {"relation": "row sum of y (exact target 0; float centering does not give bitwise 0)",
                "per_row": _fsum_rows(y), "scale": _fsum_rows(np.abs(y))}
    if family == "T6":
        y = ys["output"].reshape(x.shape)
        dim = x.shape[1]
        m2 = _fsum_rows(x * x) / dim
        target = dim * m2 / (m2 + float(inp["eps"]))
        return {"relation": "sum(y^2) - D * m2 / (m2 + eps), m2 = mean(x^2) (zero rows apart)",
                "per_row": _fsum_rows(y * y) - target, "scale": target, "zero_rows": int((m2 == 0).sum())}
    return None


# ------------------------------------------------------------------------------------------------ one job

def run_job(pid, mode, rnd, run_id, timeout):
    man = manifest()
    e = man[pid]
    if e["execution_lane"] != "measurement":
        raise SystemExit(f"{pid}: lane {e['execution_lane']}: not run (isolated opt-in not approved)")
    repeats = int(e.get("execution_repeats", 1))
    family, out_names = e["family"], list(e["output_names"])
    dev_s, conf_s = ROUNDS[rnd]
    dev_u = [s * repeats + r for s in dev_s for r in range(repeats)]
    conf_u = [s * repeats + r for s in conf_s for r in range(repeats)]
    mod, bpath = load_binding(pid, mode)
    case = AcceptanceCase(mod, pid, repeats, dev_u[0])
    job = {"schema": "structure-v1.1-rc1-job-v1", "run_id": run_id, "program": pid, "family": family, "mode": mode,
           "round": rnd, "execution_repeats": repeats, "outputs_expected": out_names,
           "seeds": {"development": [dev_s[0], dev_s[-1]], "confirmation": [conf_s[0], conf_s[-1]]},
           "statistical_unit": "input seed" + (f" (mean of {repeats} executions)" if repeats > 1 else ""),
           "binding": {"path": str(bpath.relative_to(ROOT)), "sha256": sha256_file(bpath)},
           "frozen": acceptance_run.frozen_check(), "package": package_check(), "versions": measure._versions(),
           "tool_version": check.TOOL_VERSION, "timeout_seconds": timeout}
    if not job["frozen"]["ok"] or not job["package"]["ok"]:
        job["status"] = "refused"
        job["reason"] = "tool not frozen or package changed"
        return job, {}
    keep = {}
    t0 = time.time()
    old = measure._alarm(timeout)
    try:
        rep = check.run(case, dev=dev_u, conf=conf_u, keep=keep)
    except measure._Timeout as exc:
        job.update(status="over budget", reason=str(exc), failure_class="over budget",
                   seconds=round(time.time() - t0, 1))
        return job, {}
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"[:400]
        job.update(status="error", reason=msg, failure_class=measure.classify_failure(msg),
                   traceback=traceback.format_exc()[-3000:], seconds=round(time.time() - t0, 1))
        return job, {}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    job["seconds_check_run"] = round(time.time() - t0, 1)
    t1 = time.time()
    arrays = {}
    notes = {k: rep.get(k) for k in ("outputs_not_written_by_triton", "outputs_binding_not_established",
                                      "outputs_modified_after_last_triton_write", "outputs_whose_writing_programs_aborted",
                                      "outputs_at_address_of_another_recorded_storage", "ttir_coverage_complete",
                                      "launches", "masked_lane_assumption", "external_reentries_seed0")}
    seeds_all = dev_s + conf_s
    n_dev = len(dev_s)
    outputs = {}
    for name in out_names:
        o = rep.get("outputs", {}).get(name)
        if o is None:
            why = ("not written by Triton" if name in (notes["outputs_not_written_by_triton"] or []) else
                   "binding not established" if name in (notes["outputs_binding_not_established"] or []) else
                   "; ".join((notes["outputs_whose_writing_programs_aborted"] or {}).get(name, [])) or "not evaluated")
            outputs[name] = {"status": "not established", "reason": why, "failure_class": measure.classify_failure(why)}
            continue
        rows = keep[name]
        dtype = case.dtypes.get(name, "float32")
        quality = measure.reference_quality(rows, dtype, ULP_FRACTION)
        mixed = o.get("depends_on_non_triton_intermediates") or []
        backfilled = [m for m in mixed if not m.endswith("[copy of inputs]")]
        scope = {"tool_classification": ("kernel-level: upstream non-Triton values captured (" +
                                         ", ".join(backfilled)[:200] + ")") if backfilled else "call-level",
                 "upstream_non_triton_values": mixed, "protocol_scope": "kernel-level",
                 "call_level": "not established: PROTOCOL section 1 does not upgrade storage provenance shown only by "
                               "equal content or all-zero content to call level"}
        k = np.stack([r["k"] for r in rows])
        r_lo, r_hi = np.stack([r["r_lo"] for r in rows]), np.stack([r["r_hi"] for r in rows])
        ok = np.stack([r["ok"] for r in rows])
        kr = 0.5 * (r_lo + r_hi)
        unit_ids = [r["seed"] for r in rows]
        contiguous = all(x["contiguous"] for x in case.executions)
        full = o["reference_classes"]["written_fraction"] == 1.0
        entry = {"status": "evaluated", "shape": o["shape"], "dtype": dtype, "reference": quality, "scope": scope,
                 "reference_classes": o["reference_classes"], "special_values": o.get("special_values"),
                 "not_established_reasons_seed0": o.get("not_established_reasons_seed0"),
                 "k_sha256_per_unit": [hashlib.sha256(np.asarray(r["k"], np.float32).tobytes()).hexdigest()
                                       for r in rows],
                 "comparisons": {}, "elementwise": {},
                 "equivalence_status": "not run (no equivalence delta declared in RUN_FREEZE)",
                 "localization_status": "not attempted (RUN_FREEZE localization budget 0)"}
        cmp_ = entry["comparisons"]
        n_lo, n_hi = residual_interval(k, r_lo, r_hi)
        a_lo, a_hi, a_ok = within_input_mean(n_lo, n_hi, ok, repeats)
        a_kr, kr_spread = per_input_point(kr, repeats)
        label = f"{name}: e_num = K - K_R"
        st = statistics(label, a_lo, a_hi, a_kr, a_ok, n_dev, seeds_all, True, arrays, f"{name}::FR_e_num")
        if repeats == 1:
            st["same_as_check_run_record"] = same_record(st["record"], o.get("numerical"))
        else:
            st["K_R_midpoint_spread_across_executions"] = kr_spread
        st["scale"] = (o.get("numerical") or {}).get("scale") if repeats == 1 else None
        cmp_["FR_e_num"] = st
        fas = None
        if mode == "B":
            if not (contiguous and full):
                entry["mode_B_note"] = "f not mapped: output not contiguous or not fully written"
            else:
                f_lo = np.stack([case.f[case.seed_of(u)][name][0].reshape(-1) for u in unit_ids])
                f_hi = np.stack([case.f[case.seed_of(u)][name][1].reshape(-1) for u in unit_ids])
                s_lo, s_hi = iv.isub(r_lo, r_hi, f_lo, f_hi)  # fully written: no frame fill to reproduce
                b_lo, b_hi, b_ok = within_input_mean(s_lo, s_hi, ok, repeats)
                st = statistics(f"{name}: e_sem = K_R - f", b_lo, b_hi, a_kr, b_ok, n_dev, seeds_all, True, arrays,
                                f"{name}::FR_e_sem")
                if repeats == 1:
                    st["same_as_check_run_record"] = same_record(st["record"], o.get("semantic"))
                cmp_["FR_e_sem"] = st
                fas = {u: {name: (f_lo[i], f_hi[i])} for i, u in enumerate(unit_ids)}
                entry["semantic_exact_comparison"] = measure._semantic(rows, fas, name)
                entry["semantic_exact_comparison"]["verdict_scope"] = (
                    "mixed: not a semantic verdict" if mixed else "pure Triton: K_R vs f_r exact comparison")
                # F: check.run_black_box's K - f
                t_lo, t_hi = residual_interval(k, f_lo, f_hi)
                f_fin = np.isfinite(f_lo) & np.isfinite(f_hi)
                t_ok = f_fin & np.isfinite(k)
                t_lo, t_hi = np.where(t_ok, t_lo, 0.0), np.where(t_ok, t_hi, 0.0)
                fm = np.where(f_fin, 0.5 * (f_lo + f_hi), 0.0)
                c_lo, c_hi, c_ok = within_input_mean(t_lo, t_hi, t_ok, repeats)
                c_fm, fm_spread = per_input_point(fm, repeats)
                cmp_["F_total"] = statistics(f"{name}: K - f (black box)", c_lo, c_hi, c_fm, c_ok, n_dev, seeds_all,
                                             True, arrays, f"{name}::F_total")
                entry["elementwise"]["F_K_minus_f"] = elementwise_interval(k, f_lo, f_hi, t_ok)
                # baseline: scripts/general/acceptance_run.py's definition on the same K
                ref = np.stack([case.ref[case.seed_of(u)][name] for u in unit_ids])
                eb = k - ref
                eb_ok = np.isfinite(eb) & np.isfinite(ref)
                d_lo, d_hi, d_ok = within_input_mean(np.where(eb_ok, eb, 0.0), np.where(eb_ok, eb, 0.0), eb_ok, repeats)
                d_ref, _ = per_input_point(ref, repeats)
                cmp_["baseline"] = statistics(f"baseline:{name}", d_lo, d_hi, d_ref, d_ok, n_dev, seeds_all, False,
                                              arrays, f"{name}::baseline")
                entry["elementwise"]["E_K_vs_ref64"] = elementwise_point(k, ref, np.isfinite(k) & np.isfinite(ref))
                entry["elementwise"]["FR_total_check_run"] = o.get("total")
                entry["elementwise"]["FR_e_sem_check_run"] = o.get("semantic_elementwise")
        # failure classes, as measure.run_level
        reasons = list((o.get("not_established_reasons_seed0") or {}).keys())
        fc = []
        if quality["complete_rate"] is not None and quality["complete_rate"] < 1.0:
            fc = sorted({measure.classify_failure(r) for r in reasons
                         if not str(r).startswith(("assumed:", "dot_input_precision:"))}) or ["unclassified"]
        if quality["resolved_fraction"] is not None and quality["resolved_fraction"] < 1.0:
            fc = sorted(set(fc) | {"enclosure too wide"})
        if any(v["summary"].startswith("cannot judge") for v in cmp_["FR_e_num"]["class_statistics_frozen"].values()):
            fc = sorted(set(fc) | {"statistics insufficient"})
        entry["failure_classes"] = fc
        outputs[name] = entry
    # P: property relations (raw, not scored), from the kept K and regenerated inputs
    prop = {"status": "not_scored", "family": family}
    if family in ("T2", "T5", "T6") and all(n in keep for n in out_names):
        per_unit = []
        for i, r in enumerate(keep[out_names[0]]):
            u = r["seed"]
            inp = mod.make_inputs(case.seed_of(u))
            p = property_raw(family, inp, {n: keep[n][i]["k"] for n in out_names})
            per_unit.append(p)
        prop["relation"] = per_unit[0]["relation"]
        vals = np.stack([p.get("per_row", p.get("per_row_max_abs")) for p in per_unit])
        scale = np.stack([p["scale"] for p in per_unit])
        arrays["P::per_row"], arrays["P::scale"] = vals, scale
        prop.update(units=len(per_unit), max_abs=float(np.abs(vals).max()), mean=float(vals.mean()),
                    max_abs_over_ulp_of_scale=float((np.abs(vals) / ulp32(np.maximum(scale, 1e-300))).max()),
                    mean_over_ulp_of_scale=float((vals / ulp32(np.maximum(scale, 1e-300))).mean()),
                    zero_rows=sum(p.get("zero_rows", 0) for p in per_unit),
                    note="no budget or exact-domain premise declared: raw residuals only, not a counterexample")
    elif family == "T7" and "output" in keep:
        k = np.stack([r["k"] for r in keep["output"]])
        s = k.shape[0] // repeats
        k3 = k.reshape(s, repeats, -1)
        differ = (k3 != k3[:, :1]).any(axis=1)
        spread = (k3.max(1) - k3.min(1)) / ulp32(np.maximum(np.abs(k3).max(1), 1e-300))
        arrays["P::K_per_execution"] = k3.astype(np.float32)
        prop.update(relation="real dot product independent of blocking; float atomic value may depend on the order",
                    inputs=s, executions_per_input=repeats,
                    fraction_elements_not_bitwise_identical_across_executions=float(differ.mean()),
                    inputs_with_any_difference=int(differ.any(axis=1).sum()),
                    max_spread_over_ulp=float(spread.max()), mean_spread_over_ulp=float(spread.mean()),
                    note="order-dependence evidence only; no budget or exact-domain check declared")
    elif family == "T1":
        prop["note"] = "mean and variance compared with FP64 is a reference comparison (E / baseline), not a property"
    else:
        prop["note"] = "no property listed for this family in PROTOCOL section 5"
    # the tool report; execution-level statistics of atomic programs are dropped (section 4)
    tool_report = {k: v for k, v in rep.items() if k != "outputs"}
    tool_report["outputs"] = {}
    for name, o in rep.get("outputs", {}).items():
        o = dict(o)
        if repeats > 1:
            for key in ("numerical", "semantic"):
                if key in o:
                    o[key] = {"dropped": "statistics over executions as units violate PROTOCOL section 4; the "
                                         "input-level records are in outputs.<name>.comparisons"}
        tool_report["outputs"][name] = o
    job.update(status="ok", notes={k: v for k, v in notes.items() if k not in ("launches",)},
               launches=notes["launches"], outputs=outputs, property=prop, executions=case.executions,
               tool_report=tool_report, timing_seconds=rep.get("timing_seconds"),
               seconds_derivation=round(time.time() - t1, 1), seconds=round(time.time() - t0, 1))
    return job, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--program", required=True)
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--round", choices=sorted(ROUNDS), required=True)
    ap.add_argument("--timeout", type=float, required=True)
    a = ap.parse_args()
    out_dir = RESULTS / a.run_id / "jobs"
    arr_dir = RUNS_CACHE / a.run_id / "arrays"
    out_dir.mkdir(parents=True, exist_ok=True)
    arr_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.round}_{a.mode}_{a.program}"
    job, arrays = run_job(a.program, a.mode, a.round, a.run_id, a.timeout)
    if arrays:
        p = arr_dir / f"{tag}.npz"
        np.savez_compressed(p, **{k.replace("::", "__").replace(" ", "_").replace(":", "_"): v for k, v in arrays.items()})
        job["arrays"] = {"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p), "keys": sorted(arrays)}
    (out_dir / f"{tag}.json").write_text(json.dumps(job, indent=1, default=str) + "\n")
    print(tag, job.get("status"), job.get("seconds"), flush=True)


if __name__ == "__main__":
    main()
