"""Unified entry (general-capability round, docs/protocol_general_capability_v1_20261008.md section 1).

The user declares four things -- the legal call, the input / state sources, the comparison (mode A / B and the
measured outputs) and the budget -- and the tool expands the rest: default direction rules in two classes (fixed
mean: R1, R5; reference alignment: R2, R3; Holm inside each class), the development / confirmation split, versions
and factor levels, the allowed reference classes, the resolution target.  Nothing kernel-specific is written by the
user or by this module: no reference, no suspicious node, no kernel name.

    from kernel_analyzer import measure
    report = measure.run("declaration.json")          # or a dict

Declaration::

    {"call": "path/to/file.py:function" | "package.module:function",   # function(inputs: dict) -> {name: tensor}
     "inputs": {"x": {"sampler": {"normal": [0, 1]}, "shape": [64, 64], "dtype": "float32"},
                "s": {"state": "file.py:make_state", "shape": [...], "dtype": "..."},   # state(seed, shape, dtype)
                "k": {"const": 16}},
     "compare": {"mode": "A" | "B", "measure": ["out"], "spec": "file.py:spec"},        # spec only in mode B
     "budget": {"cpu_seconds": 600, "gpu_seconds": 600, "case_timeout": 300, "max_units": 96},
     # optional: "units": {"development": 32, "confirmation": 64, "seed_offset": 0}, "alpha": 0.05,
     #           "resolution": {"ulp_fraction": 0.125}, "error_budget": "file.py:bound"  (bounded route)}

A factor is declared by giving a list where one value is expected ("dtype": ["float32", "bfloat16"]); every level is a
separate comparison.  Missing items raise ``MissingDeclaration`` with the list -- the entry never guesses an input
domain.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import signal
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from . import check

DEFAULT_RULE_CLASSES = {"fixed_mean": ["R1", "R5"], "aligned": ["R2", "R3"]}
DEFAULT_UNITS = {"development": 32, "confirmation": 64, "seed_offset": 0}
DEFAULT_RESOLUTION = {"ulp_fraction": 0.125}
ALLOWED_REFERENCE_CLASSES = ["complete_composed"]
SAMPLERS = ("normal", "uniform", "randint", "bernoulli", "lognormal", "sparse", "rare_tail")
N_MIN = 16


class MissingDeclaration(ValueError):
    """The declaration lacks items the entry needs; ``items`` lists them (dotted paths)."""

    def __init__(self, items):
        self.items = list(items)
        super().__init__("declaration incomplete: " + "; ".join(self.items))


# ------------------------------------------------------------------------------------------------ declaration

def load(decl) -> dict:
    if isinstance(decl, (str, Path)):
        p = Path(decl)
        d = json.loads(p.read_text())
        d.setdefault("_base_dir", str(p.parent))
        return d
    return copy.deepcopy(decl)


def missing_items(decl: dict) -> list:
    out = []
    if not decl.get("call"):
        out.append("call (module:function or file.py:function)")
    ins = decl.get("inputs")
    if not isinstance(ins, dict) or not ins:
        out.append("inputs (one source per input)")
    else:
        for name, src in ins.items():
            if not isinstance(src, dict):
                out.append(f"inputs.{name} (a source object)")
                continue
            if "const" in src:
                continue
            if "sampler" not in src and "state" not in src:
                out.append(f"inputs.{name}.sampler or inputs.{name}.state")
            if "shape" not in src:
                out.append(f"inputs.{name}.shape")
            if "dtype" not in src:
                out.append(f"inputs.{name}.dtype")
            for key in ("dtype", "shape"):
                v = src.get(key)
                if isinstance(v, str) and v.startswith("$") and v[1:] not in (decl.get("factors") or {}):
                    out.append(f"factors.{v[1:]} (named by inputs.{name}.{key})")
            if "sampler" in src:
                smp = src["sampler"]
                if not isinstance(smp, dict) or len(smp) != 1 or next(iter(smp)) not in SAMPLERS:
                    out.append(f"inputs.{name}.sampler (one of {', '.join(SAMPLERS)} with its parameters)")
    cmp_ = decl.get("compare")
    if not isinstance(cmp_, dict):
        out.append("compare (mode and measured outputs)")
    else:
        if cmp_.get("mode") not in ("A", "B"):
            out.append("compare.mode (A or B)")
        if not cmp_.get("measure"):
            out.append("compare.measure (output names)")
        if cmp_.get("mode") == "B" and not cmp_.get("spec"):
            out.append("compare.spec (mode B needs f)")
    bud = decl.get("budget")
    if not isinstance(bud, dict):
        out.append("budget (cpu_seconds, gpu_seconds, case_timeout, max_units)")
    else:
        for k in ("cpu_seconds", "gpu_seconds", "case_timeout", "max_units"):
            if k not in bud:
                out.append(f"budget.{k}")
    return out


def _versions():
    import triton
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"torch": torch.__version__, "triton": triton.__version__, "device": dev, "tool": check.TOOL_VERSION,
            "python": sys.version.split()[0]}


def _factor_levels(inputs: dict, factors: Optional[dict] = None) -> list:
    """Every input field given as a list (where a single value is expected) is a factor; a field written "$name"
    takes the declared factor ``factors[name]`` (one axis shared by every input that names it).  The cartesian
    product of the factors are the levels (each a separate comparison)."""

    axes = []
    for fname, values in (factors or {}).items():
        users = [(n, k) for n, src in inputs.items() for k in ("dtype", "shape") if src.get(k) == f"${fname}"]
        axes.append([tuple((n, k, tuple(x) if k == "shape" else x) for n, k in users) for x in values])
    combos = []
    for name, src in inputs.items():
        for key in ("dtype", "shape"):
            v = src.get(key)
            if key == "dtype" and isinstance(v, list):
                axes.append([((name, key, x),) for x in v])
            if key == "shape" and isinstance(v, list) and v and isinstance(v[0], list):
                axes.append([((name, key, tuple(x)),) for x in v])
    if not axes:
        return [{}]
    for combo in itertools.product(*axes):
        combos.append({f"{n}.{k}": x for group in combo for n, k, x in group})
    return combos


def expand(decl: dict) -> dict:
    """The full comparison declaration (written into the report; never changed during confirmation)."""

    miss = missing_items(decl)
    if miss:
        raise MissingDeclaration(miss)
    units = dict(DEFAULT_UNITS, **(decl.get("units") or {}))
    exp = {"call": decl["call"], "inputs": decl["inputs"], "compare": decl["compare"], "budget": decl["budget"],
           "rule_classes": copy.deepcopy(DEFAULT_RULE_CLASSES), "multiplicity": "Holm within each rule class",
           "units": units, "alpha": decl.get("alpha", 0.05),
           "reference_classes_allowed": ALLOWED_REFERENCE_CLASSES,
           "resolution": dict(DEFAULT_RESOLUTION, **(decl.get("resolution") or {})),
           "factors": decl.get("factors") or {},
           "factor_levels": _factor_levels(decl["inputs"], decl.get("factors")), "versions": _versions(),
           "error_budget": decl.get("error_budget"),
           "mixed_sources": "outputs reading a non-Triton intermediate: numerical difference only, no semantic verdict",
           "statistics": "frozen endpoint-conservative t per rule (analysis._summarize) + contract_v3 cannot-judge "
                         "rules (S0 = 2, N0 = 64, n_min = 16); bounded route when an error budget is declared"}
    body = json.dumps({k: decl[k] for k in ("call", "inputs", "compare", "budget")}, sort_keys=True, default=str)
    exp["declaration_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    exp["_base_dir"] = decl.get("_base_dir", ".")
    return exp


# ------------------------------------------------------------------------------------------------ callables, inputs

def resolve(ref: str, base_dir: str = ".") -> Callable:
    mod_part, fn = ref.rsplit(":", 1)
    if mod_part.endswith(".py"):
        p = Path(mod_part)
        if not p.is_absolute():
            p = Path(base_dir) / p
            if not p.exists():
                p = Path(mod_part)
        spec = importlib.util.spec_from_file_location(f"_measure_{p.stem}_{abs(hash(str(p)))}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(mod_part)
    return getattr(mod, fn)


_DTYPES = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16, "float16": torch.float16,
           "int32": torch.int32, "int64": torch.int64, "bool": torch.bool}


def _sample(smp: dict, shape, rng: np.random.Generator) -> np.ndarray:
    kind, par = next(iter(smp.items()))
    if kind == "normal":
        return rng.normal(par[0], par[1], shape)
    if kind == "uniform":
        return rng.uniform(par[0], par[1], shape)
    if kind == "randint":
        return rng.integers(par[0], par[1], shape)
    if kind == "bernoulli":
        return (rng.random(shape) < par).astype(np.float64)
    if kind == "lognormal":
        return rng.lognormal(par[0], par[1], shape)
    if kind == "sparse":                                 # {"sparse": [density, std]}
        return np.where(rng.random(shape) < par[0], rng.normal(0, par[1], shape), 0.0)
    if kind == "rare_tail":                              # {"rare_tail": [p, magnitude, std_of_rest]}
        big = np.where(rng.random(shape) < par[0], par[1] * rng.choice([-1.0, 1.0], shape), 0.0)
        return big + rng.normal(0, par[2], shape)
    raise ValueError(kind)


def make_inputs(exp: dict, level: dict, seed: int, device="cuda") -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(exp["declaration_sha256"][:8], 16)]))
    out = {}
    for name, src in exp["inputs"].items():
        if "const" in src:
            out[name] = src["const"]
            continue
        dtype = level.get(f"{name}.dtype", src["dtype"])
        shape = tuple(level.get(f"{name}.shape", src["shape"]))
        if "state" in src:
            fn = resolve(src["state"], exp["_base_dir"])
            out[name] = fn(seed, shape, dtype)
            continue
        a = _sample(src["sampler"], shape, rng)
        out[name] = torch.tensor(a, dtype=torch.float64).to(_DTYPES[dtype]).to(device)
    out["_seed"] = seed
    return out


# ------------------------------------------------------------------------------------------------ analysis helpers

def ulp_of(x: np.ndarray, dtype: str) -> np.ndarray:
    mant = {"float32": 23, "bfloat16": 7, "float16": 10, "float64": 52}.get(dtype, 23)
    emin = {"float32": -126, "bfloat16": -126, "float16": -14, "float64": -1022}.get(dtype, -126)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.maximum(np.abs(x), 2.0 ** emin)))
    return np.exp2(np.maximum(e, emin) - mant)


def reference_quality(keep_rows: list, dtype: str, ulp_fraction: float) -> dict:
    widths, resolved, n_ok, n_all = [], 0, 0, 0
    for r in keep_rows:
        ok = np.asarray(r["ok"], bool)
        lo, hi = np.asarray(r["r_lo"], np.float64), np.asarray(r["r_hi"], np.float64)
        n_all += ok.size
        n_ok += int(ok.sum())
        if ok.any():
            mid = 0.5 * (lo[ok] + hi[ok])
            w = (hi[ok] - lo[ok]) / ulp_of(mid, dtype)
            widths.append(w)
            resolved += int((w <= ulp_fraction).sum())
    w = np.concatenate(widths) if widths else np.zeros(0)
    q = (lambda p: float(np.quantile(w, p)) if w.size else None)
    return {"elements": n_all, "complete_finite": n_ok, "complete_rate": n_ok / n_all if n_all else None,
            "width_over_ulp": {"median": q(0.5), "p90": q(0.9), "max": float(w.max()) if w.size else None},
            "resolution_target_ulp_fraction": ulp_fraction,
            "resolved_fraction": resolved / n_ok if n_ok else None}


def _holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [1.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def class_statistics(num_rec: dict, rule_classes: dict, alpha: float) -> dict:
    """Per class: Holm inside the class over the frozen per-rule records, then the v3 cannot-judge rules."""

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "essential"))
    import contract_v3 as CV
    rules = {r.get("rule"): r for r in (num_rec or {}).get("rules", []) if isinstance(r, dict)}
    out = {}
    for cls, names in rule_classes.items():
        recs = [rules.get(n) for n in names]
        tested = [(n, r) for n, r in zip(names, recs) if r and "p_value_two_sided_conservative" in r]
        adj = dict(zip([n for n, _ in tested], _holm([r["p_value_two_sided_conservative"] for _, r in tested])))
        per = {}
        for n, r in zip(names, recs):
            if r is None:
                per[n] = {"judgment": "not established", "basis": "rule not computed"}
                continue
            j = CV.statistical_judgment(r)
            if j["judgment"].startswith("nonzero") and n in adj and adj[n] > alpha:
                j = {"judgment": "not confirmed", "basis": f"Holm in class '{cls}': adjusted p = {adj[n]:.3g} > {alpha}"}
            j["holm_adjusted_p"] = adj.get(n)
            j["mean_projection"] = r.get("mean_projection")
            j["n"] = r.get("n")
            per[n] = j
        js = [v["judgment"] for v in per.values()]
        if any(x.startswith("nonzero") for x in js):
            summary = "average effect nonzero: " + ", ".join(f"{n} {v['judgment'][9:-1]}" for n, v in per.items()
                                                             if v["judgment"].startswith("nonzero"))
        elif all(x.startswith(("cannot judge", "not established")) for x in js):
            summary = "cannot judge / not established (no statement about bias)"
        else:
            summary = "not confirmed on these projections (not a statement that the vector has no bias)"
        out[cls] = {"rules": per, "summary": summary}
    return out


def bounded_mean_test(a: np.ndarray, M: np.ndarray, alpha: float) -> dict:
    """Bounded route: per unit |a_i| <= M_i (from the declared error budget).  Hoeffding's inequality for independent
    a_i in [-M_i, M_i]: P(|mean - mu| >= t) <= 2 exp(-2 n^2 t^2 / sum (2 M_i)^2).  Two-sided at level alpha."""

    a = np.asarray(a, np.float64)
    M = np.asarray(M, np.float64)
    n = a.size
    if n < 2 or not np.all(np.isfinite(M)) or np.any(np.abs(a) > M * (1 + 1e-12)):
        return {"verdict": "NOT_ESTABLISHED", "reason": "bound violated or missing"}
    t = math.sqrt(np.sum((2 * M) ** 2) * math.log(2 / alpha) / (2 * n * n))
    m = float(a.mean())
    lo, hi = m - t, m + t
    return {"method": "Hoeffding (declared per-unit bound)", "mean": m, "half_width": t, "interval": [lo, hi],
            "verdict": "DETECTED_POSITIVE" if lo > 0 else ("DETECTED_NEGATIVE" if hi < 0 else "NOT_CONFIRMED")}


# ------------------------------------------------------------------------------------------------ failure classes

FAILURE_CLASSES = ("semantics missing", "binding", "enclosure too wide", "over budget", "statistics insufficient")
_FAILURE_TABLE = [   # (substring of the tool's reason, class) -- first match wins; printed with every report
    ("unrecognized", "semantics missing"), ("no declared semantics", "semantics missing"),
    ("not supported", "semantics missing"), ("unsupported", "semantics missing"), ("unknown op", "semantics missing"),
    ("no derivative rule", "semantics missing"), ("store through a not-established address", "semantics missing"),
    ("atomic", "semantics missing"), ("order is not declared", "semantics missing"),
    ("cross-program race", "semantics missing"), ("inline_asm", "semantics missing"),
    ("binding", "binding"), ("not written by triton", "binding"), ("modified after", "binding"),
    ("another recorded storage", "binding"), ("out-of-bounds", "binding"), ("outside the captured windows", "binding"),
    ("non-triton intermediate", "binding"), ("shape varies across units", "binding"),
    ("pinned_load", "enclosure too wide"), ("undecided", "enclosure too wide"), ("overlap", "enclosure too wide"),
    ("straddl", "enclosure too wide"), ("may be zero", "enclosure too wide"), ("may be negative", "enclosure too wide"),
    ("too wide", "enclosure too wide"), ("width", "enclosure too wide"),
    ("timeout", "over budget"), ("budget", "over budget"),
    ("cannot judge", "statistics insufficient"), ("unresolved_sample", "statistics insufficient"),
]


def classify_failure(reason: str) -> str:
    r = str(reason).lower()
    for key, cls in _FAILURE_TABLE:
        if key in r:
            return cls
    return "semantics missing" if "not_established" in r else "unclassified"


# ------------------------------------------------------------------------------------------------ run

class _Timeout(Exception):
    pass


def _alarm(seconds):
    def handler(signum, frame):
        raise _Timeout(f"timeout after {seconds} s")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(max(1, seconds)))
    return old


def _semantic(keep_rows, fas, name):
    n_ok = n_sem = 0
    worst = 0.0
    for r in keep_rows:
        f_lo, f_hi = (np.asarray(a, np.float64).ravel() for a in fas[r["seed"]][name])
        ok = np.asarray(r["ok"], bool) & np.isfinite(f_lo)
        lo, hi = np.asarray(r["r_lo"], np.float64), np.asarray(r["r_hi"], np.float64)
        with np.errstate(invalid="ignore"):
            gap = np.maximum(lo - f_hi, f_lo - hi)
        sem = ok & (gap > 0)
        n_ok += int(ok.sum())
        n_sem += int(sem.sum())
        if sem.any():
            worst = max(worst, float((gap / (1 + np.abs(0.5 * (f_lo + f_hi))))[sem].max()))
    return {"ok_elements": n_ok, "e_sem_certified": n_sem, "max_gap_relative": worst}


def run_level(exp: dict, level: dict) -> dict:
    call = resolve(exp["call"], exp["_base_dir"])
    measure_names = list(exp["compare"]["measure"])
    spec_fn = resolve(exp["compare"]["spec"], exp["_base_dir"]) if exp["compare"]["mode"] == "B" else None
    budget = exp["budget"]
    u = exp["units"]
    n_total = min(u["development"] + u["confirmation"], int(budget["max_units"]))
    seeds = [u["seed_offset"] + i for i in range(n_total)]
    n_dev = min(u["development"], max(1, n_total // 3))
    dev, conf = seeds[:n_dev], seeds[n_dev:]
    fas = {}
    shapes = {}

    class Case(check.Case):
        name = f"measure:{exp['call']}:{level}"
        implementation = exp["call"]
        specification = exp["compare"].get("spec") or "none (mode A)"

        def setup(self):
            torch._dynamo.reset()
            self.launch(self.inputs(seeds[0]))
            torch.cuda.synchronize()

        def inputs(self, seed):
            return make_inputs(exp, level, seed)

        def launch(self, inp):
            outs = call({k: v for k, v in inp.items() if k != "_seed"})
            sel = {}
            for k in measure_names:
                if k not in outs:
                    raise KeyError(f"call returned no output '{k}' (returned: {sorted(outs)})")
                sel[k] = outs[k]
            for k, v in sel.items():
                if k in shapes and shapes[k] != tuple(v.shape):
                    raise ValueError(f"output '{k}' shape varies across units ({shapes[k]} vs {tuple(v.shape)}): the "
                                     "coordinate frame must be fixed -- declare the structure (e.g. a sparsity "
                                     "pattern) as a fixed input source")
            shapes.update({k: tuple(v.shape) for k, v in sel.items()})
            self._last_dtypes = {k: str(v.dtype).replace("torch.", "") for k, v in sel.items()}
            return sel

        def spec(self, inp):
            if spec_fn is None:
                return None
            f = spec_fn({k: v for k, v in inp.items() if k != "_seed"})
            fas[inp["_seed"]] = {k: tuple(np.asarray(a, np.float64).ravel() for a in f[k]) for k in measure_names}
            return {k: tuple(np.asarray(a, np.float64).reshape(shapes[k]) for a in f[k]) for k in measure_names}

    case = Case()
    keep = {}
    t0 = time.time()
    old = _alarm(budget["case_timeout"])
    try:
        rep = check.run(case, dev=dev, conf=conf, keep=keep)
    except _Timeout as exc:
        return {"level": level, "status": "over budget", "reason": str(exc), "failure_class": "over budget",
                "seconds": round(time.time() - t0, 1)}
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"[:400]
        cls = classify_failure(msg)
        if cls == "unclassified":
            cls = "binding"                              # the call rejected the declared inputs (legal-call violation)
            msg = "the call rejected the declared inputs (legal call / input domain): " + msg
        return {"level": level, "status": "error", "reason": msg, "failure_class": cls,
                "seconds": round(time.time() - t0, 1)}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    seconds = time.time() - t0
    dtypes = getattr(case, "_last_dtypes", {})
    outputs = {}
    failures = []
    notes = {k: rep.get(k) for k in ("outputs_not_written_by_triton", "outputs_binding_not_established",
                                      "outputs_modified_after_last_triton_write", "outputs_whose_writing_programs_aborted",
                                      "outputs_at_address_of_another_recorded_storage", "ttir_coverage_complete")}
    for name in measure_names:
        o = rep.get("outputs", {}).get(name)
        if o is None:
            why = ("not written by Triton" if name in (notes["outputs_not_written_by_triton"] or []) else
                   "binding not established" if name in (notes["outputs_binding_not_established"] or []) else
                   "; ".join((notes["outputs_whose_writing_programs_aborted"] or {}).get(name, [])) or "not evaluated")
            outputs[name] = {"status": "not established", "reason": why, "failure_class": classify_failure(why)}
            failures.append(outputs[name]["failure_class"])
            continue
        quality = reference_quality(keep.get(name, []), dtypes.get(name, "float32"), exp["resolution"]["ulp_fraction"])
        stats = class_statistics(o.get("numerical"), exp["rule_classes"], exp["alpha"])
        entry = {"status": "evaluated", "reference": quality, "statistics": stats,
                 "mixed_non_triton_sources": o.get("depends_on_non_triton_intermediates") or [],
                 "not_established_reasons_seed0": o.get("not_established_reasons_seed0"),
                 "special_values": o.get("special_values")}
        reasons = list((o.get("not_established_reasons_seed0") or {}).keys())
        entry["failure_classes"] = []
        if quality["complete_rate"] is not None and quality["complete_rate"] < 1.0:
            # only then do the tool's reasons explain missing elements (otherwise they are notes such as the dot
            # input precision or a checked premise)
            entry["failure_classes"] = sorted({classify_failure(r) for r in reasons if not str(r).startswith(
                ("assumed:", "dot_input_precision:"))}) or ["unclassified"]
        if quality["resolved_fraction"] is not None and quality["resolved_fraction"] < 1.0:
            entry["failure_classes"] = sorted(set(entry["failure_classes"]) | {"enclosure too wide"})
        if any(v["summary"].startswith("cannot judge") for v in stats.values()):
            entry["failure_classes"] = sorted(set(entry["failure_classes"]) | {"statistics insufficient"})
        if spec_fn is not None and name in keep:
            entry["semantic"] = _semantic(keep[name], fas, name)
            entry["semantic"]["verdict_scope"] = ("mixed: not a semantic verdict" if entry["mixed_non_triton_sources"]
                                                  else "pure Triton: K_R vs f_r exact comparison")
        failures.extend(entry["failure_classes"])
        outputs[name] = entry
    return {"level": level, "status": "ok", "units": {"development": len(dev), "confirmation": len(conf)},
            "notes": notes, "outputs": outputs, "timing_seconds": rep.get("timing_seconds"),
            "seconds": round(seconds, 1), "failure_classes": sorted(set(failures)),
            "over_budget": seconds > float(budget["gpu_seconds"])}


def run(decl, out: Optional[str] = None) -> dict:
    d = load(decl)
    exp = expand(d)
    t0 = time.time()
    levels = []
    for level in exp["factor_levels"]:
        if time.time() - t0 > float(exp["budget"]["gpu_seconds"]):
            levels.append({"level": level, "status": "over budget", "reason": "total budget spent before this level",
                           "failure_class": "over budget"})
            continue
        levels.append(run_level(exp, level))
    counts = {c: 0 for c in FAILURE_CLASSES}
    for lv in levels:
        for c in lv.get("failure_classes", []) + ([lv["failure_class"]] if "failure_class" in lv else []):
            counts[c] = counts.get(c, 0) + 1
    report = {"tool_version": check.TOOL_VERSION, "expanded_declaration": {k: v for k, v in exp.items() if k != "_base_dir"},
              "levels": levels, "failure_counts": counts, "failure_table": _FAILURE_TABLE,
              "seconds": round(time.time() - t0, 1)}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=1, default=str) + "\n")
    return report
