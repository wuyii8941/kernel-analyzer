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
     #           "resolution": {"ulp_fraction": 0.125}, "magnitude_bound": {"elementwise": M, "basis": "..."},
     #           "equivalence": {"rel": delta, "basis": "..."}, "repeats": 2, "error_budget": "file.py:bound",
     #           "query": {"axes": ["nonzero"] | ["nonzero", "equivalence"]}}

The bias query (composition rules Q1-Q3) is fixed in ``expanded_declaration.query`` before any unit is drawn: the
comparison target, the input distribution, the observable (outputs, projections, coordinate set) and the sampling
unit, with the decision axes and the rule families.  The nonzero axis is always reported; the equivalence axis only
when requested, and then a tolerance ``equivalence.rel`` is required (declaring it requests the axis).

``declaration_sha256`` binds the whole expanded declaration -- every setting that affects a conclusion (units and
seeds, alpha, factors, resolution, M, delta, repeats, the statistics family and the policies), the versions and the
content hashes of the call / input generator / specification sources (audit F06).  The input samples are drawn from
``sampling_seed_sha256`` (call, inputs, compare, budget only), so changing a statistics setting does not change the
sampled inputs.

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
DEFAULT_RESOLUTION = {"ulp_fraction": 0.125, "max_level": 3}
ALLOWED_REFERENCE_CLASSES = ["complete_composed"]
QUERY_AXES = ("nonzero", "equivalence")
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
    eq = decl.get("equivalence")
    if eq is not None and not (isinstance(eq, dict) and isinstance(eq.get("rel"), (int, float)) and eq["rel"] > 0):
        out.append("equivalence.rel (a positive relative bound delta)")
    q = decl.get("query")
    if q is not None:
        axes = q.get("axes") if isinstance(q, dict) else None
        if not isinstance(axes, list) or not axes or not set(axes) <= set(QUERY_AXES) or "nonzero" not in axes:
            out.append("query.axes (a list of " + " / ".join(QUERY_AXES) + ", the nonzero axis always included)")
        elif "equivalence" in axes and eq is None:
            out.append("equivalence.rel (required because query.axes includes equivalence: the tolerance delta is "
                       "declared before the data)")
        elif "equivalence" not in axes and eq is not None:
            out.append("query.axes (equivalence.rel is declared but query.axes excludes the equivalence axis)")
    rep = decl.get("repeats")
    if rep is not None and not (isinstance(rep, int) and rep >= 1):
        out.append("repeats (launches per input, an integer >= 1)")
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


def _ref_file(ref: str, base_dir: str) -> Optional[Path]:
    """The source file of a "file.py:function" or "package.module:function" reference, or None."""
    mod_part = str(ref).rsplit(":", 1)[0]
    if mod_part.endswith(".py"):
        p = Path(mod_part)
        if not p.is_absolute():
            p = Path(base_dir) / p
            if not p.exists():
                p = Path(mod_part)
        return p if p.is_file() else None
    try:
        spec = importlib.util.find_spec(mod_part)
    except (ImportError, ValueError):
        return None
    return Path(spec.origin) if spec is not None and spec.origin and Path(spec.origin).is_file() else None


def source_hashes(decl: dict, base_dir: str) -> dict:
    """Content hashes of the sources a declaration names (call, input generators, specification, error budget)."""
    refs = {"call": decl.get("call")}
    for name, src in (decl.get("inputs") or {}).items():
        if isinstance(src, dict) and src.get("state"):
            refs[f"inputs.{name}.state"] = src["state"]
    if (decl.get("compare") or {}).get("spec"):
        refs["compare.spec"] = decl["compare"]["spec"]
    if isinstance(decl.get("error_budget"), str):
        refs["error_budget"] = decl["error_budget"]
    out = {}
    for key, ref in refs.items():
        if not ref:
            continue
        f = _ref_file(ref, base_dir)
        out[key] = {"ref": ref, "sha256": hashlib.sha256(f.read_bytes()).hexdigest() if f else None}
        if f is None:
            out[key]["note"] = "source file not found: the content is not bound"
    return out


def expand(decl: dict) -> dict:
    """The full comparison declaration (written into the report; never changed during confirmation)."""

    miss = missing_items(decl)
    if miss:
        raise MissingDeclaration(miss)
    units = dict(DEFAULT_UNITS, **(decl.get("units") or {}))
    base_dir = decl.get("_base_dir", ".")
    exp = {"call": decl["call"], "inputs": decl["inputs"], "compare": decl["compare"], "budget": decl["budget"],
           "rule_classes": copy.deepcopy(DEFAULT_RULE_CLASSES), "multiplicity": "Holm within each rule class",
           "units": units, "alpha": decl.get("alpha", 0.05),
           "reference_classes_allowed": ALLOWED_REFERENCE_CLASSES,
           "resolution": dict(DEFAULT_RESOLUTION, **(decl.get("resolution") or {})),
           "factors": decl.get("factors") or {},
           "factor_levels": _factor_levels(decl["inputs"], decl.get("factors")), "versions": _versions(),
           "error_budget": decl.get("error_budget"),
           "magnitude_bound": decl.get("magnitude_bound"),
           "equivalence": decl.get("equivalence"),
           "repeats": decl.get("repeats"),
           "repeats_policy": "launches per input: the declared repeats, else 2, or 8 when a launch has float atomics",
           "bounded_route": "magnitude_bound (a declared population bound M on |K - G| per element, not proven by the "
                            "tool) is passed to check.run: every rule reports the bounded route (Hoeffding on the "
                            "per-unit projections, per rule at alpha) next to the approximate route (endpoint-"
                            "conservative t); without M no distribution-free statement.  A declared "
                            "magnitude_bound.width_elementwise W (before the data) gives the pre-data detectable "
                            "effect 2 r_n + W; without it the detectable effect is conditional on the observed widths. "
                            "error_budget is recorded only (not applied)",
           "equivalence_policy": "equivalence.rel (delta relative to the reference scale q_R) is passed to check.run: "
                                 "every fixed-direction rule reports the equivalence axis (TOST); without delta no "
                                 "equivalence statement",
           "resolution_policy": "resolved_fraction is measured against resolution.ulp_fraction over the numerical "
                                "enclosures (set targets counted apart); while some output's target is not met and "
                                "can be, the level is measured again at a higher working precision (levels 1-3: SumK / "
                                "DotK K = 3, 5, 8, compensated prefix sums from level 2; up to resolution.max_level). "
                                "Stops with a category: met; budget exhausted; backend limit (float64 endpoints, the "
                                "highest level, or no precision-dependent rule on the reference path); intrinsic set "
                                "width; no numerical enclosure; pass failed.  An unchanged pass fraction is not a stop "
                                "reason.  Every level "
                                "is reported, an unmet target is 'enclosure too wide' with resolution_met = false, "
                                "never written as met",
           "mixed_sources": "outputs reading a non-Triton intermediate: numerical difference only, no semantic verdict",
           "statistics": "frozen endpoint-conservative t per rule (analysis._summarize) at alpha + contract_v3 "
                         "cannot-judge rules (S0 = 2, N0 = 64, n_min = 16)",
           "reference_scope": "call-level complete only when every float buffer the kernels read is a declared input's "
                              "own storage (unchanged), written by a recorded launch, or produced (by an unmeasured "
                              "traced run of the call, aligned launch for launch) only by copies of the declared inputs "
                              "or exact constants; otherwise kernel-level (upstream values captured) and not counted as "
                              "complete for the call.  Equal values (every element found among the inputs, an all-zero "
                              "buffer) are not provenance",
           "proof_status": "complete_rate counts proved references only; elements that hold only under an unproven "
                           "premise (CAS serializations without the lock commutativity certificate, scan bracketings "
                           "without the associativity certificate) are reported as complete_under_premise_rate and are "
                           "not used by the statistics; certificates are listed under proof_status",
           "source_hashes": source_hashes(decl, base_dir)}
    # the input sample stream: call, inputs, compare and budget only (the digest of earlier versions, kept so that the
    # same declaration draws the same inputs, and a statistics setting does not change them)
    body = json.dumps({k: decl[k] for k in ("call", "inputs", "compare", "budget")}, sort_keys=True, default=str)
    exp["sampling_seed_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    exp["query"] = query_of(decl, exp)
    from . import composition_rules
    exp["composition_rules"] = composition_rules.summary()
    exp["digest_scope"] = ("declaration_sha256: SHA-256 of the canonical JSON of every other field of this expanded "
                           "declaration")
    exp["declaration_sha256"] = hashlib.sha256(json.dumps(exp, sort_keys=True, default=str).encode()).hexdigest()
    exp["_base_dir"] = base_dir
    return exp


def unit_split(exp: dict) -> tuple:
    """(development seeds, confirmation seeds) actually drawn: at most budget.max_units units in all, of which at most
    a third (and at least one) for development."""
    u = exp["units"]
    n_total = min(u["development"] + u["confirmation"], int(exp["budget"]["max_units"]))
    seeds = [u["seed_offset"] + i for i in range(n_total)]
    n_dev = min(u["development"], max(1, n_total // 3))
    return seeds[:n_dev], seeds[n_dev:]


def query_of(decl: dict, exp: dict) -> dict:
    """The bias query fixed before the data (composition rule Q1): what is compared, over which input distribution,
    on which observable, with which sampling unit, along which axes and rule families."""
    from .reference_eval.analysis import RULES as RULE_DEFS, rule_base
    cmp_ = decl["compare"]
    mode = cmp_["mode"]
    axes = list((decl.get("query") or {}).get("axes") or
                (["nonzero", "equivalence"] if decl.get("equivalence") else ["nonzero"]))
    dev, conf = unit_split(exp)
    rules = sorted({r for names in exp["rule_classes"].values() for r in names})
    return {
        "comparison_target": {
            "mode": mode,
            "quantity": "e_num = K - G_pi" if mode == "A" else
                        f"e_num = K - G_pi and e_sem = G_pi - f (f: {cmp_.get('spec')})",
            "K": "the call's outputs on the inputs it actually received",
            "G_pi": "the reference generated from the captured program under the numerical-difference policy pi "
                    "(real arithmetic, rc3 02 1); call-level only when every float buffer read is a declared input, "
                    "a recorded launch's output or a recorded copy / exact constant (composition rules M1-M5)"},
        "input_distribution": {
            "inputs": decl["inputs"], "factors": decl.get("factors") or {},
            "sampling_seed_sha256": exp["sampling_seed_sha256"],
            "meaning": "each unit draws every input from its declared source with the unit's seed (numpy "
                       "SeedSequence of the seed and the sampling digest); state sources are called with the seed"},
        "observable": {
            "outputs": list(cmp_["measure"]),
            "projections": {r: RULE_DEFS[rule_base(r)]["definition"] for r in rules},
            "coordinate_set": "fixed on the development units: the elements whose reference is complete composed "
                              "(proved, finite, not conditional) in every development unit"},
        "sampling_unit": {
            "unit": "one input draw (one seed); the call is launched repeats times per unit and K is compared "
                    "launch by launch; residuals are averaged within a unit only for admitted float-atomic orders",
            "development_seeds": [dev[0], dev[-1]] if dev else [], "confirmation_seeds": [conf[0], conf[-1]] if conf
            else [], "development": len(dev), "confirmation": len(conf),
            "independence": "units are independent draws; launches within a unit are not units"},
        "axes": axes,
        "families": {cls: list(names) for cls, names in exp["rule_classes"].items()},
        "family_guarantee": "per declared rule class and per route: Holm over the class's rules, FWER <= alpha; the "
                            "approximate route (endpoint-conservative t, contract_v3 cannot-judge rules) and the "
                            "bounded route (Hoeffding, only with a declared magnitude_bound M) are separate families",
        "equivalence_tolerance": decl.get("equivalence") if "equivalence" in axes else None,
    }


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
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(exp["sampling_seed_sha256"][:8], 16)]))
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
    """Resolution of the numerical enclosures (complete finite elements) against ``ulp_fraction``; set targets (L_E)
    are counted apart: their width is the target's own, not an enclosure a working precision shrinks."""
    widths, resolved, n_ok, n_all, n_premise, n_set, set_w = [], 0, 0, 0, 0, 0, 0.0
    for r in keep_rows:
        ok = np.asarray(r["ok"], bool)
        lo, hi = np.asarray(r["r_lo"], np.float64), np.asarray(r["r_hi"], np.float64)
        n_all += ok.size
        n_ok += int(ok.sum())
        n_premise += int(np.asarray(r.get("premise", np.zeros(ok.size, bool)), bool).sum())
        st = np.asarray(r.get("set_target", np.zeros(ok.size, bool)), bool)
        n_set += int(st.sum())
        if st.any():
            sw = np.asarray(r.get("set_width", hi - lo), np.float64)
            set_w = max(set_w, float(np.max(sw[st])))
        if ok.any():
            mid = 0.5 * (lo[ok] + hi[ok])
            w = (hi[ok] - lo[ok]) / ulp_of(mid, dtype)
            widths.append(w)
            resolved += int((w <= ulp_fraction).sum())
    w = np.concatenate(widths) if widths else np.zeros(0)
    q = (lambda p: float(np.quantile(w, p)) if w.size else None)
    return {"elements": n_all, "complete_finite": n_ok, "complete_rate": n_ok / n_all if n_all else None,
            "complete_under_premise": n_premise,
            "complete_under_premise_rate": n_premise / n_all if n_all else None,
            "width_over_ulp": {"median": q(0.5), "p90": q(0.9), "max": float(w.max()) if w.size else None},
            "resolution_target_ulp_fraction": ulp_fraction,
            "resolved_fraction": resolved / n_ok if n_ok else None,
            "unresolved_elements": n_ok - resolved,
            "resolution_met": (resolved == n_ok) if n_ok else None,
            "set_target_elements": n_set, "set_target_max_width": set_w if n_set else None}


def _holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [1.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def class_statistics(num_rec: dict, rule_classes: dict, alpha: float, equivalence_rel=None) -> dict:
    """Per class: Holm inside the class over the frozen per-rule records, then the v3 cannot-judge rules.  ``axes``
    reports the two decision axes apart (composition rules Q2, Q3): the nonzero axis by route (approximate; bounded,
    with its own Holm family when a magnitude bound M was declared) and the equivalence axis only when it was
    requested (``equivalence_rel``)."""

    from . import contract_v3 as CV
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
            j["alpha"] = alpha
            j["equivalence"] = r.get("equivalence") or "no equivalence bound declared: no equivalence statement"
            j["mean_projection"] = r.get("mean_projection")
            j["n"] = r.get("n")
            j["mde_approximate"] = (r.get("mde_approximate") or {}).get("mde")
            if r.get("bounded"):  # DSL v2: the bounded route next to the approximate one (not merged into it)
                j["bounded"] = {k: r["bounded"].get(k) for k in ("verdict", "interval", "M", "M_basis",
                                                                 "detectable_effect_given_observed_widths",
                                                                 "pre_data_detectable_effect", "reason")}
            per[n] = j
        js = [v["judgment"] for v in per.values()]
        if any(x.startswith("nonzero") for x in js):
            summary = "average effect nonzero: " + ", ".join(f"{n} {v['judgment'][9:-1]}" for n, v in per.items()
                                                             if v["judgment"].startswith("nonzero"))
        elif all(x.startswith(("cannot judge", "not established")) for x in js):
            summary = "cannot judge / not established (no statement about bias)"
        else:
            summary = "not confirmed on these projections (not a statement that the vector has no bias)"
        out[cls] = {"rules": per, "summary": summary,
                    "axes": _axes(cls, names, recs, per, summary, alpha, equivalence_rel)}
    return out


def _axes(cls, names, recs, per, summary, alpha, equivalence_rel) -> dict:
    """The two decision axes of one rule class (Q2, Q3)."""
    nonzero = {"approximate": {"family": f"Holm within class '{cls}' at alpha = {alpha} (FWER <= alpha for the class)",
                               "rules": {n: per[n]["judgment"] for n in names}, "summary": summary}}
    bounded = [(n, (r or {}).get("bounded")) for n, r in zip(names, recs)]
    tested = [(n, b) for n, b in bounded if b and b.get("p_value_two_sided") is not None]
    if not tested:
        why = sorted({str((b or {}).get("reason")) for _, b in bounded if b}) or ["no magnitude_bound declared"]
        nonzero["bounded"] = {"available": False, "reason": "; ".join(why)}
    else:
        adj = dict(zip([n for n, _ in tested], _holm([b["p_value_two_sided"] for _, b in tested])))
        rules = {}
        for n, b in bounded:
            if not b or n not in adj:
                rules[n] = {"family_verdict": "not established", "reason": (b or {}).get("reason") or "no record"}
                continue
            direction = b.get("direction") or ("positive" if b["interval"][0] > 0 else "negative")
            rules[n] = {"p": b["p_value_two_sided"], "holm_adjusted_p": adj[n], "interval": b.get("interval"),
                        "M": b.get("M"), "M_basis": b.get("M_basis"),
                        "family_verdict": f"nonzero ({direction})" if adj[n] <= alpha else "not confirmed"}
        hits = [f"{n} {v['family_verdict'][9:-1]}" for n, v in rules.items() if v["family_verdict"].startswith("nonzero")]
        nonzero["bounded"] = {
            "available": True, "rules": rules,
            "family": f"Holm within class '{cls}' over Hoeffding p-values at alpha = {alpha} (FWER <= alpha for the "
                      "class, given |a| <= M for every unit; distribution-free, finite-sample)",
            "summary": ("average effect nonzero: " + ", ".join(hits)) if hits else
                       "not confirmed on these projections (bounded route)"}
    if equivalence_rel is None:
        equivalence = {"requested": False, "statement": "not requested: no equivalence statement"}
    else:
        verdicts = {}
        for n, r in zip(names, recs):
            e = (r or {}).get("equivalence") or {}
            v = e.get("verdict")
            verdicts[n] = {"WITHIN_DELTA": "within delta", "NOT_SHOWN": "not shown"}.get(
                v, f"cannot judge ({v or 'no record'}{': ' + str(e.get('reason')) if e.get('reason') else ''})")
        all_in = all(v == "within delta" for v in verdicts.values())
        equivalence = {"requested": True, "delta_rel": equivalence_rel, "rules": verdicts,
                       "test": "TOST on the projection endpoints at alpha per rule",
                       "joint": ("every rule of the class within delta (intersection-union test: valid at alpha "
                                 "without adjustment)") if all_in else
                                "not every rule of the class shown within delta"}
    return {"nonzero": nonzero, "equivalence": equivalence}


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

FAILURE_CLASSES = ("semantics missing", "binding", "enclosure too wide", "over budget", "statistics insufficient",
                   "set target width (L_E)", "unproven premise (conditional diagnosis)")
_FAILURE_TABLE = [   # (substring of the tool's reason, class) -- first match wins; printed with every report
    ("call rejected the declared inputs", "binding"),
    ("address outside every captured storage", "binding"), ("invalid address", "binding"),
    ("unrecognized", "semantics missing"), ("no declared semantics", "semantics missing"),
    ("unknown callee", "semantics missing"), ("no reference rule", "semantics missing"),
    ("libdevice", "semantics missing"), ("loop bound", "over budget"), ("unknown buffer", "semantics missing"),
    ("domain of", "enclosure too wide"), ("non-point", "enclosure too wide"), ("float-to-int", "enclosure too wide"),
    ("reinterpret", "enclosure too wide"),
    ("not supported", "semantics missing"), ("unsupported", "semantics missing"), ("unknown op", "semantics missing"),
    ("no derivative rule", "semantics missing"), ("store through a not-established address", "semantics missing"),
    ("atomic", "semantics missing"), ("order is not declared", "semantics missing"),
    ("cross-program race", "semantics missing"), ("inline_asm", "semantics missing"),
    ("binding", "binding"), ("not written by triton", "binding"), ("modified after", "binding"),
    ("another recorded storage", "binding"), ("out-of-bounds", "binding"), ("outside the captured windows", "binding"),
    ("non-triton intermediate", "binding"), ("shape varies across units", "binding"),
    ("pinned_load", "enclosure too wide"), ("undecided", "enclosure too wide"), ("overlap", "enclosure too wide"),
    ("straddl", "enclosure too wide"), ("may be zero", "enclosure too wide"), ("may be negative", "enclosure too wide"),
    ("too wide", "enclosure too wide"), ("interval width", "enclosure too wide"),
    ("timeout", "over budget"), ("budget", "over budget"),
    ("cannot judge", "statistics insufficient"), ("unresolved_sample", "statistics insufficient"),
]


def classify_failure(reason: str) -> str:
    r = str(reason).lower()
    for key, cls in _FAILURE_TABLE:
        if key in r:
            return cls
    return "semantics missing" if "not_established" in r else "unclassified (tool error or new reason)"


# ------------------------------------------------------------------------------------------------ run

class _Timeout(BaseException):
    """The case budget ran out.  A BaseException, so that no ``except Exception`` on the way (the call wrapper that
    reports a rejected input, a tool path that records an error) turns an exhausted budget into another failure class."""


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


REFINEMENT_CATEGORIES = {
    "met": "every numerical enclosure meets the requested resolution",
    "budget exhausted": "the time budget ran out (or a higher level ran over the case budget) before the target was met",
    "backend limit": "the backend cannot narrow the enclosures further: float64 endpoints below a float64 ulp, the "
                     "highest working-precision level reached, or no rule on the reference path depends on the level",
    "intrinsic set width": "the only elements without a narrow value are set targets (L_E), whose width is the "
                           "target's own; no precision shrinks it",
    "pass failed": "a higher level ended without a result (error); the previous level stands",
    "no numerical enclosure": "an output has no complete finite element and no set target: nothing for a working "
                              "precision to refine (its failure classes say why the reference is missing)",
}


def _output_refinement(o: dict, frac: float) -> str:
    """Per evaluated output at one level: met / intrinsic set width / backend limit (float64) / refinable."""
    q = o["reference"]
    if q.get("resolution_met") is True:      # every numerical enclosure met; set targets keep their own width
        return "intrinsic set width" if q.get("set_target_elements") else "met"
    if q.get("resolution_met") is None:     # no complete finite element
        return "intrinsic set width" if q.get("set_target_elements") else "nothing to refine"
    if o.get("dtype") == "float64" and frac < 1.0:
        return "backend limit"
    return "refinable"


def run_level(exp: dict, level: dict) -> dict:
    """One factor level with adaptive working precision (rc3 W2 / W6; external audit task book section 5).  The level
    is measured at working precision 1; while some output's requested resolution is not met and could be, it is
    measured again at the next level (``intervals.PRECISION_LEVELS``, up to ``resolution.max_level``, default 3).

    Stops, each with its own category (``REFINEMENT_CATEGORIES``): every output met; no output refinable (float64
    endpoints, set targets only); no rule on the reference path depends on the working-precision level (the pass
    reports ``precision_dependent_calls`` = 0, so a higher level gives the same enclosures exactly); the highest level
    reached; the time budget spent; a higher level that fails.  An unchanged pass fraction is never read as "more
    precision does not help": the widths can shrink by orders of magnitude while no element crosses the target yet
    (three-level summation: levels 1 and 2 resolve nothing, level 3 everything).  Every level keeps its widths.  The
    result of the last completed level is the level's result."""
    res = exp["resolution"]
    max_level = int(res.get("max_level", 3))
    frac = float(res["ulp_fraction"])
    t0 = time.time()
    steps, best, outcome, category, prec = [], None, None, None, 1
    per_output = {}
    while True:
        out = _measure_pass(exp, level, prec)
        if out.get("status") != "ok":
            if best is None:
                return out
            over = out.get("status") == "over budget"
            category = "budget exhausted" if over else "pass failed"
            outcome = (f"{category}: the pass at level {prec} ended with '{out.get('status')}' "
                       f"({str(out.get('reason'))[:120]}); the result of level {prec - 1} stands")
            prec -= 1
            break
        best = out
        evaluated = {n: o for n, o in out["outputs"].items() if o.get("status") == "evaluated"}
        per_output = {n: _output_refinement(o, frac) for n, o in evaluated.items()}
        steps.append({"level": prec, "resolution_met": all(v in ("met", "intrinsic set width")
                                                            for v in per_output.values()),
                      "per_output": dict(per_output),
                      "resolved_fraction": {n: o["reference"].get("resolved_fraction") for n, o in evaluated.items()},
                      "unresolved_elements": {n: o["reference"].get("unresolved_elements")
                                              for n, o in evaluated.items()},
                      "width_over_ulp": {n: o["reference"].get("width_over_ulp") for n, o in evaluated.items()},
                      "precision_dependent_calls": out.get("precision_dependent_calls"),
                      "seconds": out.get("seconds")})
        refinable = [n for n, v in per_output.items() if v == "refinable"]
        if not refinable:
            kinds = set(per_output.values())
            if kinds <= {"met"}:
                category, outcome = "met", f"met at level {prec}"
            elif kinds <= {"met", "intrinsic set width"}:
                category = "intrinsic set width"
                outcome = (f"intrinsic set width at level {prec}: the remaining elements are set targets (L_E); "
                           "every numerical enclosure is met")
            elif "backend limit" in kinds:
                category = "backend limit"
                outcome = ("backend limit: float64 outputs with nonzero enclosure width cannot get below a fraction "
                           "of a float64 ulp with float64 endpoints; not refined")
            else:
                category = "no numerical enclosure"
                outcome = ("no numerical enclosure: " + ", ".join(sorted(n for n, v in per_output.items()
                                                                         if v == "nothing to refine")) +
                           " without a complete finite element or set target; nothing to refine")
            break
        if out.get("precision_dependent_calls") == 0:
            category = "backend limit"
            outcome = (f"backend limit: no rule on the reference path depends on the working-precision level "
                       f"(precision_dependent_calls = 0 at level {prec}); a higher level gives the same enclosures")
            break
        if prec >= max_level:
            category = "backend limit"
            outcome = f"backend limit: highest working precision level {max_level} reached; target not met"
            break
        if time.time() - t0 > float(exp["budget"]["gpu_seconds"]):
            category = "budget exhausted"
            outcome = f"budget exhausted after level {prec}; target not met"
            break
        prec += 1
    best["refinement"] = {"levels": steps, "outcome": outcome, "category": category,
                          "category_meaning": REFINEMENT_CATEGORIES.get(category),
                          "per_output": per_output, "final_level": prec,
                          "target_ulp_fraction": frac, "max_level": max_level,
                          "levels_meaning": {str(k): v for k, v in sorted(__import__(
                              "kernel_analyzer.reference_eval.intervals", fromlist=["PRECISION_LEVELS"]
                          ).PRECISION_LEVELS.items())}}
    return best


def _measure_pass(exp: dict, level: dict, precision_level: int = 1) -> dict:
    call = resolve(exp["call"], exp["_base_dir"])
    measure_names = list(exp["compare"]["measure"])
    spec_fn = resolve(exp["compare"]["spec"], exp["_base_dir"]) if exp["compare"]["mode"] == "B" else None
    budget = exp["budget"]
    dev, conf = unit_split(exp)
    seeds = dev + conf
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
            try:
                outs = call({k: v for k, v in inp.items() if k != "_seed"})
            except Exception as exc:  # noqa: BLE001  -- the declared call itself refused the inputs
                raise RuntimeError(f"the call rejected the declared inputs (legal call / input domain): "
                                   f"{type(exc).__name__}: {exc}") from exc
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
    # import what the case setup imports before the case timeout is armed: SIGALRM inside an import (torch._dynamo
    # takes about a second) leaves a partially initialised module that breaks every later level of the process
    import torch._dynamo  # noqa: F401
    import torch._inductor.config  # noqa: F401
    t0 = time.time()
    old = _alarm(budget["case_timeout"])
    try:
        rep = check.run(case, dev=dev, conf=conf, keep=keep, magnitude_bound=exp.get("magnitude_bound"),
                        alpha=exp["alpha"], equivalence_rel=(exp.get("equivalence") or {}).get("rel"),
                        repeats=exp.get("repeats"), precision_level=precision_level)
    except _Timeout as exc:
        return {"level": level, "status": "over budget", "reason": str(exc), "failure_class": "over budget",
                "seconds": round(time.time() - t0, 1)}
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"[:400]
        cls = classify_failure(msg)
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
                                      "outputs_at_address_of_another_recorded_storage", "ttir_coverage_complete",
                                      "ir_coverage_complete", "ir_kinds")}
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
        # audit F04: a non-Triton upstream buffer keeps the reference kernel-level (equal values prove nothing),
        # unless its recorded producers are copies of the declared inputs or exact constants
        backfilled = list(o.get("depends_on_non_triton_intermediates") or [])
        recorded = list(o.get("upstream_with_producer_record") or [])
        quality["reference_scope"] = ("kernel-level: upstream non-Triton values captured (" + ", ".join(backfilled)[:200]
                                      + ")") if backfilled else (
            "call-level (upstream copies / exact constants with producer records: " + ", ".join(recorded)[:200] + ")"
            if recorded else "call-level")
        quality["complete_rate_call_level"] = 0.0 if backfilled else quality["complete_rate"]
        stats = class_statistics(o.get("numerical"), exp["rule_classes"], exp["alpha"],
                                 equivalence_rel=(exp.get("equivalence") or {}).get("rel"))
        entry = {"status": "evaluated", "dtype": dtypes.get(name, "float32"), "reference": quality,
                 "statistics": stats, "guarantee": o.get("guarantee"),
                 "proof_status": o.get("proof_status"), "execution": o.get("execution"),
                 "upstream_with_producer_record": recorded,
                 "mixed_non_triton_sources": o.get("depends_on_non_triton_intermediates") or [],
                 "not_established_reasons_seed0": o.get("not_established_reasons_seed0"),
                 "special_values": o.get("special_values")}
        reasons = list((o.get("not_established_reasons_seed0") or {}).keys())
        entry["failure_classes"] = []
        if quality["complete_rate"] is not None and quality["complete_rate"] < 1.0:
            # only then do the tool's reasons explain missing elements (otherwise they are notes such as the dot
            # input precision or a checked premise)
            entry["failure_classes"] = sorted({classify_failure(r) for r in reasons if not str(r).startswith(
                ("assumed:", "proved:", "dot_input_precision:", "set:"))})
            if quality["complete_under_premise"]:   # audit F03: a conditional diagnosis, not a complete reference
                entry["failure_classes"].append("unproven premise (conditional diagnosis)")
            entry["failure_classes"] = sorted(set(entry["failure_classes"])) or ["unclassified"]
        if quality["resolved_fraction"] is not None and quality["resolved_fraction"] < 1.0:
            # the resolution counts numerical enclosures only (complete finite elements); set targets are not among
            # them, so an unresolved element is always an enclosure that is too wide
            entry["failure_classes"] = sorted(set(entry["failure_classes"]) | {"enclosure too wide"})
        if quality.get("set_target_elements"):
            # a set target's width is part of the target, not a numerical enclosure that precision would shrink
            entry["failure_classes"] = sorted(set(entry["failure_classes"]) | {"set target width (L_E)"})
        if any(v["summary"].startswith("cannot judge") for v in stats.values()):
            entry["failure_classes"] = sorted(set(entry["failure_classes"]) | {"statistics insufficient"})
        if spec_fn is not None and name in keep:
            entry["semantic"] = _semantic(keep[name], fas, name)
            entry["semantic"]["verdict_scope"] = ("mixed: not a semantic verdict" if entry["mixed_non_triton_sources"]
                                                  else "pure Triton: K_R vs f_r exact comparison")
        failures.extend(entry["failure_classes"])
        outputs[name] = entry
    return {"level": level, "status": "ok", "precision_level": precision_level,
            "precision_dependent_calls": rep.get("precision_dependent_calls"),
            "producer_trace_per_seed": rep.get("producer_trace_per_seed"),
            "provenance_seed0": rep.get("provenance_seed0"),
            "units": {"development": len(dev), "confirmation": len(conf)},
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
