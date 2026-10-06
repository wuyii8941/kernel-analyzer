#!/usr/bin/env python3
"""External controlled corpus: The Correctness Illusion (gpuemu-corpus @ 2f15310), per the pre-registered protocol
docs/external_eval_protocol_20261006.md.

    python scripts/external_corpus.py list                      # conditions, tier-2 picks, manifest.json
    python scripts/external_corpus.py tier1 [--entries a,b]     # 22 Triton entries, every condition, seeds 0-7
    python scripts/external_corpus.py tier2 [--entries a,b]     # one condition per entry, seeds 0-95
    python scripts/external_corpus.py blackbox --tier 1|2       # the 4 NumPy entries (K - f only)
    python scripts/external_corpus.py ablation-a1 --tier 1|2    # K_R replaced by a float64 rerun (9 kernels)

Per condition: the tool report (``check.run`` / ``check.run_black_box``), one baseline row (per-seed statistics from
which B1-B3 and their sweeps are evaluated without rerunning), and in tier 1 the independent recomputation of K_R
(``external_corpus_semantics.declared``).  Existing reports are skipped (resumable).
"""
import argparse
import importlib.util
import itertools
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from kernel_analyzer import check  # noqa: E402
from kernel_analyzer.check import Case, f64_point_spec  # noqa: E402
import external_corpus_semantics as sem  # noqa: E402

CORPUS = ROOT / ".cache/src/gpuemu-corpus"
DATA = CORPUS / "gpuemu_corpus/data"
COMMIT = "2f15310"
OUT = ROOT / "results/external/gpuemu"
CAP_OUT, CAP_IN = 4096, 65536
DTYPE = "float32"
EQUIVALENCE_REL = 2.0 ** -24  # candidate delta (evaluation plan section 5), reported as provisional
A1_KERNELS = {"gelu_triton", "leaky_relu_triton", "relu_triton", "rmsnorm_triton", "softmax_triton",
              "gelu_triton_buggy", "leaky_relu_triton_buggy", "rmsnorm_triton_buggy", "softmax_triton_buggy"}
TIERS = {1: (range(0, 2), range(2, 8)), 2: (range(0, 32), range(32, 96))}
TAG = ""  # per-process suffix of the JSONL files (parallel shards)


def entries():
    out = []
    for m in sorted(DATA.glob("*/meta.json")):
        meta = json.loads(m.read_text())
        out.append({"name": meta["name"], "meta": meta, "triton": "triton" in meta["kernel"],
                    "kernel": DATA / meta["kernel"], "reference": DATA / meta["reference"]})
    return out


def conditions(meta):
    schema = meta["op_schema"]
    names = [d["name"] for d in schema["dims"]]
    kept = []
    for combo in itertools.product(*[d["candidates"] for d in schema["dims"]]):
        dims = dict(zip(names, combo))
        shapes = {t["name"]: [dims[n] for n in t["dims"]] for t in schema["inputs"]}
        n_out = math.prod(dims[n] for n in schema["output"]["dims"])
        n_in = sum(math.prod(s) for s in shapes.values())
        if n_out <= CAP_OUT and n_in <= CAP_IN:
            kept.append({"name": "_".join(f"{n}{dims[n]}" for n in names), "dims": dims, "shapes": shapes,
                         "elements": n_out, "last": dims[schema["output"]["dims"][-1]], "inputs": n_in})
    return kept


def tier2_condition(meta):
    # protocol section 2: most output elements, then the larger last dimension; deviation 1 (section 10): remaining
    # ties go to the most input elements (the first-listed tie was the degenerate N = 1 / K = 1 shape)
    return max(conditions(meta), key=lambda c: (c["elements"], c["last"], c["inputs"]))


def make_inputs(meta, cond, seed):
    """As the corpus fuzzer: each input U[-10, 10] in float32; one generator per (seed, condition)."""
    rng = np.random.default_rng([seed, zlib.crc32(cond["name"].encode())])
    out = {}
    for name in meta["input_names"]:
        shp = cond["shapes"][name]
        out[name] = ((rng.random(math.prod(shp)) * 2.0 - 1.0) * 10.0).reshape(shp).astype(DTYPE)
    return out


_MODULES = {}


def load(path, tag):
    key = (str(path), tag)
    if key not in _MODULES:
        spec = importlib.util.spec_from_file_location(f"gpuemu_{tag}_{path.parent.name}", path)
        mod = importlib.util.module_from_spec(spec)
        saved = list(sys.path)
        sys.path.insert(0, str(DATA))  # the reference scripts import _refkit from the data directory
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path[:] = saved
        _MODULES[key] = mod
    return _MODULES[key]


def reference_f64(entry, inputs):
    """The authors' ref_fp64.py on float64 inputs (so its output is not rounded back to float32)."""
    ref = load(entry["reference"], "ref")
    got = {}
    ref.read_inputs = lambda: ({k: v.astype(np.float64) for k, v in inputs.items()}, {})
    ref.emit = lambda arr: got.setdefault("out", np.asarray(arr, dtype=np.float64))
    ref.main()
    return got["out"]


def call_kernel(entry, inputs):
    """Their run(numpy inputs) -> numpy; the CUDA tensor it copies back is captured through Tensor.cpu."""
    mod = load(entry["kernel"], "kernel")
    captured = []
    orig = torch.Tensor.cpu

    def cpu(t, *a, **kw):
        if t.is_cuda:
            captured.append(t)
        return orig(t, *a, **kw)

    torch.Tensor.cpu = cpu
    try:
        out_np = mod.run(inputs)
    finally:
        torch.Tensor.cpu = orig
    return (captured[-1] if captured else None), out_np


class CorpusCase(Case):
    spec_bound = "float64 evaluation of the authors' ref_fp64.py, declared bound 2^-40 max|f| (screening)"

    def __init__(self, entry, cond):
        self.entry, self.cond = entry, cond
        self.name = f"{entry['name']}__{cond['name']}"
        self.implementation = f"gpuemu-corpus@{COMMIT} {entry['meta']['kernel']} ({entry['meta']['source']})"
        self.specification = f"{entry['meta']['reference']} on float64 inputs"
        self.log = {}  # seed -> {"K": float64, "f": float64, "returned": run()'s NumPy result}

    def inputs(self, seed):
        np_in = make_inputs(self.entry["meta"], self.cond, seed)
        self._seed = seed
        # the torch views let the engine recognise the case inputs when it looks for non-Triton intermediates
        return {"numpy": np_in, "torch": {k: torch.from_numpy(v) for k, v in np_in.items()}}

    def launch(self, inp):
        if self.entry["triton"]:
            out, out_np = call_kernel(self.entry, inp["numpy"])
        else:
            _, out_np = call_kernel(self.entry, inp["numpy"])
            out = torch.from_numpy(np.ascontiguousarray(out_np))
        self._shape = tuple(out.shape)
        self.log[self._seed] = {"returned": np.asarray(out_np, dtype=np.float64).reshape(-1)}
        return {"out": out}

    def spec(self, inp):
        f = reference_f64(self.entry, inp["numpy"]).reshape(self._shape)
        self.log[self._seed]["f"] = f.reshape(-1)
        return {"out": f64_point_spec(f)}


def baseline_row(entry, cond, case, seeds):
    """Per-seed statistics; B1 flags iff b1 > tol * m, B2 iff b2 > m, B3 iff b3 > c (protocol section 5)."""
    op = sem.op_of(entry["name"])
    tol = entry["meta"]["tolerances"][DTYPE]
    per = []
    for seed in seeds:
        inp = make_inputs(entry["meta"], cond, seed)
        k = case.log[seed]["returned"]
        f = case.log[seed]["f"]
        t0 = time.time()
        t32 = sem.trusted32(op, inp)
        t_trusted = time.time() - t0
        ours = sem.spec64(op, inp)
        fin = np.isfinite(k).all()
        rel = lambda a: float(np.linalg.norm(a - f) / max(np.linalg.norm(f), 1e-300))  # noqa: E731
        per.append({
            "seed": seed,
            "b1_max_abs_vs_rounded_f": float(np.abs(k - f.astype(np.float32)).max()) if fin else float("inf"),
            "b2_max_ratio": float((np.abs(k - f) / (1e-5 + 1.3e-6 * np.abs(f))).max()) if fin else float("inf"),
            "b3_ratio": (rel(k) / max(rel(t32), 2.0 ** -24)) if fin else float("inf"),
            "rel_l2_K": rel(k) if fin else float("inf"), "rel_l2_trusted32": rel(t32),
            "trusted32_seconds": round(t_trusted, 5),
            "spec_cross_check_max_rel": float(np.abs(ours - f).max() / max(np.abs(f).max(), 1e-300)),
        })
    return {"entry": entry["name"], "condition": cond["name"], "tolerance": tol, "per_seed": per}


def declared_check(entry, cond, keep, seeds):
    """Independent recomputation of K_R (protocol section 7): the declared real-number program in NumPy float64."""
    rows = keep.get("out", [])
    out = {"entry": entry["name"], "condition": cond["name"], "checked": 0, "violations": 0, "not_complete": 0,
           "max_excess": 0.0, "K_matches_returned": True}
    for r in rows:
        if r["seed"] not in seeds:
            continue
        inp = make_inputs(entry["meta"], cond, r["seed"])
        d = sem.declared(entry["name"], inp)
        s = 2.0 ** -40 * float(np.abs(d[np.isfinite(d)]).max(initial=0.0))
        ok = r["ok"]
        lo, hi = r["r_lo"] - s, r["r_hi"] + s
        bad = ok & ((d < lo) | (d > hi))
        out["checked"] += int(ok.sum())
        out["not_complete"] += int((~ok).sum())
        out["violations"] += int(bad.sum())
        if bad.any():
            out["max_excess"] = max(out["max_excess"], float(np.max(np.maximum(lo - d, d - hi)[bad])))
            out.setdefault("examples", []).extend(
                {"seed": r["seed"], "index": int(j), "declared": float(d[j]), "K_R": [float(r["r_lo"][j]), float(r["r_hi"][j])]}
                for j in np.flatnonzero(bad)[:3])
    return out


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")


def run_tool(tier, names, black_box=False):
    dev, conf = TIERS[tier]
    sub = ("blackbox_tier%d" % tier) if black_box else ("tier%d" % tier)
    for entry in entries():
        if entry["triton"] == black_box or (names and entry["name"] not in names):
            continue
        conds = conditions(entry["meta"]) if tier == 1 else [tier2_condition(entry["meta"])]
        for cond in conds:
            path = OUT / sub / f"{entry['name']}__{cond['name']}.json"
            if path.exists():
                continue
            case = CorpusCase(entry, cond)
            keep = {} if tier == 1 and not black_box else None
            t0 = time.time()
            try:
                if black_box:
                    report = check.run_black_box(case, dev=dev, conf=conf, equivalence_rel=EQUIVALENCE_REL)
                else:
                    report = check.run(case, dev=dev, conf=conf, keep=keep, equivalence_rel=EQUIVALENCE_REL)
            except Exception as exc:  # noqa: BLE001 -- errors count in the denominator (protocol section 7)
                import traceback

                report = {"case": case.name, "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()[-2000:]}
            report["condition"] = {"entry": entry["name"], **{k: cond[k] for k in ("name", "dims", "shapes")}}
            report["wall_seconds"] = round(time.time() - t0, 2)
            if "error" not in report:
                append(OUT / f"baselines_{sub}{TAG}.jsonl", baseline_row(entry, cond, case, list(dev) + list(conf)))
                if keep is not None:
                    dc = declared_check(entry, cond, keep, {0, 1})
                    dc["K_matches_returned"] = all(
                        np.array_equal(r["k"], case.log[r["seed"]]["returned"]) for r in keep.get("out", [])
                        if r["seed"] in case.log)
                    append(OUT / f"declared_check{TAG}.jsonl", dc)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=1, default=float) + "\n")
            print(f"{sub} {case.name} {report.get('error', '')[:160]} {report['wall_seconds']}s", flush=True)
            if tier == 2 and not black_box:
                # mode B runs the rules on e_num and e_sem; the rules on the total K - f (protocol section 8) come from
                # the same decision layer in black-box form, on the same seeds
                tot = check.run_black_box(CorpusCase(entry, cond), dev=dev, conf=conf, equivalence_rel=EQUIVALENCE_REL)
                tot["condition"] = report["condition"]
                tpath = OUT / "tier2_total" / path.name
                tpath.parent.mkdir(parents=True, exist_ok=True)
                tpath.write_text(json.dumps(tot, indent=1, default=float) + "\n")


def run_a1(tier, names):
    """Ablation A1: e_num' = K - K64 with K64 the same kernel rerun in float64, through the same decision layer."""
    from kernel_analyzer.reference_eval.analysis import assess_units

    dev, conf = TIERS[tier]
    for entry in entries():
        if entry["name"] not in A1_KERNELS or (names and entry["name"] not in names):
            continue
        mod = load(entry["kernel"], "kernel")
        mod._TORCH_DT = {**mod._TORCH_DT, "float64": "float64"}
        conds = conditions(entry["meta"]) if tier == 1 else [tier2_condition(entry["meta"])]
        for cond in conds:
            path = OUT / f"ablation_a1_tier{tier}" / f"{entry['name']}__{cond['name']}.json"
            if path.exists():
                continue
            lo, hi, ref, t0 = [], [], [], time.time()
            for seed in list(dev) + list(conf):
                inp = make_inputs(entry["meta"], cond, seed)
                k = np.asarray(mod.run(inp), np.float64).reshape(-1)
                k64 = np.asarray(mod.run({n: v.astype(np.float64) for n, v in inp.items()}), np.float64).reshape(-1)
                lo.append(k - k64)
                ref.append(k64)
            lo, ref = np.stack(lo), np.stack(ref)
            ok = np.isfinite(lo) & np.isfinite(ref)
            rec, _ = assess_units(f"{entry['name']}: K - K64 (ablation A1)", np.where(ok, lo, 0), np.where(ok, lo, 0),
                                  np.where(ok, ref, 0), ok, len(dev), check.RULES, alignment_reference=np.where(ok, ref, 0),
                                  unit_ids=list(dev) + list(conf))
            mid = np.where(ok, lo, 0)
            rec["scale"] = {"max_abs_residual": float(np.abs(mid[ok]).max()) if ok.any() else None,
                            "relative_rms": float(np.sqrt((mid[ok] ** 2).mean() / max((ref[ok] ** 2).mean(), 1e-300)))
                            if ok.any() else None}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"case": f"{entry['name']}__{cond['name']}", "ablation": "A1",
                                        "seconds": round(time.time() - t0, 2), "numerical_fp64_rerun": rec},
                                       indent=1, default=float) + "\n")
            print(f"a1 tier{tier} {entry['name']}__{cond['name']}", flush=True)


def manifest():
    import hashlib

    rows = []
    for e in entries():
        conds = conditions(e["meta"])
        rows.append({"entry": e["name"], "triton": e["triton"], "source": e["meta"]["source"],
                     "kernel_sha256": hashlib.sha256(e["kernel"].read_bytes()).hexdigest(),
                     "reference_sha256": hashlib.sha256(e["reference"].read_bytes()).hexdigest(),
                     "meta_sha256": hashlib.sha256((e["kernel"].parent / "meta.json").read_bytes()).hexdigest(),
                     "tolerance_float32": e["meta"]["tolerances"].get(DTYPE),
                     "conditions": [c["name"] for c in conds], "tier2": tier2_condition(e["meta"])["name"]})
    return {"corpus": "github.com/sarkar-dipankar/gpuemu-corpus", "commit": COMMIT, "dtype": DTYPE,
            "cap": {"output_elements": CAP_OUT, "input_elements": CAP_IN},
            "seeds": {"tier1": [0, 7, "development 0-1"], "tier2": [0, 95, "development 0-31"]},
            "conditions_triton": sum(len(r["conditions"]) for r in rows if r["triton"]),
            "conditions_numpy": sum(len(r["conditions"]) for r in rows if not r["triton"]), "entries": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["list", "tier1", "tier2", "blackbox", "ablation-a1"])
    ap.add_argument("--entries", default="")
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--tag", default="", help="suffix of the JSONL files written by this process")
    a = ap.parse_args()
    global TAG
    TAG = a.tag
    names = set(filter(None, a.entries.split(",")))
    if a.command == "list":
        m = manifest()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
        for r in m["entries"]:
            print(f"{r['entry']:30s} {'triton' if r['triton'] else 'numpy ':6s} {len(r['conditions']):3d} conditions, "
                  f"tier 2: {r['tier2']}")
        print("Triton conditions", m["conditions_triton"], "NumPy conditions", m["conditions_numpy"])
    elif a.command in ("tier1", "tier2"):
        run_tool(int(a.command[-1]), names)
    elif a.command == "blackbox":
        run_tool(a.tier, names, black_box=True)
    else:
        run_a1(a.tier, names)


if __name__ == "__main__":
    main()
