#!/usr/bin/env python3
"""Row 1 of the guarantee table (docs/protocol_closure_v3_20261008.md section 3): trigger each degrade path of the chain
once through the production code with a minimal constructed input, and save the record under results/closure/degrade/.

Paths triggered here (the others have real records already in the repository, listed in results/closure/chain_table.json):
  arrow0  pairing not provable -> output not established          (check.run, recorder keep-alive off, freed address reused)
  arrow1  kappa = conditional                                      (step-6 negative controls re-run: pinned load / branch)
  arrow1  kappa = unestablished                                    (atomic whose returned old value is used)
  arrow1  unrecognised semantics -> rejected                       (Inductor bf16 var: Welford combiner not in the mapping)
  arrow2  direction source not registered -> no formal conclusion  (apply_direction_rules with an unknown rule)
  arrow3  zero variance / distribution premise / sample size      (frozen _summarize + contract_v3.statistical_judgment)
  arrow4  no comparable pair -> shared relation not established    (classify.combine)
  arrow4  ok_elements = 0 -> not established                       (classify.fr_assess)

    PYTHONPATH=src python scripts/closure/trigger_degrades.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "essential"))
OUT = ROOT / "results/closure/degrade"


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str) + "\n")
    print(name, "->", OUT / f"{name}.json", flush=True)


def arrow0_binding():
    import triton
    import triton.language as tl
    from kernel_analyzer import check
    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder

    @triton.jit
    def _double(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) * 2.0, mask=m)

    N = 1024
    reused = []

    class Case(check.Case):
        name = "arrow0_binding_not_provable"

        def inputs(self, seed):
            return {"x": torch.randn(N, generator=torch.Generator().manual_seed(seed)).cuda()}

        def launch(self, inp):
            a = torch.empty_like(inp["x"])
            _double[(triton.cdiv(N, 256),)](inp["x"], a, N, BLOCK=256)
            addr = a.untyped_storage().data_ptr()
            del a
            b = inp["x"] * 2.0                                   # ATen output, same size and content
            reused.append(b.untyped_storage().data_ptr() == addr)
            return {"out": b}

    original = TritonLaunchRecorder.__init__

    def init(self, *a, **k):
        k.setdefault("keep_storages", False)                     # keep-alive off: identities of freed storages repeat
        original(self, *a, **k)

    TritonLaunchRecorder.__init__ = init
    try:
        rep = check.run(Case(), dev=[0], conf=[1, 2])
    finally:
        TritonLaunchRecorder.__init__ = original
    save("arrow0_binding_not_established", {
        "path": "arrow 0: pairing not provable -> output not established (not in any denominator)",
        "construction": "Triton writes a freed intermediate; an ATen output reuses its address; recorder keep-alive off",
        "address_reused": reused, "tool_version": rep.get("tool_version"),
        "outputs_binding_not_established": rep.get("outputs_binding_not_established"),
        "outputs_evaluated": sorted(rep.get("outputs", {}).keys()),
        "outputs_not_written_by_triton": rep.get("outputs_not_written_by_triton")})


def arrow1_conditional():
    out = OUT / "arrow1_step6_negative_controls_rerun.json"
    OUT.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_step6_negative_controls.py"), "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    d = json.loads(out.read_text()) if out.exists() else {"error": r.stderr[-800:]}
    csl = d.get("compute_store_load_compute", {})
    save("arrow1_kappa_conditional", {
        "path": "arrow 1: kappa = conditional -> not in the statistics, proportion reported",
        "construction": "compute -> store -> load -> compute with the load pinned to its captured value; branch on the pinned value",
        "source_run": str(out.relative_to(ROOT)),
        "load_pinned_to_capture": csl.get("load_pinned_to_capture"), "branch_on_pinned_load": csl.get("branch_on_pinned_load"),
        "composed_controls": {"composed": csl.get("composed"), "branch_composed": csl.get("branch_composed")}})


def arrow1_unestablished():
    import triton
    import triton.language as tl
    from kernel_analyzer import check

    @triton.jit
    def _atomic_ret(x_ptr, acc_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        old = tl.atomic_add(acc_ptr + offs % 4, tl.load(x_ptr + offs, mask=m), mask=m)
        tl.store(out_ptr + offs, old, mask=m)                    # the returned old value depends on the order

    N = 1024

    class Case(check.Case):
        name = "arrow1_atomic_return_used"

        def inputs(self, seed):
            return {"x": torch.randn(N, generator=torch.Generator().manual_seed(seed)).cuda()}

        def launch(self, inp):
            acc = torch.zeros(4, device="cuda")
            out = torch.empty(N, device="cuda")
            _atomic_ret[(triton.cdiv(N, 256),)](inp["x"], acc, out, N, BLOCK=256)
            return {"out": out}

    rep = check.run(Case(), dev=[0], conf=[1, 2])
    o = rep.get("outputs", {}).get("out", {})
    save("arrow1_kappa_unestablished", {
        "path": "arrow 1: kappa = unestablished -> not in the statistics, proportion reported",
        "construction": "tl.atomic_add whose returned old value is stored (order of the atomic updates not declared)",
        "reference_classes": o.get("reference_classes"), "not_established_reasons_seed0": o.get("not_established_reasons_seed0"),
        "numerical_verdicts": [(r.get("rule"), r.get("verdict")) for r in (o.get("numerical") or {}).get("rules", [])]})


def arrow1_unrecognised():
    import common
    import torch.nn.functional  # noqa: F401
    st = {}

    def fn(x):
        return x.var(-1)

    def setup():
        torch._dynamo.reset()
        st["fn"] = torch.compile(fn, dynamic=False)
        launch(inputs(0))
        torch.cuda.synchronize()

    def inputs(seed):
        g = torch.Generator().manual_seed(seed)
        return {"x": torch.randn(4, 1027, generator=g, dtype=torch.float64).float().to(torch.bfloat16).cuda(), "_seed": seed}

    def launch(t):
        return {"out": st["fn"](t["x"])}

    rep, _ = common.fr_run("arrow1_unrecognised_semantics", setup, inputs, launch, lambda _t: None)
    save("arrow1_unrecognised_semantics_rejected", {
        "path": "arrow 1: unrecognised semantics -> rejected (never guessed)",
        "construction": "torch.compile(x.var(-1)) on bfloat16, rows of 1027: Inductor emits a Welford reduction combiner",
        "ttir_coverage_complete": rep.get("ttir_coverage_complete"),
        "outputs_whose_writing_programs_aborted": rep.get("outputs_whose_writing_programs_aborted"),
        "outputs_evaluated": sorted(rep.get("outputs", {}).keys())})


def arrow2_direction():
    from kernel_analyzer.reference_eval import analysis as A
    lo = np.random.default_rng(0).normal(0, 1, (96, 8))
    try:
        res = A.apply_direction_rules("c", lo, lo, lo, {"direction_rules": ["unregistered_direction"]}, 32, 64, 0.05)
        outcome = {"result": res}
    except Exception as exc:  # noqa: BLE001
        outcome = {"raised": f"{type(exc).__name__}: {exc}"}
    save("arrow2_direction_not_registered", {
        "path": "arrow 2: direction source not registered -> no formal conclusion",
        "construction": "apply_direction_rules with the rule name 'unregistered_direction'", **outcome})


def arrow3_premises():
    from kernel_analyzer.reference_eval import analysis as A
    import contract_v3 as C
    rng = np.random.default_rng(11)
    skewed = None
    for seed in range(100):                                      # first deterministic lognormal sample with |skew| > S0
        x = np.random.default_rng(1000 + seed).lognormal(0, 1.5, 32) - 1.0
        if abs(A._t_approximation(x)["unit_skewness"]) > C.S0:
            skewed = x
            break
    cases = {
        "zero_variance": (np.full(64, 0.25), np.full(64, 0.25)),
        "distribution_premise": (skewed, skewed),
        "sample_size": tuple([rng.normal(1.0, 0.1, 8)] * 2),
    }
    for name, (l, h) in cases.items():
        rec = A._summarize("synthetic", "R1", l, h, 0.05)
        save(f"arrow3_{name}", {"path": f"arrow 3: {name} -> cannot judge", "n": int(l.size),
                                 "frozen_record": {k: rec.get(k) for k in ("verdict", "reason", "n", "unit_skewness", "t_approximation")},
                                 "contract_v3_judgment": C.statistical_judgment(rec)})


def arrow4_guards():
    import classify as Q
    rel = Q.combine("deviates", {"deviating_elements": 0, "compared_outputs": 0}, "deviates")
    cond = {"op": "index_add", "id": "t0", "shape": "1d", "m": 3, "dup": "some", "alpha": 1.0, "values": "ints"}
    keep = [{"r_lo": np.zeros(3), "r_hi": np.zeros(3), "k": np.zeros(3), "ok": np.zeros(3, dtype=bool), "shape": (3,)}] * 3
    fr = {"status": "ok", "keep": {"out": keep}, "notes": {}, "mixed": {}}
    spec = {"status": "ok", "readings": {"main": {"status": "ok", "out": (np.ones(3), np.ones(3))}}, "grad": None}
    res = Q.fr_assess("index", cond, "inductor_cuda32", fr, {0: spec, 1: spec, 2: spec})
    save("arrow4_no_comparable_pair", {"path": "arrow 4: no comparable pair -> shared relation not established",
                                         "construction": "candidate deviates, E compared no output", "classification": rel})
    save("arrow4_ok_elements_zero", {"path": "arrow 4: ok_elements = 0 -> not established",
                                      "construction": "FR keep with no complete finite element", "fr_output": res["outputs"]["out"]})


if __name__ == "__main__":
    for f in (arrow0_binding, arrow1_conditional, arrow1_unestablished, arrow1_unrecognised, arrow2_direction,
              arrow3_premises, arrow4_guards):
        try:
            f()
        except Exception:  # noqa: BLE001
            print(f.__name__, "FAILED", traceback.format_exc()[-1500:], flush=True)
