"""RQ5: real library kernels as samples -- decomposition, bitwise certificate and prevalence.

Two libraries, installed into their own directories under .cache/pydeps (downloaded packages are untrusted; this
script runs with python -I and adds their paths explicitly): Liger-Kernel 0.6.2 (its ops called directly, forward
and backward) and FlagGems 5.0.2 (a battery of torch operations under flag_gems.use_gems(), which routes them to
FlagGems' Triton kernels).  Every distinct kernel (the first two launches of each kernel name) is captured with the
tool's recorder and then:

* decomposed: every TTIR operation mapped to a catalogued primitive (op name, attribute case, element types; a
  libdevice call by its symbol) and every structure to a composition form; operations outside the catalogued and
  compiled set are counted (the package's failure code K1);
* certified: emulated with the composition rules (cells.compare_launch, every lowering choice frozen) and compared
  bit for bit with the device; elements the rules do not cover are reported with the emulator's reason.

Run with LD_LIBRARY_PATH=/data1/tzh/envs/ka_main/lib (FlagGems imports sqlite through sqlalchemy).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEPS = ROOT / ".cache/pydeps"
for p in ("liger_kernel_0_6_2", "flag_gems_5_0_2", "pyyaml_6_0_2", "sqlalchemy_2_0_48"):
    sys.path.insert(0, str(DEPS / p))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "2_tool/src"))

FORM_OF = {"tt.reduce": "C3", "tt.scan": "C4", "tt.map_elementwise": "C5", "arith.select": "C6", "scf.if": "C7",
           "scf.for": "C8", "scf.while": "C9", "tt.gather": "C11", "tt.atomic_rmw": "C14", "tt.atomic_cas": "C14",
           "tt.dot": "C15", "tt.dot_scaled": "C15", "gpu.barrier": "C12"}


def liger_calls(torch):
    from liger_kernel.ops import (cross_entropy, dyt, geglu, group_norm, jsd, kl_div, layer_norm, rms_norm,
                                  rope, softmax, sparsemax, swiglu, tvd)
    calls = []
    for dt in (torch.float32, torch.bfloat16):
        def mk(dt=dt):
            g = torch.Generator(device="cuda").manual_seed(0)
            x = torch.randn(4, 64, 256, device="cuda", dtype=dt, generator=g, requires_grad=True)
            w = torch.randn(256, device="cuda", dtype=dt, generator=g, requires_grad=True)
            return g, x, w
        def c_rms(dt=dt):
            g, x, w = mk(dt)
            y = rms_norm.LigerRMSNormFunction.apply(x, w, 1e-6, 0.0, "llama", False)
            y.sum().backward()
        def c_ln(dt=dt):
            g, x, w = mk(dt)
            b = torch.randn(256, device="cuda", dtype=dt, requires_grad=True)
            y = layer_norm.LigerLayerNormFunction.apply(x, w, b, 1e-5)
            y.sum().backward()
        def c_swiglu(dt=dt):
            g, x, w = mk(dt)
            b = torch.randn_like(x, requires_grad=True)
            swiglu.LigerSiLUMulFunction.apply(x, b).sum().backward()
        def c_geglu(dt=dt):
            g, x, w = mk(dt)
            b = torch.randn_like(x, requires_grad=True)
            geglu.LigerGELUMulFunction.apply(x, b).sum().backward()
        def c_ce(dt=dt):
            logits = torch.randn(64, 1000, device="cuda", dtype=dt, requires_grad=True)
            t = torch.randint(0, 1000, (64,), device="cuda")
            out = cross_entropy.LigerCrossEntropyFunction.apply(logits, t, None)
            (out[0] if isinstance(out, tuple) else out).backward()
        def c_rope(dt=dt):
            q = torch.randn(2, 8, 64, 64, device="cuda", dtype=dt, requires_grad=True)
            k = torch.randn(2, 4, 64, 64, device="cuda", dtype=dt, requires_grad=True)
            pos = torch.arange(64, device="cuda").float()
            inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
            f = torch.outer(pos, inv)
            emb = torch.cat([f, f], dim=-1)[None]
            qo, ko = rope.LigerRopeFunction.apply(q, k, emb.cos().to(dt), emb.sin().to(dt))
            (qo.sum() + ko.sum()).backward()
        def c_softmax(dt=dt):
            x = torch.randn(64, 1024, device="cuda", dtype=dt, requires_grad=True)
            softmax.LigerSoftmaxFunction.apply(x).sum().backward()
        def c_kl(dt=dt):
            p = torch.randn(32, 512, device="cuda", dtype=dt).log_softmax(-1).requires_grad_()
            q = torch.randn(32, 512, device="cuda", dtype=dt).log_softmax(-1)
            kl_div.LigerKLDivLossFunction.apply(p, q, "batchmean", False).backward()
        def c_jsd(dt=dt):
            p = torch.randn(32, 512, device="cuda", dtype=dt).log_softmax(-1).requires_grad_()
            q = torch.randn(32, 512, device="cuda", dtype=dt).log_softmax(-1)
            jsd.LigerJSDFunction.apply(p, q, None, 0.5, -100).backward()
        def c_gn(dt=dt):
            x = torch.randn(4, 32, 128, device="cuda", dtype=dt, requires_grad=True)
            w = torch.randn(32, device="cuda", dtype=dt, requires_grad=True)
            b = torch.randn(32, device="cuda", dtype=dt, requires_grad=True)
            group_norm.LigerGroupNormFunction.apply(x, w, b, 32, 8, 1e-5).sum().backward()
        def c_dyt(dt=dt):
            x = torch.randn(64, 256, device="cuda", dtype=dt, requires_grad=True)
            alpha = torch.tensor([0.5], device="cuda", dtype=dt, requires_grad=True)
            gamma = torch.randn(256, device="cuda", dtype=dt, requires_grad=True)
            beta = torch.randn(256, device="cuda", dtype=dt, requires_grad=True)
            dyt.LigerDyTFunction.apply(x, alpha, gamma, beta).sum().backward()
        def c_sparsemax(dt=dt):
            x = torch.randn(32, 256, device="cuda", dtype=dt, requires_grad=True)
            sparsemax.LigerSparsemaxFunction.apply(x, -1).sum().backward()
        def c_tvd(dt=dt):
            p = torch.randn(32, 512, device="cuda", dtype=dt).softmax(-1).requires_grad_()
            q = torch.randn(32, 512, device="cuda", dtype=dt).softmax(-1)
            tvd.LigerTVDLossFunction.apply(p, q, None, "batchmean", -100).backward()
        for name, fn in (("rms_norm", c_rms), ("layer_norm", c_ln), ("swiglu", c_swiglu), ("geglu", c_geglu),
                         ("cross_entropy", c_ce), ("rope", c_rope), ("softmax", c_softmax), ("kl_div", c_kl),
                         ("jsd", c_jsd), ("group_norm", c_gn), ("dyt", c_dyt), ("sparsemax", c_sparsemax),
                         ("tvd", c_tvd)):
            calls.append((f"liger.{name}.{str(dt).split('.')[-1]}", fn))
    return calls


def gems_calls(torch):
    calls = []
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        for name, fn in gems_ops(torch, dt).items():
            calls.append((f"gems.{name}.{str(dt).split('.')[-1]}", fn))
    return calls


def gems_ops(torch, dt):
    """The FlagGems calls at one dtype.  Built in their own scope so that every lambda binds this dtype's ``x_`` (the
    first version defined them in the dtype loop: the lambdas looked ``x_`` and ``dt`` up when called, so all three
    dtypes ran bf16)."""
    import torch.nn.functional as Fn

    def x_(shape=(64, 512), pos=False):
        g = torch.Generator(device="cuda").manual_seed(1)
        v = torch.randn(*shape, device="cuda", generator=g)
        return (v.abs() + 0.1 if pos else v).to(dt)
    return {
        "abs": lambda: torch.abs(x_()), "exp": lambda: torch.exp(x_()), "log": lambda: torch.log(x_(pos=True)),
        "sin": lambda: torch.sin(x_()), "cos": lambda: torch.cos(x_()), "tanh": lambda: torch.tanh(x_()),
        "sigmoid": lambda: torch.sigmoid(x_()), "gelu": lambda: Fn.gelu(x_()), "silu": lambda: Fn.silu(x_()),
        "relu": lambda: torch.relu(x_()), "rsqrt": lambda: torch.rsqrt(x_(pos=True)),
        "sqrt": lambda: torch.sqrt(x_(pos=True)), "erf": lambda: torch.erf(x_()),
        "reciprocal": lambda: torch.reciprocal(x_(pos=True)), "neg": lambda: torch.neg(x_()),
        "pow": lambda: torch.pow(x_(pos=True), 1.7), "div": lambda: x_() / x_(pos=True),
        "add": lambda: x_() + x_(), "mul": lambda: x_() * x_(), "sub": lambda: x_() - x_(),
        "where": lambda: torch.where(x_() > 0, x_(), x_() * 2), "clamp": lambda: torch.clamp(x_(), -0.5, 0.5),
        "sum": lambda: x_().sum(dim=1), "mean": lambda: x_().mean(dim=1), "amax": lambda: x_().amax(dim=1),
        "prod": lambda: x_((64, 16)).prod(dim=1), "argmax": lambda: x_().argmax(dim=1),
        "var": lambda: x_().var(dim=1), "norm": lambda: torch.linalg.vector_norm(x_(), dim=1),
        "softmax": lambda: torch.softmax(x_(), dim=1), "log_softmax": lambda: torch.log_softmax(x_(), dim=1),
        "layer_norm": lambda: Fn.layer_norm(x_(), (512,)), "cumsum": lambda: torch.cumsum(x_(), dim=1),
        "mm": lambda: x_((64, 128)) @ x_((128, 64)), "bmm": lambda: torch.bmm(x_((4, 32, 64)), x_((4, 64, 32))),
        "addmm": lambda: torch.addmm(x_((64,)), x_((64, 128)), x_((128, 64))),
        "cross_entropy": lambda: Fn.cross_entropy(x_((64, 100)).float(),
                                                  torch.randint(0, 100, (64,), device="cuda")),
        "gather": lambda: torch.gather(x_(), 1, torch.randint(0, 512, (64, 16), device="cuda")),
        "index_select": lambda: torch.index_select(x_(), 0, torch.randint(0, 64, (32,), device="cuda")),
        "sort": lambda: torch.sort(x_((16, 256)), dim=1), "topk": lambda: torch.topk(x_((16, 256)), 8, dim=1),
        "embedding": lambda: Fn.embedding(torch.randint(0, 64, (32,), device="cuda"), x_()),
        "masked_fill": lambda: x_().masked_fill(x_() > 1, 0.0), "max_dim": lambda: torch.max(x_(), dim=1),
        "min": lambda: torch.min(x_()), "isnan": lambda: torch.isnan(x_()), "floor_divide":
            lambda: torch.floor_divide(x_(), x_(pos=True)), "rms_norm": lambda: Fn.rms_norm(x_(), (512,), torch.ones(512, device="cuda", dtype=dt)),
        "group_norm": lambda: Fn.group_norm(x_((4, 32, 128)), 8), "dropout": lambda: Fn.dropout(x_(), 0.1, True),
        "linspace": lambda: torch.linspace(0, 1, 1000, device="cuda", dtype=dt),
    }


GEMS_EXCLUDE = ["copy_", "fill.Scalar", "fill.Tensor", "fill_.Scalar", "fill_.Tensor", "zero", "zero_", "zeros",
                "zeros_like", "normal_", "normal.Tensor_float", "normal.Tensor_Tensor", "normal.float_Tensor",
                "uniform_", "rand", "randn", "randn_like", "index_put", "index_put_"]


def ttir_ops(text):
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir
    m = parse_ttir(text)
    return [op for fn in m.funcs.values() for op in fn.walk()]


def classify_launch(launch, catalog_pids, catalog_names):
    """Primitives and forms of one launch (K1 / K2 come from coverage.classify; the K1 here is by name only)."""
    ops = ttir_ops(launch.asm["ttir"])
    prims, forms, k1 = Counter(), Counter(), Counter()
    for op in ops:
        f = FORM_OF.get(op.name)
        if f:
            forms[f] += 1
        if op.name == "tt.load" and len(op.operands) >= 2:
            forms["C10"] += 1
        if op.name in ("scf.yield", "tt.return", "tt.reduce.return", "tt.scan.return", "tt.map_elementwise.return",
                       "scf.condition", "tt.func", "tt.load", "tt.store", "tt.addptr", "tt.splat", "tt.broadcast",
                       "tt.expand_dims", "tt.make_range", "tt.get_program_id", "tt.get_num_programs", "tt.reshape",
                       "tt.trans", "tt.join", "tt.split", "tt.cat", "arith.constant", "tt.reduce", "tt.scan",
                       "scf.for", "scf.if", "scf.while", "tt.atomic_rmw", "tt.atomic_cas", "gpu.barrier",
                       "tt.assert", "tt.print", "tt.gather", "tt.map_elementwise", "tt.histogram", "tt.call",
                       "tt.bitcast", "ttg.convert_layout", "tt.unsplat", "tt.int_to_ptr", "tt.ptr_to_int",
                       "tt.make_tensor_ptr", "tt.advance", "ub.poison", "cf.cond_br", "cf.br", "cf.assert"):
            prims[("structural", op.name)] += 1
            continue
        if op.name == "tt.extern_elementwise":
            sym = op.attrs.get("symbol", "").strip('"')
            key = f"libdevice|{sym}"
            prims[(key,)] += 1
            if key not in catalog_pids:
                k1[key] += 1
            continue
        elem = None
        if op.result_types:
            elem = getattr(op.result_types[0], "elem", None)
        key = (op.name, elem)
        prims[key] += 1
        if op.name not in catalog_names:
            k1[op.name] += 1
    return prims, forms, k1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results/corpus.json"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    import torch
    import cells as CL
    import coverage as CV
    keys, cells_tab = CV.catalog_keys(), CV.cell_table()
    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
    cat = json.loads((HERE / "results/catalog.json").read_text())
    measured = {r["pid"] for r in cat["records"] if r.get("status") == "measured"}
    names = {r["op"] for r in cat["records"] if r.get("status") == "measured"}
    calls = []
    try:
        calls += liger_calls(torch)
    except Exception as e:
        print("liger unavailable:", e)
    try:
        import flag_gems
        gems = gems_calls(torch)
    except Exception as e:
        print("flag_gems unavailable:", e)
        gems, flag_gems = [], None
    import faulthandler
    records, by_hash = [], {}
    for name, fn in calls + gems:
        faulthandler.dump_traceback_later(900, exit=False)     # a stuck call shows where it waits
        if a.only and a.only not in name:
            continue
        per_call = Counter()

        def take_first_two(kname, idx, per_call=per_call):
            # at most two launches of each kernel per call (the first version counted across calls, so a kernel
            # already seen at one dtype was not captured at the others)
            per_call[kname] += 1
            return per_call[kname] <= 2
        rec = TritonLaunchRecorder(select=take_first_two)
        try:
            if name.startswith("gems.") and flag_gems is not None:
                # the recorder copies operands with torch ops before each launch: routed through FlagGems they would
                # launch FlagGems kernels inside the launch hook (its non-reentrant lock deadlocks), so the copy,
                # fill and random ops stay on ATen
                with rec, flag_gems.use_gems(exclude=GEMS_EXCLUDE):
                    fn()
                    torch.cuda.synchronize()
            else:
                with rec:
                    fn()
                    torch.cuda.synchronize()
        except Exception as e:
            records.append({"call": name, "status": "call failed", "error": f"{type(e).__name__}: {str(e)[:200]}"})
            print(name, "call failed", str(e)[:120], flush=True)
            continue
        for L in rec.launches:
            h = hashlib.sha256(L.asm["ttir"].encode()).hexdigest()
            if h in by_hash:
                by_hash[h]["calls"].append(name)
                continue
            try:
                prims, forms, _ = classify_launch(L, measured, names)
                cov = CV.classify(L.asm["ttir"], keys, cells_tab)
            except Exception as e:
                prims, forms = Counter(), Counter()
                cov = {"k1": {f"parse error: {type(e).__name__}": 1}, "k1_name_only": {}, "k2": {}, "pairs": {},
                       "k2_not_bit_identical": []}
            r = CL.compare_launch(L)
            entry = {"call": name, "calls": [name], "kernel": L.kernel_name, "ttir_sha256": h,
                     "status": r.get("status"), "elements": r.get("elements"), "not_emulable": r.get("not_emulable"),
                     "abort_reasons": r.get("abort_reasons"), "error": r.get("error"),
                     "forms": dict(forms), "primitives": {"|".join(map(str, k)): v for k, v in prims.items()},
                     "k1_outside_catalog": cov["k1"], "k1_name_only": cov["k1_name_only"],
                     "k2": cov["k2"], "k2_not_bit_identical": cov["k2_not_bit_identical"]}
            by_hash[h] = entry
            records.append(entry)
            print(name, L.kernel_name, entry["status"], (entry["elements"] or {}).get("bit_identical"),
                  "/", (entry["elements"] or {}).get("written"), entry["not_emulable"] or entry["abort_reasons"] or "",
                  flush=True)
        del rec
        torch.cuda.empty_cache()
    kernels = [r for r in records if "kernel" in r]
    st = Counter(r["status"] for r in kernels)
    forms = Counter()
    prims = Counter()
    k1, k2 = Counter(), Counter()
    for r in kernels:
        forms.update({k: 1 for k in r["forms"]})
        prims.update({k: 1 for k in r["primitives"]})
        k1.update({k: 1 for k in r["k1_outside_catalog"]})
        k2.update(r["k2"].values())
    el = Counter()
    for r in kernels:
        el.update(r.get("elements") or {})
    summary = {"calls": len(calls + gems), "distinct_kernels": len(kernels), "status": dict(st),
               "elements": dict(el), "kernels_with_form": dict(forms.most_common()),
               "kernels_with_primitive": dict(prims.most_common(80)), "k1_kernels": dict(k1),
               "kernels_without_k1": sum(1 for r in kernels if not r["k1_outside_catalog"]),
               "k2_pairs_by_status": dict(k2),
               "kernels_all_pairs_bit_identical": sum(1 for r in kernels if not r["k1_outside_catalog"]
                                                      and not r["k2_not_bit_identical"]),
               "call_failures": [r for r in records if r.get("status") == "call failed"]}
    Path(a.out).write_text(json.dumps({"summary": summary, "kernels": kernels}, indent=1, default=str,
                                      ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("calls", "distinct_kernels", "status", "elements")}, default=str))


if __name__ == "__main__":
    main()
