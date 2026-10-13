"""Audit of one Triton kernel launch: how far its outputs deviate, where the deviation comes from, what cannot be told.

    report = audit(lambda: my_kernel[grid](...), spec={"o_ptr": lambda: reference_tensor})
    print(report["text"])

Steps (each with the locked lowering rules of Triton 3.6.0 / sm_86):

1. prediction before the run: the launch is captured without running the kernel and its bias predicted
   (bias_predict) -- the alignment <K - G, G> / <G, G> of every float output, or "cannot be decided" with the
   failure condition that applies;
2. the run: bitwise check of the lowering rules (emulate.verify); the exact value G of the declared program
   (reference evaluator); the measured deviation K - G;
3. attribution: every rounding / approximating node of the emulation made exact in turn; its contribution to the
   alignment of each output is K_emulated - K_(node exact), on the same inputs (exact, not a model of the
   distribution; contributions of different nodes interact, so they add up only approximately);
4. the specification, if given: the semantic deviation G - S (constants, formula) and the total K - S;
5. findings: the mechanism of each large contribution and a fix, with the change the model gives when that node is
   exact (a counterfactual of the model, not a run of a modified kernel).
"""
from __future__ import annotations

import re
from typing import Callable, Optional

import numpy as np

from . import bias_predict as BP
from . import emulate as E
from . import intervals as iv
from .ttir_eval import ST_OK, KernelReferenceEvaluator, NumericMode
from .ttir_parser import parse_ttir

# mechanism and fix per lowering kind / operation (first match wins)
MECHANISMS = [
    (r"dot_tensor_core_tf32", "tf32 张量核 dot：两个操作数向零截断到 11 位（每个约 −3.5e-4 的系统收缩），累加也向零截断",
     "传 input_precision=\"ieee\"（fp32 乘积精确、较慢），或在 tl.dot 前把操作数就近取到 tf32（cvt.rna.tf32.f32），保留 tf32 速度"),
    (r"dot_tensor_core_(bf16|f16)", "张量核 dot：每组乘积与累加器向零截断（随 K 增长的小的系统收缩）",
     "通常可忽略；确有影响时分段累加到 fp32"),
    (r"dot_fma_chain", "fp32 dot 按 fma 链逐步就近舍入", "不需要"),
    (r"oracle.*math\.(exp2?|log2?|sqrt|rsqrt|sin|cos|tanh|erf)|oracle.*arith\.divf",
     "近似指令（ex2.approx / lg2.approx / div.full …）：几分之一 ulp 的系统平均误差",
     "影响确实要紧时改用正确舍入的 libdevice 函数或 tl.math.div_rn"),
    (r"reduce_tree", "按布局次序求和（每次组合就近舍入）", "影响要紧时在更宽的类型里累加"),
    (r"scan_order", "按下降次序做前缀和（每次组合就近舍入）", "影响要紧时在更宽的类型里累加"),
    (r"truncf|fp_to_fp", "转到更窄的格式（舍入）", "影响要紧时一直保留宽类型，直到写出"),
    (r"fma_contraction|fused_product", "乘加收缩成一次 fma（一次舍入）", "不需要"),
]


def _mechanism(node: str, how: str, text: str):
    key = f"{how} {node} {text}"
    for pat, mech, fix in MECHANISMS:
        if re.search(pat, key):
            return mech, fix
    return "在声明类型中就近舍入", "影响不大时不需要"


def _alignment(e, g) -> float:
    gg = float((g * g).sum())
    return float((e * g).sum() / gg) if gg > 0 else float("nan")


def _loop_carried_narrow(ttir: str, ops: dict) -> set:
    """Nodes inside an scf.for that narrow a value (truncf / fp_to_fp, or an add in a 16-bit type) whose result -- through
    extensions and layout operations -- is carried to the next iteration (an operand of the loop's scf.yield)."""
    module = parse_ttir(ttir)
    out = set()
    for fn in module.funcs.values():
        for loop in (o for o in fn.walk() if o.name == "scf.for"):
            body = [o for reg in loop.regions for blk in reg.blocks for o in blk.ops]
            yields = [o for o in body if o.name == "scf.yield"]
            if not yields:
                continue
            carried = set(yields[-1].operands)
            producer = {r: o for o in body for r in o.results}
            frontier = list(carried)
            seen = set()
            while frontier:
                v = frontier.pop()
                o = producer.get(v)
                if o is None or v in seen:
                    continue
                seen.add(v)
                narrow = o.name in ("arith.truncf", "tt.fp_to_fp") or (
                    o.name in ("arith.addf", "arith.subf") and o.result_types
                    and getattr(o.result_types[0], "elem", "") in ("bf16", "f16"))
                if narrow:
                    out.add(o.node_id)
                if o.name in ("arith.extf", "tt.splat", "tt.broadcast", "ttg.convert_layout", "tt.reshape") or narrow:
                    frontier += list(o.operands)
    return out


def audit(launch: Callable[[], None], spec: Optional[dict] = None, programs: Optional[list] = None,
          attribution_programs: int = 2, top: int = 8, predict: bool = True,
          tolerance: Optional[float] = None) -> dict:
    """Audit the first Triton launch made by ``launch()``.

    spec: {kernel argument name: callable returning the expected values of that tensor (float64, same number of
    elements)}.  programs: the program ids (x, y, z) to evaluate (default: all; large kernels should pass a sample).
    tolerance: an alignment below which a deviation does not matter to the caller (default: the unit roundoff of the
    output format -- a bias it cannot resolve is not a defect of the kernel).
    """
    import torch

    from .capture import TritonLaunchRecorder
    rep = {"prediction": None, "verify": None, "measured": {}, "attribution": {}, "spec": {}, "findings": [],
           "cannot_decide": []}
    # 1. prediction, kernel not run
    if predict:
        with TritonLaunchRecorder(execute=False) as R:
            launch()
        L0 = R.launches[0]
        pred = BP.predict(L0, programs=programs)
        rep["prediction"] = BP.summarize(pred)
        rep["cannot_decide"] += [f"预测：{f}" for f in rep["prediction"]["failures"]]
    # 2. the run, bitwise check, exact reference
    with TritonLaunchRecorder() as R:
        launch()
        torch.cuda.synchronize()
    L = R.launches[0]
    ver = E.verify(L, ungrouped=[], swapped=[], programs=programs)
    rep["verify"] = {k: ver[k] for k in ("status", "buffers", "not_emulable_nodes", "abort_reasons")}
    if ver["status"] != "bit_identical":
        rep["cannot_decide"].append(f"逐位核对：{ver['status']} {ver['not_emulable_nodes'][:3]} "
                                    f"{ver['abort_reasons'][:2]}")
    module = parse_ttir(L.asm["ttir"])
    module.source_text = L.asm["ttir"]
    ev = KernelReferenceEvaluator(module, mode=NumericMode.NUMERICAL_DIFFERENCE, ttgir=L.asm["ttgir"])
    original = iv.elementary_bounds
    iv.elementary_bounds = BP._float64_elementary(original)
    try:
        res = ev.evaluate(L, programs=programs)
    finally:
        iv.elementary_bounds = original
    G, idx = {}, {}
    for buf in res.buffers.values():
        if buf.kind != "f" or not buf.written.any():
            continue
        ok = buf.written & (buf.st == ST_OK) & (buf.actual_after_st == 0) & np.isfinite(buf.actual_after)
        G[buf.name] = 0.5 * (buf.lo[ok] + buf.hi[ok])
        idx[buf.name] = ok
        K = buf.actual_after[ok]
        g = G[buf.name]
        gg = float((g * g).sum())
        e = K - g
        rep["measured"][buf.name] = {
            "elements": int(ok.sum()), "written": int(buf.written.sum()), "alignment": _alignment(e, g),
            "format_resolution": 2.0 ** -iv.FLOAT_FORMATS[buf.elem][0] if buf.elem in iv.FLOAT_FORMATS else 0.0,
            "se": float(np.std(e * g) * np.sqrt(g.size) / gg) if gg > 0 else float("nan"),
            "energy_relative": float(np.sqrt((e * e).sum() / gg)) if gg > 0 else float("nan"),
            "max_abs": float(np.max(np.abs(e))) if g.size else 0.0}
    # 3. attribution on the emulation
    progs_a = (programs or [(x, y, z) for z in range(L.grid[2]) for y in range(L.grid[1])
                            for x in range(L.grid[0])])[:attribution_programs]
    base, emu = E.emulate(L, ungrouped=[], swapped=[], programs=progs_a)
    outs = E._outputs(base, L)
    ops = {o.node_id: o for fn in emu.module.funcs.values() for o in fn.walk()}
    texts = {n: o.text for n, o in ops.items()}
    locations = E.ttir_locations(L.asm["ttir"])
    defining = {}
    for line in L.asm["ttir"].splitlines():
        m = re.match(r"^\s*(%[\w#]+)(?::\d+)?\s*=", line)
        if m:
            defining.setdefault(m.group(1), line)

    def source(node):
        loc = E.op_location(ops[node], locations, defining) if node in ops else None
        return f"{loc[0]}:{loc[1]}" if loc else ""
    kinds = {}
    for (node, how) in emu.trace:
        kinds.setdefault(node, how)
    ref_a = None
    try:
        ref_a, _ = E.emulate(L, exact_nodes=E.rounding_nodes(emu), ungrouped=[], swapped=[], programs=progs_a)
    except Exception:
        pass
    rows = {}
    for node in E.rounding_nodes(emu):
        try:
            r, _ = E.emulate(L, exact_nodes=[node], ungrouped=[], swapped=[], programs=progs_a)
        except Exception as e:
            rep["cannot_decide"].append(f"归因 {node}：{type(e).__name__}")
            continue
        ro = E._outputs(r, L)
        for name, o in outs.items():
            g_all = E._outputs(ref_a, L)[name][1] if ref_a is not None else o[1]
            ok = (o[3] == ST_OK) & (ro[name][3] == ST_OK) & o[0]
            if not ok.any():
                continue
            c = _alignment(o[1][ok] - ro[name][1][ok], g_all[ok])
            mech, fix = _mechanism(node, kinds.get(node, ""), texts.get(node, ""))
            rows.setdefault(name, []).append({"node": node, "lowering": kinds.get(node, ""),
                                              "source": source(node),
                                              "text": re.sub(r"\s+loc\(.*$", "", texts.get(node, "").strip())[:100],
                                              "contribution": c, "mechanism": mech, "fix": fix})
    for name, rs in rows.items():
        rs.sort(key=lambda r: -abs(r["contribution"]) if np.isfinite(r["contribution"]) else 0)
        total = sum(r["contribution"] for r in rs if np.isfinite(r["contribution"]))
        gross = sum(abs(r["contribution"]) for r in rs if np.isfinite(r["contribution"]))
        rep["attribution"][name] = {"programs": progs_a, "sum": total, "gross": gross, "nodes": rs[:top]}
    # 4. specification
    for name, fn in (spec or {}).items():
        if name not in G:
            rep["cannot_decide"].append(f"规格 {name}：没有这个名字的已求值输出")
            continue
        S = np.asarray(fn().double().cpu().numpy() if hasattr(fn(), "double") else fn(), dtype=np.float64).reshape(-1)
        buf = next(b for b in res.buffers.values() if b.name == name)
        s = S[buf.global_indices()[idx[name]]]
        K = buf.actual_after[idx[name]]
        rep["spec"][name] = {"semantic_alignment": _alignment(G[name] - s, s), "total_alignment": _alignment(K - s, s),
                             "max_abs_semantic": float(np.max(np.abs(G[name] - s)))}
    # 5. verdict per output: significant bias? does the prediction agree with the measurement?
    for name, m in rep["measured"].items():
        p = (rep["prediction"] or {}).get("outputs", {}).get(name) if rep["prediction"] else None
        se = max(m["se"], p["alignment_noise_se"] if p else 0.0)
        m["statistically_significant"] = bool(np.isfinite(m["alignment"]) and abs(m["alignment"]) > 3 * se)
        # a bias below the output format's unit roundoff is below what the stored values can resolve: not a defect
        floor = tolerance if tolerance is not None else m["format_resolution"]
        m["significant"] = bool(m["statistically_significant"] and abs(m["alignment"]) > floor)
        if p and rep["prediction"]["status"] == "predicted":
            m["prediction_agrees"] = bool(abs(m["alignment"] - p["alignment"])
                                          <= 3 * np.hypot(m["se"], p["alignment_noise_se"]) + 0.1 * abs(p["alignment"]))
    # 6. findings (only for outputs with a significant bias; the semantic deviation always)
    loop_carried = _loop_carried_narrow(L.asm["ttir"], ops)
    for name, a in rep["attribution"].items():
        meas = rep["measured"].get(name, {}).get("alignment", float("nan"))
        if not rep["measured"].get(name, {}).get("significant"):
            continue
        for r in a["nodes"]:
            if r["node"] in loop_carried:
                r["mechanism"] = ("循环携带的量保留在窄类型里（低精度累加：增量不足一个 ulp 时被吞掉，长循环会停滞）")
                r["fix"] = "把循环里的累加量保持在 fp32，循环结束后再转换"
            if np.isfinite(r["contribution"]) and abs(r["contribution"]) >= max(0.1 * abs(meas), 1e-7):
                rep["findings"].append({"output": name, **r,
                                        "counterfactual": f"该位点精确时对齐 {meas:+.3g} → {meas - r['contribution']:+.3g}"
                                                          f"（模型内反事实）"})
        if a["gross"] > 3 * abs(a["sum"]) and a["gross"] > 3 * rep["measured"][name]["se"]:
            rep["findings"].append({"output": name, "node": "-", "mechanism": f"贡献相消：各节点贡献的绝对值之和 "
                                    f"{a['gross']:.3g}，净和 {a['sum']:.3g}",
                                    "fix": "-", "contribution": a["sum"], "counterfactual": "-", "lowering": "",
                                    "source": "", "text": ""})
    for name, s in rep["spec"].items():
        if abs(s["semantic_alignment"]) > 1e-6:
            rep["findings"].append({"output": name, "node": "-", "lowering": "", "source": "", "text": "",
                                    "mechanism": f"声明的程序与规格不一致：对齐 {s['semantic_alignment']:+.3g}"
                                                 f"（常数、公式、mask）——数值舍入解释不了",
                                    "fix": "核对 kernel 里的常数与公式和规格是否一致",
                                    "contribution": s["semantic_alignment"], "counterfactual": "-"})
    rep["text"] = render(rep)
    return rep


def render(rep: dict) -> str:
    out = ["══ 审查报告 ══"]
    if rep["prediction"]:
        p = rep["prediction"]
        out.append(f"【事前预测，kernel 未运行】{'已预测' if p['status'] == 'predicted' else '无法判断'}")
        for n, o in p["outputs"].items():
            out.append(f"  {n}: 对齐 {o['alignment']:+.3g}（噪声标准误 {o['alignment_noise_se']:.2g}），"
                       f"相对误差（能量加权）{o.get('energy_relative_error', float('nan')):.2g}")
    v = rep["verify"]
    out.append(f"【运行】下降规则逐位核对：{v['status']}")
    for n, m in rep["measured"].items():
        verdict = ("有显著的系统偏差" if m.get("significant") else
                   (f"统计上显著，但小于输出格式的单位舍入 {m['format_resolution']:.2g}，可忽略"
                    if m.get("statistically_significant") else "未发现显著偏差"))
        agree = {True: "，与事前预测一致", False: "，与事前预测不一致（工具需要检查）"}.get(m.get("prediction_agrees"), "")
        out.append(f"  {n}: K − G 对齐 {m['alignment']:+.3g}（标准误 {m['se']:.2g}）——{verdict}{agree}")
        out.append(f"      相对误差（能量加权）{m['energy_relative']:.2g}，最大 |K − G| {m['max_abs']:.3g}"
                   f"（{m['elements']} / {m['written']} 个元素）")
    for n, s in rep["spec"].items():
        out.append(f"  {n}: G − S（语义）{s['semantic_alignment']:+.3g}，K − S（合计）{s['total_alignment']:+.3g}")
    for n, a in rep["attribution"].items():
        label = "来源" if rep["measured"].get(n, {}).get("significant") else "来源（偏差不显著，仅供参考）"
        out.append(f"【{n} 的{label}】各节点对对齐量的贡献（program {a['programs']}；合计 {a['sum']:+.3g}）")
        for r in a["nodes"]:
            out.append(f"  {r['contribution']:+.3g}  {r['node']:<22} {r['source']:<20} {r['mechanism'][:60]}")
    if rep["findings"]:
        out.append("【发现】")
        for f in rep["findings"]:
            out.append(f"  - {f['output']}：{f['mechanism']}")
            if f["fix"] != "-":
                out.append(f"      改法：{f['fix']}；{f['counterfactual']}")
    if rep["cannot_decide"]:
        out.append("【无法判断】")
        out += [f"  - {c}" for c in rep["cannot_decide"]]
    return "\n".join(out)
