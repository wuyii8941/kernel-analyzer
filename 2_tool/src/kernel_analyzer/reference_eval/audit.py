"""Audit of one Triton kernel launch: how far its outputs deviate, where the deviation comes from, how to fix it, what
cannot be told.

    report = audit(lambda: my_kernel[grid](...), spec={"o_ptr": lambda: reference_tensor}, names={"o_ptr": "O"})
    print(report["text"])

The report starts with a summary for the kernel's author (in plain terms: which output is biased by how many percent,
whether common tolerance checks would let it through, the source lines responsible, fixes with their cost and the
predicted effect of fixing one site or all of them, the scope) and keeps the technical evidence in an appendix.

Steps (each with the locked lowering rules of Triton 3.6.0 / sm_86):

1. prediction before the run: the launch is captured without running the kernel and its bias predicted
   (bias_predict) -- the alignment <K - G, G> / <G, G> of every float output, or "cannot be decided" with the
   failure condition that applies;
2. the run: bitwise check of the lowering rules (emulate.verify); the exact value G of the declared program
   (reference evaluator); the measured deviation K - G;
3. attribution: every rounding / approximating node of the emulation made exact in turn; its contribution to the
   alignment of each output is K_emulated - K_(node exact), on the same inputs (contributions of different nodes
   interact, so they add up only approximately);
4. the specification, if given: the semantic deviation G - S (constants, formula) and the total K - S; common
   tolerance checks evaluated on K against S (or G), and on K without its systematic component (whether a failing
   check is failed by the bias or by the random error alone);
5. findings: the mechanism of each large contribution, fixes with their cost, and the deviation the model gives when
   one site or every site of the mechanism is exact (a counterfactual of the model, not a run of a modified kernel).
"""
from __future__ import annotations

import linecache
import re
from typing import Callable, Optional

import numpy as np

from . import bias_predict as BP
from . import emulate as E
from . import intervals as iv
from .ttir_eval import ST_OK, KernelReferenceEvaluator, NumericMode
from .ttir_parser import parse_ttir

# mechanism, fix options (text, cost) per lowering kind / operation (first match wins)
MECHANISMS = [
    (r"dot_tensor_core_tf32", "tf32",
     "tf32 张量核 dot 把两个操作数向零截断到 11 位（每个操作数约 −0.035% 的系统收缩），累加也向零截断",
     [("在 tl.dot 前把操作数用 cvt.rna.tf32.f32 就近取到 tf32",
       "保留张量核，几乎不增加耗时；剩下张量核累加的截断（约 1e-5 量级）"),
      ("改用 input_precision=\"ieee\"", "同样消除偏差，但不再用张量核，通常慢数倍")]),
    (r"dot_tensor_core_(bf16|f16)", "mma",
     "张量核 dot 的每组乘积与累加器向零截断（随 K 增长的小的系统收缩）",
     [("通常不需要处理；确有影响时分段累加到 fp32", "多一次加法与寄存器")]),
    (r"dot_fma_chain", "fma", "fp32 dot 按 fma 链逐步就近舍入", [("不需要", "")]),
    (r"oracle.*math\.(exp2?|log2?|sqrt|rsqrt|sin|cos|tanh|erf)|oracle.*arith\.divf", "approx",
     "近似指令（ex2.approx / lg2.approx / div.full …）带几分之一 ulp 的系统平均误差",
     [("改用正确舍入的 libdevice 函数或 tl.math.div_rn", "慢若干倍，通常只在影响要紧时值得")]),
    (r"reduce_tree", "sum", "按布局次序求和，每次组合就近舍入", [("在更宽的类型里累加", "寄存器与带宽")]),
    (r"scan_order", "scan", "按下降次序做前缀和，每次组合就近舍入", [("在更宽的类型里累加", "寄存器与带宽")]),
    (r"truncf|fp_to_fp", "narrow", "转到更窄的格式时舍入", [("一直保留宽类型，直到写出", "寄存器与带宽")]),
    (r"fma_contraction|fused_product", "fma", "乘加收缩成一次 fma（一次舍入）", [("不需要", "")]),
]
LOOP_CARRIED = ("loop", "循环携带的量保留在窄类型里：增量不足一个 ulp 时被吞掉，长循环会停滞",
                [("把循环里的累加量保持在 fp32，循环结束后再转换", "多用寄存器，几乎不增加耗时")])
DEFAULT_NAME = "在声明类型中就近舍入"
SHORT = {"tf32": "tf32 张量核 dot 的操作数截断", "mma": "张量核 dot 的截断", "fma": "fma", "approx": "近似指令",
         "sum": "归约求和的舍入", "scan": "前缀和的舍入", "narrow": "窄化转换", "loop": "循环携带的窄类型累加",
         "rn": "就近舍入"}

# common tolerance checks: (label, atol, rtol); None rtol / atol: torch.testing.assert_close's default for the dtype
TOLERANCE_CHECKS = [("assert_close(atol=1e-2, rtol=0)（Triton 教程的测试）", 1e-2, 0.0),
                    ("allclose(rtol=1e-2, atol=1e-5)", 1e-5, 1e-2),
                    ("assert_close 的 dtype 默认值", None, None)]
_ASSERT_CLOSE = {"f32": (1e-5, 1.3e-6), "f16": (1e-5, 1e-3), "bf16": (1e-5, 1.6e-2), "f64": (1e-7, 1e-7)}


def _mechanism(node: str, how: str, text: str):
    key = f"{how} {node} {text}"
    for pat, tag, mech, fixes in MECHANISMS:
        if re.search(pat, key):
            return tag, mech, fixes
    return "rn", DEFAULT_NAME, [("影响不大时不需要处理", "")]


def _alignment(e, g) -> float:
    gg = float((g * g).sum())
    return float((e * g).sum() / gg) if gg > 0 else float("nan")


def _se(e, g) -> float:
    """standard error of the alignment <e, g> / <g, g> over the elements"""
    gg = float((g * g).sum())
    return float(np.std(e * g) * np.sqrt(g.size) / gg) if gg > 0 else float("nan")


def _pct(a: float) -> str:
    return f"{abs(a) * 100:.2g}%" if abs(a) >= 1e-6 else "<0.0001%"


def _spct(a: float) -> str:
    return ("−" if a < 0 else "+") + _pct(a) if abs(a) >= 1e-6 else "<0.0001%"


def _loop_carried_narrow(ttir: str) -> set:
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
            producer = {r: o for o in body for r in o.results}
            frontier, seen = list(yields[-1].operands), set()
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


def sample_programs(grid, k: int = 2) -> tuple:
    """Programs to evaluate when the caller gives none: the first and the last program along every grid axis (the last
    block of a sequence is where masking and boundary handling differ), up to ``k`` more spread in between; and the
    rule, in words."""
    gx, gy, gz = (int(g) for g in grid)
    total = gx * gy * gz
    pick = {(0, 0, 0), (gx - 1, gy - 1, gz - 1), (gx - 1, 0, 0), (0, gy - 1, 0)}
    for i in range(1, k + 1):
        pick.add((min(gx - 1, i * gx // (k + 1)), min(gy - 1, i * gy // (k + 1)), 0))
    progs = sorted(p for p in pick if p[0] < gx and p[1] < gy and p[2] < gz)
    if len(progs) >= total:
        return [(x, y, z) for z in range(gz) for y in range(gy) for x in range(gx)], f"全部 {total} 个 program"
    return progs, f"抽查 {len(progs)}/{total} 个 program（规则：各轴首尾 program，含序列末尾的边界块，另取中间 {k} 个）"


def _sources(ttir: str) -> dict:
    """basename -> full path of the source files named in the TTIR locations (to quote source lines)."""
    out = {}
    for path in re.findall(r'loc\("([^"]+\.py)":\d+:\d+\)', ttir):
        out.setdefault(path.rsplit("/", 1)[-1], path)
    return out


def audit(launch: Callable[[], None], spec: Optional[dict] = None, programs: Optional[list] = None,
          names: Optional[dict] = None, attribution_programs: Optional[int] = None, top: int = 8, predict: bool = True,
          tolerance: Optional[float] = None, tolerance_checks: Optional[list] = None) -> dict:
    """Audit the first Triton launch made by ``launch()``.

    spec: {kernel argument name: callable returning the expected values of that tensor (float64, same number of
    elements)}.  names: {kernel argument name: name to show}.  programs: the program ids (x, y, z) to evaluate
    (default: sample_programs; "all": every program).  attribution_programs: evaluate the attribution on this many of them (the first ones and
    the last; default all -- the deviation before and after a fix then refers to the measured elements).  tolerance: an alignment below which a deviation does not matter to the caller
    (default: the unit roundoff of the output format -- a bias it cannot resolve is not a defect of the kernel).
    tolerance_checks: [(label, atol, rtol)], default TOLERANCE_CHECKS.
    """
    import torch

    from .capture import TritonLaunchRecorder
    names = names or {}
    rep = {"prediction": None, "verify": None, "measured": {}, "attribution": {}, "spec": {}, "findings": [],
           "fix_groups": [], "tolerance_checks": {}, "cannot_decide": [], "names": names}
    # 1. prediction, kernel not run
    with TritonLaunchRecorder(execute=False) as R:
        launch()
    L0 = R.launches[0]
    if programs is None:
        programs, rule = sample_programs(L0.grid)
    elif programs == "all":
        gx, gy, gz = (int(g) for g in L0.grid)
        programs = [(x, y, z) for z in range(gz) for y in range(gy) for x in range(gx)]
        rule = f"全部 {len(programs)} 个 program"
    else:
        total = int(np.prod(L0.grid))
        rule = f"指定的 {len(programs)}/{total} 个 program"
    rep["scope"] = {"programs": programs, "rule": rule, "kernel": L0.kernel_name,
                    "target": "Triton 3.6.0 / sm_86（锁定的下降规则）"}
    if predict:
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
    G, idx, K_of, elem_of = {}, {}, {}, {}
    for buf in res.buffers.values():
        if buf.kind != "f" or not buf.written.any():
            continue
        ok = buf.written & (buf.st == ST_OK) & (buf.actual_after_st == 0) & np.isfinite(buf.actual_after)
        G[buf.name] = g = 0.5 * (buf.lo[ok] + buf.hi[ok])
        idx[buf.name], K_of[buf.name], elem_of[buf.name] = ok, buf.actual_after[ok], buf.elem
        gg = float((g * g).sum())
        e = K_of[buf.name] - g
        rep["measured"][buf.name] = {
            "elements": int(ok.sum()), "written": int(buf.written.sum()), "alignment": _alignment(e, g),
            "format_resolution": 2.0 ** -iv.FLOAT_FORMATS[buf.elem][0] if buf.elem in iv.FLOAT_FORMATS else 0.0,
            "se": _se(e, g),
            "energy_relative": float(np.sqrt((e * e).sum() / gg)) if gg > 0 else float("nan"),
            "max_abs": float(np.max(np.abs(e))) if g.size else 0.0}
        w = np.asarray(buf.writer)[ok]                  # index into programs of the program that wrote each element
        rep["measured"][buf.name]["per_program"] = [
            [list(programs[j]), _alignment(e[w == j], g[w == j]), _se(e[w == j], g[w == j])] for j in np.unique(w[w >= 0])]
    # 3. specification and tolerance checks
    for name, fn in (spec or {}).items():
        if name not in G:
            rep["cannot_decide"].append(f"规格 {name}：没有这个名字的已求值输出")
            continue
        val = fn()
        S = np.asarray(val.double().cpu().numpy() if hasattr(val, "double") else val, dtype=np.float64).reshape(-1)
        buf = next(b for b in res.buffers.values() if b.name == name)
        s = S[buf.global_indices()[idx[name]]]
        rep["spec"][name] = {"semantic_alignment": _alignment(G[name] - s, s),
                             "total_alignment": _alignment(K_of[name] - s, s),
                             "max_abs_semantic": float(np.max(np.abs(G[name] - s)))}
        rep["spec"][name]["_values"] = s
    for name in G:
        # each check on K, and on K without its systematic component (K - a ref, a the alignment of K - ref): a check
        # that fails either way is failed by the random error and says nothing about the bias
        ref = rep["spec"].get(name, {}).get("_values", G[name])
        k0 = K_of[name] - _alignment(K_of[name] - ref, ref) * ref
        checks = []
        for label, atol, rtol in (tolerance_checks or TOLERANCE_CHECKS):
            if atol is None:
                atol, rtol = _ASSERT_CLOSE.get(elem_of[name], (1e-5, 1.3e-6))
            excess = np.abs(K_of[name] - ref) / (atol + rtol * np.abs(ref))
            checks.append({"check": label, "atol": atol, "rtol": rtol, "passes": bool(np.all(excess <= 1)),
                           "worst_ratio": float(excess.max()) if excess.size else 0.0,
                           "passes_without_bias": bool(np.all(np.abs(k0 - ref) <= atol + rtol * np.abs(ref))),
                           "against": "规格" if name in rep["spec"] else "声明的程序"})
        rep["tolerance_checks"][name] = checks
    for s in rep["spec"].values():
        s.pop("_values", None)
    # 4. verdict per output: significant bias? does the prediction agree with the measurement?
    for name, m in rep["measured"].items():
        p = (rep["prediction"] or {}).get("outputs", {}).get(name) if rep["prediction"] else None
        se = max(m["se"], p["alignment_noise_se"] if p else 0.0)
        m["statistically_significant"] = bool(np.isfinite(m["alignment"]) and abs(m["alignment"]) > 3 * se)
        floor = tolerance if tolerance is not None else m["format_resolution"]
        m["significant"] = bool(m["statistically_significant"] and abs(m["alignment"]) > floor)
        per = [(a, s_) for _, a, s_ in m["per_program"] if np.isfinite(a)]
        if len(per) > 1:
            (lo, s_lo), (hi, s_hi) = min(per), max(per)
            m["program_range"] = [lo, hi]
            m["programs_differ"] = bool(hi - lo > max(0.5 * abs(m["alignment"]), 3 * np.hypot(s_lo, s_hi), floor))
        if p and rep["prediction"]["status"] == "predicted":
            m["predicted"] = p["alignment"]
            m["prediction_agrees"] = bool(abs(m["alignment"] - p["alignment"])
                                          <= 3 * np.hypot(m["se"], p["alignment_noise_se"]) + 0.1 * abs(p["alignment"]))
    # 5. attribution on the emulation (by default on the same programs as the measurement, so that the deviation
    #    before and after a fix refer to the same elements)
    if attribution_programs is None or attribution_programs >= len(programs):
        progs_a = list(programs)
    else:
        progs_a = list(dict.fromkeys(programs[:max(attribution_programs - 1, 0)] + [programs[-1]]))   # incl. the last
    base, emu = E.emulate(L, ungrouped=[], swapped=[], programs=progs_a)
    outs = E._outputs(base, L)
    ops = {o.node_id: o for fn in emu.module.funcs.values() for o in fn.walk()}
    locations = E.ttir_locations(L.asm["ttir"])
    files = _sources(L.asm["ttir"])
    defining = {}
    for line in L.asm["ttir"].splitlines():
        m_ = re.match(r"^\s*(%[\w#]+)(?::\d+)?\s*=", line)
        if m_:
            defining.setdefault(m_.group(1), line)

    def source(node):
        loc = E.op_location(ops[node], locations, defining) if node in ops else None
        if not loc:
            return "", ""
        code = linecache.getline(files.get(loc[0], loc[0]), loc[1]).strip()
        return f"{loc[0]}:{loc[1]}", code
    kinds = {}
    for (node, how) in emu.trace:
        kinds.setdefault(node, how)
    loop_carried = _loop_carried_narrow(L.asm["ttir"])
    nodes = E.rounding_nodes(emu)
    exact_all = None
    try:
        exact_all, _ = E.emulate(L, exact_nodes=nodes, ungrouped=[], swapped=[], programs=progs_a)
    except Exception as e:
        rep["cannot_decide"].append(f"归因：所有位点同时精确的仿真失败（{type(e).__name__}），贡献以设备值为参照")
    g_exact = E._outputs(exact_all, L) if exact_all is not None else None
    before = {}                                   # the deviation on the attribution programs, from the emulation
    for name, o in outs.items():
        ok = (o[3] == ST_OK) & o[0]
        if g_exact is not None and ok.any():
            before[name] = _alignment(o[1][ok] - g_exact[name][1][ok], g_exact[name][1][ok])
        elif name in rep["measured"]:
            before[name] = rep["measured"][name]["alignment"]

    def contribution(exact_nodes) -> dict:
        r, _ = E.emulate(L, exact_nodes=list(exact_nodes), ungrouped=[], swapped=[], programs=progs_a)
        ro = E._outputs(r, L)
        out = {}
        for name, o in outs.items():
            g_all = g_exact[name][1] if g_exact is not None else o[1]
            ok = (o[3] == ST_OK) & (ro[name][3] == ST_OK) & o[0]
            if ok.any():
                out[name] = _alignment(o[1][ok] - ro[name][1][ok], g_all[ok])
        return out
    rows, single = {}, {}
    for node in nodes:
        try:
            cs = contribution([node])
        except Exception as e:
            rep["cannot_decide"].append(f"归因 {node}：{type(e).__name__}")
            continue
        tag, mech, fixes = LOOP_CARRIED if node in loop_carried else _mechanism(node, kinds.get(node, ""),
                                                                               ops[node].text if node in ops else "")
        src, code = source(node)
        for name, c in cs.items():
            single.setdefault(name, {})[node] = c
            rows.setdefault(name, []).append({"node": node, "lowering": kinds.get(node, ""), "source": src,
                                              "code": code, "contribution": c, "tag": tag, "mechanism": mech,
                                              "fixes": fixes})
    for name, rs in rows.items():
        rs.sort(key=lambda r: -abs(r["contribution"]) if np.isfinite(r["contribution"]) else 0)
        finite = [r["contribution"] for r in rs if np.isfinite(r["contribution"])]
        rep["attribution"][name] = {"programs": progs_a, "before": before.get(name, float("nan")),
                                    "sum": sum(finite), "gross": sum(abs(c) for c in finite),
                                    "nodes": [r for r in rs if r["contribution"] != 0][:top]}
    # 6. fix groups: the sites with a large contribution to an output with a significant bias, grouped by mechanism
    #    across outputs; the deviation after fixing every site of a group comes from one emulation with all of them
    #    exact, after fixing one site from that site's own emulation
    sig = [n for n in rep["attribution"] if rep["measured"].get(n, {}).get("significant")]
    groups, big_of = {}, {}
    for name in sig:
        ref = abs(before.get(name, rep["measured"][name]["alignment"]))
        big_of[name] = [r for r in rows[name] if np.isfinite(r["contribution"])
                        and abs(r["contribution"]) >= max(0.1 * ref, 1e-7)]
        for r in big_of[name]:
            gr = groups.setdefault(r["tag"], {"tag": r["tag"], "mechanism": r["mechanism"], "fixes": r["fixes"],
                                              "sites": {}})
            gr["sites"].setdefault(r["node"], {"node": r["node"], "source": r["source"], "code": r["code"]})
    for gr in groups.values():
        site_nodes = list(gr["sites"])
        combined = None
        if len(site_nodes) > 1:
            try:
                combined = contribution(site_nodes)
            except Exception:
                combined = None
        gr["outputs"] = {}
        for name in sig:
            b = before[name]
            c_each = {n_: single[name].get(n_, 0.0) for n_ in site_nodes}
            c_all = combined.get(name, float("nan")) if combined is not None else sum(c_each.values())
            gr["outputs"][name] = {"before": b, "after_all": b - c_all, "contribution": c_each,
                                   "after_each": {n_: b - c for n_, c in c_each.items()},
                                   "combined_emulated": combined is not None or len(site_nodes) == 1}
        gr["sites"] = list(gr["sites"].values())
        rep["fix_groups"].append(gr)
    for name in sig:
        rep["attribution"][name]["rest"] = sum(c for n_, c in single[name].items()
                                               if np.isfinite(c) and n_ not in {r["node"] for r in big_of[name]})
    for name, a in rep["attribution"].items():
        m = rep["measured"].get(name, {})
        if a["gross"] > 3 * abs(a["sum"]) and a["gross"] > 3 * m.get("se", 0.0) and a["gross"] > m.get(
                "format_resolution", 0.0):
            rep["findings"].append({"output": name, "kind": "cancel",
                                    "text": f"贡献相消：各位点贡献的绝对值之和 {_pct(a['gross'])}，净和 {_pct(a['sum'])}"})
    for name, s_ in rep["spec"].items():
        if abs(s_["semantic_alignment"]) > max(1e-6, rep["measured"][name]["format_resolution"]):
            rep["findings"].append({"output": name, "kind": "semantic",
                                    "text": f"按声明的程序精确计算，结果就{'偏小' if s_['semantic_alignment'] < 0 else '偏大'}"
                                            f"约 {_pct(s_['semantic_alignment'])}，数值舍入解释不了；核对常数、公式与 mask"})
    rep["text"] = render(rep)
    return rep


def _show(rep, name):
    return rep["names"].get(name, name)


def _short(mechanism: str) -> str:
    return re.split(r"[（：，]", mechanism)[0]


def _site(r: dict) -> str:
    return r["source"] or r["node"]


def _in(group: dict, order: list) -> list:
    return [(n, group["outputs"][n]) for n in order if n in group["outputs"]]


def render(rep: dict) -> str:
    """The report: a summary per output for the kernel's author, causes per source line, fixes per mechanism with cost
    and predicted effect, the scope and what cannot be decided; the technical evidence in an appendix."""
    out = []
    meas, spec = rep["measured"], rep["spec"]
    sig = [n for n, m in meas.items() if m.get("significant")]
    sem = {f["output"]: f["text"] for f in rep["findings"] if f["kind"] == "semantic"}
    order = sorted(meas, key=lambda n: (n not in spec, -abs(spec.get(n, {}).get("total_alignment", 0.0)
                                                             if n in sem else meas[n]["alignment"])))
    pred_failed = rep["prediction"] is not None and rep["prediction"]["status"] != "predicted"
    verify_failed = rep["verify"]["status"] != "bit_identical"

    def size(a):
        return f"{'偏小' if a < 0 else '偏大'}约 {_pct(a)}" if abs(a) >= 1e-6 else "约为 0（<0.0001%）"
    head = []
    for n in order:
        if n in sem:
            head.append(f"输出 {_show(rep, n)} 比规格{size(spec[n]['total_alignment'])}（公式或常数与规格不一致）")
        elif n in sig:
            head.append(f"输出 {_show(rep, n)} 系统性{size(meas[n]['alignment'])}（不是噪声）")
    if head:
        title = "；".join(head)
    elif pred_failed or verify_failed:
        parts = [p_ for p_, f_ in (("事前预测", pred_failed), ("逐位核对与归因", verify_failed)) if f_]
        title = f"实测未发现显著偏差，但{'、'.join(parts)}没能完成（原因见「无法判断」）"
    else:
        title = "未发现需要处理的系统偏差"
    out.append(f"══ {rep['scope']['kernel']}：{title} ══")

    # per output
    for n in order:
        m = meas[n]
        name = _show(rep, n)
        pred = (f"运行前预测{size(m['predicted'])}，实测{size(m['alignment'])}，"
                f"{'一致' if m.get('prediction_agrees') else '不一致（工具需要检查）'}") if "predicted" in m else ""
        if n not in sig and n not in sem:
            if m.get("statistically_significant"):
                state = f"统计上可辨（{_spct(m['alignment'])}），但小于输出格式的单位舍入，可忽略"
            else:
                state = f"未发现显著偏差（实测 {_spct(m['alignment'])}，噪声标准误 {_pct(m['se'])}）"
            out.append(f"{name}：{state}{'；' + pred if pred else ''}。")
            continue
        out.append("")
        if n in sem:
            out.append(f"{name}：比规格{size(spec[n]['total_alignment'])}——来自公式或常数，不是数值误差")
            out.append(f"  · {sem[n]}（工具不能定位到具体的常数）。")
        else:
            out.append(f"{name}：系统性{size(m['alignment'])}（不是噪声）")
        checks = rep["tolerance_checks"].get(n, [])
        if checks:
            ok_ = [c["check"] for c in checks if c["passes"]]
            caught = [c["check"] for c in checks if not c["passes"] and c.get("passes_without_bias")]
            noisy = [c["check"] for c in checks if not c["passes"] and not c.get("passes_without_bias")]
            against = "" if n in spec else "（未给规格，对声明的程序）"
            parts = []
            if ok_:
                parts.append(f"{'、'.join(ok_)} 会通过——这个偏差会被漏掉")
            if caught:
                parts.append(f"{'、'.join(caught)} 会因为它报错")
            if noisy:
                parts.append(f"{'、'.join(noisy)} 去掉这个偏差也通不过（随机误差已超出），说明不了它")
            out.append(f"  · 常用容差检查{against}：{'；'.join(parts)}。")
        if pred:
            out.append(f"  · {'数值误差部分：' if n in sem else ''}{pred}{'，可忽略' if n in sem and n not in sig else ''}。")
        if "program_range" in m and n in sig:
            lo, hi = m["program_range"]
            if m.get("programs_differ"):
                more = "" if rep["scope"]["rule"].startswith("全部") else \
                    "抽查值不一定代表整个 kernel，要全量评估传 programs=\"all\""
                out.append(f"  · 各 program 的偏差差别大（{_spct(lo)} 到 {_spct(hi)}）{'：' + more if more else ''}。")
            else:
                out.append(f"  · 各 program 的偏差在 {_spct(lo)} 到 {_spct(hi)} 之间，差别在随机误差范围内。")
        if n in sig and n not in sem:
            out.append("  · 公式与规格一致，问题出在数值实现。" if n in spec else
                       "  · 未给规格：以上相对声明的程序；公式本身对不对，要给规格才能判断。")

    # causes per source line, fixes per mechanism
    groups = rep["fix_groups"]
    if groups:
        out.append("")
        out.append("原因")
        for gr in groups:
            for r in gr["sites"]:
                contrib = [f"{_show(rep, n)} {_spct(o['contribution'][r['node']])}" for n, o in _in(gr, order)
                           if abs(o["contribution"][r["node"]]) >= 1e-6]
                out.append(f"  - {_site(r)}  `{r['code'][:80]}`")
                out.append(f"      {gr['mechanism']}；贡献：{'，'.join(contrib) or '<0.0001%'}")
        rest = [f"{_show(rep, n)} {_spct(rep['attribution'][n].get('rest', 0.0))}" for n in order if n in sig]
        small = all(abs(rep["attribution"][n].get("rest", 0.0)) < 0.1 * abs(rep["attribution"][n]["before"])
                    for n in sig)
        out.append(f"  - 其他位点合计：{'，'.join(rest)}{'，可忽略' if small else ''}")
        out.append("")
        out.append("建议")
        for i, gr in enumerate(groups, 1):
            sites = gr["sites"]
            out.append(f"  {i}. {SHORT.get(gr['tag'], _short(gr['mechanism']))}（{'、'.join(_site(r) for r in sites)}）")
            for text, cost in gr["fixes"]:
                out.append(f"     - {'各处' if len(sites) > 1 else ''}{text}{'：' + cost if cost else ''}")
            eff = [f"{_show(rep, n)} {_spct(o['before'])} → {_spct(o['after_all'])}" for n, o in _in(gr, order)]
            out.append(f"     预测效果，{'全部 ' + str(len(sites)) + ' 处都改' if len(sites) > 1 else '改这一处'}："
                       f"{'；'.join(eff)}")
            if len(sites) > 1:
                for r in sites:
                    each = [f"{_show(rep, n)} → {_spct(o['after_each'][r['node']])}" for n, o in _in(gr, order)]
                    out.append(f"       只改 {_site(r)}：{'；'.join(each)}")
        progs_a, progs = rep["attribution"][sig[0]]["programs"] if sig else [], rep["scope"]["programs"]
        same = len(progs_a) == len(progs)
        out.append(f"  （预测效果是模型内的反事实：这些位点不再引入误差时的偏差，在{'同一组' if same else f'其中 {len(progs_a)} 个'}"
                   f" program 上算得；修改后再跑一次 audit 即可在 GPU 上核对。）")
    if set(sig) - set(sem):
        out.append("")
        out.append("数值偏差是否值得修，取决于它进入下游（例如训练）后会不会超过随机种子之间的波动；这需要下游测量，工具不判断。")
    if sem:
        out.append("")
        out.append("公式或常数与规格不一致的偏差应当修正（除非规格本身写错）。")
    out.append("")
    out.append(f"范围：{rep['scope']['target']}，{rep['scope']['rule']}。")
    if rep["cannot_decide"]:
        out.append("无法判断：")
        out += [f"  - {c}" for c in rep["cannot_decide"]]

    # appendix
    out.append("")
    out.append("──── 附录 ────")
    if rep["prediction"]:
        p = rep["prediction"]
        out.append(f"【事前预测，kernel 未运行】{'已预测' if p['status'] == 'predicted' else '无法判断'}")
        for n, o in p["outputs"].items():
            out.append(f"  {_show(rep, n)}: 对齐 {o['alignment']:+.3g}（噪声标准误 {o['alignment_noise_se']:.2g}），"
                       f"相对误差（能量加权）{o.get('energy_relative_error', float('nan')):.2g}")
    v = rep["verify"]
    out.append(f"【运行】下降规则逐位核对：{v['status']}")
    for n, m in meas.items():
        verdict = ("有显著的系统偏差" if m.get("significant") else
                   (f"统计上显著，但小于输出格式的单位舍入 {m['format_resolution']:.2g}，可忽略"
                    if m.get("statistically_significant") else "未发现显著偏差"))
        out.append(f"  {_show(rep, n)}: K − G 对齐 {m['alignment']:+.3g}（标准误 {m['se']:.2g}）——{verdict}；"
                   f"相对误差（能量加权）{m['energy_relative']:.2g}，最大 |K − G| {m['max_abs']:.3g}"
                   f"（{m['elements']} / {m['written']} 个元素）")
    for n, s_ in spec.items():
        out.append(f"  {_show(rep, n)}: G − S（语义）{s_['semantic_alignment']:+.3g}，K − S（合计）{s_['total_alignment']:+.3g}")
    for n, checks in rep["tolerance_checks"].items():
        for c in checks:
            out.append(f"  {_show(rep, n)}: {c['check']}（对{c['against']}，atol {c['atol']:g}, rtol {c['rtol']:g}）："
                       f"{'通过' if c['passes'] else '不通过'}，最差比值 {c['worst_ratio']:.2g}；去掉系统分量后"
                       f"{'通过' if c.get('passes_without_bias') else '不通过'}")
    for n, a in rep["attribution"].items():
        out.append(f"【{_show(rep, n)} 的各位点贡献】（{len(a['programs'])} 个 program；仿真偏差 {a['before']:+.3g}，"
                   f"逐点贡献合计 {a['sum']:+.3g}；不为零的前 {len(a['nodes'])} 个）")
        for r in a["nodes"]:
            out.append(f"  {r['contribution']:+.3g}  {r['node']:<22} {r['source']:<20} {_short(r['mechanism'])}")
    for f in rep["findings"]:
        if f["kind"] == "cancel":
            out.append(f"  {_show(rep, f['output'])}：{f['text']}")
    out.append("说明：对齐 = ⟨K − G, G⟩ / ⟨G, G⟩，K 为设备输出，G 为声明程序的精确值，S 为规格；负值即系统性偏小。"
               "贡献 = 仿真输出与「该位点精确」的仿真输出之差的对齐；位点之间有相互作用，相加只是近似。")
    return "\n".join(out)
