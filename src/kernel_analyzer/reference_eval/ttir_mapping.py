"""Version-locked mapping from TTIR operations to internal reference operations.

Every operation registered in the locked Triton build
(``results/reference_eval/ttir_op_registry.json``) has exactly one entry:
SUPPORTED with an internal reference operation and a coverage category, or
REJECTED with a reason.  An operation outside the table is a coverage failure.

Coverage categories (stage summary section 5):

A  non-numerical: addressing, integers, booleans, layout
B  exact real arithmetic
C  exact but discontinuous (comparisons, selection, rounding to integers, min/max)
D  elementary real functions
E  reductions, scans and dot products
F  conversions and declared rounding
G  bit-level reinterpretation
H  external state: memory, atomics, program ids, diagnostics
I  undeclared or composite semantics
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .ttir_parser import TFunc, TModule, TOp

LOCKED_TRITON = "3.6.0"
# canonical copy in the repository; an installed package uses the copy shipped as package data (kept identical,
# tests/test_packaging.py)
_REPO_REGISTRY = Path(__file__).resolve().parents[3] / "results" / "reference_eval" / "ttir_op_registry.json"
_PACKAGED_REGISTRY = Path(__file__).resolve().parent / "data" / "ttir_op_registry.json"
REGISTRY_PATH = _REPO_REGISTRY if _REPO_REGISTRY.exists() else _PACKAGED_REGISTRY


@dataclass(frozen=True)
class Rule:
    status: str  # "SUPPORTED" | "REJECTED"
    category: str
    internal: Optional[str] = None
    reason: str = ""


def S(category: str, internal: str) -> Rule:
    return Rule("SUPPORTED", category, internal)


def R(category: str, reason: str) -> Rule:
    return Rule("REJECTED", category, None, reason)


_ELEMENTARY = {"exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "sqrt", "rsqrt", "sin", "cos",
               "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan", "asinh", "acosh", "atanh", "erf",
               "erfc", "cbrt"}

MAPPING: dict[str, Rule] = {
    # ---- tt -----------------------------------------------------------------
    "tt.addptr": S("A", "addptr"),
    "tt.advance": R("A", "block pointers are not supported in this version"),
    "tt.assert": S("H", "assert"),
    "tt.atomic_cas": S("H", "atomic_cas"),  # DSL v2 increment 3: deterministic without contention; contention -> race
    "tt.atomic_rmw": S("H", "atomic_rmw"),
    # DSL v2 increment 5 (official main): atomic load / store (point or set target), poll (termination premise and
    # happens-before)
    "tt.atomic_load": S("H", "atomic_load"),
    "tt.atomic_store": S("H", "atomic_store"),
    "tt.atomic_poll": S("H", "atomic_poll"),
    "tt.bitcast": S("G", "bitcast"),
    "tt.broadcast": S("A", "broadcast"),
    "tt.call": S("I", "call"),
    "tt.cat": R("A", "tt.cat may reorder elements; the element order is not declared"),
    "tt.clampf": S("C", "clamp"),
    "tt.descriptor_gather": R("H", "tensor descriptors (TMA) are not supported"),
    "tt.descriptor_load": R("H", "tensor descriptors (TMA) are not supported"),
    "tt.descriptor_reduce": R("H", "tensor descriptors (TMA) are not supported"),
    "tt.descriptor_scatter": R("H", "tensor descriptors (TMA) are not supported"),
    "tt.descriptor_store": R("H", "tensor descriptors (TMA) are not supported"),
    "tt.dot": S("E", "dot"),
    "tt.dot_scaled": S("E", "dot_scaled"),  # DSL v2 increment 3: exact MX decoding, exact real dot (rc3 02 6.5)
    "tt.elementwise_inline_asm": S("I", "inline_asm"),
    "tt.expand_dims": S("A", "expand_dims"),
    "tt.extern_elementwise": S("I", "extern"),
    "tt.fp_to_fp": S("F", "cast"),
    "tt.func": S("A", "func"),
    "tt.gather": S("A", "gather"),
    "tt.get_num_programs": S("H", "num_programs"),
    "tt.get_program_id": S("H", "program_id"),
    "tt.histogram": S("E", "histogram"),  # DSL v2 increment 5: exact counts; inputs outside [0, bins) not established
    "tt.int_to_ptr": S("H", "int_to_ptr"),
    "tt.join": S("A", "join"),
    "tt.load": S("H", "load"),
    "tt.make_range": S("A", "make_range"),
    "tt.make_tensor_descriptor": R("H", "tensor descriptors (TMA) are not supported"),
    "tt.make_tensor_ptr": R("A", "block pointers are not supported in this version"),
    "tt.map_elementwise": S("A", "map_elementwise"),  # DSL v2 increment 6: the region per pack group
    "tt.map_elementwise.return": S("A", "map_elementwise_return"),  # terminator, read by _run_map_region
    "tt.mulhiui": S("A", "mulhiui"),
    "tt.precise_divf": S("B", "div"),
    "tt.approx_divf": S("B", "approx_div"),  # rc3 02 6.5: promoted to x / y in the numerical-difference mode
    "tt.precise_sqrt": S("D", "sqrt"),
    "tt.print": S("H", "nop"),
    "tt.ptr_to_int": S("H", "ptr_to_int"),
    "tt.reduce": S("E", "reduce"),
    "tt.reduce.return": S("E", "region_return"),
    "tt.reshape": S("A", "reshape"),
    "tt.return": S("A", "return"),
    "tt.scan": S("E", "scan"),
    "tt.scan.return": S("E", "region_return"),
    "tt.splat": S("A", "splat"),
    "tt.split": S("A", "split"),
    "tt.store": S("H", "store"),
    "tt.trans": S("A", "trans"),
    "tt.unsplat": S("A", "unsplat"),
    # ---- arith --------------------------------------------------------------
    "arith.addf": S("B", "add"),
    "arith.addi": S("A", "addi"),
    "arith.addui_extended": R("A", "extended-precision integer add is not supported"),
    "arith.andi": S("A", "andi"),
    "arith.bitcast": S("G", "bitcast"),
    "arith.ceildivsi": S("A", "ceildivsi"),
    "arith.ceildivui": S("A", "ceildivui"),
    "arith.cmpf": S("C", "cmpf"),
    "arith.cmpi": S("A", "cmpi"),
    "arith.constant": S("A", "constant"),
    "arith.divf": S("B", "div"),
    "arith.divsi": S("A", "divsi"),
    "arith.divui": S("A", "divui"),
    "arith.extf": S("F", "cast"),
    "arith.extsi": S("A", "extsi"),
    "arith.extui": S("A", "extui"),
    "arith.floordivsi": S("A", "floordivsi"),
    "arith.fptosi": S("C", "fptosi"),
    "arith.fptoui": S("C", "fptoui"),
    "arith.index_cast": S("A", "extsi"),
    "arith.index_castui": S("A", "extui"),
    "arith.maximumf": S("C", "maximum"),
    "arith.maxnumf": S("C", "maxnum"),
    "arith.maxsi": S("A", "maxsi"),
    "arith.maxui": S("A", "maxui"),
    "arith.minimumf": S("C", "minimum"),
    "arith.minnumf": S("C", "minnum"),
    "arith.minsi": S("A", "minsi"),
    "arith.minui": S("A", "minui"),
    "arith.mulf": S("B", "mul"),
    "arith.muli": S("A", "muli"),
    "arith.mulsi_extended": R("A", "extended-precision integer multiply is not supported"),
    "arith.mului_extended": R("A", "extended-precision integer multiply is not supported"),
    "arith.negf": S("B", "neg"),
    "arith.ori": S("A", "ori"),
    "arith.remf": S("C", "remf"),
    "arith.remsi": S("A", "remsi"),
    "arith.remui": S("A", "remui"),
    "arith.scaling_extf": R("F", "microscaling conversions are not supported"),
    "arith.scaling_truncf": R("F", "microscaling conversions are not supported"),
    "arith.select": S("C", "select"),
    "arith.shli": S("A", "shli"),
    "arith.shrsi": S("A", "shrsi"),
    "arith.shrui": S("A", "shrui"),
    "arith.sitofp": S("F", "sitofp"),
    "arith.subf": S("B", "sub"),
    "arith.subi": S("A", "subi"),
    "arith.truncf": S("F", "cast"),
    "arith.trunci": S("A", "trunci"),
    "arith.uitofp": S("F", "uitofp"),
    "arith.xori": S("A", "xori"),
    # ---- math ---------------------------------------------------------------
    "math.absf": S("B", "abs"),
    "math.absi": S("A", "absi"),
    "math.atan2": R("D", "atan2 is not supported in this version"),
    "math.ceil": S("C", "ceil"),
    "math.clampf": S("C", "math_clampf"),  # DSL v2 increment 8: NaN or min > max (poison) not established
    "math.copysign": S("C", "copysign"),
    "math.ctlz": R("G", "bit counting is not supported in this version"),
    "math.ctpop": R("G", "bit counting is not supported in this version"),
    "math.cttz": R("G", "bit counting is not supported in this version"),
    "math.floor": S("C", "floor"),
    "math.fma": S("B", "fma"),
    "math.fpowi": R("D", "fpowi is not supported in this version"),
    "math.ipowi": R("A", "ipowi is not supported in this version"),
    "math.isfinite": S("C", "isfinite"),
    "math.isinf": S("C", "isinf"),
    "math.isnan": S("C", "isnan"),
    "math.isnormal": R("C", "isnormal depends on the storage format; not supported"),
    "math.powf": S("D", "pow"),
    "math.round": S("C", "round"),
    "math.roundeven": S("C", "roundeven"),
    "math.trunc": S("C", "trunc"),
    # ---- scf / cf / ub -------------------------------------------------------
    "scf.condition": S("A", "condition"),
    "scf.execute_region": R("A", "execute_region is not produced by the Triton frontend"),
    "scf.for": S("A", "for"),
    "scf.forall": R("A", "not produced by the Triton frontend"),
    "scf.forall.in_parallel": R("A", "not produced by the Triton frontend"),
    "scf.if": S("A", "if"),
    "scf.index_switch": R("A", "index_switch is not supported in this version"),
    "scf.parallel": R("A", "not produced by the Triton frontend"),
    "scf.reduce": R("A", "not produced by the Triton frontend"),
    "scf.reduce.return": R("A", "not produced by the Triton frontend"),
    "scf.while": S("A", "while"),
    "scf.yield": S("A", "yield"),
    "cf.assert": S("H", "assert"),
    "cf.br": S("A", "br"),
    "cf.cond_br": S("A", "cond_br"),
    "cf.switch": R("A", "cf.switch is not supported in this version"),
    "ub.poison": S("A", "poison"),
    "gpu.barrier": S("H", "barrier"),  # orders the threads of one program (block): starts a new access epoch
    "ttg.barrier": S("H", "barrier"),  # official main emits this for tl.debug_barrier (DSL v2 increment 3)
    "llvm.intr.assume": S("A", "assume"),  # rc3 02 12: a per-sample premise checked on the reference path
}
for _name in _ELEMENTARY:
    MAPPING[f"math.{_name}"] = S("D", _name)


def _gpu_rule(name: str) -> Rule:
    return R("H", "gpu dialect operation not produced at the TTIR stage")


# libdevice symbols -> internal operation (declared mathematical function).
# The fast/approximate variants declare the same mathematical function; the
# approximation is part of K and appears in K - K_R.
LIBDEVICE = {}
for _fn in _ELEMENTARY | {"floor", "ceil", "trunc", "round", "abs", "fabs"}:
    base = "abs" if _fn in ("abs", "fabs") else _fn
    for _sym in (f"__nv_{_fn}f", f"__nv_fast_{_fn}f", f"__nv_{_fn}"):
        LIBDEVICE[_sym] = base
LIBDEVICE.update({
    "__nv_powf": "pow", "__nv_fast_powf": "pow", "__nv_pow": "pow",
    "__nv_fmaxf": "maxnum", "__nv_fminf": "minnum", "__nv_fmax": "maxnum", "__nv_fmin": "minnum",
    "__nv_fmaf": "fma", "__nv_fma": "fma", "__nv_fdividef": "div", "__nv_fdiv_rn": "div",
    "__nv_div_rn": "div", "__nv_frcp_rn": "rcp", "__nv_rcp64h": "rcp", "__nv_fsqrt_rn": "sqrt",
    "__nv_dsqrt_rn": "sqrt", "__nv_isnanf": "isnan", "__nv_isnand": "isnan", "__nv_isinff": "isinf",
    "__nv_isinfd": "isinf", "__nv_finitef": "isfinite", "__nv_isfinited": "isfinite",
    "__nv_signbitf": "signbit", "__nv_copysignf": "copysign", "__nv_rintf": "roundeven",
    "__nv_nearbyintf": "roundeven", "__nv_exp10f": "exp10", "__nv_fast_exp10f": "exp10",
    "__nv_fmodf": "remf", "__nv_saturatef": "saturate", "__nv_fadd_rn": "add", "__nv_fmul_rn": "mul",
    "__nv_fsub_rn": "sub", "__nv_dadd_rn": "add", "__nv_dmul_rn": "mul", "__nv_fmaf_rn": "fma",
    # DSL v2 increment 6: Bessel functions, enclosed by Arb ball arithmetic (intervals.bessel_bounds)
    "__nv_j0f": "bessel_j0", "__nv_j0": "bessel_j0", "__nv_j1f": "bessel_j1", "__nv_j1": "bessel_j1",
    "__nv_y0f": "bessel_y0", "__nv_y0": "bessel_y0", "__nv_y1f": "bessel_y1", "__nv_y1": "bessel_y1",
    "__nv_cyl_bessel_i0f": "bessel_i0", "__nv_cyl_bessel_i0": "bessel_i0",
    "__nv_cyl_bessel_i1f": "bessel_i1", "__nv_cyl_bessel_i1": "bessel_i1",
})

# Rounding-suffixed libdevice operations: the declared semantics is the real operation (the
# suffix is the implementation's rounding, used only in rounding-check mode).
LIBDEVICE_ROUNDING = {}
for _base, _internal in (("fadd", "add"), ("fsub", "sub"), ("fmul", "mul"), ("fdiv", "div"), ("fmaf", "fma"),
                         ("fsqrt", "sqrt"), ("frcp", "rcp"), ("dadd", "add"), ("dsub", "sub"), ("dmul", "mul"),
                         ("ddiv", "div"), ("fma", "fma"), ("dsqrt", "sqrt"), ("drcp", "rcp")):
    for _suffix, _mode in (("rn", "rtne"), ("rz", "rtz"), ("rd", "rd"), ("ru", "ru")):
        LIBDEVICE[f"__nv_{_base}_{_suffix}"] = _internal
        LIBDEVICE_ROUNDING[f"__nv_{_base}_{_suffix}"] = _mode

# Inline PTX templates with one declared mathematical operation.  "approx"
# instructions declare the real function; the approximation lands in K - K_R.
INLINE_ASM = {
    r"ex2\.approx(\.ftz)?\.f32 \$0, \$1;": "exp2",
    r"lg2\.approx(\.ftz)?\.f32 \$0, \$1;": "log2",
    r"rsqrt\.approx(\.ftz)?\.f32 \$0, \$1;": "rsqrt",
    r"sqrt\.approx(\.ftz)?\.f32 \$0, \$1;": "sqrt",
    r"sqrt\.rn(\.ftz)?\.f32 \$0, \$1;": "sqrt",
    r"rcp\.approx(\.ftz)?\.f32 \$0, \$1;": "rcp",
    r"rcp\.rn(\.ftz)?\.f32 \$0, \$1;": "rcp",
    r"tanh\.approx\.f32 \$0, \$1;": "tanh",
    r"sin\.approx(\.ftz)?\.f32 \$0, \$1;": "sin",
    r"cos\.approx(\.ftz)?\.f32 \$0, \$1;": "cos",
    r"mov\.b32 \$0, \$1;": "identity",
    r"div\.full(\.ftz)?\.f32 \$0, \$1, \$2;": "approx_div",  # DSL v2 increment 5 (rc3 02 6.5)
}


# Arithmetic with an explicit rounding / flush modifier: the declared semantics is the real operation
# (numerical-difference mode); the modifier describes the executed instruction and is used only for the
# rounding in rounding-check mode (ftz is not modelled there).
_INLINE_ARITH = re.compile(r"(add|sub|mul)\.(rn|rz|rm|rp)(\.ftz)?\.f32 \$0, \$1, \$2;")
_INLINE_FMA = re.compile(r"fma\.(rn|rz|rm|rp)(\.ftz)?\.f32 \$0, \$1, \$2, \$3;")


def _normalize_asm(asm: str) -> str:
    return " ".join(asm.replace("\\n", " ").split())


def inline_asm_internal(asm: str) -> Optional[str]:
    text = _normalize_asm(asm)
    for pattern, internal in INLINE_ASM.items():
        if re.fullmatch(pattern, text):
            return internal
    m = _INLINE_ARITH.fullmatch(text)
    if m:
        return m.group(1)
    if _INLINE_FMA.fullmatch(text):
        return "fma"
    return None


def inline_asm_rounding(asm: str) -> Optional[str]:
    """The rounding modifier of an inline arithmetic instruction (rn / rz / rm / rp), if any."""

    text = _normalize_asm(asm)
    m = _INLINE_ARITH.fullmatch(text)
    if m:
        return m.group(2)
    m = _INLINE_FMA.fullmatch(text)
    return m.group(1) if m else None


def rule_for(name: str) -> Optional[Rule]:
    if name in MAPPING:
        return MAPPING[name]
    if name.startswith("gpu."):
        return _gpu_rule(name)
    return None


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(Path(path).read_text())


def registry_coverage(registry: Optional[dict] = None) -> dict:
    """Every registered operation is SUPPORTED or REJECTED (enumeration test)."""

    registry = registry or load_registry()
    missing, supported, rejected = [], [], []
    for dialect, names in registry["operations"].items():
        for name in names:
            rule = rule_for(name)
            if rule is None:
                missing.append(name)
            elif rule.status == "SUPPORTED":
                supported.append(name)
            else:
                rejected.append(name)
    known = {x for v in registry["operations"].values() for x in v}
    other = sorted(n for n in MAPPING if n not in known and n in OTHER_PROFILE_NAMES)
    stale = sorted(n for n in MAPPING if n not in known and n not in OTHER_PROFILE_NAMES)
    return {"triton": registry["triton"], "missing": sorted(missing), "supported": len(supported),
            "rejected": len(rejected), "stale_table_entries": stale, "other_profile_entries": other,
            "complete": not missing}


# Table entries for names that the locked 3.6.0 build does not register but another recorded profile does (DSL v2
# increment 3; registration evidence: results/dsl_v2/w0/registered_e50b186e8bd2.json, official main e50b186e).
OTHER_PROFILE_NAMES = {"ttg.barrier": "e50b186e", "llvm.intr.assume": "e50b186e", "tt.atomic_load": "e50b186e",
                       "tt.atomic_store": "e50b186e", "tt.atomic_poll": "e50b186e", "tt.approx_divf": "e50b186e"}


# ---------------------------------------------------------------------------
# Reduction combiners
# ---------------------------------------------------------------------------

_SIMPLE_COMBINERS = {
    "arith.addf": "sum", "arith.mulf": "prod", "arith.addi": "sum_int", "arith.muli": "prod_int",
    "arith.maxnumf": "max_num", "arith.minnumf": "min_num", "arith.maximumf": "max_nan",
    "arith.minimumf": "min_nan", "arith.maxsi": "max_int", "arith.minsi": "min_int",
    "arith.maxui": "max_uint", "arith.minui": "min_uint", "arith.andi": "and", "arith.ori": "or",
    "arith.xori": "xor",
}


def recognize_combiner(op: TOp) -> Optional[str]:
    """Name of a recognized associative-commutative combiner, else None.

    Only combiners whose real-number semantics is independent of the
    combination tree are recognized; any other region leaves the reduction's
    reference not established.
    """

    region = op.regions[0]
    block = region.entry
    args = [a[0] for a in block.args]
    ops = [o for o in block.ops if o.name not in ("tt.reduce.return", "tt.scan.return")]
    ret = block.ops[-1]
    k = len(args) // 2
    if k == 1 and len(ops) == 1:
        inner = ops[0]
        if inner.name in _SIMPLE_COMBINERS and sorted(inner.operands) == sorted(args) \
                and ret.operands == inner.results:
            return _SIMPLE_COMBINERS[inner.name]
    if k == 1:
        kind = _match_select_minmax(ops, args, ret)
        if kind:
            return kind
    if k == 2:
        kind = _match_arg_minmax(ops, args, ret)
        if kind:
            return kind
    if k == 3 and match_welford(op) is not None:
        return "welford"
    return None


def _leaves(defs, v, opname, used):
    """Leaves of the tree of binary ``opname`` ops rooted at ``v`` inside the region; ops visited go to ``used``."""

    o = defs.get(v)
    if o is not None and o.name == opname and len(o.operands) == 2:
        used.add(id(o))
        return _leaves(defs, o.operands[0], opname, used) + _leaves(defs, o.operands[1], opname, used)
    return [v]


def match_welford(op: TOp) -> Optional[dict]:
    """The Welford merge of (mean, M2, weight) triples, as Inductor's ``welford_combine`` writes it::

        delta = mean_b - mean_a;  W = w_a + w_b;  r = select(W == 0, 0, w_b / W)   (or r = w_b / W)
        mean = mean_a + delta * r;  M2 = M2_a + M2_b + delta * delta * w_a * r;  weight = W

    Matched on the dataflow (sums and products as multisets, every op of the region accounted for), never on a
    kernel or function name.  Returns {"zero_constants": [...]} -- outer values that must be exactly 0.0 at
    evaluation time -- or None."""

    region = op.regions[0]
    block = region.entry
    args = [a[0] for a in block.args]
    if len(args) != 6:
        return None
    m_a, s_a, w_a, m_b, s_b, w_b = args
    ops = [o for o in block.ops if o.name not in ("tt.reduce.return", "tt.scan.return")]
    ret = block.ops[-1]
    if len(ret.operands) != 3:
        return None
    defs = _defs(ops)
    used = set()
    mean_v, m2_v, w_v = ret.operands
    wd = defs.get(w_v)
    if wd is None or wd.name != "arith.addf" or sorted(wd.operands) != sorted([w_a, w_b]):
        return None
    used.add(id(wd))
    mean_leaves = _leaves(defs, mean_v, "arith.addf", used)
    if len(mean_leaves) != 2 or m_a not in mean_leaves:
        return None
    prod = [v for v in mean_leaves if v != m_a][0]
    prod_leaves = _leaves(defs, prod, "arith.mulf", used)
    if len(prod_leaves) != 2:
        return None
    zeros = []

    def is_delta(v):
        d = defs.get(v)
        if d is not None and d.name == "arith.subf" and d.operands == [m_b, m_a]:
            used.add(id(d))
            return True
        return False

    def is_ratio(v):
        d = defs.get(v)
        if d is None:
            return False
        if d.name == "arith.divf" and d.operands == [w_b, w_v]:
            used.add(id(d))
            return True
        if d.name == "arith.select" and len(d.operands) == 3:
            c, z, q = d.operands
            cd, qd = defs.get(c), defs.get(q)
            if cd is None or qd is None or z in defs or z in args:
                return False
            if cd.name != "arith.cmpf" or cd.attrs.get("predicate") != "oeq" or w_v not in cd.operands:
                return False
            z2 = [x for x in cd.operands if x != w_v]
            if len(z2) != 1 or z2[0] in defs or z2[0] in args:
                return False
            if qd.name != "arith.divf" or qd.operands != [w_b, w_v]:
                return False
            used.update({id(d), id(cd), id(qd)})
            zeros.extend([z, z2[0]])
            return True
        return False

    a, b = prod_leaves
    if is_delta(a) and is_ratio(b):
        delta, ratio = a, b
    elif is_delta(b) and is_ratio(a):
        delta, ratio = b, a
    else:
        return None
    m2_leaves = _leaves(defs, m2_v, "arith.addf", used)
    if len(m2_leaves) != 3 or sorted(v for v in m2_leaves if v in (s_a, s_b)) != sorted([s_a, s_b]):
        return None
    q = [v for v in m2_leaves if v not in (s_a, s_b)]
    if len(q) != 1:
        return None
    q_leaves = _leaves(defs, q[0], "arith.mulf", used)
    if sorted(q_leaves) != sorted([delta, delta, w_a, ratio]):
        return None
    if {id(o) for o in ops} != used:                 # an op the pattern does not account for
        return None
    return {"zero_constants": zeros}


def _defs(ops):
    return {r: o for o in ops for r in o.results}


def _match_select_minmax(ops, args, ret) -> Optional[str]:
    """select(a > b [or a != a], a, b) style max/min (TorchInductor helpers)."""

    a, b = args
    defs = _defs(ops)
    if len(ret.operands) != 1 or ret.operands[0] not in defs:
        return None
    sel = defs[ret.operands[0]]
    if sel.name != "arith.select":
        return None
    cond, x, y = sel.operands
    if (x, y) not in ((a, b), (b, a)):
        return None
    cmp = defs.get(cond)
    nan_aware = False
    if cmp is not None and cmp.name == "arith.ori":
        parts = [defs.get(v) for v in cmp.operands]
        nan_parts = [p for p in parts if p is not None and p.name == "arith.cmpf"
                     and p.attrs.get("predicate") == "une" and p.operands[0] == p.operands[1] == x]
        others = [p for p in parts if p not in nan_parts]
        if len(nan_parts) == 1 and len(others) == 1:
            nan_aware, cmp = True, others[0]
    if cmp is None or cmp.name != "arith.cmpf" or len(ops) > (5 if nan_aware else 2):
        return None
    pred = cmp.attrs.get("predicate")
    l, r = cmp.operands
    if (l, r) == (x, y) and pred in ("ogt", "oge"):
        base = "max"
    elif (l, r) == (x, y) and pred in ("olt", "ole"):
        base = "min"
    elif (l, r) == (y, x) and pred in ("olt", "ole"):
        base = "max"
    elif (l, r) == (y, x) and pred in ("ogt", "oge"):
        base = "min"
    else:
        return None
    return f"{base}_nan" if nan_aware else f"{base}_select"


def _match_arg_minmax(ops, args, ret) -> Optional[str]:
    """Triton's argmax/argmin combiner: value compare with index tie-break."""

    va, ia, vb, ib = args
    defs = _defs(ops)
    if len(ret.operands) != 2:
        return None
    v_sel, i_sel = (defs.get(v) for v in ret.operands)
    if v_sel is None or i_sel is None or v_sel.name != "arith.select" or i_sel.name != "arith.select":
        return None
    if v_sel.operands[0] != i_sel.operands[0] or v_sel.operands[1:] != [va, vb] or i_sel.operands[1:] != [ia, ib]:
        return None
    cond = defs.get(v_sel.operands[0])
    if cond is None or cond.name != "arith.ori":
        return None
    preds = {}
    for v in cond.operands:
        d = defs.get(v)
        if d is None:
            return None
        if d.name == "arith.cmpf" and d.operands == [va, vb]:
            preds["value"] = d.attrs["predicate"]
        elif d.name == "arith.andi":
            sub = [defs.get(x) for x in d.operands]
            for s in sub:
                if s is None:
                    return None
                if s.name == "arith.cmpf" and s.operands == [va, vb]:
                    preds["tie"] = s.attrs["predicate"]
                elif s.name == "arith.cmpi" and s.operands == [ia, ib]:
                    preds["index"] = s.attrs["predicate"]
    if preds.get("tie") != "oeq" or preds.get("index") != "slt":
        return None
    if preds.get("value") == "ogt":
        return "argmax"
    if preds.get("value") == "olt":
        return "argmin"
    return None


# ---------------------------------------------------------------------------
# Static coverage report for one kernel
# ---------------------------------------------------------------------------


def kernel_coverage(module: TModule, func: Optional[TFunc] = None) -> dict:
    """Instruction-processing coverage of one kernel: each op accepted or rejected."""

    func = func or module.entry()
    rows, rejected = [], []
    bitcast_sources = set()
    for op in _walk_with_called(module, func):
        rule = rule_for(op.name)
        row = {"node": op.node_id, "op": op.name, "category": rule.category if rule else None}
        reason = None
        if rule is None:
            reason = "operation not in the mapping table"
        elif rule.status == "REJECTED":
            reason = rule.reason
        elif op.name == "tt.extern_elementwise":
            symbol = op.attrs.get("symbol", "").strip('"')
            if symbol not in LIBDEVICE:
                reason = f"libdevice symbol {symbol} has no declared semantics"
            row["internal"] = LIBDEVICE.get(symbol)
        elif op.name == "tt.elementwise_inline_asm":
            internal = inline_asm_internal(op.attrs.get("asm", ""))
            packed = op.attrs.get("packed_element", "1 : i32").split(":")[0].strip()
            if internal is None:
                from .ptx_bits import classify
                from .ttir_eval import parse_ptx_program  # straight-line PTX snippets have lane-wise semantics

                if parse_ptx_program(op.attrs.get("asm", "")) is not None and packed == "1" and len(op.results) == 1:
                    internal = "ptx_program"
                else:
                    why = classify(op.attrs.get("asm", ""))   # DSL v2 increment 9: bit-level PTX interpreter
                    if why is None:
                        internal = "ptx_bits"
                    else:
                        reason = f"inline asm {op.attrs.get('asm')!r}: {why}"
            elif packed != "1" or len(op.results) != 1:
                reason = "packed or multi-result inline asm is not supported"
            row["internal"] = internal
        elif op.name in ("tt.reduce", "tt.scan"):
            combiner = recognize_combiner(op)
            row["combiner"] = combiner
            if combiner is None:
                # DSL v2 increment 1: evaluated along the TTGIR lowering order (declared premise); without a TTGIR
                # the evaluator reports it not established
                row["combiner"] = "generic: TTGIR lowering order (declared premise)"
        elif op.name == "tt.reshape" and "allow_reorder" in op.attrs:
            row["note"] = "allow_reorder: element order is not declared; only order-insensitive uses are valid"
        if op.name in ("tt.bitcast", "arith.bitcast"):
            bitcast_sources.update(op.results)
        if any(v in bitcast_sources for v in op.operands):
            row["bit_level_source"] = True
            bitcast_sources.update(op.results)
        if reason:
            row["rejected"] = reason
            rejected.append(row)
        rows.append(row)
    categories = {}
    for row in rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    return {"kernel": func.name, "triton": LOCKED_TRITON, "operations": len(rows),
            "categories": categories, "rejected": rejected, "complete": not rejected, "rows": rows}


def _walk_with_called(module: TModule, func: TFunc, seen=None):
    seen = seen or set()
    seen.add(func.name)
    for op in func.walk():
        yield op
        if op.name == "tt.call":
            callee = op.attrs.get("callee")
            if callee in module.funcs and callee not in seen:
                yield from _walk_with_called(module, module.funcs[callee], seen)
