"""Bitwise emulation of a captured launch's FP32 execution, and in-kernel localization by shadow execution.

Values inside a kernel are not observable on the device, so numerical evidence
alone localizes only to the kernel.  This module emulates the device execution
from the captured TTIR plus the lowering rules of the locked Triton build
(3.6.0, sm_86), checks that the emulated outputs equal the device outputs bit
for bit, and then makes one node exact at a time; the change of the outputs is
that node's contribution.

Lowering rules (each one is checked only through the bitwise comparison):

* IEEE operations (add, sub, mul, fma, precise sqrt / div, conversions,
  rounding-suffixed libdevice calls) round once per operation in the declared
  mode, as in rounding-check mode;
* contraction: with ``enable_fp_fusion`` an ``arith.addf`` / ``arith.subf`` with
  an ``arith.mulf`` operand defined in the same block is one fma (LLVM's
  aggressive fusion on NVPTX: every such add is fused; when both operands are
  products, the one with fewer uses);
* approximate instructions (``arith.divf`` = div.full, ``math.exp`` / ``exp2`` /
  ``log`` / ``sqrt`` / ``rsqrt`` / ..., libdevice functions, inline asm) are
  evaluated by running the same TTIR operation on the device with the emulated
  operands (a hardware oracle; the instruction is deterministic, so equal
  operands give equal results);
* reductions follow the layout in the captured TTGIR: sequential within a
  thread in register order, butterfly over the lanes along the axis, butterfly
  over the warps along the axis;
* an FP32 ``tt.dot`` with input precision ieee is a sequential fma chain over k.

Not emulable (localization stops there; the affected elements are reported):
tensor-core dots (tf32, bf16, f16), scans, atomics.

"Node exact" replaces every execution of one TTIR operation by its exact
real result from its emulated operands; operations downstream keep their
device behaviour (IEEE operations round the exact input; an approximate
operation keeps its own approximation error: hw(RN(x)) + f(x) - f(RN(x))).
"""

from __future__ import annotations

import collections
import hashlib
import re
from pathlib import Path
from typing import Optional

import numpy as np

from . import intervals as iv
from .ttir_eval import (ST_NE, ST_OK, ST_UNDEF, TV, KernelReferenceEvaluator, NumericMode, ProgramAbort, _ftv,
                        _merge_cond, _merge_reasons, _merge_status)
from .ttir_mapping import LIBDEVICE, LIBDEVICE_ROUNDING, inline_asm_internal, recognize_combiner
from .ttir_parser import TModule, TOp

ORACLE_OPS = {"arith.divf", "arith.remf", "math.exp", "math.exp2", "math.log", "math.log2", "math.log10",
              "math.log1p", "math.expm1", "math.sqrt", "math.rsqrt", "math.sin", "math.cos", "math.tan",
              "math.tanh", "math.erf", "math.powf", "math.atan", "math.sinh", "math.cosh",
              "tt.extern_elementwise", "tt.elementwise_inline_asm"}
_TORCH = {"f32": "float32", "f16": "float16", "bf16": "bfloat16", "f64": "float64", "i32": "int32", "i64": "int64",
          "i8": "int8", "i16": "int16", "i1": "bool"}
_TT_TYPE = {"f32": "f32", "f16": "f16", "bf16": "bf16", "f64": "f64", "i32": "i32", "i64": "i64", "i8": "i8",
            "i16": "i16", "i1": "i1"}
BLOCK = 1024
LAYOUT_OPS = {"tt.splat", "tt.broadcast", "tt.expand_dims", "ttg.convert_layout", "tt.reshape"}
SUBSTITUTIONS = ("mid", "lo", "hi", "rn")
_BITS = {"f32": (np.float32, np.uint32), "f16": (np.float16, np.uint16), "f64": (np.float64, np.uint64)}


# ---------------------------------------------------------------------------
# Hardware oracle for approximate instructions
# ---------------------------------------------------------------------------


class HardwareOracle:
    """Runs one TTIR elementwise operation on the device, compiled by the installed Triton."""

    def __init__(self, cache_dir: Optional[Path] = None, options: Optional[dict] = None, device: str = "cuda"):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[4] / ".cache" / "oracle"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.options = {"num_warps": 4, **(options or {})}
        self.device = device
        self._kernels = {}
        self.calls = 0

    @staticmethod
    def rewrite(op: TOp, names: dict, result: str) -> str:
        """The operation as a 1-D tensor operation; operands renamed by ``names``, result named ``result``."""

        text = re.sub(r"\s+loc\([^()]*(\([^()]*\))?[^()]*\)\s*$", "", op.text.strip())
        text = re.sub(r"^%[\w#]+(:\d+)?\s*=\s*", "", text)  # the result is renamed after the operands
        for name in sorted(set(op.operands), key=len, reverse=True):
            text = re.sub(re.escape(name) + r"(?![\w#.$-])", "%__" + names[name][1:], text)
        text = text.replace("%__", "%")
        text = re.sub(r"tensor<(?:\d+x)+", f"tensor<{BLOCK}x", text)
        head, sep, types = text.rpartition(" : ")
        if sep:
            types = re.sub(r"(?<![\w<])(f16|bf16|f32|f64|i1|i8|i16|i32|i64)(?![\w>])",
                           lambda m: f"tensor<{BLOCK}x{m.group(1)}>", types)
            text = head + sep + types
        return f"{result} = {text}"

    @staticmethod
    def operand_elems(op: TOp) -> list:
        elems = [t.elem for t in op.operand_types] if op.operand_types else []
        if len(elems) != len(op.operands):
            elems = [elems[0]] * len(op.operands) if elems else []
        return elems

    def _module(self, line: str, in_elems: list, out_elem: str) -> str:
        args = [f"%in{i}: !tt.ptr<{_TT_TYPE[e]}> {{tt.divisibility = 16 : i32}}" for i, e in enumerate(in_elems)]
        args.append(f"%out: !tt.ptr<{_TT_TYPE[out_elem]}> {{tt.divisibility = 16 : i32}}")
        args.append("%n: i32")
        body = [
            "%pid = tt.get_program_id x : i32",
            f"%cb = arith.constant {BLOCK} : i32",
            "%base = arith.muli %pid, %cb : i32",
            f"%range = tt.make_range {{end = {BLOCK} : i32, start = 0 : i32}} : tensor<{BLOCK}xi32>",
            f"%bases = tt.splat %base : i32 -> tensor<{BLOCK}xi32>",
            f"%offs = arith.addi %bases, %range : tensor<{BLOCK}xi32>",
            f"%ns = tt.splat %n : i32 -> tensor<{BLOCK}xi32>",
            f"%mask = arith.cmpi slt, %offs, %ns : tensor<{BLOCK}xi32>",
        ]
        for i, e in enumerate(in_elems):
            t = _TT_TYPE[e]
            body += [f"%p{i} = tt.splat %in{i} : !tt.ptr<{t}> -> tensor<{BLOCK}x!tt.ptr<{t}>>",
                     f"%a{i} = tt.addptr %p{i}, %offs : tensor<{BLOCK}x!tt.ptr<{t}>>, tensor<{BLOCK}xi32>",
                     f"%x{i} = tt.load %a{i}, %mask : tensor<{BLOCK}x!tt.ptr<{t}>>"]
        t = _TT_TYPE[out_elem]
        body += [line,
                 f"%po = tt.splat %out : !tt.ptr<{t}> -> tensor<{BLOCK}x!tt.ptr<{t}>>",
                 f"%ao = tt.addptr %po, %offs : tensor<{BLOCK}x!tt.ptr<{t}>>, tensor<{BLOCK}xi32>",
                 f"tt.store %ao, %y, %mask : tensor<{BLOCK}x!tt.ptr<{t}>>",
                 "tt.return"]
        return ("module {\n  tt.func public @oracle(" + ", ".join(args) + ") attributes {noinline = false} {\n    "
                + "\n    ".join(body) + "\n  }\n}\n")

    def __call__(self, op: TOp, inputs: list, out_elem: str) -> np.ndarray:
        """One operation; ``inputs`` per operand position."""

        return self.group([op], {}, dict(zip(op.operands, inputs)), out_elem)

    def group(self, ops: list, alias: dict, inputs: dict, out_elem: str) -> np.ndarray:
        """Several operations compiled together (the last one is the output), so that contraction
        between them happens as in the kernel.  ``alias`` maps a layout-only value (splat, broadcast)
        to the value it repeats; ``inputs`` maps every external operand name to its values."""

        import torch
        import triton
        from triton.backends.compiler import GPUTarget

        names, ext, elems_of, lines = {}, [], {}, []
        for k, op in enumerate(ops):
            for name, elem in zip(op.operands, self.operand_elems(op) or [out_elem] * len(op.operands)):
                src = alias.get(name, name)
                if src not in names:
                    names[src] = f"%x{len(ext)}"
                    ext.append(src)
                    elems_of[src] = elem
                names[name] = names[src]
            res = "%y" if k == len(ops) - 1 else f"%t{k}"
            lines.append(self.rewrite(op, names, res))
            for r in op.results:
                names[r] = res
        elems = [elems_of[n] for n in ext]
        key = (tuple(lines), tuple(elems), out_elem, tuple(sorted(self.options.items())))
        kernel = self._kernels.get(key)
        if kernel is None:
            text = self._module("\n    ".join(lines), elems, out_elem)
            path = self.cache_dir / f"oracle_{hashlib.sha256(repr(key).encode()).hexdigest()[:16]}.ttir"
            path.write_text(text)
            kernel = triton.compile(str(path), target=GPUTarget("cuda", 86, 32), options=self.options)
            self._kernels[key] = kernel
        arrays = [np.asarray(inputs[n]) for n in ext]
        shape = np.broadcast_shapes(*(a.shape for a in arrays))
        n = int(np.prod(shape)) if shape else 1
        tensors = [torch.as_tensor(np.broadcast_to(a, shape).reshape(-1).copy(), device=self.device)
                   .to(getattr(torch, _TORCH[e])) for a, e in zip(arrays, elems)]
        out = torch.empty(max(n, 1), device=self.device, dtype=getattr(torch, _TORCH[out_elem]))
        kernel[(triton.cdiv(max(n, 1), BLOCK), 1, 1)](*tensors, out, n)
        self.calls += 1
        return out[:n].double().cpu().numpy().reshape(shape)


# ---------------------------------------------------------------------------
# TTGIR reduction layouts and the contraction plan
# ---------------------------------------------------------------------------

from .layouts import _axis_params, parse_layouts, reduce_layouts  # noqa: E402,F401  (moved; names kept)


def contraction_plan(module: TModule) -> dict:
    """node_id of each fused add/sub -> (index of the product operand, the mulf op, alternative or None).

    When both operands are products, LLVM folds the one with fewer uses (the left one on a tie); the
    use counts it sees are those of the selection DAG, which the TTIR does not always predict, so the
    other product is kept as the alternative."""

    plan = {}
    for fn in module.funcs.values():
        blocks = []

        def collect(region):
            for block in region.blocks:
                blocks.append(block)
                for op in block.ops:
                    for r in op.regions:
                        collect(r)
        collect(fn.body)
        uses = collections.Counter(v for op in fn.walk() for v in op.operands)
        for block in blocks:
            defs = {}

            def product(v):  # the mulf behind v, looking through layout-only ops (a splat is the same register)
                op = defs.get(v)
                while op is not None and op.name in LAYOUT_OPS:
                    v = op.operands[0]
                    op = defs.get(v)
                return (op, v) if op is not None and op.name == "arith.mulf" else (None, None)

            for op in block.ops:
                if op.name in ("arith.addf", "arith.subf") and len(op.operands) == 2:
                    found = [product(v) for v in op.operands]
                    cands = [i for i, (m, _) in enumerate(found) if m is not None]
                    alt = None
                    if len(cands) == 2:
                        u0, u1 = uses[found[0][1]], uses[found[1][1]]
                        cands = [1] if u0 > u1 else [0]
                        alt = (1 - cands[0], found[1 - cands[0]][0])
                    if cands:
                        plan[op.node_id] = (cands[0], found[cands[0]][0], alt)
                for r in op.results:
                    defs[r] = op
    return plan


# ---------------------------------------------------------------------------
# Evidence from the compiled PTX (shared source locations)
# ---------------------------------------------------------------------------


def ttir_locations(text: str) -> dict:
    """#locN -> (file basename, line, column), resolving named and call-site locations."""

    raw = dict(re.findall(r"^(#loc\d*)\s*=\s*loc\((.*)\)\s*$", text, re.M))
    cache = {}

    def resolve(ref, depth=0):
        if ref in cache or depth > 50:
            return cache.get(ref)
        body = raw.get(ref)
        out = None
        if body is not None:
            m = re.match(r'^"([^"]*)":(\d+):(\d+)$', body)
            if m:
                out = (Path(m.group(1)).name, int(m.group(2)), int(m.group(3)))
            else:
                inner = re.findall(r"#loc\d*", body)
                if body.startswith("callsite(") and inner:
                    out = resolve(inner[0], depth + 1)  # the callee location
                elif inner:
                    out = resolve(inner[-1], depth + 1)
        cache[ref] = out
        return out

    return {ref: resolve(ref) for ref in raw}


def op_location(op: TOp, locations: dict, defining_lines: Optional[dict] = None):
    """Source location of an op, from its raw TTIR line (the parser strips loc(...) from op.text);
    ``defining_lines`` maps an SSA result name to the raw line that defines it."""

    candidates = [op.text]
    if defining_lines is not None and op.results and op.results[0] in defining_lines:
        candidates = [defining_lines[op.results[0]]] + candidates
    for text in candidates:
        m = re.search(r"loc\((#loc\d*)\)\s*$", text.strip())
        if m:
            return locations.get(m.group(1))
        m = re.search(r'loc\("([^"]*)":(\d+):(\d+)\)\s*$', text.strip())
        if m:
            return (Path(m.group(1)).name, int(m.group(2)), int(m.group(3)))
    return None


def ptx_instructions_by_location(ptx: str) -> dict:
    """(file basename, line, column) -> opcodes of the PTX instructions emitted for it."""

    files = {m.group(1): Path(m.group(2)).name
             for m in re.finditer(r'^\s*\.file\s+(\d+)\s+"([^"]*)"', ptx, re.M)}  # listed after the code
    out, current = collections.defaultdict(list), None
    for line in ptx.splitlines():
        s = line.split("//")[0].strip()
        if s.startswith(".file"):
            continue
        m = re.match(r"^\.loc\s+(\d+)\s+(\d+)\s+(\d+)", s)
        if m:
            current = (files.get(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue
        if current and s and not s.startswith((".", "$", "{", "}")) and not s.endswith(":"):
            tok = s.split()[0]
            out[current].append(s.split()[1] if tok.startswith("@") else tok)
    return out


def ptx_product_choices(module: TModule, ptx: str, plan: dict) -> dict:
    """For adds / subs with a product on both sides, read LLVM's choice from the PTX.

    A product whose only use is this add and which still appears as a standalone multiply at its own
    source location was not folded, so the other product was.  Returns node_id -> "swap" / "default";
    nodes without conclusive evidence are left out."""

    text = _module_text(module)
    locations = ttir_locations(text)
    defining = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(%[\w#]+)(?::\d+)?\s*=", line)
        if m:
            defining.setdefault(m.group(1), line)
    by_loc = ptx_instructions_by_location(ptx)
    ops = [o for fn in module.funcs.values() for o in fn.walk()]
    uses = collections.Counter(v for o in ops for v in o.operands)
    loc_of = {o.node_id: op_location(o, locations, defining) for o in ops}
    shared = collections.Counter(loc_of[o.node_id] for o in ops if o.name == "arith.mulf")

    def materialized(mul) -> Optional[bool]:
        loc = loc_of.get(mul.node_id)
        if loc is None or loc not in by_loc or shared[loc] != 1 or uses[mul.results[0]] != 1:
            return None
        return any(t.startswith("mul.") and (".f32" in t or ".f16" in t) for t in by_loc[loc])

    out = {}
    for node, (idx, mul, alt) in plan.items():
        if alt is None:
            continue
        default_kept, other_kept = materialized(mul), materialized(alt[1])
        if default_kept and other_kept is not True:
            out[node] = "swap"
        elif other_kept and default_kept is not True:
            out[node] = "default"
    return out


def sass_instructions_by_line(sass: str) -> dict:
    """(file basename, line) -> SASS opcodes, from nvdisasm --print-line-info output."""

    out, current = collections.defaultdict(list), None
    for line in sass.splitlines():
        m = re.search(r'//## File "([^"]*)", line (\d+)', line)
        if m:
            current = (Path(m.group(1)).name, int(m.group(2)))
            continue
        m = re.match(r"^\s*/\*[0-9a-f]+\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if m and current:
            out[current].append(m.group(2))
    return out


_SASS_CACHE = {}


def launch_sass(launch) -> Optional[str]:
    """SASS with line information of the captured cubin (nvdisasm shipped with Triton), if available."""

    import subprocess

    directory = getattr(launch, "directory", None)
    if directory is None:
        return None
    cubins = sorted(Path(directory, "compiled").glob("*.cubin"))
    if not cubins:
        return None
    key = str(cubins[0])
    if key not in _SASS_CACHE:
        try:
            import triton

            tool = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "nvdisasm"
            _SASS_CACHE[key] = subprocess.run([str(tool), "-c", "-g", key], capture_output=True, text=True,
                                              check=True).stdout
        except Exception:
            _SASS_CACHE[key] = None
    return _SASS_CACHE[key]


def _module_text(module: TModule) -> str:
    return getattr(module, "source_text", "")


# ---------------------------------------------------------------------------
# The emulator
# ---------------------------------------------------------------------------


def _fp32_representable(x: np.ndarray, elem: str) -> np.ndarray:
    if elem == "f32":
        return np.float32(x).astype(np.float64) == x
    if elem == "f16":
        return np.float16(x).astype(np.float64) == x
    if elem in iv.FLOAT_FORMATS:
        r, overflow = iv.round_nearest_even(x, elem)
        return (r == x) & ~overflow
    return np.zeros(np.shape(x), dtype=bool)


def _zero_sign(out: TV, kind: str, ops: list) -> TV:
    """IEEE sign of a zero result: an exact zero sum of opposite-signed (or equal nonzero opposite)
    operands is +0 under round to nearest, -0 only when both addends are -0; a nonzero result that
    rounds to zero keeps its sign.  float64 arithmetic on the FP32 operands follows the same rule (the
    product of two FP32 values is exact in float64), so the sign of p + c in float64 is the IEEE sign."""

    z = (out.st == ST_OK) & (out.lo == 0)
    if not z.any():
        return out
    xs = [np.broadcast_to(np.asarray(o.lo, dtype=np.float64), out.lo.shape) for o in ops]
    if kind == "add":
        exact = xs[0] + xs[1]
    elif kind == "sub":
        exact = xs[0] + (-xs[1])
    else:
        exact = xs[0] * xs[1] + xs[2]
    lo = np.where(z, np.copysign(0.0, exact), out.lo)
    hi = np.where(z, lo, out.hi)
    res = _ftv(out.elem, lo, hi, out.st, out.cond, out.reasons)
    res.d = out.d
    return res


class GpuEmulator(KernelReferenceEvaluator):
    """Rounding-check evaluation extended with the lowering rules of the locked build (see module doc)."""

    def __init__(self, module: TModule, ttgir: str, enable_fp_fusion: bool = True, exact_nodes=(),
                 oracle: Optional[HardwareOracle] = None, func_name: Optional[str] = None, ungrouped=(),
                 swapped=(), substitution: str = "mid", ptx: Optional[str] = None, sass: Optional[str] = None):
        super().__init__(module, mode=NumericMode.ROUNDING_CHECK, func_name=func_name, ttgir=ttgir)
        if substitution not in SUBSTITUTIONS:
            raise ValueError(f"substitution must be one of {SUBSTITUTIONS}")
        self.exact_nodes = set(exact_nodes)
        self.substitution = substitution
        self.ungrouped = set(ungrouped)  # add / sub nodes whose approximate producer is not contracted
        self.oracle = oracle or HardwareOracle()
        self.layouts = reduce_layouts(module, ttgir)
        self.enable_fp_fusion = enable_fp_fusion
        plan = contraction_plan(module) if enable_fp_fusion else {}
        # Product choices read from the PTX take precedence; the rest can still be given (swapped).
        self.ptx_choices = ptx_product_choices(module, ptx, plan) if (ptx and enable_fp_fusion) else {}
        swapped = set(swapped) | {n for n, c in self.ptx_choices.items() if c == "swap"}
        self.swappable = [n for n, (_, _, alt) in plan.items() if alt is not None and n not in self.ptx_choices]
        self.fusion = {n: (alt if (n in swapped and alt is not None) else (i, m)) for n, (i, m, alt) in plan.items()}
        self.fused_products = {mul.node_id for _, mul in self.fusion.values()}
        self._uses = collections.Counter(v for fn in module.funcs.values() for o in fn.walk() for v in o.operands)
        self._def, self._block = {}, {}
        self._sass = sass
        for fn in module.funcs.values():
            def collect(region):
                for block in region.blocks:
                    for op in block.ops:
                        self._block[id(op)] = id(block)
                        for r in op.results:
                            self._def[r] = op
                        for reg in op.regions:
                            collect(reg)
            collect(fn.body)
        self.sass_choices = self._sass_contractions(sass) if (sass and enable_fp_fusion) else {}
        self.ungrouped = set(self.ungrouped) | {n for n, c in self.sass_choices.items() if c == "uncontracted"}
        self.trace = collections.Counter()
        self._env = None
        self._no_round = False

    # -- plumbing ---------------------------------------------------------------

    def _exec(self, op, env, state):
        saved = self._env
        self._env = env
        try:
            return super()._exec(op, env, state)
        finally:
            self._env = saved

    def _maybe_round(self, v, op, fmt, declared):
        if self._no_round:
            return v
        return super()._maybe_round(v, op, fmt, declared)

    def _unrounded(self, fn, *a) -> TV:
        """The rigorous enclosure [lo, hi] of the operation's exact real result (no rounding)."""

        self._no_round = True
        try:
            return fn(*a)
        finally:
            self._no_round = False

    def _estimate(self, fn, *a) -> TV:
        """A float64 estimate (midpoint of the enclosure) of the exact result, used inside a model."""

        out = self._unrounded(fn, *a)
        mid = 0.5 * (out.lo + out.hi)
        return _ftv(out.elem, mid, mid.copy(), out.st, out.cond, out.reasons)

    def _substitute(self, elem, lo, hi, st, cond, reasons) -> TV:
        """The value a substituted node passes on.  "mid" / "lo" / "hi": a float64 value of the exact
        result (midpoint or one endpoint of its enclosure).  Running both endpoints is a sensitivity
        check, not an enclosure: equal downstream outputs at the two endpoints do not imply equal outputs
        for every value in between (F(z) = 1 - z^2 on [-1, 1]) nor for mixed endpoints across elements.
        "rn": the exact result rounded to the node's format -- the output of a correctly rounded
        implementation, which a real kernel change can reproduce.  Its uniqueness RN(lo) == RN(hi) does
        follow from the enclosure (rounding is monotone); where the endpoints round apart the element is
        marked not established."""

        if self.substitution == "lo":
            v = lo
        elif self.substitution == "hi":
            v = hi
        elif self.substitution == "mid":
            v = 0.5 * (lo + hi)
        else:
            r_lo, r_hi = iv.round_nearest_even(lo, elem)[0], iv.round_nearest_even(hi, elem)[0]
            st = np.where((r_lo != r_hi) & (st == ST_OK), ST_NE, st).astype(np.int8)
            v = np.where(r_lo == r_hi, r_lo, 0.0)
        return _ftv(elem, v, np.array(v, copy=True), st, cond, reasons)

    def _exact_value(self, fn, *a) -> TV:
        out = self._unrounded(fn, *a)
        return self._substitute(out.elem, out.lo, out.hi, out.st, out.cond, out.reasons)

    def _not_emulable(self, op, shape, elem, args, what) -> TV:
        self.trace[(op.node_id, "not_emulable:" + what)] += 1
        st = np.full(shape, ST_NE, dtype=np.int8)
        z = np.zeros(shape)
        return _ftv(elem, z, z.copy(), st, np.zeros(shape, dtype=bool), frozenset({f"not_emulable:{what}@{op.node_id}"}))

    # -- elementwise --------------------------------------------------------------

    def _float_op(self, name, op, args, out_elem):
        node = op.node_id
        if node in self.fusion:
            idx, mul = self.fusion[node]
            a, b = (self._env[v] for v in mul.operands)
            c = args[1 - idx]
            if op.name == "arith.subf":
                if idx == 0:  # a*b - c
                    c = TV("f", c.elem, -c.hi, -c.lo, None, c.st, c.cond, c.reasons)
                else:  # c - a*b
                    a = TV("f", a.elem, -a.hi, -a.lo, None, a.st, a.cond, a.reasons)
            fused = TOp("math.fma", op.results, op.result_types, [mul.operands[0], mul.operands[1], "%c"], op.attrs,
                        [], op.line, op.text, op.operand_types)
            self.trace[(node, "fma_contraction")] += 1
            if node in self.exact_nodes:
                return self._exact_value(super()._float_op, "fma", fused, [a, b, c], out_elem)
            return _zero_sign(super()._float_op("fma", fused, [a, b, c], out_elem), "fma", [a, b, c])
        if node in self.exact_nodes:
            self.trace[(node, "exact")] += 1
            return self._exact_value(super()._float_op, name, op, args, out_elem)
        if op.name in ("arith.addf", "arith.subf"):
            grouped = self._group_oracle(op, args, out_elem)
            if grouped is not None:
                return grouped
        if op.name in ORACLE_OPS and not (op.name == "tt.extern_elementwise"
                                         and op.attrs.get("symbol", "").strip('"') in LIBDEVICE_ROUNDING):
            return self._oracle(name, op, args, out_elem)
        self.trace[(node, "fused_product" if node in self.fused_products else "ieee")] += 1
        out = super()._float_op(name, op, args, out_elem)
        if name in ("add", "sub", "fma"):
            out = _zero_sign(out, name, args)
        return out

    def _is_oracle(self, op) -> bool:
        return op.name in ORACLE_OPS and not (op.name == "tt.extern_elementwise"
                                             and op.attrs.get("symbol", "").strip('"') in LIBDEVICE_ROUNDING)

    def _group_producers(self, op):
        """Approximate operations of the same block feeding this add / sub (through layout-only ops),
        each with a single use; returns (producers, alias of layout values to producer results)."""

        block = self._block.get(id(op))
        producers, alias = [], {}
        for v in op.operands:
            prod, chain = self._def.get(v), []
            while prod is not None and prod.name in LAYOUT_OPS and self._block.get(id(prod)) == block:
                chain.append(prod)
                prod = self._def.get(prod.operands[0])
            # Only a single-use result: with other uses the rounded quotient is kept and the add stays plain.
            single = all(self._uses[c.results[0]] == 1 for c in chain) and prod is not None \
                and self._uses[prod.results[0]] == 1
            if prod is not None and self._is_oracle(prod) and self._block.get(id(prod)) == block \
                    and single and prod not in producers:
                producers.append(prod)
                for c in chain:
                    alias[c.results[0]] = prod.results[0]
        return producers, alias

    def _sass_contractions(self, sass: str) -> dict:
        """For each approximate-then-add pair, read ptxas' decision from the SASS line information:
        at the add's source line an FADD and no FFMA means not contracted, an FFMA and no FADD means
        contracted.  Lines with more than one TTIR add / sub, or both or neither instruction, are left
        undecided.  Returns node_id -> "uncontracted" / "contracted"."""

        text = getattr(self.module, "source_text", "")
        locations = ttir_locations(text)
        defining = {}
        for line in text.splitlines():
            m = re.match(r"^\s*(%[\w#]+)(?::\d+)?\s*=", line)
            if m:
                defining.setdefault(m.group(1), line)
        by_line = sass_instructions_by_line(sass)
        ops = [o for fn in self.module.funcs.values() for o in fn.walk()]
        line_of = {o.node_id: (lambda loc: (loc[0], loc[1]) if loc else None)(op_location(o, locations, defining))
                   for o in ops}
        adds_per_line = collections.Counter(line_of[o.node_id] for o in ops if o.name in ("arith.addf", "arith.subf"))
        out = {}
        for o in ops:
            if o.name not in ("arith.addf", "arith.subf") or o.node_id in self.fusion:
                continue
            producers, _ = self._group_producers(o)
            key = line_of[o.node_id]
            if not producers or key is None or adds_per_line[key] != 1 or key not in by_line:
                continue
            if any(line_of[p.node_id] == key for p in producers):
                continue  # producer and add on one line: the instructions cannot be told apart
            codes = by_line[key]
            fadd = any(c.startswith("FADD") for c in codes)
            ffma = any(c.startswith("FFMA") for c in codes)
            if fadd and not ffma:
                out[o.node_id] = "uncontracted"
            elif ffma and not fadd:
                out[o.node_id] = "contracted"
        return out

    def _group_oracle(self, op, args, out_elem) -> Optional[TV]:
        """An add / sub fed by an approximate operation of the same block (through layout-only ops):
        ptxas may contract the multiply that ends the approximate instruction's expansion into the
        plain add, so the pair runs on the device together."""

        if op.node_id in self.ungrouped:
            return None
        producers, alias = self._group_producers(op)
        if not producers or any(p.node_id in self.exact_nodes for p in producers):
            return None
        inputs = {}
        produced = {r for p in producers for r in p.results} | set(alias)
        for p in producers:
            for v in p.operands:
                inputs[v] = self._env[v]
        for v, a in zip(op.operands, args):
            if v not in produced:
                inputs[v] = a
        arrays = {}
        for v, t in inputs.items():
            if (t.st != ST_OK).any():
                return None
            if t.kind == "f":
                if t.elem not in iv.FLOAT_FORMATS or not np.all(_fp32_representable(t.lo, t.elem)):
                    return None
            arrays[v] = t.lo
        self.trace[(op.node_id, "oracle_group")] += 1
        hw = self.oracle.group(producers + [op], alias, arrays, out_elem)
        st = np.where(np.isnan(hw), 1, np.where(hw == np.inf, 2, np.where(hw == -np.inf, 3, ST_OK))).astype(np.int8)
        vals = np.where(st == ST_OK, hw, 0.0)
        return _ftv(out_elem, vals, vals.copy(), st, _merge_cond(*args), _merge_reasons(*args))

    def _oracle(self, name, op, args, out_elem) -> TV:
        self.trace[(op.node_id, "oracle")] += 1
        shape = np.broadcast_shapes(*(a.shape for a in args))
        elems = [a.elem for a in args]
        st = _merge_status(*args)
        rounded, exact_inputs = [], True
        for a, e in zip(args, elems):
            if a.kind != "f":
                rounded.append(np.broadcast_to(a.lo, shape))
                continue
            if e not in iv.FLOAT_FORMATS:
                return self._not_emulable(op, list(shape), out_elem, args, f"no rounding model for {e}")
            vals = np.where(a.st == ST_OK, a.lo, 0.0)
            vals = np.where(a.st == 1, np.nan, np.where(a.st == 2, np.inf, np.where(a.st == 3, -np.inf, vals)))
            r, _ = iv.round_nearest_even(np.where(np.isfinite(vals), vals, 0.0), e)
            r = np.where(np.isfinite(vals), r, vals)
            exact_inputs &= bool(np.all(r[np.isfinite(vals)] == vals[np.isfinite(vals)]))
            rounded.append(np.broadcast_to(r, shape))
        hw = self.oracle(op, rounded, out_elem)
        if not exact_inputs:
            # An upstream node was made exact: propagate the input change through the declared
            # function and keep this node's own approximation error.
            f_exact = self._estimate(super()._float_op, name, op, args, out_elem)
            r_args = [TV("f", a.elem, r, r.copy(), None, a.st, a.cond, a.reasons) if a.kind == "f" else a
                      for a, r in zip(args, rounded)]
            f_round = self._estimate(super()._float_op, name, op, r_args, out_elem)
            hw = np.where((f_exact.st == ST_OK) & (f_round.st == ST_OK) & np.isfinite(hw),
                          hw + (f_exact.lo - f_round.lo), hw)
        out_st = np.where(np.isnan(hw), 1, np.where(hw == np.inf, 2, np.where(hw == -np.inf, 3, ST_OK)))
        out_st = np.where(st >= ST_UNDEF, st, out_st).astype(np.int8)
        vals = np.where(out_st == ST_OK, hw, 0.0)
        return _ftv(out_elem, vals, vals.copy(), out_st, _merge_cond(*args), _merge_reasons(*args))

    def _op_extern(self, op, args, env, state):
        symbol = op.attrs.get("symbol", "").strip('"')
        internal = LIBDEVICE.get(symbol)
        if symbol in LIBDEVICE_ROUNDING or internal is None or op.result_types[0].elem not in iv.FLOAT_FORMATS:
            return super()._op_extern(op, args, env, state)
        return self._float_op(internal, op, args, op.result_types[0].elem)

    def _op_inline_asm(self, op, args, env, state):
        internal = inline_asm_internal(op.attrs.get("asm", "")) or "identity"
        return self._float_op(internal, op, args, op.result_types[0].elem)

    # -- reductions, dot, scan, atomics -------------------------------------------------

    def _op_reduce(self, op, args, env, state):
        combiner = recognize_combiner(op)
        if combiner == "welford":  # float Welford depends on the merge tree: not order-free, not emulable here
            raise ProgramAbort(f"{op.node_id}: Welford merge order is not modelled by the emulator")
        if combiner is None:  # the reference interprets the region in real arithmetic; that is not the device's bits
            raise ProgramAbort(f"{op.node_id}: a generic combine region is not emulated bitwise")
        if combiner not in ("sum", "prod") or args[0].kind != "f":
            mode, self.mode = self.mode, NumericMode.NUMERICAL_DIFFERENCE  # max / min / arg: order-free
            try:
                return super()._op_reduce(op, args, env, state)
            finally:
                self.mode = mode
        (x,) = args
        axis = int(op.attrs["axis"].split(":")[0])
        if op.node_id in self.exact_nodes or combiner == "prod":
            if combiner == "prod":
                return self._not_emulable(op, np.delete(np.array(x.shape), axis).tolist(), x.elem, [x], "prod")
            self.trace[(op.node_id, "exact")] += 1
            finite = x.st == ST_OK
            lo, hi = iv.isum(np.where(finite, x.lo, 0.0), np.where(finite, x.hi, 0.0), axis)
            st = np.where(np.any(~finite, axis=axis), ST_NE, ST_OK).astype(np.int8)
            return self._substitute(x.elem, lo, hi, st, np.any(x.cond, axis=axis), x.reasons)
        lay = self.layouts.get(op.node_id)
        if lay is None:
            return self._not_emulable(op, list(np.delete(np.array(x.shape), axis)), x.elem, [x], "reduce layout")
        vals = np.moveaxis(np.where(x.st == ST_OK, x.lo, 0.0), axis, -1)
        n = vals.shape[-1]
        spt = min(lay["spt"], n)
        tpw = min(lay["tpw"], max(1, n // spt))
        wpc = min(lay["wpc"], max(1, n // (spt * tpw)))
        tile = spt * tpw * wpc
        reps = max(1, n // tile)
        fmt = x.elem
        rn = lambda v: iv.round_nearest_even(v, fmt)[0]  # noqa: E731

        def fma(a, b, c):  # RN(a*b + c), or RN(a + c) when b is None
            lo, hi = iv.ifma(a, a, b, b, c, c) if b is not None else iv.iadd(a, a, c, c)
            r_lo, r_hi = rn(lo), rn(hi)
            r = np.where(r_lo == r_hi, r_lo, np.nan)
            return np.where(r == 0, np.copysign(0.0, (a * b if b is not None else a) + c), r)

        # Contraction into the combine: the operand is a product defined in the same block, so the
        # first combine that sees a raw product is an fma (LLVM folds the product with fewer uses,
        # the left one on a tie; a product also sent through a shuffle still folds).
        producer = self._def.get(op.operands[0])
        fused = (self.enable_fp_fusion and producer is not None and producer.name == "arith.mulf"
                 and self._block.get(id(producer)) == self._block.get(id(op)) and op.node_id not in self.exact_nodes)
        if fused:
            fa, fb = (np.moveaxis(np.broadcast_to(np.where(self._env[v].st == ST_OK, self._env[v].lo, 0.0), x.shape),
                                  axis, -1) for v in producer.operands)
        self.trace[(op.node_id, "reduce_tree_fused" if fused else "reduce_tree")] += 1
        # thread partials: [..., warp, lane]
        idx = (np.arange(reps)[:, None, None, None] * tile + np.arange(wpc)[None, :, None, None] * (spt * tpw)
               + np.arange(tpw)[None, None, :, None] * spt + np.arange(spt)[None, None, None, :])
        order = [idx[rep, :, :, j] for rep in range(reps) for j in range(spt)]
        acc = vals[..., order[0]]
        raw = fused  # acc is still the unrounded product of (fa, fb) at order[0]
        for k, pos in enumerate(order[1:]):
            if raw:
                acc = fma(fa[..., order[0]], fb[..., order[0]], vals[..., pos])
                raw = False
            elif fused:
                acc = fma(fa[..., pos], fb[..., pos], acc)
            else:
                acc = fma(acc, None, vals[..., pos])
        lane_of = np.arange(tpw)
        first_raw = raw
        stride = tpw // 2
        while stride >= 1:
            other = acc[..., lane_of ^ stride]
            if raw:
                acc = fma(fa[..., order[0]], fb[..., order[0]], other)
                raw = False
            else:
                acc = fma(acc, None, other)
            stride //= 2
        if wpc > 1:
            part = acc[..., 0]  # the lane at offset 0 along the axis writes the warp partial
            stride = wpc // 2
            while stride >= 1:
                part = fma(part, None, part[..., np.arange(wpc) ^ stride])
                stride //= 2
            res = part[..., 0]  # broadcast to every thread through shared memory
            divergent = np.zeros(res.shape, dtype=bool)
        else:
            lanes = acc[..., 0, :]  # warp-synchronous: every lane keeps its own result
            res = lanes[..., 0]
            divergent = np.any(lanes != lanes[..., :1], axis=-1) & ~np.isnan(res)
            if first_raw and tpw == 1:
                divergent[:] = False
        bad = np.any(np.moveaxis(x.st != ST_OK, axis, -1), axis=-1) | np.isnan(res) | divergent
        st = np.where(bad, ST_NE, ST_OK).astype(np.int8)
        res = np.where(bad, 0.0, res)
        reasons = x.reasons | ({f"not_emulable:lane-divergent reduction@{op.node_id}"} if divergent.any() else set())
        return _ftv(x.elem, res, res.copy(), st, np.any(x.cond, axis=axis), frozenset(reasons))

    def _op_dot(self, op, args, env, state):
        a, b, c = args
        precision = op.attrs.get("inputPrecision", "ieee").strip()  # the default (ieee) is not printed
        shape = np.broadcast_shapes(c.shape)
        if not (a.elem == b.elem == c.elem == "f32" and precision.startswith("ieee")):
            return self._not_emulable(op, list(shape), c.elem, [a, b, c], f"tensor-core dot ({precision})")
        if op.node_id in self.exact_nodes:
            self.trace[(op.node_id, "exact")] += 1
            lo, hi = iv.idot(a.lo, a.hi, b.lo, b.hi)
            lo, hi = iv.iadd(lo, hi, c.lo, c.hi)
            return self._substitute(c.elem, lo, hi, _merge_status(c).copy(), c.cond, c.reasons)
        self.trace[(op.node_id, "dot_fma_chain")] += 1
        acc = c.lo.copy()
        for k in range(a.shape[-1]):
            x = np.broadcast_to(a.lo[..., :, k][..., :, None], shape)
            y = np.broadcast_to(b.lo[..., k, :][..., None, :], shape)
            lo_acc = acc
            lo, hi = iv.ifma(x, x, y, y, acc, acc)
            r_lo, r_hi = iv.round_nearest_even(lo, "f32")[0], iv.round_nearest_even(hi, "f32")[0]
            acc = np.where(r_lo == r_hi, r_lo, np.nan)
            acc = np.where(acc == 0, np.copysign(0.0, x * y + lo_acc), acc)
        bad = np.isnan(acc) | (np.any(a.st != ST_OK, axis=-1)[..., :, None]) | (np.any(b.st != ST_OK, axis=-2)[..., None, :])
        st = np.where(bad, ST_NE, ST_OK).astype(np.int8)
        acc = np.where(bad, 0.0, acc)
        return _ftv(c.elem, acc, acc.copy(), st, c.cond | np.any(a.cond, axis=-1)[..., :, None], c.reasons)

    def _op_scan(self, op, args, env, state):
        if args[0].kind == "f":
            return self._not_emulable(op, list(args[0].shape), args[0].elem, args, "scan")
        return super()._op_scan(op, args, env, state)

    def _op_atomic_rmw(self, op, args, env, state):
        raise ProgramAbort(f"not_emulable:atomic@{op.node_id}")


# ---------------------------------------------------------------------------
# Driver: verify, then localize
# ---------------------------------------------------------------------------


def _encode(values: np.ndarray, elem: str) -> Optional[np.ndarray]:
    """Bit patterns of values (already representable) in the storage format."""

    if elem in _BITS:
        ftype, utype = _BITS[elem]
        return values.astype(ftype).view(utype).astype(np.uint64)
    if elem == "bf16":
        return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint64)
    return None


def _device_bits(b) -> Optional[np.ndarray]:
    raw = np.asarray(b.after_raw, dtype=np.uint8)
    width = {"f32": np.uint32, "f16": np.uint16, "bf16": np.uint16, "f64": np.uint64}.get(b.elem)
    if width is None:
        return None
    size = np.dtype(width).itemsize
    return raw[: (raw.size // size) * size].view(width).astype(np.uint64)[: b.lo.size]


def _outputs(result, launch) -> dict:
    out = {}
    for ptr, b in result.buffers.items():
        if b.kind != "f" or not b.written.any():
            continue
        actual = b.actual_after.copy()
        if b.actual_after_st is not None:  # specials are stored as 0 with a status
            actual = np.where(b.actual_after_st == 1, np.nan, np.where(b.actual_after_st == 2, np.inf,
                              np.where(b.actual_after_st == 3, -np.inf, actual)))
        out[b.name] = (b.written.copy(), b.lo.copy(), b.hi.copy(), b.st.copy(), actual, b.elem, _device_bits(b))
    return out


def _compare(written, lo, hi, st, actual, elem, bits) -> dict:
    """Element masks: emulated, bit-identical (finite and infinite values, compared as storage bits),
    numerically equal (+0 and -0 equal), NaN on both sides (the payload is not modelled)."""

    finite = written & (st == ST_OK) & (lo == hi)
    inf = written & np.isin(st, (2, 3))
    nan = written & (st == 1)
    emulated = finite | inf | nan
    value = np.where(st == 2, np.inf, np.where(st == 3, -np.inf, lo))
    enc = _encode(np.where(finite | inf, value, 0.0), elem)
    if enc is not None and bits is not None:
        bit_equal = (finite | inf) & (enc == bits)
    else:  # formats without a bit model: numerical equality only
        bit_equal = (finite | inf) & (value == actual)
    return {"emulated": emulated, "bit_equal": bit_equal,
            "value_equal": (finite | inf) & (value == actual), "nan_both": nan & np.isnan(actual)}


def _bitwise_count(result, launch) -> tuple:
    equal = emulated = 0
    for name, o in _outputs(result, launch).items():
        m = _compare(*o)
        emulated += int(m["emulated"].sum())
        equal += int((m["bit_equal"] | m["nan_both"]).sum())
    return equal, emulated


def emulate(launch, exact_nodes=(), oracle: Optional[HardwareOracle] = None, ungrouped=(), swapped=(),
            substitution: str = "mid", programs: Optional[list] = None):
    from .ttir_parser import parse_ttir

    module = parse_ttir(launch.asm["ttir"])
    fusion = bool(launch.metadata.get("enable_fp_fusion", True))
    if oracle is None:
        options = {k: launch.metadata[k] for k in ("enable_fp_fusion", "enable_reflect_ftz") if k in launch.metadata}
        oracle = HardwareOracle(options={"num_warps": 4, **options})
    module.source_text = launch.asm["ttir"]
    emu = GpuEmulator(module, launch.asm["ttgir"], enable_fp_fusion=fusion, exact_nodes=exact_nodes, oracle=oracle,
                      ungrouped=ungrouped, swapped=swapped, substitution=substitution, ptx=launch.asm.get("ptx"),
                      sass=launch_sass(launch))
    result = emu.evaluate(launch, programs=programs)
    return result, emu


def verify(launch, oracle: Optional[HardwareOracle] = None, ungrouped=None, swapped=None,
           programs: Optional[list] = None) -> dict:
    """Emulate and compare with the device outputs bit for bit (storage bit patterns).

    Every lowering rule is fixed in advance except two binary choices per affected node, which the TTIR
    does not determine:

    * whether ptxas contracts the multiply that ends an approximate instruction's expansion (div.full:
      reciprocal times numerator) into the plain add / sub that consumes it -- a scheduling decision
      below PTX, visible only in SASS; such a pair starts contracted;
    * which product LLVM folds when both operands of an add / sub are products -- decided by use
      counts in the selection DAG; it starts with the TTIR use counts.

    With ``ungrouped`` / ``swapped`` given, the choices are frozen (held-out validation): nothing is
    adjusted.  Otherwise a choice is flipped only if that raises the number of bit-identical outputs,
    and the flipped choices are reported -- this fits the model to the same outputs it is checked
    against, so such a kernel needs held-out inputs before it counts as validated.

    status: "bit_identical" (every emulated element has the device's bits; NaN only as a class),
    "mismatch", or "not_established" (no emulated output, e.g. every program stopped at an atomic).
    """

    frozen = ungrouped is not None or swapped is not None
    ungrouped, swapped = list(ungrouped or []), list(swapped or [])
    result, emu = emulate(launch, oracle=oracle, ungrouped=ungrouped, swapped=swapped, programs=programs)
    oracle = emu.oracle
    best = _bitwise_count(result, launch)
    groups = [n for (n, how) in emu.trace if how == "oracle_group"]
    swappable = list(emu.swappable)
    fitted_u, fitted_s = [], []
    if not frozen and best[0] < best[1]:
        for kind, node in [("group", n) for n in groups] + [("swap", n) for n in swappable]:
            ug = ungrouped + [node] if kind == "group" else ungrouped
            sw = swapped + [node] if kind == "swap" else swapped
            trial_res, trial_emu = emulate(launch, oracle=oracle, ungrouped=ug, swapped=sw, programs=programs)
            count = _bitwise_count(trial_res, launch)
            if count[0] > best[0]:
                best, result, emu, ungrouped, swapped = count, trial_res, trial_emu, ug, sw
                (fitted_u if kind == "group" else fitted_s).append(node)
            if best[0] == best[1]:
                break
    report = {"ttir_sha256": hashlib.sha256(launch.asm["ttir"].encode()).hexdigest(),
              "buffers": {}, "aborted_programs": len(result.aborted),
              "abort_reasons": sorted(set(result.aborted.values()))[:3],
              "nodes": collections.Counter(), "not_emulable_nodes": [],
              "lowering_choices": {"frozen": frozen, "approximate_then_add_pairs": len(groups),
                                   "two_product_adds": len(swappable) + len(emu.ptx_choices),
                                   "product_choices_from_ptx": emu.ptx_choices,
                                   "contractions_from_sass": emu.sass_choices, "uncontracted": ungrouped,
                                   "swapped": swapped, "uncontracted_from_output": fitted_u,
                                   "swapped_from_output": fitted_s}}
    for (node, how), _ in emu.trace.items():
        report["nodes"][how.split(":")[0]] += 1
        if how.startswith("not_emulable"):
            report["not_emulable_nodes"].append(f"{node} ({how.split(':', 1)[1]})")
    report["nodes"] = dict(report["nodes"])
    totals = collections.Counter()
    for name, o in _outputs(result, launch).items():
        m = _compare(*o)
        written = o[0]
        entry = {"written": int(written.sum()), "emulated": int(m["emulated"].sum()),
                 "bit_identical": int(m["bit_equal"].sum()), "value_equal": int(m["value_equal"].sum()),
                 "nan_both": int(m["nan_both"].sum()), "not_emulated": int((written & ~m["emulated"]).sum())}
        entry["mismatch"] = entry["emulated"] - entry["bit_identical"] - entry["nan_both"]
        report["buffers"][name] = entry
        totals.update(entry)
    if totals["emulated"] == 0:
        report["status"] = "not_established"
    elif totals["mismatch"] == 0:
        report["status"] = "bit_identical"
    else:
        report["status"] = "mismatch"
    report["bitwise_reproduced"] = report["status"] == "bit_identical"
    report["fitted_to_output"] = bool(fitted_u or fitted_s)
    return report


def rounding_nodes(emu: GpuEmulator) -> list:
    """Nodes whose execution rounds or approximates (candidates for localization), in program order."""

    seen = []
    for (node, how) in emu.trace:
        if how in ("ieee", "oracle", "oracle_group", "fma_contraction", "reduce_tree", "reduce_tree_fused",
                   "dot_fma_chain") \
                and node not in seen:
            seen.append(node)
    return seen


def localize(launch, nodes: Optional[list] = None, oracle: Optional[HardwareOracle] = None,
             ungrouped=(), swapped=(), endpoint_nodes=None, substitution: str = "mid",
             programs: Optional[list] = None) -> dict:
    """Per-node model contribution c_j = K_emulated - K_(node j substituted) on the bit-identical outputs.

    ``substitution`` "mid" passes a float64 value of the exact result downstream (an estimate in the
    declared execution model, not a deployable change); "rn" passes the correctly rounded result, which
    a real kernel change can reproduce.  For the nodes in ``endpoint_nodes`` (all when "all") the
    substitution is repeated with all-lower and all-upper endpoints of the exact result's enclosure
    ("endpoint_check").  This is a sensitivity check: outputs that differ show where carrying a float64
    value instead of the exact real matters; outputs that agree are stable under that perturbation, which
    is not an enclosure of the downstream result (no monotonicity is assumed or checked).
    """

    base, emu = emulate(launch, oracle=oracle, ungrouped=ungrouped, swapped=swapped, programs=programs)
    oracle = emu.oracle
    outs = _outputs(base, launch)
    masks = {name: _compare(*o)["bit_equal"] & (o[3] == ST_OK) for name, o in outs.items()}
    nodes = nodes if nodes is not None else rounding_nodes(emu)
    kinds = {}
    for (node, how) in emu.trace:
        kinds.setdefault(node, how)
    texts = {o.node_id: re.sub(r"\s+loc\(.*$", "", o.text.strip())[:140]
             for fn in emu.module.funcs.values() for o in fn.walk()}

    def run(node, mode):
        res, _ = emulate(launch, exact_nodes=[node], oracle=oracle, ungrouped=ungrouped, swapped=swapped,
                         substitution=mode, programs=programs)
        return _outputs(res, launch)

    rows = []
    for node in nodes:
        o = run(node, substitution)
        ends = (run(node, "lo"), run(node, "hi")) if (endpoint_nodes == "all" or
                                                      (endpoint_nodes and node in endpoint_nodes)) else None
        entry = {"node": node, "lowering": kinds.get(node), "text": texts.get(node, ""), "substitution": substitution,
                 "buffers": {}}
        for name, m in masks.items():
            k = outs[name][1]
            e_lo, e_st = o[name][1], o[name][3]
            ok = m & (e_st == ST_OK)
            c = (k - e_lo)[ok]
            if c.size == 0:
                continue
            kk = k[ok]
            rel = c[kk != 0] / kk[kk != 0]
            scale = np.abs(kk).mean() if np.abs(kk).mean() > 0 else 1.0
            b = {"elements": int(c.size), "nonzero": int((c != 0).sum()),
                 "mean": float(c.mean()), "mean_abs": float(np.abs(c).mean()),
                 "mean_relative": float(rel.mean()) if rel.size else 0.0,
                 "aligned": float((c * kk).sum() / np.linalg.norm(kk)) if np.linalg.norm(kk) > 0 else 0.0,
                 "positive": int((c > 0).sum()), "negative": int((c < 0).sum()),
                 "mean_relative_to_output_scale": float(c.mean() / scale)}
            if ends is not None:
                lo_out, hi_out = ends[0][name], ends[1][name]
                both = ok & (lo_out[3] == ST_OK) & (hi_out[3] == ST_OK)
                d = np.abs(lo_out[1] - hi_out[1])[both]
                ulp = np.spacing(np.abs(k[both]).astype(np.float32)).astype(np.float64)
                in_ulp = d / ulp
                b["endpoint_check"] = {"elements": int(both.sum()),
                                       "outputs_differ_by_half_ulp32_or_more": int((in_ulp >= 0.5).sum()),
                                       "max_difference_in_ulp32": float(in_ulp.max()) if d.size else 0.0,
                                       "max_abs_difference": float(d.max()) if d.size else 0.0,
                                       "mean_lo": float((k - lo_out[1])[both].mean()) if both.any() else 0.0,
                                       "mean_hi": float((k - hi_out[1])[both].mean()) if both.any() else 0.0}
            entry["buffers"][name] = b
        rows.append(entry)
    return {"verified": {name: int(m.sum()) for name, m in masks.items()}, "nodes": rows,
            "oracle_calls": oracle.calls}
