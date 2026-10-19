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
  mode, as in rounding-check mode; with ``enable_reflect_ftz`` (the default) the
  rounding-suffixed fp32 libdevice calls and the libdevice-lowered math.floor /
  math.ceil flush subnormal operands and results to signed zero, and an exact zero
  sum under round toward -inf is -0;
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
* an FP32 ``tt.dot`` with input precision ieee is a sequential fma chain over k;
* a tensor-core ``tt.dot`` (tf32, bf16 or f16 operands, fp32 accumulator) follows the mma rule in
  ``tensor_core.py``: tf32 operands cut toward zero, exact products, groups of 4 (tf32) / 8 (bf16, f16) aligned
  to the largest operand-exponent sum with one extra bit, every term and the group sum cut toward zero.

* a 1-D float ``tt.scan`` (sum or product, forward, blocked layout without replication) follows the order in
  ``scan_order.py``: sequential inside a thread, Hillis-Steele over lanes, warp totals folded left to right.

Not emulable (localization stops there; the affected elements are reported):
tf32x3 / bf16x3 / bf16x6 dots and fp16 accumulators, other scans (reverse, multi-operand, rank > 1, a product operand
that may be contracted), atomics.

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
from . import scan_order, tensor_core
from .ttir_eval import (INT_WIDTH, ST_NAN, ST_NE, ST_NINF, ST_OK, ST_PINF, ST_UNDEF, TV, KernelReferenceEvaluator, NumericMode,
                        ProgramAbort, _ftv,
                        _merge_cond, _merge_reasons, _merge_status)
from .ttir_mapping import LIBDEVICE, LIBDEVICE_ROUNDING, inline_asm_internal, recognize_combiner
from .ttir_parser import TModule, TOp

ORACLE_OPS = {"arith.divf", "arith.remf", "math.exp", "math.exp2", "math.log", "math.log2", "math.log10",
              "math.log1p", "math.expm1", "math.sqrt", "math.rsqrt", "math.sin", "math.cos", "math.tan",
              "math.tanh", "math.erf", "math.powf", "math.atan", "math.sinh", "math.cosh",
              "tt.extern_elementwise", "tt.elementwise_inline_asm"}
_TORCH = {"f32": "float32", "f16": "float16", "bf16": "bfloat16", "f64": "float64", "i32": "int32", "i64": "int64",
          "i8": "int8", "i16": "int16", "i1": "bool", "f8E5M2": "float8_e5m2"}
_TT_TYPE = {"f32": "f32", "f16": "f16", "bf16": "bf16", "f64": "f64", "i32": "i32", "i64": "i64", "i8": "i8",
            "i16": "i16", "i1": "i1", "f8E5M2": "f8E5M2"}
BLOCK = 1024
LAYOUT_OPS = {"tt.splat", "tt.broadcast", "tt.expand_dims", "ttg.convert_layout", "tt.reshape"}
SUBSTITUTIONS = ("mid", "lo", "hi", "rn")
_BITS = {"f32": (np.float32, np.uint32), "f16": (np.float16, np.uint16), "f64": (np.float64, np.uint64)}


# ---------------------------------------------------------------------------
# Hardware oracle for approximate instructions
# ---------------------------------------------------------------------------


def default_cache_dir() -> Path:
    """KA_CACHE_DIR, else $XDG_CACHE_HOME/kernel_analyzer, else ~/.cache/kernel_analyzer -- never a path relative to
    the installed package."""
    import os
    if os.environ.get("KA_CACHE_DIR"):
        return Path(os.environ["KA_CACHE_DIR"])
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "kernel_analyzer"


class HardwareOracle:
    """Runs one TTIR elementwise operation on the device, compiled by the installed Triton."""

    def __init__(self, cache_dir: Optional[Path] = None, options: Optional[dict] = None, device: str = "cuda"):
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir() / "oracle"
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
            # write-then-rename: parallel emulations share the cache directory and must never read a partial file
            import os
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp.write_text(text)
            os.replace(tmp, path)
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
        if out_elem in ("i8", "i16", "i32", "i64"):  # integers exactly (a float64 copy loses i64 bits beyond 2^53)
            return out[:n].to(torch.int64).cpu().numpy().reshape(shape)
        return out[:n].double().cpu().numpy().reshape(shape)


# ---------------------------------------------------------------------------
# TTGIR reduction layouts and the contraction plan
# ---------------------------------------------------------------------------

from .layouts import _axis_params, parse_layouts, reduce_layouts, scan_layouts  # noqa: E402,F401  (moved; names kept)


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


_NEG_ZERO = {"0x8000", "0x80008000", "0f80000000", "-0f00000000"}


def ptx_dataflow_multiplies(ptx: str) -> list:
    """The standalone floating-point multiplies of a PTX listing, each as the pair of source-location sets its two
    multiplicands come from (the locations of the instructions that define them, through mov / cvt).  A multiply is
    mul.* on a float type, or an fma whose addend is -0 (sm_86 has no bf16 multiply: x * y is fma.rn.bf16 x, y, -0)."""

    files = {m.group(1): Path(m.group(2)).name
             for m in re.finditer(r'^\s*\.file\s+(\d+)\s+"([^"]*)"', ptx, re.M)}
    instrs, current = [], None
    for line in ptx.splitlines():
        s = line.split("//")[0].strip()
        m = re.match(r"^\.loc\s+(\d+)\s+(\d+)\s+(\d+)", s)
        if m:
            current = (files.get(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue
        if not s or s.startswith((".", "$", "{", "}")) or s.endswith(":"):
            continue
        if s.startswith("@"):
            s = s.split(None, 1)[1] if len(s.split(None, 1)) > 1 else ""
        parts = s.rstrip(";").split(None, 1)
        if len(parts) < 2:
            continue
        args = [a.strip() for a in re.split(r",(?![^{]*\})", parts[1])]
        instrs.append((current, parts[0], re.findall(r"%\w+", args[0]), args[1:]))
    defs = {}
    for loc, op, dsts, srcs in instrs:
        for d in dsts:
            defs[d] = (loc, op, srcs)

    def origin(arg, depth=0) -> frozenset:
        regs = re.findall(r"%\w+", arg)
        out = set()
        for r in regs:
            d = defs.get(r)
            if d is None or depth > 16:
                continue
            loc, op, srcs = d
            if op.startswith(("mov.", "cvt.")):
                for a in srcs:
                    out |= origin(a, depth + 1)
            elif loc is not None:
                out.add(loc)
        return frozenset(out)

    def is_neg_zero(arg) -> bool:
        if arg in _NEG_ZERO:
            return True
        d = defs.get(arg)
        return d is not None and d[1].startswith("mov.") and len(d[2]) == 1 and d[2][0] in _NEG_ZERO

    out = []
    for loc, op, dsts, srcs in instrs:
        is_mul = op.startswith("mul.") and re.search(r"\.(f32|f16|f16x2|bf16|bf16x2)$", op) is not None
        is_mul |= op.startswith("fma.") and len(srcs) == 3 and is_neg_zero(srcs[2])
        if is_mul and len(srcs) >= 2:
            out.append((origin(srcs[0]), origin(srcs[1])))
    return out


def ptx_product_choices(module: TModule, ptx: str, plan: dict) -> dict:
    """For adds / subs with a product on both sides, read LLVM's choice from the PTX.

    Dataflow evidence first: a standalone multiply whose multiplicands come from the source locations of one
    product's operands is that product, not folded -- so the other one was (LLVM can give the standalone multiply
    the first product's column whichever product it keeps, so the location alone can mislead).  Otherwise (operands
    without distinct locations), location evidence: a product whose only use is this add and which still appears as
    a standalone multiply at its own source location was not folded.  Returns node_id -> "swap" / "default"; nodes
    without conclusive evidence are left out."""

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
    producer = {r: o for o in ops for r in o.results}
    multiplies = ptx_dataflow_multiplies(ptx)

    def operand_locs(mul):
        locs = []
        for v in mul.operands:
            o = producer.get(v)
            while o is not None and o.name in LAYOUT_OPS:
                o = producer.get(o.operands[0])
            locs.append(None if o is None else loc_of.get(o.node_id))
        return locs

    def materialized_by_dataflow(mul, other) -> Optional[bool]:
        mine, theirs = operand_locs(mul), operand_locs(other)
        if None in mine or None in theirs or set(mine) == set(theirs):
            return None
        seen = False
        for o1, o2 in multiplies:
            src = o1 | o2
            if all(l in src for l in mine) and not all(l in src for l in theirs):
                seen = True
        return seen

    def materialized(mul) -> Optional[bool]:
        loc = loc_of.get(mul.node_id)
        if loc is None or loc not in by_loc or shared[loc] != 1 or uses[mul.results[0]] != 1:
            return None
        return any(t.startswith("mul.") and (".f32" in t or ".f16" in t) for t in by_loc[loc])

    out = {}
    for node, (idx, mul, alt) in plan.items():
        if alt is None:
            continue
        d_def, d_alt = materialized_by_dataflow(mul, alt[1]), materialized_by_dataflow(alt[1], mul)
        if d_def is not None and d_alt is not None and d_def != d_alt:
            out[node] = "swap" if d_def else "default"
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
        # an in-memory capture keeps its compiled kernel: disassemble its cubin from the cache directory
        kernel = getattr(launch, "kernel", None)
        cubin = kernel.asm.get("cubin") if kernel is not None and hasattr(kernel, "asm") else None
        if not isinstance(cubin, (bytes, bytearray)):
            return None
        target = default_cache_dir() / "sass" / f"{hashlib.sha256(cubin).hexdigest()[:24]}.cubin"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(cubin)
        cubins = [target]
    else:
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


def _zero_sign(out: TV, kind: str, ops: list, mode: str = "rtne") -> TV:
    """IEEE sign of a zero result: an exact zero sum of opposite-signed (or equal nonzero opposite)
    operands is +0 under round to nearest (and toward zero / +inf), -0 under round toward -inf, and -0
    when both addends are -0; a nonzero result that rounds to zero keeps its sign.  float64 arithmetic on the FP32 operands follows the same rule (the
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
    sign = np.copysign(1.0, exact)
    if mode == "rd":  # roundTowardNegative: an exact zero sum is -0 unless both addends are +0
        if kind == "add":
            first, second = xs[0], xs[1]
        elif kind == "sub":
            first, second = xs[0], -xs[1]
        else:
            first, second = xs[0] * xs[1], xs[2]
        both_plus = (first == 0) & (second == 0) & ~np.signbit(first) & ~np.signbit(second)
        sign = np.where(exact == 0, np.where(both_plus, 1.0, -1.0), sign)
    lo = np.where(z, np.copysign(0.0, sign), out.lo)
    hi = np.where(z, lo, out.hi)
    res = _ftv(out.elem, lo, hi, out.st, out.cond, out.reasons)
    res.d = out.d
    return res


_F64_MODES = {"rtne": "RoundToNearest", "rtz": "RoundToZero", "rd": "RoundDown", "ru": "RoundUp"}


_MPFR_FORMAT = {"f64": (53, -1073, 1024), "f32": (24, -148, 128), "f16": (11, -23, 16), "bf16": (8, -132, 128)}


def _f64_ieee(name: str, args: list, mode: str = "rtne", fmt: str = "f64") -> TV:
    """An IEEE binary64 / binary32 operation evaluated exactly (MPFR with the format's precision, exponent range and
    subnormals).  For binary64 the float64 interval arithmetic of rounding-check mode cannot decide the rounding (it
    would leave the element undecided, or claim the wrong neighbour in a directed mode); for binary32 in a directed
    mode the outward-rounded interval of an exact zero rounds to the smallest subnormal instead of zero."""

    import gmpy2

    shape = np.broadcast_shapes(*(a.shape for a in args))
    vals = []
    unknown = np.zeros(shape, dtype=bool)
    for a in args:  # NaN / +-inf operands are values MPFR takes as they are; unknown operands stop the element
        v = np.broadcast_to(np.asarray(a.lo, dtype=np.float64), shape)
        s_ = np.broadcast_to(a.st, shape)
        vals.append(np.where(s_ == ST_NAN, np.nan, np.where(s_ == ST_PINF, np.inf, np.where(s_ == ST_NINF, -np.inf, v))))
        unknown |= np.isin(s_, (ST_UNDEF, ST_NE))
        if a.hi is not None:  # an operand known only as an interval is not a point: no exact evaluation
            unknown |= np.broadcast_to((np.asarray(a.hi) != np.asarray(a.lo)) & (s_ == ST_OK), shape)
    st = np.where(unknown, ST_NE, ST_OK).astype(np.int8)
    out = np.zeros(shape)
    out_st = np.array(st, dtype=np.int8, copy=True)
    prec, emin, emax = _MPFR_FORMAT[fmt]
    ctx = gmpy2.context(precision=prec, emin=emin, emax=emax, subnormalize=True,
                        round=getattr(gmpy2, _F64_MODES[mode]))
    with gmpy2.local_context(ctx):
        for idx in np.ndindex(shape):
            if st[idx] != ST_OK:
                continue
            x = [gmpy2.mpfr(float(v[idx])) for v in vals]
            if name == "add":
                r = x[0] + x[1]
            elif name == "sub":
                r = x[0] - x[1]
            elif name == "mul":
                r = x[0] * x[1]
            elif name == "div":
                r = x[0] / x[1]
            elif name == "rcp":
                r = 1 / x[0]
            elif name == "sqrt":
                r = gmpy2.sqrt(x[0])
            else:
                r = gmpy2.fma(x[0], x[1], x[2])
            f = float(r)
            if f != f:
                out_st[idx] = ST_NAN
            elif f == np.inf:
                out_st[idx] = ST_PINF
            elif f == -np.inf:
                out_st[idx] = ST_NINF
            else:
                out[idx] = f
    return _ftv(fmt, out, out.copy(), out_st, _merge_cond(*args), _merge_reasons(*args))


def _all_plus_zero(v: TV) -> bool:
    return (v.kind == "f" and bool(np.all(v.st == ST_OK)) and bool(np.all(v.lo == 0))
            and not bool(np.any(np.signbit(v.lo))) and (v.hi is None or bool(np.all(v.hi == 0))))


def _flush_subnormal(v: TV) -> TV:
    """Values below the fp32 normal range to zero of the same sign (both enclosure endpoints)."""

    if v.kind != "f":
        return v
    tiny = np.finfo(np.float32).tiny

    def flush(x):
        x = np.asarray(x, dtype=np.float64)
        return np.where((x != 0) & (np.abs(x) < tiny), np.copysign(0.0, x), x)
    out = TV(v.kind, v.elem, flush(v.lo), None if v.hi is None else flush(v.hi), v.base, v.st, v.cond, v.reasons, v.d)
    return out


class GpuEmulator(KernelReferenceEvaluator):
    """Rounding-check evaluation extended with the lowering rules of the locked build (see module doc)."""

    def __init__(self, module: TModule, ttgir: str, enable_fp_fusion: bool = True, exact_nodes=(),
                 oracle: Optional[HardwareOracle] = None, func_name: Optional[str] = None, ungrouped=(),
                 swapped=(), substitution: str = "mid", ptx: Optional[str] = None, sass: Optional[str] = None,
                 enable_reflect_ftz: bool = True):
        super().__init__(module, mode=NumericMode.ROUNDING_CHECK, func_name=func_name, ttgir=ttgir)
        if substitution not in SUBSTITUTIONS:
            raise ValueError(f"substitution must be one of {SUBSTITUTIONS}")
        self.exact_nodes = set(exact_nodes)
        self.substitution = substitution
        self.ungrouped = set(ungrouped)  # add / sub nodes whose approximate producer is not contracted
        self.oracle = oracle or HardwareOracle()
        self.layouts = reduce_layouts(module, ttgir)
        self.scan_layouts = scan_layouts(module, ttgir)
        self.enable_fp_fusion = enable_fp_fusion
        self.reflect_ftz = enable_reflect_ftz
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
        if elem not in iv.FLOAT_FORMATS:  # an integer result: an integer value with the not-established status
            return TV("i", elem, np.zeros(shape, dtype=np.int64), None, None, st, np.zeros(shape, dtype=bool),
                      frozenset({f"not_emulable:{what}@{op.node_id}"}))
        z = np.zeros(shape)
        return _ftv(elem, z, z.copy(), st, np.zeros(shape, dtype=bool), frozenset({f"not_emulable:{what}@{op.node_id}"}))

    # -- elementwise --------------------------------------------------------------

    def _float_op(self, name, op, args, out_elem):
        node = op.node_id
        if node in self.fusion:
            idx, mul = self.fusion[node]
            a, b = (self._env[v] for v in mul.operands)
            c = args[1 - idx]
            if op.name == "arith.addf" and node not in self.exact_nodes:
                # An accumulator that starts at +0 (Inductor's reductions): LLVM folds 0 + x*x to x*x -- a square is
                # never -0 -- and the unrounded square stays in a register.  Consumers that do not contract see
                # RN(x*x) (the value kept here); a later add or the reduction can fuse the square (raw_square).
                if mul.operands[0] == mul.operands[1] and _all_plus_zero(c):
                    self.trace[(node, "folded_zero_add")] += 1
                    sq = self._env[mul.results[0]]
                    out = TV(sq.kind, sq.elem, sq.lo.copy(), None if sq.hi is None else sq.hi.copy(), None,
                             sq.st.copy(), sq.cond.copy(), sq.reasons)
                    out.raw_square = a
                    return out
                # The other operand is such an unrounded square and stands on the left: LLVM fuses that product and
                # rounds this one (the left one of two single-use products).
                raw = getattr(c, "raw_square", None)
                if raw is not None and idx == 1:
                    self.trace[(node, "fma_contraction_raw_square")] += 1
                    fused = TOp("math.fma", op.results, op.result_types, ["%x", "%x", mul.results[0]], op.attrs,
                                [], op.line, op.text, op.operand_types)
                    rn_product = self._env[mul.results[0]]
                    return _zero_sign(super()._float_op("fma", fused, [raw, raw, rn_product], out_elem), "fma",
                                      [raw, raw, rn_product])
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
            if out_elem == "f64":
                return _f64_ieee("fma", [a, b, c])
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
        if (op.name == "tt.fp_to_fp" and (out_elem == "f8E5M2" or any(a.elem == "f8E5M2" for a in args))) \
                or op.name == "tt.clampf":
            # lowerings that are not the declared IEEE operation: the software fp8 e5m2 conversion of sm_86 rounds
            # ties away from zero (and goes through an RZ conversion to f16 from f32); tt.clampf's min / max
            # instructions have their own signed-zero rule -- the device evaluates them on the emulated operands
            return self._oracle(name, op, args, out_elem)
        self.trace[(node, "fused_product" if node in self.fused_products else "ieee")] += 1
        symbol = op.attrs.get("symbol", "").strip('"') if op.name == "tt.extern_elementwise" else ""
        ftz = self.reflect_ftz and (
            (op.name in ("math.floor", "math.ceil") and out_elem in ("f32", "bf16"))
            or (symbol in LIBDEVICE_ROUNDING and out_elem == "f32"))
        if ftz:  # libdevice's FTZ path (cvt.rmi.ftz, add.rn.ftz, ...): subnormal operands and results to signed zero
            self.trace[(node, "ftz")] += 1
            args = [_flush_subnormal(a) for a in args]
        if name in ("add", "sub", "mul", "div", "rcp", "sqrt", "fma") and all(a.kind == "f" for a in args) and (
                out_elem == "f64" or (out_elem == "f32" and symbol in LIBDEVICE_ROUNDING)):
            self.trace[(node, f"{out_elem}_exact")] += 1
            out = _f64_ieee(name, args, LIBDEVICE_ROUNDING.get(symbol, "rtne"), out_elem)
            return _flush_subnormal(out) if ftz else out
        out = super()._float_op(name, op, args, out_elem)
        if name in ("add", "sub", "fma"):
            out = _zero_sign(out, name, args, LIBDEVICE_ROUNDING.get(symbol, "rtne"))
        if ftz:
            out = _flush_subnormal(out)
        return out

    def _op_select(self, op, args, env, state):
        out = self._elementwise("select", op, args)
        raw = getattr(args[1], "raw_square", None)
        c = args[0]
        if raw is not None and np.all(c.st == ST_OK) and np.all(c.lo == 1):   # every lane takes the unrounded square
            out.raw_square = raw
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
            # Only a single-use result: with other uses the rounded quotient is kept and the add stays plain
            # (ptxas, for div.full).  A libdevice call is inlined LLVM IR: LLVM's aggressive fusion on NVPTX folds
            # its final multiply into every add that consumes it, whatever the use count (composition study: a
            # fast_logf result added to itself and reused in a loop).
            single = prod is not None and (
                prod.name == "tt.extern_elementwise"
                or (all(self._uses[c.results[0]] == 1 for c in chain) and self._uses[prod.results[0]] == 1))
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
            # NaN / +-inf operands are known values and run on the device as such (an all-or-nothing check would
            # drop the whole program to a plain add when one lane holds a NaN); unknown values still stop the group
            if np.isin(t.st, (ST_UNDEF, ST_NE)).any():
                return None
            vals = t.lo
            if t.kind == "f":
                special = t.st != ST_OK
                vals = np.where(t.st == ST_NAN, np.nan, np.where(t.st == ST_PINF, np.inf,
                                                                 np.where(t.st == ST_NINF, -np.inf, t.lo)))
                if t.elem not in iv.FLOAT_FORMATS or not np.all(_fp32_representable(t.lo, t.elem) | special):
                    return None
            elif (t.st != ST_OK).any():
                return None
            arrays[v] = vals
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

    def _int_to_float(self, name, op, x, out_elem):
        """Integer to float rounded exactly (MPFR): the evaluator's float64 path leaves an interval for integers
        beyond 2^53 (and would round twice for narrow formats)."""
        if out_elem not in _MPFR_FORMAT or op.node_id in self.exact_nodes:
            return super()._int_to_float(name, op, x, out_elem)
        import gmpy2
        self.trace[(op.node_id, "int_to_float_exact")] += 1
        width = INT_WIDTH[x.elem]
        ints = np.asarray(x.lo).astype(np.int64)
        prec, emin, emax = _MPFR_FORMAT[out_elem]
        out = np.zeros(ints.shape)
        st = np.where(x.st == ST_OK, ST_OK, ST_NE).astype(np.int8)
        ctx = gmpy2.context(precision=prec, emin=emin, emax=emax, subnormalize=True, round=gmpy2.RoundToNearest)
        with gmpy2.local_context(ctx):
            for idx in np.ndindex(ints.shape):
                if st[idx] != ST_OK:
                    continue
                v = int(ints[idx])
                if name == "uitofp":
                    v &= (1 << width) - 1
                f = float(gmpy2.mpfr(v))
                if f in (np.inf, -np.inf):
                    st[idx] = ST_PINF if f > 0 else ST_NINF
                else:
                    out[idx] = f
        return _ftv(out_elem, out, out.copy(), st, x.cond, x.reasons)

    def _op_extern(self, op, args, env, state):
        symbol = op.attrs.get("symbol", "").strip('"')
        internal = LIBDEVICE.get(symbol)
        out_elem = op.result_types[0].elem
        if internal is None and op.node_id not in self.exact_nodes and len(op.results) == 1:
            # no declared semantics (a reference cannot be formed), but bitwise emulation needs none: the device
            # evaluates the call on the emulated operands (deterministic, so equal operands give equal results)
            return self._oracle_any(op, args, out_elem)
        if symbol in LIBDEVICE_ROUNDING or internal is None or out_elem not in iv.FLOAT_FORMATS:
            return super()._op_extern(op, args, env, state)
        return self._float_op(internal, op, args, out_elem)

    def _oracle_any(self, op, args, out_elem) -> TV:
        """Run one operation on the device on the emulated operands; float or integer result."""
        shape = np.broadcast_shapes(*(a.shape for a in args))
        # lanes with an unknown operand (undefined, not established) are not emulated; the others run as they are
        unknown = np.zeros(shape, dtype=bool)
        for a in args:
            unknown |= np.broadcast_to(np.isin(a.st, (ST_UNDEF, ST_NE)), shape)
        if out_elem not in (*iv.FLOAT_FORMATS, "i8", "i16", "i32", "i64"):
            return self._not_emulable(op, list(shape), out_elem, args, f"device evaluation of {out_elem}")
        self.trace[(op.node_id, "oracle_no_semantics")] += 1
        inputs = []
        for a in args:
            v = np.broadcast_to(a.lo, shape)
            if a.kind == "f":
                v = np.where(a.st == ST_NAN, np.nan, np.where(a.st == ST_PINF, np.inf,
                                                             np.where(a.st == ST_NINF, -np.inf, v)))
            inputs.append(np.where(unknown, 0, v))
        hw = np.asarray(self.oracle(op, inputs, out_elem))
        if out_elem in iv.FLOAT_FORMATS:
            hw = hw.astype(np.float64)
            st = np.where(unknown, ST_NE, np.where(np.isnan(hw), ST_NAN, np.where(hw == np.inf, ST_PINF,
                                                                             np.where(hw == -np.inf, ST_NINF, ST_OK))))
            st = st.astype(np.int8)
            vals = np.where(st == ST_OK, hw, 0.0)
            return _ftv(out_elem, vals, vals.copy(), st, _merge_cond(*args), _merge_reasons(*args))
        st = np.where(unknown, ST_NE, ST_OK).astype(np.int8)
        return TV("i", out_elem, np.where(unknown, 0, hw).astype(np.int64), None, None, st,
                  _merge_cond(*args), _merge_reasons(*args))

    def _op_inline_asm(self, op, args, env, state):
        internal = inline_asm_internal(op.attrs.get("asm", "")) or "identity"
        return self._float_op(internal, op, args, op.result_types[0].elem)

    # -- reductions, dot, scan, atomics -------------------------------------------------

    @staticmethod
    def _single_op_region(op):
        """(inner operation, [first block argument, second block argument]) of a combine region made of one
        operation on the two block arguments, else None."""
        block = op.regions[0].entry
        names = [a[0] for a in block.args]
        ops = [o for o in block.ops if o.name not in ("tt.reduce.return", "tt.scan.return")]
        if len(names) != 2 or len(ops) != 1 or len(ops[0].results) != 1:
            return None
        inner = ops[0]
        if sorted(set(inner.operands)) != sorted(set(names)) or block.ops[-1].operands != inner.results:
            return None
        return inner, names

    # integer combines with undefined results (division by zero, shifts by the width or more): LLVM may fold the
    # undefined lanes differently in the kernel and in a separately compiled combine, so they are not emulated
    _UB_INT_COMBINES = {"arith.divsi", "arith.divui", "arith.remsi", "arith.remui", "arith.ceildivsi", "arith.ceildivui",
                        "arith.floordivsi", "arith.shli", "arith.shrsi", "arith.shrui"}

    def _device_combine(self, inner, names, elem):
        """combine(acc, cur) evaluated on the device elementwise (the region's operation; acc is the first block
        argument, cur the second, as the lowering applies them)."""
        def combine(a, b):
            shape = np.broadcast_shapes(np.shape(a), np.shape(b))
            a, b = np.broadcast_to(a, shape), np.broadcast_to(b, shape)
            bind = {names[0]: a, names[1]: b}
            inputs = [bind[v] for v in inner.operands]
            flat = [np.ascontiguousarray(v).reshape(-1) for v in inputs]
            out = self.oracle(inner, flat, elem)
            return np.asarray(out, dtype=np.int64 if elem in ("i8", "i16", "i32", "i64") else np.float64).reshape(shape)
        return combine

    def _reduce_tree(self, vals, lay, combine):
        """The layout order of the sum path for any combine: sequential in a thread, butterfly over lanes, butterfly
        over warps; returns (result, lane-divergent mask)."""
        n = vals.shape[-1]
        spt = min(lay["spt"], n)
        tpw = min(lay["tpw"], max(1, n // spt))
        wpc = min(lay["wpc"], max(1, n // (spt * tpw)))
        tile = spt * tpw * wpc
        reps = max(1, n // tile)
        idx = (np.arange(reps)[:, None, None, None] * tile + np.arange(wpc)[None, :, None, None] * (spt * tpw)
               + np.arange(tpw)[None, None, :, None] * spt + np.arange(spt)[None, None, None, :])
        order = [idx[rep, :, :, j] for rep in range(reps) for j in range(spt)]
        acc = vals[..., order[0]]
        for pos in order[1:]:
            acc = combine(acc, vals[..., pos])
        lane_of = np.arange(tpw)
        stride = tpw // 2
        while stride >= 1:
            acc = combine(acc, acc[..., lane_of ^ stride])
            stride //= 2
        if wpc > 1:
            part = acc[..., 0]
            stride = wpc // 2
            while stride >= 1:
                part = combine(part, part[..., np.arange(wpc) ^ stride])
                stride //= 2
            return part[..., 0], np.zeros(part[..., 0].shape, dtype=bool)
        lanes = acc[..., 0, :]
        res = lanes[..., 0]
        divergent = np.any((lanes != lanes[..., :1]) & ~(np.isnan(lanes) & np.isnan(lanes[..., :1])), axis=-1)
        return res, divergent

    def _reduce_generic(self, op, x, inner, names):
        """A float or (<= 32-bit) integer reduce whose region is one operation: the layout order with the combine
        evaluated on the device."""
        axis = int(op.attrs["axis"].split(":")[0])
        out_shape = list(np.delete(np.array(x.shape), axis))
        lay = self.layouts.get(op.node_id)
        if (lay is None or op.node_id in self.exact_nodes or inner.name in self._UB_INT_COMBINES
                or x.elem not in (*iv.FLOAT_FORMATS, "i8", "i16", "i32", "i64")):
            return self._not_emulable(op, out_shape, x.elem, [x], "reduce region")
        self.trace[(op.node_id, "reduce_tree_device_combine")] += 1
        ok = np.moveaxis(x.st == ST_OK, axis, -1)
        vals = np.moveaxis(np.where(x.st == ST_OK, x.lo, 0), axis, -1).astype(np.int64 if x.kind == "i" else np.float64)
        if "perm" in lay:                       # a flattening reshape: the source layout's register order
            vals = vals[..., lay["perm"]]
        res, divergent = self._reduce_tree(vals, lay, self._device_combine(inner, names, x.elem))
        bad = ~np.all(ok, axis=-1) | divergent
        if x.kind == "i":
            st = np.where(bad, ST_NE, ST_OK).astype(np.int8)
            return TV("i", x.elem, np.where(bad, 0, res).astype(np.int64), None, None, st,
                      np.any(x.cond, axis=axis), x.reasons)
        st = np.where(bad, ST_NE, np.where(np.isnan(res), ST_NAN, np.where(res == np.inf, ST_PINF,
                                                                        np.where(res == -np.inf, ST_NINF, ST_OK))))
        st = st.astype(np.int8)
        res = np.where(st == ST_OK, res, 0.0)
        return _ftv(x.elem, res, res.copy(), st, np.any(x.cond, axis=axis), x.reasons)

    def _op_reduce(self, op, args, env, state):
        combiner = recognize_combiner(op)
        if combiner == "welford":  # float Welford depends on the merge tree: not order-free, not emulable here
            raise ProgramAbort(f"{op.node_id}: Welford merge order is not modelled by the emulator")
        single = self._single_op_region(op) if len(args) == 1 else None
        if (combiner is None or combiner == "prod") and single is not None:
            # one operation in the region: the sum path's layout order with the device evaluating the combine
            return self._reduce_generic(op, args[0], *single)
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
        perm = lay.get("perm")
        if perm is not None:                    # a flattening reshape: the source layout's register order
            vals = vals[..., perm]
        n = vals.shape[-1]
        spt = min(lay["spt"], n)
        tpw = min(lay["tpw"], max(1, n // spt))
        wpc = min(lay["wpc"], max(1, n // (spt * tpw)))
        tile = spt * tpw * wpc
        reps = max(1, n // tile)
        fmt = x.elem

        def rn(v):  # an overflowing partial becomes inf (and may meet -inf): outside this model, marked undecided
            r, overflow = iv.round_nearest_even(v, fmt)
            return np.where(overflow | ~np.isfinite(v), np.nan, r)

        def fma(a, b, c):  # RN(a*b + c), or RN(a + c) when b is None
            lo, hi = iv.ifma(a, a, b, b, c, c) if b is not None else iv.iadd(a, a, c, c)
            r_lo, r_hi = rn(lo), rn(hi)
            r = np.where(r_lo == r_hi, r_lo, np.nan)
            return np.where(r == 0, np.copysign(0.0, (a * b if b is not None else a) + c), r)

        # Contraction into the combine: the operand is a product defined in the same block, so the
        # first combine that sees a raw product is an fma (LLVM folds the product with fewer uses,
        # the left one on a tie; a product also sent through a shuffle still folds).
        producer = self._def.get(op.operands[0])
        # a reshape with allow_reorder (or a layout conversion) of the same shape keeps the product in its register:
        # LLVM still folds it into the combine (composition study: Liger's group norm sums X * X this way)
        while (producer is not None and producer.name in ("tt.reshape", "ttg.convert_layout")
               and self._env.get(producer.operands[0]) is not None
               and (self._env[producer.operands[0]].shape == x.shape
                    or (perm is not None
                        and int(np.prod(self._env[producer.operands[0]].shape)) == int(np.prod(x.shape))))):
            # a flattening reshape (allow_reorder) keeps the values in their registers as well
            producer = self._def.get(producer.operands[0])
        # sum(where(c, x*y, 0)) with c the same along the reduced axis (Inductor's row mask): LLVM moves the select
        # after the in-thread partial sum, so the products are still contracted into it; masked rows give 0
        hoisted_cond = None
        if (producer is not None and producer.name == "arith.select" and self.enable_fp_fusion
                and op.node_id not in self.exact_nodes):
            c_tv, f_tv = self._env.get(producer.operands[0]), self._env.get(producer.operands[2])
            if c_tv is not None and f_tv is not None and np.all(c_tv.st == ST_OK) and _all_plus_zero(f_tv):
                cv = np.moveaxis(np.broadcast_to(c_tv.lo, x.shape), axis, -1)
                if np.all(cv == cv[..., :1]):
                    hoisted_cond = cv[..., 0] == 1
                    producer = self._def.get(producer.operands[1])
                    while (producer is not None and producer.name in ("tt.broadcast", "tt.reshape", "ttg.convert_layout")
                           and self._env.get(producer.operands[0]) is not None
                           and int(np.prod(self._env[producer.operands[0]].shape)) == int(np.prod(x.shape))):
                        producer = self._def.get(producer.operands[0])
        fused = (self.enable_fp_fusion and producer is not None and producer.name == "arith.mulf"
                 and self._block.get(id(producer)) == self._block.get(id(op)) and op.node_id not in self.exact_nodes)
        if hoisted_cond is not None and not fused:
            hoisted_cond = None                 # no product behind the select: the plain path, as before
        raw = getattr(x, "raw_square", None) if (self.enable_fp_fusion and op.node_id not in self.exact_nodes) else None
        if fused:
            pshape = tuple(producer.result_types[0].shape)
            fa, fb = (np.moveaxis(np.broadcast_to(np.where(self._env[v].st == ST_OK, self._env[v].lo, 0.0),
                                                  pshape).reshape(x.shape), axis, -1) for v in producer.operands)
            if perm is not None:
                fa, fb = fa[..., perm], fb[..., perm]
        elif raw is not None:   # the operand holds unrounded squares (folded 0 + x*x of a loop that ran once)
            fused = True
            fa = np.moveaxis(np.broadcast_to(np.where(raw.st == ST_OK, raw.lo, 0.0), x.shape), axis, -1)
            if perm is not None:
                fa = fa[..., perm]
            fb = fa
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
        if hoisted_cond is not None:
            res = np.where(hoisted_cond, res, 0.0)
            divergent = divergent & hoisted_cond
        bad = np.any(np.moveaxis(x.st != ST_OK, axis, -1), axis=-1) | np.isnan(res) | divergent
        st = np.where(bad, ST_NE, ST_OK).astype(np.int8)
        res = np.where(bad, 0.0, res)
        reasons = x.reasons | ({f"not_emulable:lane-divergent reduction@{op.node_id}"} if divergent.any() else set())
        return _ftv(x.elem, res, res.copy(), st, np.any(x.cond, axis=axis), frozenset(reasons))

    def _op_dot(self, op, args, env, state):
        a, b, c = args
        precision = op.attrs.get("inputPrecision", "ieee").strip()  # the default (ieee) is not printed
        shape = np.broadcast_shapes(c.shape)
        tc = tensor_core.mma_format(a.elem, b.elem, c.elem, precision)
        if tc is None and not (a.elem == b.elem == c.elem == "f32" and precision.startswith("ieee")):
            return self._not_emulable(op, list(shape), c.elem, [a, b, c], f"tensor-core dot {a.elem} x {b.elem} -> {c.elem} (inputPrecision {precision})")
        if op.node_id in self.exact_nodes:
            self.trace[(op.node_id, "exact")] += 1
            lo, hi = iv.idot(a.lo, a.hi, b.lo, b.hi)
            lo, hi = iv.iadd(lo, hi, c.lo, c.hi)
            return self._substitute(c.elem, lo, hi, _merge_status(c).copy(), c.cond, c.reasons)
        if tc is not None:  # tensor-core rule of the locked build (tensor_core.py)
            self.trace[(op.node_id, f"dot_tensor_core_{tc}")] += 1
            acc, covered = tensor_core.mma_chain(a.lo, b.lo, c.lo, tc)
            bad = (~covered | (c.st != ST_OK) | np.any(a.st != ST_OK, axis=-1)[..., :, None]
                   | np.any(b.st != ST_OK, axis=-2)[..., None, :])
            st = np.where(bad, ST_NE, ST_OK).astype(np.int8)
            acc = np.where(bad, 0.0, acc)
            return _ftv(c.elem, acc, acc.copy(), st, c.cond | np.any(a.cond, axis=-1)[..., :, None], c.reasons)
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

    ORDER_FREE = ("sum_int", "prod_int", "max_int", "min_int", "max_uint", "min_uint", "and", "or", "xor",
                  "max_num", "min_num", "max_nan", "min_nan")

    def _op_scan(self, op, args, env, state):
        x = args[0]
        combiner = recognize_combiner(op)
        if len(args) == 1 and combiner in self.ORDER_FREE:
            # Prefer the existing explicit lowering-order model for supported
            # integer scans. The generic reference may only assume associativity
            # when its optional proof engine is unavailable; that assumption
            # must not be silently accepted by bitwise verification.
            info = self.scan_layouts.get(op.node_id)
            single = self._single_op_region(op)
            if (x.kind == "i" and info is not None and single is not None
                    and single[0].name not in self._UB_INT_COMBINES
                    and len(x.shape) == 1 and x.shape[0] > 0 and info["axis"] == 0
                    and not info["reverse"] and info["tpw"] == 32
                    and x.shape[0] % (info["spt"] * 32 * info["wpc"]) == 0
                    and op.node_id not in self.exact_nodes):
                return self._scan_device_combine(op, args)
            # associative and commutative without rounding (integers, float max / min): any order gives the
            # device's values, so the evaluator's own fold is exact
            mode, self.mode = self.mode, NumericMode.NUMERICAL_DIFFERENCE
            try:
                self.trace[(op.node_id, "scan_order_free")] += 1
                return super()._op_scan(op, args, env, state)
            finally:
                self.mode = mode
        if x.kind != "f" or (len(args) == 1 and combiner not in ("sum", "prod")):
            return self._scan_device_combine(op, args)
        info = self.scan_layouts.get(op.node_id)
        producer = self._def.get(op.operands[0])
        contracted = (self.enable_fp_fusion and producer is not None and producer.name == "arith.mulf"
                      and self._block.get(id(producer)) == self._block.get(id(op)))
        n = x.shape[0] if len(x.shape) == 1 else 0
        if (info is None or len(args) != 1 or len(x.shape) != 1 or info["axis"] != 0 or info["reverse"]
                or combiner not in ("sum", "prod") or info["tpw"] != 32 or contracted
                or n % (info["spt"] * 32 * info["wpc"])):
            return self._not_emulable(op, list(x.shape), x.elem, args, "scan")
        if op.node_id in self.exact_nodes:
            return self._not_emulable(op, list(x.shape), x.elem, args, "scan (exact node)")
        self.trace[(op.node_id, "scan_order")] += 1
        fmt = x.elem
        prod = combiner == "prod"

        def combine(a, b):  # RN(a + b) or RN(a * b); an overflow leaves the model (undecided)
            lo, hi = (iv.imul(a, a, b, b) if prod else iv.iadd(a, a, b, b))
            (r_lo, o_lo), (r_hi, o_hi) = iv.round_nearest_even(lo, fmt), iv.round_nearest_even(hi, fmt)
            r = np.where((r_lo == r_hi) & ~o_lo & ~o_hi & np.isfinite(lo) & np.isfinite(hi), r_lo, np.nan)
            return np.where(r == 0, np.copysign(0.0, a * b if prod else a + b), r)

        ok = x.st == ST_OK
        vals = np.where(ok, x.lo, 0.0)
        res = scan_order.scan_1d(vals, info["spt"], info["wpc"], combine)
        bad = np.isnan(res) | ~np.logical_and.accumulate(ok)   # an input outside the model spoils what follows it
        st = np.where(bad, ST_NE, ST_OK).astype(np.int8)
        res = np.where(bad, 0.0, res)
        return _ftv(fmt, res, res.copy(), st, x.cond, x.reasons)

    def _scan_device_combine(self, op, args):
        """A 1-D forward scan whose region is one operation (float, or integer up to 32 bits): the scan_order.py
        order with the device evaluating the combine."""
        x = args[0]
        info = self.scan_layouts.get(op.node_id)
        single = self._single_op_region(op) if len(args) == 1 else None
        n = x.shape[0] if len(x.shape) == 1 else 0
        if (single is None or single[0].name in self._UB_INT_COMBINES or info is None or len(x.shape) != 1
                or info["axis"] != 0 or info["reverse"]
                or info["tpw"] != 32 or n % (info["spt"] * 32 * info["wpc"]) or op.node_id in self.exact_nodes
                or x.elem not in (*iv.FLOAT_FORMATS, "i8", "i16", "i32", "i64")):
            return self._not_emulable(op, list(x.shape), x.elem, args, "scan")
        self.trace[(op.node_id, "scan_order_device_combine")] += 1
        ok = x.st == ST_OK
        vals = np.where(ok, x.lo, 0).astype(np.int64 if x.kind == "i" else np.float64)
        res = scan_order.scan_1d(vals, info["spt"], info["wpc"], self._device_combine(*single, x.elem))
        bad = ~np.logical_and.accumulate(ok)
        if x.kind == "i":
            st = np.where(bad, ST_NE, ST_OK).astype(np.int8)
            return TV("i", x.elem, np.where(bad, 0, res).astype(np.int64), None, None, st, x.cond, x.reasons)
        st = np.where(bad, ST_NE, np.where(np.isnan(res), ST_NAN, np.where(res == np.inf, ST_PINF,
                                                                        np.where(res == -np.inf, ST_NINF, ST_OK))))
        st = st.astype(np.int8)
        res = np.where(st == ST_OK, res, 0.0)
        return _ftv(x.elem, res, res.copy(), st, x.cond, x.reasons)

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


def _verification_outputs(result, launch):
    """Comparison masks for every written storage, including discrete outputs.

    Integers are already decoded in their storage representation by the IR
    evaluator. Compare those arrays directly: a float64 conversion loses i64
    differences above 2**53. Unknown/set-valued outputs remain unverified.
    """
    floating = _outputs(result, launch)
    for b in result.buffers.values():
        written = b.written.copy()
        if not written.any():
            continue
        if b.kind == "f":
            masks = _compare(*floating[b.name])
        else:
            exact = np.zeros(written.shape, dtype=bool)
            equal = exact.copy()
            actual = b.actual_after
            if (b.kind in ("i", "b") and actual is not None
                    and np.asarray(actual).shape == b.lo.shape
                    and np.asarray(actual).dtype.kind in "biu" and b.lo.dtype.kind in "biu"):
                exact = written & (b.st == ST_OK)
                if b.hi is not None:
                    exact &= b.lo == b.hi
                if b.kind == "b":
                    exact &= np.isin(b.lo, (0, 1))
                if getattr(b, "iset", None) is not None:
                    exact &= ~b.iset
                equal = exact & (b.lo == actual)
            masks = {"emulated": exact, "bit_equal": equal, "value_equal": equal.copy(),
                     "nan_both": np.zeros(written.shape, dtype=bool)}
        unconditional = ~np.asarray(b.cond, dtype=bool)
        if getattr(b, "premise", None) is not None:
            unconditional &= ~np.asarray(b.premise, dtype=bool)
        masks = {key: value & unconditional for key, value in masks.items()}
        yield b.name, written, masks


def _bitwise_count(result, launch) -> tuple:
    equal = emulated = 0
    for name, written, m in _verification_outputs(result, launch):
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
                      sass=launch_sass(launch),
                      enable_reflect_ftz=bool(launch.metadata.get("enable_reflect_ftz", True)))
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

    status: "bit_identical" (every written element is emulated and has the device's bits; NaN only as a class; no
    program stopped), "partial" (every emulated element agrees but some written elements are not emulated or a
    program stopped), "mismatch", or "not_established" (no emulated output, e.g. every program stopped at an
    atomic).
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
    for name, written, m in _verification_outputs(result, launch):
        entry = {"written": int(written.sum()), "emulated": int(m["emulated"].sum()),
                 "bit_identical": int(m["bit_equal"].sum()), "value_equal": int(m["value_equal"].sum()),
                 "nan_both": int(m["nan_both"].sum()), "not_emulated": int((written & ~m["emulated"]).sum())}
        entry["mismatch"] = entry["emulated"] - entry["bit_identical"] - entry["nan_both"]
        report["buffers"][name] = entry
        totals.update(entry)
    if totals["emulated"] == 0:
        report["status"] = "not_established"
    elif totals["mismatch"]:
        report["status"] = "mismatch"
    elif totals["not_emulated"] or result.aborted:
        # every emulated element agrees, but some written elements have no emulated value (no declared value, a
        # not-emulable node) or a program stopped: not a bitwise reproduction of the launch
        report["status"] = "partial"
    else:
        report["status"] = "bit_identical"
    report["bitwise_reproduced"] = report["status"] == "bit_identical"
    report["fitted_to_output"] = bool(fitted_u or fitted_s)
    return report


def rounding_nodes(emu: GpuEmulator) -> list:
    """Nodes whose execution rounds or approximates (candidates for localization), in program order."""

    seen = []
    for (node, how) in emu.trace:
        rounds = (how in ("ieee", "oracle", "oracle_group", "fma_contraction", "ftz", "int_to_float_exact")
                  or how.startswith(("reduce_tree", "dot_", "scan_order")) or how.endswith("_exact"))
        # not candidates: nodes already exact, a product folded into an fma (no rounding of its own), a device
        # evaluation without declared semantics (no exact value to substitute), an order-free scan (exact)
        if rounds and how != "scan_order_free" and node not in seen:
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
