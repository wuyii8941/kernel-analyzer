"""Per-signature evidence for the rule contracts (DSL v2 rc3 04 W1): positive, boundary and premise-violation cases.

A one-op elementwise kernel is generated per signature, compiled for sm_86 without a launch, and evaluated on synthetic
captures (CPU only).  Every lane is checked against an exact value computed with mpmath (200 bits) or exact integers:

* positive: inputs inside the domain; every lane complete and its enclosure contains the exact value;
* boundary: signed zeros, subnormals, the largest finite values, infinities, NaN, domain end points; a lane may be
  special or not established, but a lane reported complete must enclose the exact value and a special class must be
  the exact one;
* premise violation: inputs outside the domain; no lane may be reported as a complete finite value.

Helper module, not a test file.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import importlib.util
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KDIR = ROOT / ".cache" / "tmp" / "sigkernels"
N = 64
mp.mp.prec = 200

ST_OK, ST_NAN, ST_PINF, ST_NINF, ST_UNDEF, ST_NE = 0, 1, 2, 3, 4, 5
NP = {"fp32": np.float32, "fp64": np.float64, "int32": np.int32, "int64": np.int64, "uint32": np.uint32,
      "uint64": np.uint64, "fp16": np.float16, "int8": np.int8}
SIG = {"fp32": "*fp32", "fp64": "*fp64", "int32": "*i32", "int64": "*i64", "uint32": "*u32", "uint64": "*u64",
       "fp16": "*fp16", "int8": "*i8"}
TORCH = {"fp32": "float32", "fp64": "float64", "int32": "int32", "int64": "int64", "uint32": "uint32",
         "uint64": "uint64", "fp16": "float16", "int8": "int8"}


def libdevice_table() -> dict:
    """symbol -> (python function name, argument dtypes, result dtype), read from the installed 3.6.0 source."""
    import triton.language.extra.cuda.libdevice as lib
    tree = ast.parse(Path(lib.__file__).read_text())
    out = {}
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    try:
                        args = tuple(e.args[0].value for e in k.elts)
                        sym, ret = v.elts[0].value, v.elts[1].args[0].value
                    except (AttributeError, IndexError):
                        continue
                    out[sym] = (fn.name, args, ret)
    return out


def kernel_module(name: str, expr: str, in_dtypes: tuple, out_dtype: str):
    """Write and import a one-op kernel: o = EXPR(a0, a1, a2)."""
    KDIR.mkdir(parents=True, exist_ok=True)
    loads = "\n".join(f"    a{i} = tl.load(p{i} + i)" for i in range(len(in_dtypes)))
    params = ", ".join(f"p{i}" for i in range(len(in_dtypes)))
    src = f"""import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def kernel({params}, out, N: tl.constexpr):
    i = tl.arange(0, N)
{loads}
    o = {expr}
    tl.store(out + i, o)
"""
    digest = hashlib.sha256(src.encode()).hexdigest()[:12]
    path = KDIR / f"k_{name}_{digest}.py"
    if not path.exists():
        path.write_text(src)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=None)
def compiled(name: str, expr: str, in_dtypes: tuple, out_dtype: str):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    mod = kernel_module(name, expr, in_dtypes, out_dtype)
    sig = {f"p{i}": SIG[d] for i, d in enumerate(in_dtypes)}
    sig.update({"out": SIG[out_dtype], "N": "constexpr"})
    ck = triton.compile(ASTSource(fn=mod.kernel, signature=sig, constexprs={"N": N}), target=GPUTarget("cuda", 86, 32),
                        options={"num_warps": 1})
    return {k: ck.asm[k] for k in ("ttir", "ttgir", "ptx")}


def evaluate(name, expr, in_dtypes, out_dtype, inputs, rename=None):
    """inputs: list of arrays (len N) -> (lo, hi, st) of the reference output and the executed op names.  ``rename``
    = (old, new) replaces a text in the compiled TTIR / TTGIR (carrier kernels for symbols the frontend does not
    expose; evaluation only)."""
    from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence
    asm = dict(compiled(name if rename is None else f"carrier_{expr}", expr, tuple(in_dtypes), out_dtype))
    if rename is not None:
        assert rename[0] in asm["ttir"], rename
        asm = {k: v.replace(rename[0], rename[1]) for k, v in asm.items()}
    args, base = [], 1 << 20
    arrays = [np.asarray(a, NP[d]) for a, d in zip(inputs, in_dtypes)] + [np.zeros(N, NP[out_dtype])]
    dts = list(in_dtypes) + [out_dtype]
    for i, (arr, d) in enumerate(zip(arrays, dts)):
        raw = np.ascontiguousarray(arr).view(np.uint8).copy()
        args.append(CapturedArg(index=i, name=f"p{i}" if i < len(in_dtypes) else "out", kind="tensor", constexpr=False,
                                signature_type=SIG[d], dtype=TORCH[d], shape=arr.shape, stride=None,
                                element_size=arr.itemsize, data_ptr=base * (i + 1), storage_ptr=base * (i + 1),
                                storage_nbytes=raw.size, storage_id=i, before=raw, after=raw.copy()))
    args.append(CapturedArg(index=len(args), name="N", kind="int", constexpr=True, signature_type="constexpr", value=N))
    launch = CapturedLaunch(index=0, kernel_name=name, kernel_hash="", grid=(1, 1, 1), args=args, asm=asm,
                            cubin_sha256=None, metadata={}, libtriton_sha256=None)
    ref = evaluate_sequence([launch]).launches[0]
    out = ref.buffers[base * len(dts)]
    hi = out.hi if out.hi is not None else out.lo
    return np.asarray(out.lo), np.asarray(hi), np.asarray(out.st), ref


def pad(values, n=N):
    v = list(values)
    return (v * (n // len(v) + 1))[:n]


def exact_class(x):
    """mpmath value or None (undefined) -> (status, value)."""
    if x is None:
        return ST_UNDEF, None
    if isinstance(x, mp.mpc) and x.imag != 0:
        return ST_UNDEF, None
    if isinstance(x, mp.mpc):
        x = x.real
    if isinstance(x, (int,)):
        return ST_OK, x
    if mp.isnan(x):
        return ST_NAN, None
    if mp.isinf(x):
        return (ST_PINF if x > 0 else ST_NINF), None
    return ST_OK, x


def check_lanes(category, lo, hi, st, exact, is_int=False):
    """Returns a list of violations (empty when the lanes satisfy the category's contract)."""
    bad = []
    for j, ex in enumerate(exact):
        cls, val = exact_class(ex)
        s = int(st[j])
        if s == ST_OK:
            if cls != ST_OK:
                bad.append(f"lane {j}: complete [{lo[j]}, {hi[j]}] but exact is {'undefined' if val is None else cls}")
            elif is_int:
                if int(lo[j]) != int(val):
                    bad.append(f"lane {j}: {int(lo[j])} != exact {val}")
            elif not (mp.mpf(float(lo[j])) <= val <= mp.mpf(float(hi[j]))):
                bad.append(f"lane {j}: [{lo[j]!r}, {hi[j]!r}] does not contain {mp.nstr(val, 20)}")
        elif s in (ST_NAN, ST_PINF, ST_NINF):
            if cls != s and not (s == ST_NAN and cls == ST_UNDEF):  # IEEE NaN for a real value that does not exist
                bad.append(f"lane {j}: special class {s} but exact class {cls}")
        if category == "positive" and s != ST_OK:
            bad.append(f"lane {j}: positive case not complete (status {s})")
        if category == "premise_violation" and s == ST_OK and cls != ST_OK:
            bad.append(f"lane {j}: premise violation reported complete")
    return bad
