"""The universe of primitive instances of the catalog (RQ1) and their declared semantics.

Sources:
  F1         the 705 operation instances of the package (space.F1), structural ones excluded here;
  libdevice  every (function, signature) of triton.language.extra.cuda.libdevice: Triton's frontend lowers the
             elementary functions the backend does not legalize (tanh, erfc, pow, ...) to these calls, so each one is a
             numerical primitive of its own although F1 counts them all as the single name tt.extern_elementwise;
  literal    decimal float literals of arith.constant (the declared constant is the decimal; the program holds its
             rounding);
  scalar     arith.index_cast / index_castui (scalar only).

``Prim`` records what the answer machine needs: operand and result types, the semantic name in refs.py, the rounding
mode of the declared result, and how operands are drawn (domain).
"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field

import space

LIBDEVICE_TYPES = {"fp32": "f32", "fp64": "f64", "int32": "i32", "int64": "i64", "uint32": "u32", "uint64": "u64",
                   "int1": "i1", "fp16": "f16", "bf16": "bf16"}


@dataclass
class Prim:
    pid: str
    source: str
    op: str
    attr: str
    sig: str
    ins: list
    outs: list
    sem: str                    # semantic name (refs.REAL / integer table / conversion / predicate ...)
    family: str                 # real | int | pred | convert_ff | convert_fi | convert_if | bits | select | dot | literal | scalar
    mode: str = "rne"
    domain: tuple = ()
    extra: dict = field(default_factory=dict)


def _op_sem(op):
    return op.split(".", 1)[1]


def f1_prims() -> list:
    out = []
    for inst in space.F1:
        op, attr, sig = inst
        k = space.kind(inst)
        if k == "structural":
            continue
        pid = f"F1|{op}|{attr}|{sig}"
        if k == "float":
            if op in ("arith.maximumf", "arith.minimumf", "arith.maxnumf", "arith.minnumf", "math.copysign",
                      "arith.negf", "math.absf", "math.ceil", "math.floor", "math.round", "math.roundeven",
                      "math.trunc"):
                fam = "real"
            else:
                fam = "real"
            nin = 3 if op == "math.fma" else (2 if (op in space.FLOAT_BIN or op in ("arith.divf", "tt.precise_divf"))
                                              else (2 if op == "math.fpowi" else 1))
            ins = [sig] * nin
            sem = _op_sem(op)
            if op in ("tt.clampf", "math.clampf"):
                ins, sem = [sig] * 3, ("clampf_nan" if attr.endswith("true") else "clampf")
            if op == "math.fpowi":
                ins, sem = [sig, "i32"], "powi"
            out.append(Prim(pid, "F1", op, attr, sig, ins, [sig], sem, fam))
        elif k == "int":
            nin = 1 if op in space.INT_UN else 2
            outs = [sig] * (2 if op in ("arith.mulsi_extended", "arith.mului_extended") else 1)
            if op == "arith.addui_extended":
                outs = [sig, "i1"]
            out.append(Prim(pid, "F1", op, attr, sig, [sig] * nin, outs, _op_sem(op), "int", extra={"flags": attr}))
        elif k == "predicate":
            if op in ("arith.cmpf", "arith.cmpi"):
                out.append(Prim(pid, "F1", op, attr, sig, [sig, sig], ["i1"], f"{_op_sem(op)}.{attr}", "pred"))
            else:
                t = sig.split("->")[0]
                out.append(Prim(pid, "F1", op, attr, sig, [t], ["i1"], _op_sem(op), "pred"))
        elif k == "select":
            out.append(Prim(pid, "F1", op, attr, sig, ["i1", sig, sig], [sig], "select", "select"))
        elif k == "dot":
            fam = "mx" if op == "tt.dot_scaled" else "dot"
            out.append(Prim(pid, "F1", op, attr, sig, [], ["f32"], "dot" if fam == "dot" else "mx", fam))
        elif k == "convert":
            if op in ("arith.index_cast", "arith.index_castui"):
                out.append(Prim(pid, "scalar", op, attr, sig, [sig.split("<->")[0]], [sig.split("<->")[0]],
                                "index_roundtrip", "scalar"))
                continue
            if op in ("arith.scaling_extf", "arith.scaling_truncf", "tt.dot_scaled"):
                out.append(Prim(pid, "F1", op, attr, sig, [], [], "mx", "mx"))
                continue
            a, b = sig.split("->")
            if op in ("arith.bitcast", "tt.bitcast"):
                fam = "bits"
            elif op in ("arith.extf", "arith.truncf", "tt.fp_to_fp"):
                fam = "convert_ff"
            elif op in ("arith.sitofp", "arith.uitofp"):
                fam = "convert_if"
            elif op in ("arith.fptosi", "arith.fptoui"):
                fam = "convert_fi"
            else:
                fam = "convert_ii"
            mode = {"rtz": "rz", "rtne": "rne"}.get(attr, "rne")
            out.append(Prim(pid, "F1", op, attr, sig, [a], [b], _op_sem(op), fam, mode=mode))
        elif k == "other":
            out.append(Prim(pid, "F1", op, attr, sig, [], [], "mx", "mx"))
    return out


def libdevice_entries() -> list:
    import triton.language.extra.cuda.libdevice as ld
    src = inspect.getsource(ld)
    out = []
    for b in re.split(r"\n@core\.extern\n", src)[1:]:
        name = re.match(r"def (\w+)\(", b).group(1)
        for args, sym, ret in re.findall(r"\(\s*((?:core\.dtype\(\"\w+\"\),?\s*)+)\):\s*\(\"(__nv_\w+)\",\s*"
                                         r"core\.dtype\(\"(\w+)\"\)\)", b):
            ins = [LIBDEVICE_TYPES[t] for t in re.findall(r"dtype\(\"(\w+)\"\)", args)]
            out.append((name, sym, ins, LIBDEVICE_TYPES[ret]))
    return out


_MODE = {"rn": "rne", "rz": "rz", "rd": "rd", "ru": "ru"}


def libdevice_prims() -> list:
    out = []
    for name, sym, ins, ret in libdevice_entries():
        pid = f"libdevice|{sym}"
        sig = ",".join(ins) + "->" + ret
        mode = "rne"
        m = re.match(r"^(\w+?)_(rn|rz|rd|ru)$", name)
        floats_in = all(t in ("f32", "f64") for t in ins)
        if ret in ("f32", "f64") and floats_in:
            sem = name
            if m and m.group(1) in ("add", "sub", "mul", "div", "fma", "rcp", "sqrt"):
                mode = _MODE[m.group(2)]
            if name == "rsqrt_rn":
                mode = "rne"
            if name == "abs":
                sem = "fabs"
            if name in ("double2float_rn", "double2float_rz", "double2float_rd", "double2float_ru"):
                sem, mode = "convert", _MODE[m.group(2)]
                out.append(Prim(pid, "libdevice", "tt.extern_elementwise", sym, sig, ins, [ret], sem, "convert_ff",
                                mode=mode, extra={"name": name}))
                continue
            out.append(Prim(pid, "libdevice", "tt.extern_elementwise", sym, sig, ins, [ret], sem, "real", mode=mode,
                            extra={"name": name}))
            continue
        fam, sem = "libint", name
        if re.match(r"^(double|float)2(int|uint|ll|ull)_(rn|rz|rd|ru)$", name):
            fam, mode = "convert_fi", _MODE[name.rsplit("_", 1)[1]]
        elif re.match(r"^(int|uint|ll|ull)2(float|double)_(rn|rz|rd|ru)$", name):
            fam, mode = "convert_if", _MODE[name.rsplit("_", 1)[1]]
        elif name in ("int_as_float", "float_as_int", "uint_as_float", "float_as_uint", "longlong_as_double",
                      "double_as_longlong", "hiloint2double", "double2loint", "double2hiint"):
            fam = "bits"
        elif ret in ("f32", "f64"):
            fam = "real_mixed"              # ldexp, scalbn, jn, yn, pow with an integer exponent
        out.append(Prim(pid, "libdevice", "tt.extern_elementwise", sym, sig, ins, [ret], sem, fam, mode=mode,
                        extra={"name": name}))
    return out


LITERALS = {
    "f32": ["0.1", "0.9999", "0.999", "1e-8", "3.14159265358979", "1e-40", "3.4e38", "6.02214076e23"],
    "f16": ["0.1", "0.9999", "1e-5", "65504.0", "3.14159265358979", "6e-8"],
    "bf16": ["0.1", "0.9999", "0.999", "1e-8", "3.14159265358979", "3.0e38"],
    "f64": ["0.1", "0.9999", "1e-300", "3.14159265358979323846"],
}


def literal_prims() -> list:
    out = []
    for t, lits in LITERALS.items():
        for v in lits:
            out.append(Prim(f"literal|{t}|{v}", "literal", "arith.constant", v, t, [], [t], "literal", "literal"))
    for t in space.INT + ["i1"]:
        v = {"i8": "-128", "i16": "-32768", "i32": "-2147483648", "i64": "-9223372036854775808", "i1": "1"}[t]
        out.append(Prim(f"literal|{t}|{v}", "literal", "arith.constant", v, t, [], [t], "literal", "literal"))
    return out


def all_prims() -> list:
    return f1_prims() + libdevice_prims() + literal_prims()


if __name__ == "__main__":
    from collections import Counter
    ps = all_prims()
    print(len(ps), Counter(p.source for p in ps), Counter(p.family for p in ps))
