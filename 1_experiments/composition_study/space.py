"""The factor space of the composition study.

F1 (operation instances) is regenerated from the tool's Triton 3.6.0 operation registry with the rules of the
three_plans_v3 package (inputs/three_plans_v3_20261013.tar.gz, factor_space.py): the denominator is every registry name
outside the gpu dialect (165 names), expanded by attribute case and type signature.  ``check_against_package`` compares
the regenerated list with the package's rows, so the two stay the same space.

F2 (composition forms) keeps the package's 23 forms and adds the factors the device lowering depends on (section 2 of
protocol.md): the compile options, the layout parameters and the element type of the carried value.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "2_tool/src/kernel_analyzer/reference_eval/data/ttir_op_registry.json"

FLT = ["f16", "bf16", "f32", "f64"]
INT = ["i8", "i16", "i32", "i64"]
FP8 = ["f8e4m3", "f8e5m2"]
OVF = ["none", "nsw", "nuw", "nsw+nuw"]
CMPF = ["oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq", "ugt", "uge", "ult", "ule", "une", "uno", "true", "false"]
CMPI = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]

FLOAT_BIN = {"arith.addf", "arith.subf", "arith.mulf", "arith.maximumf", "arith.minimumf", "arith.maxnumf",
             "arith.minnumf", "arith.remf", "math.powf", "math.copysign", "math.atan2"}
FLOAT_UN = {"arith.negf", "math.absf", "math.ceil", "math.floor", "math.round", "math.roundeven", "math.trunc",
            "math.cbrt", "math.acos", "math.acosh", "math.asin", "math.asinh", "math.atan", "math.atanh", "math.cos",
            "math.cosh", "math.erf", "math.erfc", "math.exp", "math.exp2", "math.expm1", "math.log", "math.log10",
            "math.log1p", "math.log2", "math.rsqrt", "math.sin", "math.sinh", "math.sqrt", "math.tan", "math.tanh"}
INT_OVF = {"arith.addi", "arith.subi", "arith.muli", "arith.shli"}
INT_BIN = {"arith.andi", "arith.ori", "arith.xori", "arith.shrsi", "arith.shrui", "arith.divsi", "arith.divui",
           "arith.remsi", "arith.remui", "arith.ceildivsi", "arith.ceildivui", "arith.floordivsi", "arith.maxsi",
           "arith.maxui", "arith.minsi", "arith.minui"}
INT_UN = {"math.absi", "math.ctlz", "math.ctpop", "math.cttz"}
PREDICATE_OPS = {"arith.cmpf", "arith.cmpi", "math.isfinite", "math.isinf", "math.isnan", "math.isnormal"}


GPU_IN_DENOMINATOR = ("gpu.barrier", "gpu.wait")      # the two gpu-dialect names Triton's own passes emit


def denominator() -> list:
    reg = json.loads(REGISTRY.read_text())
    return sorted(n for d, names in reg["operations"].items() for n in names
                  if d != "gpu" or n in GPU_IN_DENOMINATOR)


def instances(op: str) -> list:
    """The package's expansion rule, unchanged (factor_space.instances)."""
    out = []

    def add(a, t):
        out.append((op, a, t))
    if op in FLOAT_BIN or op in FLOAT_UN:
        for t in FLT:
            add("default", t)
    elif op == "arith.divf":
        for t in FLT:
            add("approx(div.full)" if t == "f32" else "rn", t)
    elif op in ("tt.precise_divf", "tt.precise_sqrt"):
        add("rn", "f32")
    elif op == "math.fma":
        for t in FLT:
            add("default", t)
    elif op in INT_OVF:
        for t in INT:
            for f in OVF:
                add(f, t)
    elif op in INT_BIN or op in INT_UN:
        for t in INT:
            add("default", t)
    elif op in ("arith.mulsi_extended", "arith.mului_extended", "arith.addui_extended", "tt.mulhiui"):
        for t in (INT if op != "tt.mulhiui" else ["i32", "i64"]):
            add("default", t)
    elif op == "math.ipowi":
        for t in INT:
            add("default", t)
    elif op == "math.fpowi":
        for t in FLT:
            add("i32 exponent", t)
    elif op == "arith.cmpf":
        for p in CMPF:
            for t in FLT:
                add(p, t)
    elif op == "arith.cmpi":
        for p in CMPI:
            for t in INT:
                add(p, t)
    elif op == "arith.select":
        for t in FLT + INT + ["i1"]:
            add("default", t)
    elif op in ("math.isfinite", "math.isinf", "math.isnan", "math.isnormal"):
        for t in FLT:
            add("default", t + "->i1")
    elif op in ("tt.clampf", "math.clampf"):
        for t in FLT:
            for pn in ("propagate_nan=false", "propagate_nan=true"):
                add(pn, t)
    elif op == "arith.extf":
        for a, b in itertools.combinations(["f16", "f32", "f64"], 2):
            add("default", f"{a}->{b}")
        for b in ("f32", "f64"):
            add("default", f"bf16->{b}")
    elif op == "arith.truncf":
        for a, b in [("f32", "f16"), ("f32", "bf16"), ("f64", "f32"), ("f64", "f16"), ("f64", "bf16")]:
            add("rn", f"{a}->{b}")
    elif op == "tt.fp_to_fp":
        for s in ("f16", "bf16", "f32"):
            for d in FP8:
                for r in ("rtne", "rtz"):
                    add(r, f"{s}->{d}")
        for s in FP8:
            for d in ("f16", "bf16", "f32"):
                add("exact", f"{s}->{d}")
    elif op in ("arith.sitofp", "arith.uitofp"):
        for a in INT:
            for b in FLT:
                add("rn", f"{a}->{b}")
    elif op in ("arith.fptosi", "arith.fptoui"):
        for a in FLT:
            for b in INT:
                add("trunc toward zero", f"{a}->{b}")
    elif op in ("arith.extsi", "arith.extui"):
        for a, b in itertools.combinations(INT, 2):
            add("default", f"{a}->{b}")
    elif op == "arith.trunci":
        for a, b in itertools.combinations(INT, 2):
            add("modular", f"{b}->{a}")
    elif op in ("arith.index_cast", "arith.index_castui"):
        for t in ("i32", "i64"):
            add("default", f"{t}<->index")
    elif op in ("arith.bitcast", "tt.bitcast"):
        for a, b in [("f16", "i16"), ("bf16", "i16"), ("f32", "i32"), ("f64", "i64")]:
            add("default", f"{a}->{b}")
            add("default", f"{b}->{a}")
    elif op in ("arith.scaling_extf", "arith.scaling_truncf", "tt.dot_scaled"):
        for d in FP8 + ["f4e2m1"]:
            add("mx block scale", d)
    elif op == "tt.dot":
        for ip in ("ieee", "tf32", "tf32x3"):
            add(f"input_precision={ip}", "f32xf32->f32")
        for t in ("f16", "bf16"):
            add("default", f"{t}x{t}->f32")
        for t in FP8:
            add("max_num_imprecise_acc", f"{t}x{t}->f32")
    elif op == "arith.constant":
        for t in FLT + INT + ["i1"]:
            add("default", t)
    else:
        add("default", "structural")
    return out


def kind(inst) -> str:
    op, attr, sig = inst
    if sig == "structural":
        return "structural"
    if op in PREDICATE_OPS:
        return "predicate"
    if op == "arith.select":
        return "select"
    if op in ("tt.dot", "tt.dot_scaled"):
        return "dot"
    if "->" in sig or op.startswith(("arith.scaling_extf", "arith.scaling_truncf")):
        return "convert"
    if sig in INT or sig == "i1":
        return "int"
    if sig in FLT:
        return "float"
    return "other"


F1 = [i for op in denominator() for i in instances(op)]

# F2: the package's composition forms (name, parameter)
F2 = ([("C1 straight-line", f"depth={d}") for d in (1, 2, 4)]
      + [("C2 fan-out sharing", f"reuse={k}") for k in (2, 3)]
      + [("C3 reduce region", o) for o in ("sequential order", "tree order")]
      + [("C4 scan region", s) for s in ("inclusive", "exclusive by shift")]
      + [("C5 map_elementwise region", "-"), ("C6 select branch", "elementwise"), ("C7 scf.if branch", "program-level")]
      + [("C8 scf.for loop-carried", t) for t in ("static trip", "integer-data trip")]
      + [("C9 scf.while", "termination certificate")]
      + [("C10 masked load/store", m) for m in ("mask with other", "mask without other, lanes unobserved")]
      + [("C11 computed index", k) for k in ("gather", "scatter by store")]
      + [("C12 store then load in program", "with barrier"),
         ("C13 cross-launch state", "kernel 1 writes, kernel 2 reads"),
         ("C14 atomic fold", "returned value unused"), ("C15 dot", "tile accumulate")])

# Factors the package does not have; the device lowering depends on them (protocol.md section 2).
F7_OPTIONS = {"enable_fp_fusion": (True, False), "num_warps": (1, 4, 8)}
F8_LAYOUT = {"block": (32, 128, 1024), "reduce_axis_extent": (32, 128, 1024)}


def check_against_package(rows_json: Path) -> dict:
    """Compare F1 with the operation instances of the package's covering rows."""
    rows = json.loads(Path(rows_json).read_text())
    rows = rows["rows"] if isinstance(rows, dict) else [r["row"] for r in rows]
    pkg = {(r["op"], r["attr"], r["sig"]) for r in rows}
    mine = set(F1)
    return {"regenerated": len(F1), "distinct": len(mine), "package": len(pkg), "only_regenerated": sorted(mine - pkg),
            "only_package": sorted(pkg - mine), "equal": mine == pkg}


if __name__ == "__main__":
    import sys
    print(len(denominator()), "names;", len(F1), "instances;", dict(Counter(kind(i) for i in F1)))
    if len(sys.argv) > 1:
        print(json.dumps(check_against_package(Path(sys.argv[1])), indent=1))
