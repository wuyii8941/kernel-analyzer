"""TTIR text for single operation instances (catalog, RQ1).

Every non-structural F1 instance becomes a 1-D kernel: masked loads of its operands, the operation line exactly as the
instance names it (attribute case and type signature), masked stores of its results.  i1 values are carried as i8 in
memory (loaded i8, ``arith.cmpi ne`` to i1; stored through ``arith.extui``), so the instance under test is the only
operation that is not an exact carrier.  Structural instances get their own templates (structural.py).
"""
from __future__ import annotations

from typing import Optional

import space

BLOCK = 1024
TT = {"f16": "f16", "bf16": "bf16", "f32": "f32", "f64": "f64", "i1": "i1", "i8": "i8", "i16": "i16", "i32": "i32",
      "i64": "i64", "f8e4m3": "f8E4M3FN", "f8e5m2": "f8E5M2", "f4e2m1": "f4E2M1FN", "e8m0": "f8E8M0FNU"}
MEM = {**TT, "i1": "i8"}            # element type in memory


def tensor(e: str, block: int = BLOCK) -> str:
    return f"tensor<{block}x{TT[e]}>"


class Unbuildable(Exception):
    """The instance has no single-operation elementwise form (it is tested elsewhere)."""


def op_lines(inst) -> tuple:
    """(input element types, TTIR lines producing %y [, %y1], output element types) for one instance."""
    op, attr, sig = inst
    T = tensor
    if op in space.FLOAT_BIN or op in ("arith.divf", "tt.precise_divf") or op in space.INT_BIN \
            or op in ("math.ipowi", "tt.mulhiui"):
        t = sig
        return [t, t], [f"%y = {op} %x0, %x1 : {T(t)}"], [t]
    if op in space.FLOAT_UN or op in space.INT_UN or op == "tt.precise_sqrt":
        t = sig
        return [t], [f"%y = {op} %x0 : {T(t)}"], [t]
    if op == "math.fma":
        t = sig
        return [t, t, t], [f"%y = math.fma %x0, %x1, %x2 : {T(t)}"], [t]
    if op in space.INT_OVF:
        t = sig
        flags = {"none": "", "nsw": " overflow<nsw>", "nuw": " overflow<nuw>", "nsw+nuw": " overflow<nsw, nuw>"}[attr]
        return [t, t], [f"%y = {op} %x0, %x1{flags} : {T(t)}"], [t]
    if op == "math.fpowi":
        t = sig
        return [t, "i32"], [f"%y = math.fpowi %x0, %x1 : {T(t)}, {T('i32')}"], [t]
    if op in ("arith.mulsi_extended", "arith.mului_extended"):
        t = sig
        return [t, t], [f"%y, %y1 = {op} %x0, %x1 : {T(t)}"], [t, t]
    if op == "arith.addui_extended":
        t = sig
        return [t, t], [f"%y, %y1 = {op} %x0, %x1 : {T(t)}, {T('i1')}"], [t, "i1"]
    if op in ("arith.cmpf", "arith.cmpi"):
        t = sig
        return [t, t], [f"%y = {op} {attr}, %x0, %x1 : {T(t)}"], ["i1"]
    if op == "arith.select":
        t = sig
        return ["i1", t, t], [f"%y = arith.select %x0, %x1, %x2 : {T('i1')}, {T(t)}"], [t]
    if op in ("math.isfinite", "math.isinf", "math.isnan", "math.isnormal"):
        t = sig.split("->")[0]
        return [t], [f"%y = {op} %x0 : {T(t)}"], ["i1"]
    if op == "tt.clampf":
        t = sig
        pn = "all" if attr.endswith("true") else "none"
        return [t, t, t], [f"%y = tt.clampf %x0, %x1, %x2, propagateNan = {pn} : {T(t)}"], [t]
    if op == "math.clampf":
        t = sig
        return [t, t, t], [f"%y = math.clampf %x0 to [%x1, %x2] : {T(t)}"], [t]
    if "->" in sig and op in ("arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp", "arith.fptosi",
                              "arith.fptoui", "arith.extsi", "arith.extui", "arith.trunci", "arith.bitcast"):
        a, b = sig.split("->")
        return [a], [f"%y = {op} %x0 : {T(a)} to {T(b)}"], [b]
    if op == "tt.bitcast":
        a, b = sig.split("->")
        return [a], [f"%y = tt.bitcast %x0 : {T(a)} -> {T(b)}"], [b]
    if op == "tt.fp_to_fp":
        a, b = sig.split("->")
        rnd = f", rounding = {attr}" if attr in ("rtne", "rtz") else ""
        return [a], [f"%y = tt.fp_to_fp %x0{rnd} : {T(a)} -> {T(b)}"], [b]
    if op == "arith.scaling_extf":           # the e8m0 scale is carried as i8 in memory and bit-cast in the kernel
        d = sig
        return [d, "i8"], [f"%sc = tt.bitcast %x1 : {T('i8')} -> {T('e8m0')}",
                           f"%y = arith.scaling_extf %x0, %sc : {T(d)}, {T('e8m0')} to {T('f32')}"], ["f32"]
    if op == "arith.scaling_truncf":
        d = sig
        return ["f32", "i8"], [f"%sc = tt.bitcast %x1 : {T('i8')} -> {T('e8m0')}",
                               f"%y = arith.scaling_truncf %x0, %sc : {T('f32')}, {T('e8m0')} to {T(d)}"], [d]
    raise Unbuildable(f"{op} {attr} {sig}")


def elementwise_module(in_elems: list, lines: list, out_elems: list, name: str = "probe", block: int = BLOCK) -> str:
    args = [f"%in{i}: !tt.ptr<{MEM[e]}> {{tt.divisibility = 16 : i32}}" for i, e in enumerate(in_elems)]
    args += [f"%out{j}: !tt.ptr<{MEM[e]}> {{tt.divisibility = 16 : i32}}" for j, e in enumerate(out_elems)]
    args.append("%n: i32")
    b = [
        "%pid = tt.get_program_id x : i32",
        f"%cb = arith.constant {block} : i32",
        "%base = arith.muli %pid, %cb : i32",
        f"%range = tt.make_range {{end = {block} : i32, start = 0 : i32}} : tensor<{block}xi32>",
        f"%bases = tt.splat %base : i32 -> tensor<{block}xi32>",
        f"%offs = arith.addi %bases, %range : tensor<{block}xi32>",
        f"%ns = tt.splat %n : i32 -> tensor<{block}xi32>",
        f"%mask = arith.cmpi slt, %offs, %ns : tensor<{block}xi32>",
    ]
    for i, e in enumerate(in_elems):
        m = MEM[e]
        b += [f"%p{i} = tt.splat %in{i} : !tt.ptr<{m}> -> tensor<{block}x!tt.ptr<{m}>>",
              f"%a{i} = tt.addptr %p{i}, %offs : tensor<{block}x!tt.ptr<{m}>>, tensor<{block}xi32>"]
        if e == "i1":
            b += [f"%l{i} = tt.load %a{i}, %mask : tensor<{block}x!tt.ptr<i8>>",
                  f"%z{i} = arith.constant dense<0> : tensor<{block}xi8>",
                  f"%x{i} = arith.cmpi ne, %l{i}, %z{i} : tensor<{block}xi8>"]
        else:
            b += [f"%x{i} = tt.load %a{i}, %mask : tensor<{block}x!tt.ptr<{m}>>"]
    b += lines
    for j, e in enumerate(out_elems):
        m = MEM[e]
        v = "%y" if j == 0 else f"%y{j}"
        if e == "i1":
            b.append(f"%w{j} = arith.extui {v} : tensor<{block}xi1> to tensor<{block}xi8>")
            v = f"%w{j}"
        b += [f"%po{j} = tt.splat %out{j} : !tt.ptr<{m}> -> tensor<{block}x!tt.ptr<{m}>>",
              f"%ao{j} = tt.addptr %po{j}, %offs : tensor<{block}x!tt.ptr<{m}>>, tensor<{block}xi32>",
              f"tt.store %ao{j}, {v}, %mask : tensor<{block}x!tt.ptr<{m}>>"]
    b.append("tt.return")
    return ("module {\n  tt.func public @" + name + "(" + ", ".join(args) + ") attributes {noinline = false} {\n    "
            + "\n    ".join(b) + "\n  }\n}\n")


def dot_module(a: str, b: str, attrs: str, m: int = 16, n: int = 16, k: int = 16, acc: str = "f32",
               name: str = "dot") -> str:
    """C = dot(A, B, C0) on one tile, row-major A (m x k), B (k x n), C0 / C (m x n)."""
    lines = [f"%ra = tt.make_range {{end = {m} : i32, start = 0 : i32}} : tensor<{m}xi32>",
             f"%rk = tt.make_range {{end = {k} : i32, start = 0 : i32}} : tensor<{k}xi32>",
             f"%rn = tt.make_range {{end = {n} : i32, start = 0 : i32}} : tensor<{n}xi32>"]

    def grid(ptr, e, rows, cols, rr, rc, tag):
        t = TT[e]
        return [f"%{tag}r = tt.expand_dims {rr} {{axis = 1 : i32}} : tensor<{rows}xi32> -> tensor<{rows}x1xi32>",
                f"%{tag}cs = arith.constant dense<{cols}> : tensor<{rows}x1xi32>",
                f"%{tag}ro = arith.muli %{tag}r, %{tag}cs : tensor<{rows}x1xi32>",
                f"%{tag}c = tt.expand_dims {rc} {{axis = 0 : i32}} : tensor<{cols}xi32> -> tensor<1x{cols}xi32>",
                f"%{tag}rb = tt.broadcast %{tag}ro : tensor<{rows}x1xi32> -> tensor<{rows}x{cols}xi32>",
                f"%{tag}cb = tt.broadcast %{tag}c : tensor<1x{cols}xi32> -> tensor<{rows}x{cols}xi32>",
                f"%{tag}o = arith.addi %{tag}rb, %{tag}cb : tensor<{rows}x{cols}xi32>",
                f"%{tag}p = tt.splat {ptr} : !tt.ptr<{t}> -> tensor<{rows}x{cols}x!tt.ptr<{t}>>",
                f"%{tag}a = tt.addptr %{tag}p, %{tag}o : tensor<{rows}x{cols}x!tt.ptr<{t}>>, tensor<{rows}x{cols}xi32>"]
    lines += grid("%A", a, m, k, "%ra", "%rk", "ga") + [f"%xa = tt.load %gaa : tensor<{m}x{k}x!tt.ptr<{TT[a]}>>"]
    lines += grid("%B", b, k, n, "%rk", "%rn", "gb") + [f"%xb = tt.load %gba : tensor<{k}x{n}x!tt.ptr<{TT[b]}>>"]
    lines += grid("%C0", acc, m, n, "%ra", "%rn", "gc") + [f"%xc = tt.load %gca : tensor<{m}x{n}x!tt.ptr<{TT[acc]}>>"]
    lines += [f"%y = tt.dot %xa, %xb, %xc{attrs} : tensor<{m}x{k}x{TT[a]}> * tensor<{k}x{n}x{TT[b]}> -> "
              f"tensor<{m}x{n}x{TT[acc]}>"]
    lines += grid("%C", acc, m, n, "%ra", "%rn", "go") + [f"tt.store %goa, %y : tensor<{m}x{n}x!tt.ptr<{TT[acc]}>>",
                                                          "tt.return"]
    args = [f"%A: !tt.ptr<{TT[a]}> {{tt.divisibility = 16 : i32}}", f"%B: !tt.ptr<{TT[b]}> {{tt.divisibility = 16 : i32}}",
            f"%C0: !tt.ptr<{TT[acc]}> {{tt.divisibility = 16 : i32}}", f"%C: !tt.ptr<{TT[acc]}> {{tt.divisibility = 16 : i32}}"]
    return ("module {\n  tt.func public @" + name + "(" + ", ".join(args) + ") attributes {noinline = false} {\n    "
            + "\n    ".join(lines) + "\n  }\n}\n")


def dot_scaled_module(fmt: str, m: int = 32, n: int = 32, k: int = 64, name: str = "dot_scaled") -> str:
    """C = dot_scaled(A, sA, B, sB): A (m x k) and B (k x n) in an mx element format (e2m1 packed two per i8 along k),
    one e8m0 scale (carried as i8) per 32 elements of k; layout as the frontend emits it."""
    et = {"e4m3": "f8E4M3FN", "e5m2": "f8E5M2", "e2m1": "i8"}[fmt]
    ka = k // 2 if fmt == "e2m1" else k
    kb = ka
    L = [f"%ra = tt.make_range {{end = {m} : i32, start = 0 : i32}} : tensor<{m}xi32>",
         f"%rka = tt.make_range {{end = {ka} : i32, start = 0 : i32}} : tensor<{ka}xi32>",
         f"%rn = tt.make_range {{end = {n} : i32, start = 0 : i32}} : tensor<{n}xi32>",
         f"%rs = tt.make_range {{end = {k // 32} : i32, start = 0 : i32}} : tensor<{k // 32}xi32>"]

    def grid(tag, rows, cols, rr, rc, stride, ptr, t):
        return [f"%{tag}r = tt.expand_dims {rr} {{axis = 1 : i32}} : tensor<{rows}xi32> -> tensor<{rows}x1xi32>",
                f"%{tag}s = arith.constant dense<{stride}> : tensor<{rows}x1xi32>",
                f"%{tag}ro = arith.muli %{tag}r, %{tag}s : tensor<{rows}x1xi32>",
                f"%{tag}c = tt.expand_dims {rc} {{axis = 0 : i32}} : tensor<{cols}xi32> -> tensor<1x{cols}xi32>",
                f"%{tag}rb = tt.broadcast %{tag}ro : tensor<{rows}x1xi32> -> tensor<{rows}x{cols}xi32>",
                f"%{tag}cb = tt.broadcast %{tag}c : tensor<1x{cols}xi32> -> tensor<{rows}x{cols}xi32>",
                f"%{tag}o = arith.addi %{tag}rb, %{tag}cb : tensor<{rows}x{cols}xi32>",
                f"%{tag}p = tt.splat {ptr} : !tt.ptr<{t}> -> tensor<{rows}x{cols}x!tt.ptr<{t}>>",
                f"%{tag}a = tt.addptr %{tag}p, %{tag}o : tensor<{rows}x{cols}x!tt.ptr<{t}>>, tensor<{rows}x{cols}xi32>",
                f"%{tag}v = tt.load %{tag}a : tensor<{rows}x{cols}x!tt.ptr<{t}>>"]
    L += grid("a", m, ka, "%ra", "%rka", ka, "%A", et)
    L += grid("b", kb, n, "%rka", "%rn", n, "%B", et)
    L += grid("sa", m, k // 32, "%ra", "%rs", k // 32, "%SA", "i8")
    L += grid("sb", n, k // 32, "%rn", "%rs", k // 32, "%SB", "i8")
    L += [f"%z = arith.constant dense<0.000000e+00> : tensor<{m}x{n}xf32>",
          f"%y = tt.dot_scaled %av scale %sav, %bv scale %sbv, %z lhs = {fmt} rhs = {fmt} {{fastMath = false}} : "
          f"tensor<{m}x{ka}x{et}>, tensor<{m}x{k // 32}xi8> * tensor<{kb}x{n}x{et}>, tensor<{n}x{k // 32}xi8> -> "
          f"tensor<{m}x{n}xf32>"]
    rm_ = [l.replace("%ra", "%ra").replace("tt.load", "tt.load") for l in grid("o", m, n, "%ra", "%rn", n, "%C", "f32")]
    L += rm_[:-1] + [f"tt.store %oa, %y : tensor<{m}x{n}x!tt.ptr<f32>>", "tt.return"]
    args = [f"%A: !tt.ptr<{et}> {{tt.divisibility = 16 : i32}}", f"%B: !tt.ptr<{et}> {{tt.divisibility = 16 : i32}}",
            "%SA: !tt.ptr<i8> {tt.divisibility = 16 : i32}", "%SB: !tt.ptr<i8> {tt.divisibility = 16 : i32}",
            "%C: !tt.ptr<f32> {tt.divisibility = 16 : i32}"]
    return ("module {\n  tt.func public @" + name + "(" + ", ".join(args) + ") attributes {noinline = false} {\n    "
            + "\n    ".join(L) + "\n  }\n}\n")


def dot_attrs(inst) -> tuple:
    """(A elem, B elem, attribute text) of a tt.dot instance."""
    op, attr, sig = inst
    a = sig.split("x")[0]
    if attr.startswith("input_precision="):
        return "f32", "f32", f", inputPrecision = {attr.split('=')[1]}"
    if attr == "max_num_imprecise_acc":
        return a, a, ", inputPrecision = tf32 {maxNumImpreciseAcc = 32 : i32}"
    return a, a, ""


def module_for(inst, name: str = "probe") -> Optional[tuple]:
    """(module text, input elems, output elems, launch kind) or None for structural instances."""
    if space.kind(inst) == "structural":
        return None
    op = inst[0]
    if op == "tt.dot":
        a, b, attrs = dot_attrs(inst)
        return dot_module(a, b, attrs, name=name), [a, b, "f32"], ["f32"], "dot"
    if op == "tt.dot_scaled":
        raise Unbuildable("tt.dot_scaled: tested by its own template")
    if op == "arith.constant":
        raise Unbuildable("arith.constant: tested by the literal template")
    if op in ("arith.index_cast", "arith.index_castui"):
        raise Unbuildable("index casts are scalar: tested by the scalar template")
    ins, lines, outs = op_lines(inst)
    return elementwise_module(ins, lines, outs, name=name), ins, outs, "elementwise"
