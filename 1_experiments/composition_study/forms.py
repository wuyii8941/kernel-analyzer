"""Composition forms (F2) as TTIR templates around one primitive (RQ2).

``build(prim, form, param, block)`` returns a ``Cell``: the module text, the inputs (name, element type, role) and
outputs, the grid, and whether the form is applicable to the primitive.  The primitive is inserted exactly as in the
catalog (ttir_build.op_lines for F1 instances, tt.extern_elementwise for libdevice); every other operation of a
template is a carrier: exact (loads, stores, pointer arithmetic, select, integer counters, exact conversions) or a
fixed rounding carrier named in the template (the combine of a sharing / loop form when the primitive itself cannot be
chained).  Applicability follows the package's rule: a reduce / scan combine must be a binary operation of one type,
an atomic fold an atomic-eligible update, a dot form a dot instance; everything else applies to every primitive.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import ttir_build as tb

TT = tb.TT
MEMT = {"u32": "i32", "u64": "i64"}


def mem(e):
    return tb.MEM[MEMT.get(e, e)]


def tt_name(e):
    return TT[MEMT.get(e, e)]


@dataclass
class Cell:
    form: str
    param: str
    text: str = ""
    inputs: list = field(default_factory=list)      # (arg name, elem, role)
    outputs: list = field(default_factory=list)     # (arg name, elem, length in elements)
    grid: int = 1
    block: int = 128
    applicable: bool = True
    reason: str = ""
    launches: int = 1
    second_text: str = ""                            # C13: the second kernel


FLOATS = ("f16", "bf16", "f32", "f64")


def is_binary_same(p) -> bool:
    return len(p.ins) == 2 and p.ins[0] == p.ins[1] == p.outs[0] and p.outs[0] != "i1" and len(p.outs) == 1


ATOMIC = {"arith.addf": "fadd", "arith.addi": "add", "arith.maxsi": "max", "arith.minsi": "min", "arith.maxui": "umax",
          "arith.minui": "umin", "arith.andi": "and", "arith.ori": "or", "arith.xori": "xor"}


class B:
    """Small SSA builder."""

    def __init__(self, block):
        self.block = block
        self.lines = []
        self.n = 0
        self.args = []

    def v(self, base="v"):
        self.n += 1
        return f"%{base}{self.n}"

    def ten(self, e, shape=None):
        return f"tensor<{shape or self.block}x{tt_name(e)}>"

    def ptrt(self, e, shape=None):
        return f"tensor<{shape or self.block}x!tt.ptr<{mem(e)}>>"

    def arg(self, name, e):
        self.args.append(f"%{name}: !tt.ptr<{mem(e)}> {{tt.divisibility = 16 : i32}}")
        return f"%{name}"

    def emit(self, *ls):
        self.lines.extend(ls)

    def const(self, val, e, scalar=False):
        r = self.v("c")
        t = tt_name(e) if scalar else self.ten(e)
        lit = val
        if not scalar:
            lit = f"dense<{val}>"
        self.emit(f"{r} = arith.constant {lit} : {t}")
        return r

    def offsets(self, stride_programs=True):
        pid, cb, base, rng, bs, offs = (self.v(x) for x in ("pid", "cb", "base", "rng", "bs", "offs"))
        self.emit(f"{pid} = tt.get_program_id x : i32",
                  f"{cb} = arith.constant {self.block} : i32",
                  f"{base} = arith.muli {pid}, {cb} : i32",
                  f"{rng} = tt.make_range {{end = {self.block} : i32, start = 0 : i32}} : tensor<{self.block}xi32>",
                  f"{bs} = tt.splat {base} : i32 -> tensor<{self.block}xi32>",
                  f"{offs} = arith.addi {bs}, {rng} : tensor<{self.block}xi32>")
        self.pid, self.rng, self.offs = pid, rng, offs
        return offs

    def ptrs(self, ptr, e, offs):
        p, a = self.v("p"), self.v("a")
        self.emit(f"{p} = tt.splat {ptr} : !tt.ptr<{mem(e)}> -> {self.ptrt(e)}",
                  f"{a} = tt.addptr {p}, {offs} : {self.ptrt(e)}, tensor<{self.block}xi32>")
        return a

    def load(self, ptr, e, offs, mask=None, other=None):
        a = self.ptrs(ptr, e, offs)
        x = self.v("x")
        extra = (f", {mask}" if mask else "") + (f", {other}" if other else "")
        if e == "i1":
            l, z = self.v("l"), self.const("0", "i8")
            self.emit(f"{l} = tt.load {a}{extra} : {self.ptrt('i8')}",
                      f"{x} = arith.cmpi ne, {l}, {z} : {self.ten('i8')}")
        else:
            self.emit(f"{x} = tt.load {a}{extra} : {self.ptrt(e)}")
        return x

    def store(self, ptr, e, offs, val, mask=None):
        a = self.ptrs(ptr, e, offs)
        if e == "i1":
            w = self.v("w")
            self.emit(f"{w} = arith.extui {val} : {self.ten('i1')} to {self.ten('i8')}")
            val = w
        self.emit(f"tt.store {a}, {val}{', ' + mask if mask else ''} : {self.ptrt(e)}")

    def module(self, name="cell"):
        return ("module {\n  tt.func public @" + name + "(" + ", ".join(self.args) + ") attributes {noinline = false} {\n    "
                + "\n    ".join(self.lines + ["tt.return"]) + "\n  }\n}\n")


def apply(p, args, res, ty) -> list:
    """Lines applying primitive p to SSA args, result(s) named res (res_1 for a second result); ty(tt_elem) is the
    type text for an element type name (tensor or scalar)."""
    if p.source == "libdevice":
        ins = [tt_name(e) for e in p.ins]
        return [f'{res} = tt.extern_elementwise {", ".join(args)} {{libname = "", libpath = "", pure = true, '
                f'symbol = "{p.attr}"}} : ({", ".join(ty(e) for e in ins)}) -> {ty(tt_name(p.outs[0]))}']
    ins, lines, outs = tb.op_lines((p.op, p.attr, p.sig))
    text = "\n".join(lines)
    text = re.sub(r"tensor<1024x([\w]+)>", lambda m: ty(m.group(1)), text)
    text = re.sub(r"%y1\b", res + "_1", text)
    text = re.sub(r"%y\b", res, text)
    text = re.sub(r"%x(\d)\b", lambda m: args[int(m.group(1))], text)
    return text.splitlines()


def back_to(bld, val, src, dst, alts):
    """Carrier from the primitive's result type back to its first operand type (for chains)."""
    if src == dst:
        return val
    r = bld.v("bk")
    T = bld.ten
    fs, fd = src in FLOATS, dst in FLOATS
    if src == "i1":
        a, b2 = alts
        bld.emit(f"{r} = arith.select {val}, {a}, {b2} : {T('i1')}, {T(dst)}")
    elif fs and fd:
        wid = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}
        if src == "bf16" and dst == "f16" or src == "f16" and dst == "bf16":
            m = bld.v("bk")
            bld.emit(f"{m} = arith.extf {val} : {T(src)} to {T('f32')}",
                     f"{r} = arith.truncf {m} : {T('f32')} to {T(dst)}")
        else:
            op = "arith.extf" if wid[dst] > wid[src] else "arith.truncf"
            bld.emit(f"{r} = {op} {val} : {T(src)} to {T(dst)}")
    elif src.startswith("f8") or dst.startswith("f8"):
        if src.startswith("f8") and fd:
            bld.emit(f"{r} = tt.fp_to_fp {val} : {T(src)} -> {T(dst)}")
        elif dst.startswith("f8") and fs:
            bld.emit(f"{r} = tt.fp_to_fp {val}, rounding = rtne : {T(src)} -> {T(dst)}")
        else:
            return None
    elif fs and not fd:
        bld.emit(f"{r} = arith.fptosi {val} : {T(src)} to {T(dst)}")
    elif fd and not fs:
        bld.emit(f"{r} = arith.sitofp {val} : {T(src)} to {T(dst)}")
    else:
        ws = int(MEMT.get(src, src)[1:])
        wd = int(MEMT.get(dst, dst)[1:])
        if ws == wd:
            return val
        op = "arith.extsi" if wd > ws else "arith.trunci"
        bld.emit(f"{r} = {op} {val} : {T(src)} to {T(dst)}")
    return r


def combine(bld, a, b2, e):
    """The fixed carrier that combines two values of type e (sharing and loop forms)."""
    r = bld.v("cm")
    T = bld.ten(e)
    if e in FLOATS:
        bld.emit(f"{r} = arith.addf {a}, {b2} : {T}")
    elif e == "i1":
        bld.emit(f"{r} = arith.xori {a}, {b2} : {T}")
    elif e.startswith("f8"):
        u1, u2, s = bld.v("u"), bld.v("u"), bld.v("u")
        bld.emit(f"{u1} = tt.fp_to_fp {a} : {T} -> {bld.ten('f16')}",
                 f"{u2} = tt.fp_to_fp {b2} : {T} -> {bld.ten('f16')}",
                 f"{s} = arith.addf {u1}, {u2} : {bld.ten('f16')}",
                 f"{r} = tt.fp_to_fp {s}, rounding = rtne : {bld.ten('f16')} -> {T}")
    else:
        bld.emit(f"{r} = arith.addi {a}, {b2} : {T}")
    return r


FORMS = [("C1", "depth=1"), ("C1", "depth=2"), ("C1", "depth=4"), ("C2", "reuse=2"), ("C2", "reuse=3"),
         ("C3", "sequential order"), ("C3", "tree order"), ("C4", "inclusive"), ("C4", "exclusive by shift"),
         ("C5", "-"), ("C6", "elementwise"), ("C7", "program-level"), ("C8", "static trip"),
         ("C8", "integer-data trip"), ("C9", "termination certificate"), ("C10", "mask with other"),
         ("C10", "mask without other, lanes unobserved"), ("C11", "gather"), ("C11", "scatter by store"),
         ("C12", "with barrier"), ("C13", "kernel 1 writes, kernel 2 reads"), ("C14", "returned value unused"),
         ("C15", "tile accumulate")]
GRID = 6          # programs per launch (each program's inputs come from one distribution block)
TRIP = 4


def chain_position(p) -> int:
    """The operand a chain feeds the previous result into: the first one of the result's type (a select's value
    operand, not its condition), else the first operand (through a carrier conversion)."""
    for j, e in enumerate(p.ins):
        if e == p.outs[0]:
            return j
    return 0


def build(p, form, param, block=128) -> Cell:
    c = Cell(form, param, block=block, grid=GRID)
    if p.family in ("dot", "literal", "scalar", "mx") and form != "C15":
        c.applicable, c.reason = False, f"family {p.family} has its own template"
        return c
    if form == "C15":
        if p.family != "dot":
            c.applicable, c.reason = False, "a dot form needs a dot instance"
            return c
        return build_dot(p, c)
    if form in ("C3", "C4") and not is_binary_same(p):
        c.applicable, c.reason = False, "a reduce / scan combine is a binary operation of one type"
        return c
    if form == "C14" and not (p.op in ATOMIC and p.ins[0] in ("f32", "f16", "bf16", "i32", "i64")):
        c.applicable, c.reason = False, "an atomic fold needs an atomic-eligible update"
        return c
    if form == "C3" and param == "sequential order":
        block = c.block = 1024
    bld = B(block)
    ins = []

    def inp(e, role):
        name = f"in{len(ins)}"
        ins.append((name, e, role))
        return bld.arg(name, e)

    outs = []

    def out(e, n=None):
        name = f"out{len(outs)}"
        outs.append((name, e, n))
        return bld.arg(name, e)

    T = bld.ten
    tyt = (lambda tt: f"tensor<{block}x{tt}>")
    tys = (lambda tt: tt)
    if form == "C1":
        d = int(param.split("=")[1])
        cp = chain_position(p)
        ptrs = [inp(e, f"stage0.op{j}") for j, e in enumerate(p.ins)]
        alts = None
        if p.outs[0] == "i1" and d > 1 and p.ins[cp] != "i1":
            alts = (inp(p.ins[cp], "alt_true"), inp(p.ins[cp], "alt_false"))
        extra = [[inp(e, f"stage{k}.op{j}") for j, e in enumerate(p.ins) if j != cp] for k in range(1, d)]
        po = out(p.outs[0])
        offs = bld.offsets()
        vals = [bld.load(q, e, offs) for q, e in zip(ptrs, p.ins)]
        altv = tuple(bld.load(q, p.ins[cp], offs) for q in alts) if alts else None
        r = bld.v("r")
        bld.emit(*apply(p, vals, r, tyt))
        for k in range(1, d):
            carried = back_to(bld, r, p.outs[0], p.ins[cp], altv)
            if carried is None:
                c.applicable, c.reason = False, "no carrier back to the operand type"
                return c
            others = [bld.load(q, e, offs) for q, e in zip(extra[k - 1], [t for j, t in enumerate(p.ins) if j != cp])]
            operands = others[:cp] + [carried] + others[cp:]
            r = bld.v("r")
            bld.emit(*apply(p, operands, r, tyt))
        bld.store(po, p.outs[0], offs, r)
    elif form == "C2":
        k = int(param.split("=")[1])
        ptrs = [inp(e, f"op{j}") for j, e in enumerate(p.ins)]
        po = out(p.outs[0])
        offs = bld.offsets()
        vals = [bld.load(q, e, offs) for q, e in zip(ptrs, p.ins)]
        r = bld.v("r")
        bld.emit(*apply(p, vals, r, tyt))
        acc = r
        for _ in range(k - 1):
            acc = combine(bld, acc, r, p.outs[0])
        bld.store(po, p.outs[0], offs, acc)
    elif form in ("C3", "C4"):
        e = p.ins[0]
        q = inp(e, "op0")
        po = out(e, 1 if form == "C3" else None)
        offs = bld.offsets()
        x = bld.load(q, e, offs)
        r = bld.v("r")
        if form == "C4" and param.startswith("exclusive"):
            one = bld.const("1", "i32")
            sh, m, zero = bld.v("sh"), bld.v("m"), bld.const("0", "i32")
            zf = bld.const("0" if e not in FLOATS else "0.000000e+00", e)
            bld.emit(f"{sh} = arith.subi {offs}, {one} : {T('i32')}",
                     f"{m} = arith.cmpi sgt, {bld.rng}, {zero} : {T('i32')}")
            x = bld.load(q, e, sh, mask=m, other=zf)
        s = tt_name(e)
        body = apply(p, ["%ra", "%rb"], "%rc", tys)
        region = ["^bb0(%ra: " + s + ", %rb: " + s + "):"] + ["  " + l for l in body]
        if form == "C3":
            region.append(f"  tt.reduce.return %rc : {s}")
            bld.emit(f'{r} = "tt.reduce"({x}) <{{axis = 0 : i32}}> ({{', *["  " + l for l in region],
                     f"}}) : ({T(e)}) -> {s}")
            pidp = bld.v("pp")
            bld.emit(f"{pidp} = tt.addptr {po}, {bld.pid} : !tt.ptr<{mem(e)}>, i32",
                     f"tt.store {pidp}, {r} : !tt.ptr<{mem(e)}>")
            outs[-1] = (outs[-1][0], e, GRID)
        else:
            region.append(f"  tt.scan.return %rc : {s}")
            bld.emit(f'{r} = "tt.scan"({x}) <{{axis = 0 : i32, reverse = false}}> ({{', *["  " + l for l in region],
                     f"}}) : ({T(e)}) -> {T(e)}")
            bld.store(po, e, offs, r)
    elif form == "C5":
        ptrs = [inp(e, f"op{j}") for j, e in enumerate(p.ins)]
        po = out(p.outs[0])
        offs = bld.offsets()
        vals = [bld.load(q, e, offs) for q, e in zip(ptrs, p.ins)]
        names = [f"%me{j}" for j in range(len(p.ins))]
        body = apply(p, names, "%mr", tys)
        if len(p.outs) > 1:
            c.applicable, c.reason = False, "map_elementwise returns one value per region result here"
            return c
        r = bld.v("r")
        sig = ", ".join(f"{n}: {tt_name(e)}" for n, e in zip(names, p.ins))
        bld.emit(f'{r} = "tt.map_elementwise"({", ".join(vals)}) <{{pack = 1 : i32}}> ({{',
                 f"^bb0({sig}):", *["  " + l for l in body], f"  tt.map_elementwise.return %mr : {tt_name(p.outs[0])}",
                 f"}}) : ({', '.join(T(e) for e in p.ins)}) -> {T(p.outs[0])}")
        bld.store(po, p.outs[0], offs, r)
    elif form in ("C6", "C7"):
        ptrs = [inp(e, f"op{j}") for j, e in enumerate(p.ins)]
        alt = inp(p.outs[0], "alternative")
        sel = inp("f32", "condition source") if form == "C6" else None
        po = out(p.outs[0])
        offs = bld.offsets()
        vals = [bld.load(q, e, offs) for q, e in zip(ptrs, p.ins)]
        a = bld.load(alt, p.outs[0], offs)
        r = bld.v("r")
        if form == "C6":
            s = bld.load(sel, "f32", offs)
            z = bld.const("0.000000e+00", "f32")
            cnd = bld.v("cnd")
            bld.emit(*apply(p, vals, r, tyt))
            y = bld.v("y")
            bld.emit(f"{cnd} = arith.cmpf ogt, {s}, {z} : {T('f32')}",
                     f"{y} = arith.select {cnd}, {r}, {a} : {T('i1')}, {T(p.outs[0])}")
        else:
            two, zero, rem, cnd, y = bld.v("two"), bld.v("zero"), bld.v("rem"), bld.v("cnd"), bld.v("y")
            bld.emit(f"{two} = arith.constant 2 : i32", f"{zero} = arith.constant 0 : i32",
                     f"{rem} = arith.remsi {bld.pid}, {two} : i32", f"{cnd} = arith.cmpi eq, {rem}, {zero} : i32",
                     f"{y} = scf.if {cnd} -> ({T(p.outs[0])}) {{", *["  " + l for l in apply(p, vals, r, tyt)],
                     f"  scf.yield {r} : {T(p.outs[0])}", "} else {", f"  scf.yield {a} : {T(p.outs[0])}", "}")
        bld.store(po, p.outs[0], offs, y)
    elif form in ("C8", "C9"):
        binary_chain = is_binary_same(p)
        init = inp(p.outs[0], "initial accumulator")
        ptrs = [inp(e, f"op{j} per iteration") for j, e in enumerate(p.ins) if not (binary_chain and j == 0)]
        trip_ptr = inp("i32", "trip count") if (form == "C8" and param.startswith("integer")) or form == "C9" else None
        po = out(p.outs[0])
        offs = bld.offsets()
        acc0 = bld.load(init, p.outs[0], offs)
        tot = bld.v("tot")
        bld.emit(f"{tot} = arith.constant {GRID * block} : i32")
        if trip_ptr:
            n = bld.v("n")
            bld.emit(f"{n} = tt.load {trip_ptr} : !tt.ptr<i32>")
        else:
            n = bld.v("n")
            bld.emit(f"{n} = arith.constant {TRIP} : i32")
        c0, c1 = bld.v("c0"), bld.v("c1")
        bld.emit(f"{c0} = arith.constant 0 : i32", f"{c1} = arith.constant 1 : i32")
        tmp = B(block)
        tmp.n = bld.n + 1000
        tmp.pid, tmp.rng, tmp.offs = bld.pid, bld.rng, offs
        it = "%it"
        so, sh = tmp.v("so"), tmp.v("sh")
        tmp.emit(f"{so} = arith.muli {it}, {tot} : i32", f"{sh} = tt.splat {so} : i32 -> {T('i32')}")
        o2 = tmp.v("o2")
        tmp.emit(f"{o2} = arith.addi {offs}, {sh} : {T('i32')}")
        es = [e for j, e in enumerate(p.ins) if not (binary_chain and j == 0)]
        xs = [tmp.load(q, e, o2) for q, e in zip(ptrs, es)]
        r = tmp.v("r")
        if binary_chain:
            tmp.emit(*apply(p, ["%acc"] + xs, r, tyt))
            nxt = r
        else:
            tmp.emit(*apply(p, xs, r, tyt))
            nxt = combine(tmp, "%acc", r, p.outs[0])
        y = bld.v("y")
        if form == "C8":
            bld.emit(f"{y} = scf.for {it} = {c0} to {n} step {c1} iter_args(%acc = {acc0}) -> ({T(p.outs[0])}) : i32 {{",
                     *["  " + l for l in tmp.lines], f"  scf.yield {nxt} : {T(p.outs[0])}", "}")
        else:
            k = bld.v("k")
            bld.emit(f"{y}, {k} = scf.while (%wa = {acc0}, %wk = {c0}) : ({T(p.outs[0])}, i32) -> ({T(p.outs[0])}, i32) {{",
                     f"  %wc = arith.cmpi slt, %wk, {n} : i32",
                     f"  scf.condition(%wc) %wa, %wk : {T(p.outs[0])}, i32",
                     "} do {", f"^bb0(%acc: {T(p.outs[0])}, {it}: i32):",
                     *["  " + l for l in tmp.lines],
                     f"  %wn = arith.addi {it}, {c1} : i32",
                     f"  scf.yield {nxt}, %wn : {T(p.outs[0])}, i32", "}")
            y = y
        bld.n = tmp.n + 10
        bld.store(po, p.outs[0], offs, y)
        c.launches = 1
    elif form == "C10":
        ptrs = [inp(e, f"op{j}") for j, e in enumerate(p.ins)]
        po = out(p.outs[0])
        offs = bld.offsets()
        three, z, rem, m = bld.const("3", "i32"), bld.const("0", "i32"), bld.v("rem"), bld.v("m")
        bld.emit(f"{rem} = arith.remsi {bld.rng}, {three} : {T('i32')}", f"{m} = arith.cmpi ne, {rem}, {z} : {T('i32')}")
        vals = []
        for q, e in zip(ptrs, p.ins):
            if param == "mask with other":
                o = bld.const("1" if e not in FLOATS else "1.500000e+00", e) if e != "i1" else None
                vals.append(bld.load(q, e, offs, mask=m, other=o) if o else bld.load(q, e, offs))
            else:
                vals.append(bld.load(q, e, offs, mask=m) if e != "i1" else bld.load(q, e, offs))
        r = bld.v("r")
        bld.emit(*apply(p, vals, r, tyt))
        if param == "mask with other":
            bld.store(po, p.outs[0], offs, r)
        else:
            bld.store(po, p.outs[0], offs, r, mask=m)
    elif form == "C11":
        ptrs = [inp(e, f"op{j}") for j, e in enumerate(p.ins)]
        perm = inp("i32", "permutation within the program")
        po = out(p.outs[0])
        offs = bld.offsets()
        pv = bld.load(perm, "i32", offs)
        bs = bld.v("bs")
        bld.emit(f"{bs} = tt.splat {bld.v('dummy')} : i32 -> {T('i32')}") if False else None
        base = bld.v("pb")
        cbk = bld.v("cbk")
        bld.emit(f"{cbk} = arith.constant {block} : i32", f"{base} = arith.muli {bld.pid}, {cbk} : i32")
        bsp = bld.v("bsp")
        bld.emit(f"{bsp} = tt.splat {base} : i32 -> {T('i32')}")
        idx = bld.v("idx")
        bld.emit(f"{idx} = arith.addi {bsp}, {pv} : {T('i32')}")
        if param == "gather":
            vals = [bld.load(q, e, idx) for q, e in zip(ptrs, p.ins)]
            r = bld.v("r")
            bld.emit(*apply(p, vals, r, tyt))
            bld.store(po, p.outs[0], offs, r)
        else:
            vals = [bld.load(q, e, offs) for q, e in zip(ptrs, p.ins)]
            r = bld.v("r")
            bld.emit(*apply(p, vals, r, tyt))
            bld.store(po, p.outs[0], idx, r)
    elif form == "C12":
        ptrs = [inp(e, f"op{j}") for j, e in enumerate(p.ins)]
        scratch = out(p.outs[0])
        po = out(p.outs[0])
        offs = bld.offsets()
        vals = [bld.load(q, e, offs) for q, e in zip(ptrs, p.ins)]
        r = bld.v("r")
        bld.emit(*apply(p, vals, r, tyt))
        bld.store(scratch, p.outs[0], offs, r)
        bld.emit("gpu.barrier")
        last, rev, revo, base = bld.const(str(block - 1), "i32"), bld.v("rev"), bld.v("revo"), bld.v("b2")
        cbk = bld.v("cbk")
        bld.emit(f"{rev} = arith.subi {last}, {bld.rng} : {T('i32')}", f"{cbk} = arith.constant {block} : i32",
                 f"{base} = arith.muli {bld.pid}, {cbk} : i32")
        bsp = bld.v("bsp")
        bld.emit(f"{bsp} = tt.splat {base} : i32 -> {T('i32')}", f"{revo} = arith.addi {bsp}, {rev} : {T('i32')}")
        back = bld.load(scratch, p.outs[0], revo)
        bld.store(po, p.outs[0], offs, back)
    elif form == "C13":
        # kernel 1: state = p(x); kernel 2: out = p(back(state), y...) -- the second launch reads the first's output
        ptrs = [inp(e, f"op{j}") for j, e in enumerate(p.ins)]
        st = out(p.outs[0])
        offs = bld.offsets()
        vals = [bld.load(q, e, offs) for q, e in zip(ptrs, p.ins)]
        r = bld.v("r")
        bld.emit(*apply(p, vals, r, tyt))
        bld.store(st, p.outs[0], offs, r)
        c.text = bld.module("cell_k1")
        cp = chain_position(p)
        b2 = B(block)
        state = b2.arg("state", p.outs[0])
        others = [b2.arg(f"y{j}", e) for j, e in enumerate(p.ins) if j != cp]
        alts = [b2.arg("alt_t", p.ins[cp]), b2.arg("alt_f", p.ins[cp])] if (p.outs[0] == "i1" and p.ins[cp] != "i1") \
            else None
        o2 = b2.arg("out2", p.outs[0])
        offs2 = b2.offsets()
        sv = b2.load(state, p.outs[0], offs2)
        av = tuple(b2.load(q, p.ins[cp], offs2) for q in alts) if alts else None
        carried = back_to(b2, sv, p.outs[0], p.ins[cp], av)
        if carried is None:
            c.applicable, c.reason = False, "no carrier back to the operand type"
            return c
        ov = [b2.load(q, e, offs2) for q, e in zip(others, [t for j, t in enumerate(p.ins) if j != cp])]
        r2 = b2.v("r")
        b2.emit(*apply(p, ov[:cp] + [carried] + ov[cp:], r2, (lambda tt: f"tensor<{block}x{tt}>")))
        b2.store(o2, p.outs[0], offs2, r2)
        c.second_text = b2.module("cell_k2")
        c.inputs, c.outputs, c.launches = ins, outs, 2
        c.second_inputs = [(f"y{j}", e, f"second launch op{j}") for j, e in enumerate(p.ins) if j != cp] + \
            ([("alt_t", p.ins[cp], "alt"), ("alt_f", p.ins[cp], "alt")] if alts else [])
        return c
    elif form == "C14":
        e = p.ins[0]
        q = inp(e, "update value")
        po = out(e, 4)
        offs = bld.offsets()
        x = bld.load(q, e, offs)
        four, rem = bld.const("4", "i32"), bld.v("rem")
        bld.emit(f"{rem} = arith.remsi {bld.rng}, {four} : {T('i32')}")
        a = bld.ptrs(po, e, rem)
        t = bld.const("true", "i1")
        old = bld.v("old")
        bld.emit(f"{old} = tt.atomic_rmw {ATOMIC[p.op]}, relaxed, gpu, {a}, {x}, {t} : ({bld.ptrt(e)}, {T(e)}, "
                 f"{T('i1')}) -> {T(e)}")
        outs[-1] = (outs[-1][0], e, 4)
    else:
        raise ValueError(form)
    c.text = bld.module()
    c.inputs, c.outputs = ins, outs
    return c


def build_dot(p, c, K=64) -> Cell:
    """C15: acc = dot(A_t, B_t, acc) over K / 16 k-tiles in an scf.for (tile accumulate), one 16 x 16 output tile."""
    a_t, b_t, attrs = tb.dot_attrs((p.op, p.attr, p.sig))
    m = n = k = 16
    c.grid, c.block = 1, m * n
    A, Bt = TT[a_t], TT[b_t]

    def grid(tag, rows, cols, rr, rc, stride, ptr, et):
        return [f"%{tag}r = tt.expand_dims {rr} {{axis = 1 : i32}} : tensor<{rows}xi32> -> tensor<{rows}x1xi32>",
                f"%{tag}s = arith.constant dense<{stride}> : tensor<{rows}x1xi32>",
                f"%{tag}ro = arith.muli %{tag}r, %{tag}s : tensor<{rows}x1xi32>",
                f"%{tag}c = tt.expand_dims {rc} {{axis = 0 : i32}} : tensor<{cols}xi32> -> tensor<1x{cols}xi32>",
                f"%{tag}rb = tt.broadcast %{tag}ro : tensor<{rows}x1xi32> -> tensor<{rows}x{cols}xi32>",
                f"%{tag}cb = tt.broadcast %{tag}c : tensor<1x{cols}xi32> -> tensor<{rows}x{cols}xi32>",
                f"%{tag}o = arith.addi %{tag}rb, %{tag}cb : tensor<{rows}x{cols}xi32>",
                f"%{tag}p = tt.splat {ptr} : !tt.ptr<{et}> -> tensor<{rows}x{cols}x!tt.ptr<{et}>>",
                f"%{tag}a = tt.addptr %{tag}p, %{tag}o : tensor<{rows}x{cols}x!tt.ptr<{et}>>, tensor<{rows}x{cols}xi32>"]
    L = [f"%ra = tt.make_range {{end = {m} : i32, start = 0 : i32}} : tensor<{m}xi32>",
         f"%rk = tt.make_range {{end = {k} : i32, start = 0 : i32}} : tensor<{k}xi32>",
         f"%rn = tt.make_range {{end = {n} : i32, start = 0 : i32}} : tensor<{n}xi32>"]
    L += grid("ga", m, k, "%ra", "%rk", K, "%A", A)
    L += grid("gb", k, n, "%rk", "%rn", n, "%B", Bt)
    L += grid("gc", m, n, "%ra", "%rn", n, "%C0", "f32") + [f"%xc = tt.load %gca : tensor<{m}x{n}x!tt.ptr<f32>>"]
    L += ["%c0 = arith.constant 0 : i32", "%c1 = arith.constant 1 : i32", f"%ct = arith.constant {K // k} : i32",
          f"%ck = arith.constant {k} : i32", f"%ckn = arith.constant {k * n} : i32",
          f"%acc_out = scf.for %t = %c0 to %ct step %c1 iter_args(%acc = %xc) -> (tensor<{m}x{n}xf32>) : i32 {{",
          "  %ta = arith.muli %t, %ck : i32", f"  %tas = tt.splat %ta : i32 -> tensor<{m}x{k}xi32>",
          f"  %pa = tt.addptr %gaa, %tas : tensor<{m}x{k}x!tt.ptr<{A}>>, tensor<{m}x{k}xi32>",
          f"  %xa = tt.load %pa : tensor<{m}x{k}x!tt.ptr<{A}>>",
          "  %tb = arith.muli %t, %ckn : i32", f"  %tbs = tt.splat %tb : i32 -> tensor<{k}x{n}xi32>",
          f"  %pb = tt.addptr %gba, %tbs : tensor<{k}x{n}x!tt.ptr<{Bt}>>, tensor<{k}x{n}xi32>",
          f"  %xb = tt.load %pb : tensor<{k}x{n}x!tt.ptr<{Bt}>>",
          f"  %y = tt.dot %xa, %xb, %acc{attrs} : tensor<{m}x{k}x{A}> * tensor<{k}x{n}x{Bt}> -> tensor<{m}x{n}xf32>",
          f"  scf.yield %y : tensor<{m}x{n}xf32>", "}"]
    L += grid("go", m, n, "%ra", "%rn", n, "%C", "f32") + [f"tt.store %goa, %acc_out : tensor<{m}x{n}x!tt.ptr<f32>>",
                                                           "tt.return"]
    args = [f"%A: !tt.ptr<{A}> {{tt.divisibility = 16 : i32}}", f"%B: !tt.ptr<{Bt}> {{tt.divisibility = 16 : i32}}",
            "%C0: !tt.ptr<f32> {tt.divisibility = 16 : i32}", "%C: !tt.ptr<f32> {tt.divisibility = 16 : i32}"]
    c.text = ("module {\n  tt.func public @cell(" + ", ".join(args) + ") attributes {noinline = false} {\n    "
              + "\n    ".join(L) + "\n  }\n}\n")
    c.inputs = [("A", a_t, f"A {m}x{K}"), ("B", b_t, f"B {K}x{n}"), ("C0", "f32", "initial accumulator")]
    c.input_sizes = {"A": m * K, "B": K * n, "C0": m * n}
    c.outputs = [("C", "f32", m * n)]
    c.dot_shape = (m, n, K)
    return c
