"""Bit-precise interpretation of inline PTX (DSL v2 increment 9, rc3 02 6.10: per-instruction inline asm with
constraints, packing and several outputs).

``run(asm, constraints, pack, inputs, outputs)`` interprets one ``tt.elementwise_inline_asm`` over all element groups at
once.  Registers are numpy arrays over the groups and hold either bit patterns (with a "definite" mask) or real values
(interval enclosures, for float arithmetic); a value is converted on demand: bits -> real by exact decoding, real ->
bits only for a point exactly representable in the register format (the tt.bitcast policy).

Packing follows the Triton lowering: each tensor operand of element width w occupies ``max(1, pack * w // R)``
consecutive asm operands of register width R (from the constraint letter), element k of a group in bits
``(k * w) % R`` of register ``(k * w) // R``; outputs first, then inputs.

Supported: mov (registers, immediates, vector pack / unpack), and / or / xor / not, shl / shr (logical and arithmetic),
add / sub / mul.lo (wrapping), prmt.b32 (default mode), lop3.b32, shf.{l,r}.{wrap,clamp}.b32, selp, setp on integers
and f16x2, cvt between integer widths and from integers to f32, min / max on f32 and f16x2, predication @p / @!p,
.reg declarations, braces, line comments.  Reads of clocks or SM ids are environment observations without a reference
value (``EnvironmentRead``); loads or stores inside the asm are memory effects not modelled here (``Unsupported``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np


class Unsupported(Exception):
    """An instruction or operand form outside the interpreted subset."""


class EnvironmentRead(Exception):
    """The snippet reads the environment (clock, SM id): no reference value."""


REG_BITS = {"r": 32, "h": 16, "l": 64, "f": 32, "d": 64, "c": 16, "n": 32}
ENV_REGS = ("%globaltimer", "%clock", "%clock64", "%smid", "%nsmid", "%warpid", "%laneid", "%gridid")


@dataclass
class Reg:
    """bits: uint64 array (low ``width`` bits used) with ``definite``; or real: (lo, hi) float arrays, ``fmt`` f32/f16."""
    width: int
    bits: Optional[np.ndarray] = None
    definite: Optional[np.ndarray] = None
    lo: Optional[np.ndarray] = None
    hi: Optional[np.ndarray] = None
    fmt: str = ""


def _mask(width):
    return np.uint64((1 << width) - 1) if width < 64 else np.uint64(0xFFFFFFFFFFFFFFFF)


def _to_bits(reg: Reg, width: int, n: int):
    """Bit view of a register (definite mask)."""
    if reg.bits is not None:
        return reg.bits & _mask(width), reg.definite.copy()
    fmt = reg.fmt or ("f16" if reg.width == 16 else "f32")
    point = reg.lo == reg.hi
    x = np.where(point, reg.lo, 0.0)
    if fmt == "f16":
        f = x.astype(np.float16)
        bits = f.view(np.uint16).astype(np.uint64)
    else:
        f = x.astype(np.float32)
        bits = f.view(np.uint32).astype(np.uint64)
    definite = point & (f.astype(np.float64) == x)
    return bits & _mask(width), definite


def _to_real(reg: Reg, fmt: str):
    """Real view of a register as a float of format f32 / f16 (bits decoded exactly)."""
    if reg.lo is not None:
        return reg.lo, reg.hi, np.ones(reg.lo.shape, dtype=bool)
    if fmt == "f16":
        v = (reg.bits & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16).astype(np.float64)
    else:
        v = (reg.bits & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.float32).astype(np.float64)
    return v, v.copy(), reg.definite & np.isfinite(v)


def _signed(bits, width):
    b = bits.astype(np.uint64) & _mask(width)
    if width >= 64:
        return b.view(np.int64)
    sign = np.uint64(1 << (width - 1))
    return np.where(b & sign, b.astype(np.int64) - (1 << width), b.astype(np.int64))


def _statements(asm: str) -> list:
    text = asm.replace("\\0A", "\n").replace("\\n", "\n").replace("\\t", " ")
    text = " ".join(ln.split("//")[0] for ln in text.split("\n"))
    out = []
    for stmt in text.split(";"):
        s = stmt.strip()
        # drop scope braces (a "{" or "}" not inside an operand list)
        s = re.sub(r"^[{}\s]+", "", s)
        s = re.sub(r"[{}\s]+$", "", s) if not re.search(r"\{[^}]*$", s) else s
        if s:
            out.append(s)
    return out


def _split_ops(rest: str) -> list:
    ops, depth, cur = [], 0, []
    for ch in rest:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            ops.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        ops.append("".join(cur).strip())
    return ops


class Machine:
    def __init__(self, n: int):
        self.n = n
        self.regs = {}
        self.preds = {}
        self.decl = {}

    def imm(self, tok: str):
        t = tok.strip()
        if re.fullmatch(r"-?0[xX][0-9a-fA-F]+", t) or re.fullmatch(r"-?\d+", t):
            v = int(t, 0)
            return Reg(64, bits=np.full(self.n, v & 0xFFFFFFFFFFFFFFFF, dtype=np.uint64),
                       definite=np.ones(self.n, dtype=bool))
        if re.fullmatch(r"0[fF][0-9a-fA-F]{8}", t):
            v = np.array([int(t[2:], 16)], dtype=np.uint32).view(np.float32).astype(np.float64)[0]
            return Reg(32, lo=np.full(self.n, v), hi=np.full(self.n, v), fmt="f32")
        return None

    def get(self, tok: str) -> Reg:
        t = tok.strip()
        if t in ENV_REGS or t.startswith("%globaltimer") or t.startswith("%clock"):
            raise EnvironmentRead(t)
        r = self.imm(t)
        if r is not None:
            return r
        if t not in self.regs:
            raise Unsupported(f"register {t} read before it is written")
        return self.regs[t]

    def width_of(self, tok, default):
        return self.decl.get(tok.strip(), default)

    def put(self, tok: str, reg: Reg, pred):
        t = tok.strip()
        if pred is not None and t in self.regs:
            p, neg = pred
            take = ~p if neg else p
            old = self.regs[t]
            ob, od = _to_bits(old, reg.width, self.n) if old.bits is not None or reg.bits is not None else (None, None)
            if reg.bits is not None:
                nb, nd = reg.bits, reg.definite
                self.regs[t] = Reg(reg.width, bits=np.where(take, nb, ob), definite=np.where(take, nd, od))
                return
            olo, ohi, _ = _to_real(old, reg.fmt or "f32")
            self.regs[t] = Reg(reg.width, lo=np.where(take, reg.lo, olo), hi=np.where(take, reg.hi, ohi), fmt=reg.fmt)
            return
        if pred is not None:
            raise Unsupported(f"predicated first write of {t}")
        self.regs[t] = reg


def _bits_reg(width, bits, definite):
    return Reg(width, bits=bits & _mask(width), definite=definite)


def _exec(m: Machine, stmt: str):
    if stmt.startswith(".reg"):
        mm = re.fullmatch(r"\.reg\s+\.(\w+)\s+(.+)", stmt)
        if not mm:
            raise Unsupported(stmt)
        ty = mm.group(1)
        width = {"pred": 1, "b8": 8, "b16": 16, "b32": 32, "b64": 64, "u8": 8, "u16": 16, "u32": 32, "u64": 64,
                 "s8": 8, "s16": 16, "s32": 32, "s64": 64, "f16": 16, "f32": 32, "f64": 64}.get(ty)
        if width is None:
            raise Unsupported(stmt)
        for name in (x.strip() for x in mm.group(2).split(",")):
            rng = re.fullmatch(r"(\w+)<(\d+)>", name)
            names = [f"{rng.group(1)}{i}" for i in range(int(rng.group(2)))] if rng else [name]
            for nm in names:
                m.decl[nm] = width
        return
    mm = re.fullmatch(r"(?:@(!?)(\w+)\s+)?([\w.]+)\s+(.+)", stmt)
    if not mm:
        raise Unsupported(stmt)
    neg, pname, opcode, rest = mm.group(1) == "!", mm.group(2), mm.group(3), mm.group(4)
    pred = None
    if pname:
        if pname not in m.preds:
            raise Unsupported(f"predicate {pname}")
        pred = (m.preds[pname], neg)
    ops = _split_ops(rest)
    parts = opcode.split(".")
    base, mods = parts[0], parts[1:]
    n = m.n
    if base in ("ld", "st", "atom", "red", "cp", "bar", "membar", "fence"):
        raise Unsupported(f"memory or synchronization effect inside inline asm ({opcode})")
    ty = mods[-1] if mods else ""
    width = int(re.sub(r"\D", "", ty) or 32) if ty and ty[0] in "bus" else (32 if ty in ("f32",) else 16 if ty in ("f16", "f16x2") else 64 if ty == "f64" else 32)

    if base == "mov":
        dst, src = ops
        if dst.startswith("{"):        # unpack: mov.b32 {t0, t1, ...}, src
            names = [x.strip() for x in dst.strip("{}").split(",")]
            bits, d = _to_bits(m.get(src), width, n)
            w = width // len(names)
            for k, nm in enumerate(names):
                m.put(nm, _bits_reg(w, (bits >> np.uint64(k * w)), d.copy()), pred)
            return
        if src.startswith("{"):        # pack: mov.b32 dst, {c0, c1}
            names = [x.strip() for x in src.strip("{}").split(",")]
            w = width // len(names)
            bits = np.zeros(n, dtype=np.uint64)
            d = np.ones(n, dtype=bool)
            for k, nm in enumerate(names):
                b, dd = _to_bits(m.get(nm), w, n)
                bits |= (b & _mask(w)) << np.uint64(k * w)
                d &= dd
            m.put(dst, _bits_reg(width, bits, d), pred)
            return
        r = m.get(src)
        if ty in ("f32", "f16") and r.bits is None:
            m.put(dst, Reg(width, lo=r.lo, hi=r.hi, fmt=ty), pred)
        else:
            b, d = _to_bits(r, width, n)
            m.put(dst, _bits_reg(width, b, d), pred)
        return
    if base in ("and", "or", "xor"):
        a, d1 = _to_bits(m.get(ops[1]), width, n)
        b, d2 = _to_bits(m.get(ops[2]), width, n)
        f = {"and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor}[base]
        m.put(ops[0], _bits_reg(width, f(a, b), d1 & d2), pred)
        return
    if base == "not":
        a, d1 = _to_bits(m.get(ops[1]), width, n)
        m.put(ops[0], _bits_reg(width, ~a, d1), pred)
        return
    if base in ("shl", "shr"):
        a, d1 = _to_bits(m.get(ops[1]), width, n)
        s, d2 = _to_bits(m.get(ops[2]), 32, n)
        s = np.minimum(s, np.uint64(width))
        if base == "shl":
            out = np.where(s >= width, np.uint64(0), a << np.minimum(s, np.uint64(63)))
        elif ty.startswith("s"):
            sa = _signed(a, width)
            out = (sa >> np.minimum(s, np.uint64(width - 1)).astype(np.int64)).astype(np.uint64)
        else:
            out = np.where(s >= width, np.uint64(0), a >> np.minimum(s, np.uint64(63)))
        m.put(ops[0], _bits_reg(width, out, d1 & d2), pred)
        return
    if base in ("add", "sub") and ty and ty[0] in "bus" and "f" not in ty:
        a, d1 = _to_bits(m.get(ops[1]), width, n)
        b, d2 = _to_bits(m.get(ops[2]), width, n)
        out = (a + b) if base == "add" else (a - b)
        m.put(ops[0], _bits_reg(width, out, d1 & d2), pred)
        return
    if base == "mul" and "lo" in mods and ty[0] in "us":
        a, d1 = _to_bits(m.get(ops[1]), width, n)
        b, d2 = _to_bits(m.get(ops[2]), width, n)
        out = np.array([(int(x) * int(y)) & ((1 << width) - 1) for x, y in zip(a.tolist(), b.tolist())],
                       dtype=np.uint64)
        m.put(ops[0], _bits_reg(width, out, d1 & d2), pred)
        return
    if base == "prmt" and ty == "b32" and len(mods) == 1:
        a, d1 = _to_bits(m.get(ops[1]), 32, n)
        b, d2 = _to_bits(m.get(ops[2]), 32, n)
        c, d3 = _to_bits(m.get(ops[3]), 32, n)
        src = a | (b << np.uint64(32))
        out = np.zeros(n, dtype=np.uint64)
        for i in range(4):
            sel = (c >> np.uint64(4 * i)) & np.uint64(0xF)
            byte = (src >> ((sel & np.uint64(7)) * np.uint64(8))) & np.uint64(0xFF)
            byte = np.where(sel & np.uint64(8), np.where(byte & np.uint64(0x80), np.uint64(0xFF), np.uint64(0)), byte)
            out |= byte << np.uint64(8 * i)
        m.put(ops[0], _bits_reg(32, out, d1 & d2 & d3), pred)
        return
    if base == "lop3" and ty == "b32":
        a, d1 = _to_bits(m.get(ops[1]), 32, n)
        b, d2 = _to_bits(m.get(ops[2]), 32, n)
        c, d3 = _to_bits(m.get(ops[3]), 32, n)
        lut = int(ops[4], 0)
        full = np.uint64(0xFFFFFFFF)
        out = np.zeros(n, dtype=np.uint64)
        for i in range(8):
            if lut >> i & 1:
                term = (a if i & 4 else ~a & full) & (b if i & 2 else ~b & full) & (c if i & 1 else ~c & full)
                out |= term
        m.put(ops[0], _bits_reg(32, out, d1 & d2 & d3), pred)
        return
    if base == "shf" and ty == "b32" and len(mods) == 3 and mods[0] in ("l", "r") and mods[1] in ("wrap", "clamp"):
        a, d1 = _to_bits(m.get(ops[1]), 32, n)
        b, d2 = _to_bits(m.get(ops[2]), 32, n)
        c, d3 = _to_bits(m.get(ops[3]), 32, n)
        s = (c & np.uint64(31)) if mods[1] == "wrap" else np.minimum(c, np.uint64(32))
        both = [(int(hi_) << 32) | int(lo_) for lo_, hi_ in zip(a.tolist(), b.tolist())]
        if mods[0] == "l":
            out = [((v << int(k)) >> 32) & 0xFFFFFFFF for v, k in zip(both, s.tolist())]
        else:
            out = [(v >> int(k)) & 0xFFFFFFFF for v, k in zip(both, s.tolist())]
        m.put(ops[0], _bits_reg(32, np.array(out, dtype=np.uint64), d1 & d2 & d3), pred)
        return
    if base == "selp":
        a = m.get(ops[1])
        b = m.get(ops[2])
        p = m.preds.get(ops[3].strip())
        if p is None:
            raise Unsupported(f"selp predicate {ops[3]}")
        ab, ad = _to_bits(a, width, n)
        bb, bd = _to_bits(b, width, n)
        m.put(ops[0], _bits_reg(width, np.where(p, ab, bb), np.where(p, ad, bd)), pred)
        return
    if base == "setp":
        cmp = mods[0]
        if ty == "f16x2":
            dsts = ops[0].split("|")
            for k, dname in enumerate(dsts):
                av, _, ad = _to_real(_half(m.get(ops[1]), k, n), "f16")
                bv, _, bd = _to_real(_half(m.get(ops[2]), k, n), "f16")
                m.preds[dname.strip()] = _cmp(cmp, av, bv) & ad & bd
                if not (ad & bd).all():
                    raise Unsupported("setp on an operand without definite bits")
            return
        if ty and ty[0] in "bus":
            a, d1 = _to_bits(m.get(ops[1]), width, n)
            b, d2 = _to_bits(m.get(ops[2]), width, n)
            if not (d1 & d2).all():
                raise Unsupported("setp on an operand without definite bits")
            if ty[0] == "s":
                a, b = _signed(a, width), _signed(b, width)
            m.preds[ops[0].strip()] = _cmp(cmp, a, b)
            return
        raise Unsupported(stmt)
    if base == "cvt":
        dty, sty = mods[-2], mods[-1]
        if dty[0] in "bus" and sty[0] in "bus" and dty[0] != "b":
            dw, sw = int(dty[1:]), int(sty[1:])
            a, d1 = _to_bits(m.get(ops[1]), sw, n)
            v = _signed(a, sw).astype(np.uint64) if sty[0] == "s" else a
            m.put(ops[0], _bits_reg(dw, v, d1), pred)
            return
        if dty == "f32" and sty[0] in "us" and set(mods[:-2]) <= {"rn", "rz", "rm", "rp"}:
            sw = int(sty[1:])
            a, d1 = _to_bits(m.get(ops[1]), sw, n)
            v = (_signed(a, sw) if sty[0] == "s" else a.astype(np.int64)).astype(np.float64)
            # numerical-difference mode: the real value of the integer (the sitofp rule)
            m.put(ops[0], Reg(32, lo=v, hi=v.copy(), fmt="f32"), pred)
            if not d1.all():
                raise Unsupported("cvt of an operand without definite bits")
            return
        raise Unsupported(stmt)
    if base in ("min", "max") and ty in ("f32", "f16x2"):
        if ty == "f32":
            alo, ahi, ad = _to_real(m.get(ops[1]), "f32")
            blo, bhi, bd = _to_real(m.get(ops[2]), "f32")
            f = np.maximum if base == "max" else np.minimum
            m.put(ops[0], Reg(32, lo=f(alo, blo), hi=f(ahi, bhi), fmt="f32"), pred)
            if not (ad & bd).all():
                raise Unsupported("min / max on NaN or non-definite operands")
            return
        halves = []
        for k in range(2):
            av, _, ad = _to_real(_half(m.get(ops[1]), k, n), "f16")
            bv, _, bd = _to_real(_half(m.get(ops[2]), k, n), "f16")
            if not (ad & bd).all():
                raise Unsupported("min / max f16x2 on NaN or non-definite operands")
            pick_a = (av >= bv) if base == "max" else (av <= bv)
            ab, _ = _to_bits(_half(m.get(ops[1]), k, n), 16, n)
            bb, _ = _to_bits(_half(m.get(ops[2]), k, n), 16, n)
            halves.append(np.where(pick_a, ab, bb))
        m.put(ops[0], _bits_reg(32, halves[0] | (halves[1] << np.uint64(16)), np.ones(n, dtype=bool)), pred)
        return
    raise Unsupported(f"instruction {opcode}")


def _half(reg: Reg, k: int, n: int) -> Reg:
    b, d = _to_bits(reg, 32, n)
    return Reg(16, bits=(b >> np.uint64(16 * k)) & np.uint64(0xFFFF), definite=d)


def _cmp(cmp, a, b):
    return {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b,
            "lo": a < b, "ls": a <= b, "hi": a > b, "hs": a >= b}[cmp]


def run(asm: str, constraints: str, pack: int, inputs: list, outputs: list, total: Optional[int] = None):
    """inputs: [(kind, width, bits, definite, lo, hi)] per tensor operand, flattened over elements (length G * pack);
    kind "bits" (integers, bit patterns) or "real" (floats: lo / hi, the format named by width 16 / 32).
    outputs: [(kind, width)] per result tensor.  Returns [(bits, definite)] or [(lo, hi, ok)] per output."""
    why = classify(asm)
    if why is not None and why.startswith("reads the environment"):
        raise EnvironmentRead(why)
    cons = [c.strip() for c in constraints.split(",")]
    out_cons = [c for c in cons if c.startswith("=")]
    in_cons = [c for c in cons if not c.startswith("=")]
    if inputs:
        total = len(inputs[0][2] if inputs[0][0] == "bits" else inputs[0][4])
    if total is None or total % pack:
        raise Unsupported("element count not a multiple of pack")
    g = total // pack
    m = Machine(g)
    slot = 0

    def regs_for(width, letter):
        return max(1, pack * width // REG_BITS.get(letter, 32))

    # outputs occupy the first operands; record their register ranges
    out_slots = []
    ci = 0
    for kind, width in outputs:
        letter = out_cons[ci].lstrip("=&")[:1] if ci < len(out_cons) else "r"
        k = regs_for(width, letter)
        out_slots.append(list(range(slot, slot + k)))
        slot += k
        ci += k
    ci = 0
    for kind, width, bits, definite, lo, hi in inputs:
        letter = in_cons[ci][:1] if ci < len(in_cons) else "r"
        rbits = REG_BITS.get(letter, 32)
        k = regs_for(width, letter)
        per = max(1, rbits // width) if pack > 1 else 1
        for r in range(k):
            name = f"${slot + r}"
            if kind == "real" and per == 1:
                sel = np.arange(g) * pack + r
                m.regs[name] = Reg(width, lo=np.asarray(lo)[sel], hi=np.asarray(hi)[sel], fmt="f16" if width == 16 else "f32")
                continue
            if kind == "real":
                rb, rd = _to_bits(Reg(width, lo=np.asarray(lo), hi=np.asarray(hi), fmt="f16" if width == 16 else "f32"),
                                  width, total)
            else:
                rb, rd = np.asarray(bits, dtype=np.uint64) & _mask(width), np.asarray(definite)
            word = np.zeros(g, dtype=np.uint64)
            dd = np.ones(g, dtype=bool)
            for e in range(per):
                elem = np.arange(g) * pack + r * per + e
                word |= (rb[elem] & _mask(width)) << np.uint64(e * width)
                dd &= rd[elem]
            m.regs[name] = Reg(rbits, bits=word, definite=dd)
        slot += k
        ci += k
    for stmt in _statements(asm):
        _exec(m, stmt)
    results = []
    for (kind, width), slots in zip(outputs, out_slots):
        letter = "r"
        rbits = REG_BITS.get(letter, 32)
        per = max(1, rbits // width) if pack > 1 else 1
        if kind == "real":
            lo = np.zeros(total)
            hi = np.zeros(total)
            ok = np.zeros(total, dtype=bool)
        else:
            bits = np.zeros(total, dtype=np.uint64)
            definite = np.zeros(total, dtype=bool)
        for r, s in enumerate(slots):
            reg = m.regs.get(f"${s}")
            if reg is None:
                raise Unsupported(f"output ${s} never written")
            for e in range(per):
                elem = np.arange(g) * pack + r * per + e
                if kind == "real" and per == 1 and reg.lo is not None:
                    lo[elem], hi[elem], ok[elem] = reg.lo, reg.hi, True
                    continue
                b, d = _to_bits(reg, rbits if per > 1 else width, g)
                part = (b >> np.uint64(e * width)) & _mask(width)
                if kind == "real":
                    v, _, okk = _to_real(Reg(width, bits=part, definite=d), "f16" if width == 16 else "f32")
                    lo[elem], hi[elem], ok[elem] = v, v, okk
                else:
                    bits[elem], definite[elem] = part, d
        results.append((lo, hi, ok) if kind == "real" else (bits, definite))
    return results


_OPCODES = {"mov", "and", "or", "xor", "not", "shl", "shr", "add", "sub", "mul", "prmt", "lop3", "shf", "selp", "setp",
            "cvt", "min", "max"}


def classify(asm: str) -> Optional[str]:
    """Static check without running: None when every statement is in the interpreted subset, else the reason (an
    environment read, a memory effect, or the first unsupported instruction)."""
    for stmt in _statements(asm):
        if stmt.startswith(".reg"):
            continue
        if any(env in stmt for env in ENV_REGS):
            return "reads the environment (clock / SM id): no reference value"
        mm = re.fullmatch(r"(?:@!?\w+\s+)?([\w.]+)\s+.+", stmt)
        if not mm:
            return f"unparsed statement {stmt[:40]!r}"
        base = mm.group(1).split(".")[0]
        if base in ("ld", "st", "atom", "red", "cp", "bar", "membar", "fence"):
            return f"memory or synchronization effect inside inline asm ({mm.group(1)})"
        if base not in _OPCODES:
            return f"instruction {mm.group(1)} is not modelled"
    return None
