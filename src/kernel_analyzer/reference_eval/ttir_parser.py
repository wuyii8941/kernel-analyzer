"""Parser for TTIR as printed by Triton (``asm['ttir']``).

The printer emits one operation per line and opens regions with a trailing
``{``.  The parser works line by line with a region stack, and parses each
operation's operands, attributes and types with a small tokenizer.  It does
not decide semantics: an unknown operation is parsed into a generic record
and rejected later by the mapping (instruction-processing coverage).
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from typing import Optional, Union


class TTIRParseError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PtrType:
    pointee: "Union[str, PtrType]"
    address_space: int = 1


@dataclass(frozen=True)
class TType:
    """``shape == ()`` for scalars.  ``elem`` is ``'f32'``, ``'i1'``, ... or a PtrType."""

    shape: tuple
    elem: Union[str, PtrType]

    @property
    def is_tensor(self) -> bool:
        return self.shape != ()

    @property
    def is_ptr(self) -> bool:
        return isinstance(self.elem, PtrType)

    def with_elem(self, elem) -> "TType":
        return TType(self.shape, elem)

    def __str__(self):
        elem = f"!tt.ptr<{self.elem.pointee}>" if self.is_ptr else self.elem
        if not self.shape:
            return str(elem)
        return "tensor<" + "x".join(str(d) for d in self.shape) + f"x{elem}>"


def parse_type(text: str) -> TType:
    text = text.strip()
    if text.startswith("tensor<") and text.endswith(">"):
        inner = text[len("tensor<"):-1]
        dims = []
        rest = inner
        while True:
            m = re.match(r"(\d+)x", rest)
            if not m:
                break
            dims.append(int(m.group(1)))
            rest = rest[m.end():]
        return TType(tuple(dims), _parse_elem(rest))
    return TType((), _parse_elem(text))


def _parse_elem(text: str):
    text = text.strip()
    if text.startswith("!tt.ptr<") and text.endswith(">"):
        inner = text[len("!tt.ptr<"):-1]
        parts = _split_top(inner, ",")
        pointee = _parse_elem(parts[0])
        space = int(parts[1]) if len(parts) > 1 else 1
        return PtrType(pointee.elem if isinstance(pointee, TType) else pointee, space)
    if text.startswith("tensor<"):
        return parse_type(text)  # pointer to tensor (block pointers)
    return text


def _split_top(text: str, sep: str) -> list[str]:
    """Split on ``sep`` outside of brackets and strings."""

    parts, depth, current, in_string = [], 0, [], False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            current.append(ch)
            if ch == "\\":
                current.append(text[i + 1])
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
            current.append(ch)
        elif ch in "<([{":
            depth += 1
            current.append(ch)
        elif ch in ">)]}":
            if ch == ">" and i > 0 and text[i - 1] == "-":
                current.append(ch)  # part of '->'
            else:
                depth -= 1
                current.append(ch)
        elif text.startswith(sep, i) and depth == 0:
            parts.append("".join(current).strip())
            current = []
            i += len(sep)
            continue
        else:
            current.append(ch)
        i += 1
    if current or parts:
        parts.append("".join(current).strip())
    return [p for p in parts if p != ""]


# ---------------------------------------------------------------------------
# IR records
# ---------------------------------------------------------------------------


@dataclass
class TOp:
    name: str
    results: list
    result_types: list
    operands: list
    attrs: dict
    regions: list
    line: int
    text: str
    operand_types: list = field(default_factory=list)
    successors: list = field(default_factory=list)  # cf branches: (label, [operands])

    @property
    def node_id(self) -> str:
        return f"{self.name}@{self.line}"


@dataclass
class TBlock:
    label: Optional[str]
    args: list  # [(name, TType)]
    ops: list


@dataclass
class TRegion:
    blocks: list

    @property
    def entry(self) -> TBlock:
        return self.blocks[0]


@dataclass
class TFunc:
    name: str
    params: list  # [(name, TType, attr_text)]
    result_types: list
    body: TRegion
    visibility: str = "public"

    def walk(self):
        yield from _walk_region(self.body)


@dataclass
class TModule:
    funcs: dict

    def entry(self, name: Optional[str] = None) -> TFunc:
        if name is not None:
            return self.funcs[name]
        publics = [f for f in self.funcs.values() if f.visibility == "public"]
        if len(publics) != 1:
            raise TTIRParseError(f"expected one public function, found {[f.name for f in publics]}")
        return publics[0]


def _walk_region(region: TRegion):
    for block in region.blocks:
        for op in block.ops:
            yield op
            for sub in op.regions:
                yield from _walk_region(sub)


# ---------------------------------------------------------------------------
# Line-level helpers
# ---------------------------------------------------------------------------


def strip_locations(text: str) -> str:
    """Remove ``#loc`` definitions and every balanced ``loc(...)`` suffix."""

    lines = [line for line in text.splitlines() if not line.startswith("#loc")]
    text = "\n".join(lines)
    out, i = [], 0
    while True:
        j = text.find("loc(", i)
        if j < 0:
            out.append(text[i:])
            break
        # Only strip when 'loc(' is a standalone token.
        if j > 0 and (text[j - 1].isalnum() or text[j - 1] in "_."):
            out.append(text[i:j + 4])
            i = j + 4
            continue
        out.append(text[i:j].rstrip(" "))
        depth, k, in_string = 0, j + 3, False
        while k < len(text):
            ch = text[k]
            if in_string:
                if ch == "\\":
                    k += 1
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        i = k + 1
    return "".join(out)


_VALUE = r"%[A-Za-z0-9_$.\-]+(?:#\d+)?"
_RESULTS = re.compile(r"^((?:%[A-Za-z0-9_$.\-]+(?::\d+)?)(?:\s*,\s*%[A-Za-z0-9_$.\-]+(?::\d+)?)*)\s*=\s*(.*)$")


def _expand_results(text: str) -> list[str]:
    names = []
    for part in _split_top(text, ","):
        m = re.fullmatch(r"(%[A-Za-z0-9_$.\-]+)(?::(\d+))?", part.strip())
        if not m:
            raise TTIRParseError(f"bad result list {text!r}")
        if m.group(2):
            names.extend(f"{m.group(1)}#{i}" for i in range(int(m.group(2))))
        else:
            names.append(m.group(1))
    return names


def parse_attr_dict(text: str) -> dict:
    """``{a = 1 : i32, b = "x", unit}`` or ``<{...}>`` -> {key: raw value}."""

    text = text.strip()
    if text.startswith("<{") and text.endswith("}>"):
        text = text[1:-1]
    if not (text.startswith("{") and text.endswith("}")):
        raise TTIRParseError(f"not an attribute dictionary: {text!r}")
    attrs = {}
    for item in _split_top(text[1:-1], ","):
        if "=" in item and not item.strip().startswith('"'):
            key, value = item.split("=", 1)
            attrs[key.strip()] = value.strip()
        else:
            attrs[item.strip()] = "unit"
    return attrs


def attr_int(raw: str) -> int:
    return int(raw.split(":")[0].strip())


def _find_top(text: str, token: str, start: int = 0) -> int:
    """Index of ``token`` outside brackets/strings, or -1."""

    depth, in_string, i = 0, False, start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif text.startswith(token, i) and depth == 0:
            return i
        elif ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            if not (ch == ">" and i > 0 and text[i - 1] == "-"):
                depth -= 1
        i += 1
    return -1


def _extract_attr_dicts(text: str) -> tuple[str, dict]:
    """Remove top-level ``{...}`` / ``<{...}>`` dictionaries from an op line."""

    attrs = {}
    out, i, depth_other, in_string = [], 0, 0, False
    while i < len(text):
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\":
                out.append(text[i + 1])
                i += 1
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        starts_dict = (ch == "{" or text.startswith("<{", i)) and depth_other == 0
        if starts_dict:
            open_len = 2 if text.startswith("<{", i) else 1
            depth, k = 0, i + open_len - 1
            in_s = False
            while k < len(text):
                c = text[k]
                if in_s:
                    if c == "\\":
                        k += 1
                    elif c == '"':
                        in_s = False
                elif c == '"':
                    in_s = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            end = k + (2 if open_len == 2 else 1)
            attrs.update(parse_attr_dict(text[i:end]))
            i = end
            continue
        if ch in "<([":
            depth_other += 1
        elif ch in ">)]":
            if not (ch == ">" and i > 0 and text[i - 1] == "-"):
                depth_other -= 1
        out.append(ch)
        i += 1
    return "".join(out), attrs


def _values(text: str) -> list[str]:
    return re.findall(_VALUE, text)


def _signature(text: str) -> tuple[list, list]:
    """Parse a trailing type signature ``: T`` / ``: (A, B) -> C`` / ``: A -> B``.

    Returns (operand_or_main_types, result_types).
    """

    idx = _find_top(text, " : ")
    if idx < 0:
        if text.rstrip().endswith(":"):
            return [], []
        return [], []
    sig = text[idx + 3:].strip()
    arrow = _find_top(sig, "->")
    if arrow >= 0:
        left, right = sig[:arrow].strip(), sig[arrow + 2:].strip()
        return _type_list(left), _type_list(right)
    return _type_list(sig), []


def _type_list(text: str) -> list:
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    if not text:
        return []
    out = []
    for part in _split_top(text, ","):
        part = part.strip()
        if " * " in part:  # tt.dot: A * B
            out.extend(parse_type(p) for p in part.split(" * "))
        else:
            out.append(parse_type(part))
    return out


# ---------------------------------------------------------------------------
# Operation parsing
# ---------------------------------------------------------------------------

_CAST_OPS = {"arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui",
             "arith.extsi", "arith.extui", "arith.trunci", "arith.bitcast", "arith.index_cast",
             "arith.index_castui"}
_ARROW_OPS = {"tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape", "tt.trans", "tt.bitcast",
              "tt.fp_to_fp", "tt.int_to_ptr", "tt.ptr_to_int", "tt.unsplat", "tt.join", "tt.split",
              "tt.cat", "tt.histogram", "tt.gather"}


def _parse_op_line(body: str, results: list, line_no: int, raw: str) -> TOp:
    body = body.strip()
    # Generic form: "name"(operands) <{props}> ({  or  ... : (types) -> types
    m = re.match(r'^"([A-Za-z0-9_.]+)"\((.*?)\)\s*(.*)$', body)
    if m:
        name = m.group(1)
        operands = _values(m.group(2))
        rest, attrs = _extract_attr_dicts(m.group(3))
        op = TOp(name, results, [], operands, attrs, [], line_no, raw)
        operand_types, result_types = _signature(" " + rest.strip())
        op.operand_types, op.result_types = operand_types, result_types
        return op
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*)\s*(.*)$", body)
    if not m:
        raise TTIRParseError(f"line {line_no}: cannot parse {raw!r}")
    name, rest = m.group(1), m.group(2)
    # Strings are kept intact by the attribute extractor; keep the raw text.
    rest_no_attrs, attrs = _extract_attr_dicts(rest)
    op = TOp(name, results, [], [], attrs, [], line_no, raw)
    head_end = _find_top(rest_no_attrs, " : ")
    head = rest_no_attrs if head_end < 0 else rest_no_attrs[:head_end]
    fm = re.search(r"fastmath<([^>]*)>", head)
    if fm:
        op.attrs["fastmath"] = fm.group(1)
    # String literals (asm text, messages) never contain operands.
    head = re.sub(r'"(?:[^"\\]|\\.)*"', '""', head)
    operand_types, result_types = _signature(rest_no_attrs)

    if name in ("arith.cmpf", "arith.cmpi"):
        pred, *_ = [p.strip() for p in head.split(",")]
        op.attrs["predicate"] = pred
        op.operands = _values(head)
        op.operand_types = operand_types * 2
        op.result_types = [operand_types[0].with_elem("i1")]
    elif name == "arith.constant":
        value_text = head.strip()
        op.attrs["value"] = value_text
        op.result_types = operand_types
        if not operand_types and value_text in ("true", "false"):
            op.result_types = [parse_type("i1")]  # MLIR prints i1 constants without a type
    elif name in _CAST_OPS:
        idx = _find_top(rest_no_attrs, " : ")
        sig = rest_no_attrs[idx + 3:]
        src, dst = sig.split(" to ")
        op.operands = _values(head)
        op.operand_types = [parse_type(src)]
        op.result_types = [parse_type(dst)]
    elif name == "arith.select":
        op.operands = _values(head)
        op.operand_types = operand_types
        op.result_types = [operand_types[-1]]
    elif name in _ARROW_OPS:
        for key in ("rounding", "allow_reorder", "efficient_layout"):
            mm = re.search(rf"\b{key}\b(?:\s*=\s*([A-Za-z0-9_]+))?", head)
            if mm:
                op.attrs[key] = mm.group(1) or "unit"
        op.operands = _values(head)
        op.operand_types, op.result_types = operand_types, result_types
    elif name == "tt.addptr":
        op.operands = _values(head)
        op.operand_types = operand_types
        op.result_types = [operand_types[0]]
    elif name == "tt.load":
        op.operands = _values(head)
        for key in ("cacheModifier", "evictionPolicy", "isVolatile", "padding"):
            mm = re.search(rf"\b{key}\s*=\s*([A-Za-z0-9_]+)", head)
            if mm:
                op.attrs[key] = mm.group(1)
        ptr_type = operand_types[0]
        op.operand_types = [ptr_type]
        op.result_types = [ptr_type.with_elem(ptr_type.elem.pointee)] if ptr_type.is_ptr else []
    elif name == "tt.store":
        op.operands = _values(head)
        op.operand_types = operand_types
    elif name in ("tt.make_range",):
        op.result_types = operand_types
    elif name in ("tt.get_program_id", "tt.get_num_programs"):
        op.attrs["axis"] = head.strip()
        op.result_types = operand_types
    elif name == "tt.dot":
        parts = _split_top(head, ",")
        op.operands = [p for p in parts if p.startswith("%")]
        for p in parts:
            if "=" in p:
                key, value = p.split("=", 1)
                op.attrs[key.strip()] = value.strip()
        op.operand_types, op.result_types = operand_types, result_types
    elif name in ("tt.atomic_rmw", "tt.atomic_cas"):
        parts = [p.strip() for p in _split_top(head, ",")]
        words = [p for p in parts if not p.startswith("%")]
        if name == "tt.atomic_rmw":
            op.attrs["rmw_op"], op.attrs["sem"], op.attrs["scope"] = words[:3]
        else:
            op.attrs["sem"], op.attrs["scope"] = words[:2]
        op.operands = [p for p in parts if p.startswith("%")]
        op.operand_types, op.result_types = operand_types, result_types
    elif name == "tt.extern_elementwise":
        op.operands = _values(head)
        op.operand_types, op.result_types = operand_types, result_types
    elif name == "tt.elementwise_inline_asm":
        sm = re.match(r'\s*"((?:[^"\\]|\\.)*)"', rest)
        op.attrs["asm"] = sm.group(1) if sm else ""
        op.operands = _values(head)
        op.operand_types, op.result_types = operand_types, result_types
    elif name == "tt.clampf":
        mm = re.search(r"propagateNan\s*=\s*([A-Za-z0-9_]+)", head)
        op.attrs["propagateNan"] = mm.group(1) if mm else "none"
        op.operands = _values(head)
        op.operand_types = operand_types
        op.result_types = operand_types[:1]
    elif name in ("tt.assert", "tt.print"):
        sm = re.search(r'"((?:[^"\\]|\\.)*)"', rest)
        op.attrs["message"] = sm.group(1) if sm else ""
        op.operands = _values(head)
        op.operand_types = operand_types
    elif name in ("tt.return", "scf.yield", "tt.reduce.return", "tt.scan.return"):
        op.operands = _values(head)
        op.operand_types = operand_types
    elif name == "scf.condition":
        op.operands = _values(head)
        op.operand_types = operand_types
    elif name == "scf.for":
        mm = re.match(r"\s*(%[\w$.\-]+)\s*=\s*(%[\w$.\-]+)\s+to\s+(%[\w$.\-]+)\s+step\s+(%[\w$.\-]+)(.*)$", head)
        if not mm:
            raise TTIRParseError(f"line {line_no}: bad scf.for {raw!r}")
        op.attrs["induction_var"] = mm.group(1)
        op.operands = [mm.group(2), mm.group(3), mm.group(4)]
        iter_part = mm.group(5)
        im = re.search(r"iter_args\((.*?)\)\s*->\s*\((.*)\)", iter_part)
        op.attrs["iter_args"] = []
        if im:
            for pair in _split_top(im.group(1), ","):
                arg, init = [s.strip() for s in pair.split("=")]
                op.attrs["iter_args"].append(arg)
                op.operands.append(init)
            op.result_types = _type_list(im.group(2))
        op.attrs["iv_type"] = operand_types[0] if operand_types else TType((), "i32")
    elif name == "scf.if":
        mm = re.match(r"\s*(%[\w$.\-]+)\s*(?:->\s*\((.*)\))?\s*$", head)
        if not mm:
            raise TTIRParseError(f"line {line_no}: bad scf.if {raw!r}")
        op.operands = [mm.group(1)]
        op.result_types = _type_list(mm.group(2) or "")
    elif name == "scf.while":
        mm = re.match(r"\s*\((.*?)\)\s*$", head)
        op.attrs["before_args"] = []
        if mm and mm.group(1).strip():
            for pair in _split_top(mm.group(1), ","):
                arg, init = [s.strip() for s in pair.split("=")]
                op.attrs["before_args"].append(arg)
                op.operands.append(init)
        op.operand_types, op.result_types = operand_types, result_types
    elif name in ("cf.cond_br", "cf.br"):
        targets = re.findall(r"(\^[\w]+)(?:\(([^)]*)\))?", head)
        cond = re.match(r"\s*(%[\w$.\-]+)\s*,", head) if name == "cf.cond_br" else None
        op.operands = [cond.group(1)] if cond else []
        for label, args in targets:
            vals = _values(args.split(":")[0]) if args else []
            op.successors.append((label, vals))
    elif name == "tt.call":
        mm = re.match(r"\s*@([\w$.\-]+)\((.*?)\)", head)
        op.attrs["callee"] = mm.group(1) if mm else ""
        op.operands = _values(mm.group(2) if mm else head)
        op.operand_types, op.result_types = operand_types, result_types
    else:
        # Default pretty form: operands, then ': T' (result type = last type).
        op.operands = _values(head)
        op.operand_types = operand_types
        if result_types:
            op.result_types = result_types
        elif results and operand_types:
            op.result_types = [operand_types[-1]] * len(results)
    if results and len(op.result_types) < len(results) and op.result_types:
        op.result_types = op.result_types + [op.result_types[-1]] * (len(results) - len(op.result_types))
    return op


_FUNC = re.compile(r"^tt\.func\s+(public|private)?\s*@([\w$.\-]+)\((.*)\)\s*(?:->\s*(.*?))?\s*(attributes\s*\{.*\})?\s*\{$")


def _parse_params(text: str) -> list:
    params = []
    for part in _split_top(text, ","):
        mm = re.match(r"(%[\w$.\-]+)\s*:\s*(.*)$", part.strip())
        if not mm:
            raise TTIRParseError(f"bad parameter {part!r}")
        type_text, attrs = _extract_attr_dicts(mm.group(2))
        params.append((mm.group(1), parse_type(type_text.strip()), attrs))
    return params


def _strip_comment(line: str) -> str:
    """Drop a trailing ``// ...`` comment that is not inside a string literal."""

    in_string = False
    for i, ch in enumerate(line):
        if in_string:
            if ch == '"' and line[i - 1] != "\\":
                in_string = False
        elif ch == '"':
            in_string = True
        elif line.startswith("//", i):
            return line[:i]
    return line


def parse_ttir(text: str) -> TModule:
    """Parse TTIR text; results are cached by text (modules are not mutated by the evaluator)."""

    return _parse_ttir_cached(text)


@functools.lru_cache(maxsize=128)
def _parse_ttir_cached(text: str) -> TModule:
    return _parse_ttir(text)


def _parse_ttir(text: str) -> TModule:
    lines = strip_locations(text).splitlines()
    funcs = {}
    # Stack entries: (kind, payload).  kind in {"func", "region"}.
    stack: list = []
    current_func: Optional[TFunc] = None

    def current_block() -> TBlock:
        region = stack[-1][1]
        if not region.blocks:
            region.blocks.append(TBlock(None, [], []))
        return region.blocks[-1]

    def open_region(owner: Optional[TOp]):
        region = TRegion([])
        if owner is not None:
            owner.regions.append(region)
        stack.append(("region", region, owner))
        return region

    for line_no, raw in enumerate(lines, 1):
        line = _strip_comment(raw).strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("module"):
            continue
        if current_func is None:
            if line == "}":
                continue
            m = _FUNC.match(line)
            if not m:
                raise TTIRParseError(f"line {line_no}: expected tt.func, got {line!r}")
            ret = _type_list(m.group(4)) if m.group(4) else []
            current_func = TFunc(m.group(2), _parse_params(m.group(3)), ret, TRegion([]),
                                 m.group(1) or "public")
            stack = [("region", current_func.body, None)]
            current_func.body.blocks.append(TBlock(None, [(p[0], p[1]) for p in current_func.params], []))
            continue
        # Block label.
        bm = re.match(r"^(\^[\w]+)(?:\((.*)\))?:$", line)
        if bm:
            args = []
            if bm.group(2):
                for part in _split_top(bm.group(2), ","):
                    name, type_text = part.split(":", 1)
                    args.append((name.strip(), parse_type(type_text.strip())))
            region = stack[-1][1]
            if region.blocks and not region.blocks[-1].ops and region.blocks[-1].label is None \
                    and not region.blocks[-1].args:
                region.blocks[-1] = TBlock(bm.group(1), args, [])
            else:
                region.blocks.append(TBlock(bm.group(1), args, []))
            continue
        if line.startswith("}"):
            kind, region, owner = stack.pop()
            tail = line[1:].strip()
            if not stack:
                funcs[current_func.name] = current_func
                current_func = None
                continue
            if tail.startswith("else {") or tail.startswith("do {"):
                open_region(owner)
                continue
            if tail.startswith(", {") or tail.startswith(",{"):
                open_region(owner)
                continue
            if tail.startswith(")"):
                # End of a generic op's region list: ') : (types) -> types'
                rest, attrs = _extract_attr_dicts(tail[1:])
                if owner is not None:
                    owner.attrs.update(attrs)
                    operand_types, result_types = _signature(" " + rest.strip())
                    owner.operand_types = operand_types
                    owner.result_types = result_types
                continue
            if tail.startswith("{") and owner is not None:
                _, attrs = _extract_attr_dicts(tail)
                owner.attrs.update(attrs)
            continue
        m = _RESULTS.match(line)
        results, body = ([], line) if not m else (_expand_results(m.group(1)), m.group(2))
        opens_generic_region = body.endswith("({")
        opens_region = body.endswith("{") and not opens_generic_region
        head = body[:-2] if opens_generic_region else (body[:-1] if opens_region else body)
        op = _parse_op_line(head, results, line_no, raw)
        current_block().ops.append(op)
        if opens_region or opens_generic_region:
            region = open_region(op)
            # Pretty forms declare entry-block arguments in the op header.
            if op.name == "scf.for":
                types = [op.attrs["iv_type"]] + list(op.result_types)
                names = [op.attrs["induction_var"]] + op.attrs["iter_args"]
                region.blocks.append(TBlock(None, list(zip(names, types)), []))
            elif op.name == "scf.while":
                region.blocks.append(TBlock(None, list(zip(op.attrs["before_args"], op.operand_types)), []))
    if current_func is not None:
        raise TTIRParseError("unterminated function")
    return TModule(funcs)
