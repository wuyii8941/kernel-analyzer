"""K1 / K2 of an unseen kernel against the primitive catalog and the composition cells (RQ4).

K1: operations of the kernel that match no measured catalog instance by full signature -- operation name, the
attributes that select a lowering (predicate, overflow flags, rounding, input precision, NaN propagation, libdevice
symbol, ...), operand element types and result element types.  The catalog side is built the same way, by parsing
each measured instance's own probe module, so the two sides use one key function.  (The first version compared
operation names only.)

K2: pairs (catalogued primitive, enclosing form) of the kernel whose composition cells are not all bit identical under
the strict rule (cells.rescore).  The enclosing form of an operation is the innermost region that composes it: a
reduce combine (C3), a scan combine (C4), a map_elementwise body (C5), scf.if (C7), scf.for (C8), scf.while (C9),
otherwise the straight-line chain (C1); an atomic's update is the primitive of its kind in C14, a dot is its own
instance in C15.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "2_tool/src"))
sys.path.insert(0, str(HERE))

# operations that move, index or structure data: their values are exact by construction or checked by the forms
STRUCTURAL = {"scf.yield", "tt.return", "tt.reduce.return", "tt.scan.return", "tt.map_elementwise.return",
              "scf.condition", "tt.func", "tt.load", "tt.store", "tt.addptr", "tt.splat", "tt.broadcast",
              "tt.expand_dims", "tt.make_range", "tt.get_program_id", "tt.get_num_programs", "tt.reshape", "tt.trans",
              "tt.join", "tt.split", "tt.cat", "arith.constant", "tt.reduce", "tt.scan", "scf.for", "scf.if",
              "scf.while", "tt.atomic_rmw", "tt.atomic_cas", "gpu.barrier", "tt.assert", "tt.print", "tt.gather",
              "tt.map_elementwise", "tt.histogram", "tt.call", "ttg.convert_layout", "tt.unsplat", "tt.int_to_ptr",
              "tt.ptr_to_int", "tt.make_tensor_ptr", "tt.advance", "ub.poison", "cf.cond_br", "cf.br", "cf.assert",
              "tt.descriptor_load", "tt.descriptor_store", "tt.make_tensor_descriptor", "llvm.intr.assume"}
REGION_FORM = {"tt.reduce": "C3", "tt.scan": "C4", "tt.map_elementwise": "C5", "scf.if": "C7", "scf.for": "C8",
               "scf.while": "C9"}
DROP_ATTRS = {"libname", "libpath", "value"}
HINT_ATTRS = {"tt.contiguity", "tt.divisibility", "tt.constancy"}      # analysis hints: no change to element values
ATOMIC_OP = {"fadd": "arith.addf", "add": "arith.addi", "max": "arith.maxsi", "min": "arith.minsi",
             "umax": "arith.maxui", "umin": "arith.minui", "and": "arith.andi", "or": "arith.ori",
             "xor": "arith.xori"}


def _elem(t) -> str:
    e = getattr(t, "elem", t)
    return str(e).lower() if isinstance(e, str) else "ptr"


def op_key(op) -> tuple:
    """Full signature of one operation (shapes and layouts dropped: the lowering is per element)."""
    ins = tuple(_elem(t) for t in op.operand_types)
    outs = tuple(_elem(t) for t in op.result_types)
    if op.name == "arith.select" and outs:
        ins = ("i1", outs[0], outs[0])     # a scalar condition prints one type (": f32"), a tensor one three
    attrs = {}
    for k, v in op.attrs.items():
        v = str(v).strip()
        if k in DROP_ATTRS or k in HINT_ATTRS or (k == "fastmath" and "none" in v):
            continue
        attrs[k] = v
    if op.name == "tt.dot":
        if not all(e == "f32" for e in ins[:2]):
            attrs.pop("inputPrecision", None)            # input precision only selects the f32 lowering
        if not any(e.startswith("f8") for e in ins[:2]):
            attrs.pop("maxNumImpreciseAcc", None)        # only fp8 dots accumulate imprecisely
        elif attrs.get("maxNumImpreciseAcc", "0 : i32").split(":")[0].strip() == "0":
            attrs.pop("maxNumImpreciseAcc", None)
        if all(e == "f32" for e in ins[:2]):
            attrs.setdefault("inputPrecision", "tf32")   # the printed default
    return (op.name, tuple(sorted(attrs.items())), ins, outs)


def catalog_keys(catalog_json=HERE / "results/catalog.json") -> dict:
    """Full signature -> pid for every measured instance."""
    import catalog as C
    import primitives as P
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir
    prims = {p.pid: p for p in P.all_prims()}
    keys = {}
    for r in json.loads(Path(catalog_json).read_text())["records"]:
        if r.get("status") != "measured" or r["pid"] not in prims:
            continue
        p = prims[r["pid"]]
        if p.family in ("literal", "scalar"):
            continue
        try:
            text, _ = C.module_text(p)
        except Exception:
            continue
        want = "tt.extern_elementwise" if p.source == "libdevice" else p.op
        for fn in parse_ttir(text).funcs.values():
            for op in fn.walk():
                if op.name == want:
                    keys.setdefault(op_key(op), r["pid"])
    return keys


def _walk_ctx(region, ctx):
    for block in region.blocks:
        for op in block.ops:
            yield op, ctx
            inner = REGION_FORM.get(op.name, ctx)
            for reg in op.regions:
                yield from _walk_ctx(reg, inner)


def cell_table(paths=(HERE / "results/cells_default.jsonl.gz", HERE / "results/cells_options.jsonl.gz")) -> dict:
    """(pid, form) -> Counter of strict cell statuses."""
    from cells import rescore
    tab = defaultdict(Counter)
    for p in paths:
        for line in gzip.decompress(Path(p).read_bytes()).decode().splitlines():
            if line.strip():
                r = rescore(json.loads(line))
                tab[(r["pid"], r["form"])][r["status"]] += 1
    return tab


def k2_status(statuses: Counter) -> str:
    if not statuses or statuses.get("not applicable", 0) == sum(statuses.values()):
        return "no cell"
    applicable = {k: v for k, v in statuses.items() if k != "not applicable"}
    if set(applicable) == {"bit identical"}:
        return "bit identical"
    if "mismatch" in applicable:
        return "mismatch"
    if "bit identical" in applicable or "partial" in applicable:
        return "partial"
    return "not established"


def classify(ttir_text: str, keys: dict, cells: dict) -> dict:
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir
    m = parse_ttir(ttir_text)
    k1, k1_name_only, pairs, forms = Counter(), Counter(), Counter(), Counter()
    names = {k[0] for k in keys}
    for fn in m.funcs.values():
        for op, ctx in _walk_ctx(fn.body, "C1"):
            if op.name in REGION_FORM:
                forms[REGION_FORM[op.name]] += 1
            if op.name == "tt.load" and len(op.operands) >= 2:
                forms["C10"] += 1
            if op.name == "tt.atomic_rmw":
                forms["C14"] += 1
                kind = str(op.attrs.get("atomic_rmw_op", op.attrs.get("rmw_op", ""))).strip().lower()
                base = ATOMIC_OP.get(kind.split()[-1] if kind else "")
                e = _elem(op.result_types[0]) if op.result_types else "?"
                if base:
                    attr = "none" if base in ("arith.addi",) else "default"
                    pairs[(f"F1|{base}|{attr}|{e}", "C14")] += 1
                continue
            if op.name in STRUCTURAL:
                continue
            key = op_key(op)
            if op.name == "tt.bitcast" and key[2][:1] == key[3][:1]:
                continue                        # a bitcast to its own type (pointer or integer casts): no value change
            if op.name not in names:
                k1_name_only[op.name] += 1
            pid = keys.get(key)
            if pid is None:
                k1["|".join([key[0], ",".join(f"{a}={v}" for a, v in key[1]), "x".join(key[2]), "x".join(key[3])])] += 1
                continue
            if op.name in ("tt.dot", "tt.dot_scaled"):
                forms["C15"] += 1
                pairs[(pid, "C15")] += 1
            else:
                pairs[(pid, ctx)] += 1
    k2 = {}
    for (pid, form), n in pairs.items():
        k2[f"{pid} @ {form}"] = k2_status(cells.get((pid, form), Counter()))
    return {"forms": dict(forms), "k1": dict(k1), "k1_name_only": dict(k1_name_only),
            "pairs": {f"{p} @ {f}": n for (p, f), n in pairs.items()}, "k2": k2,
            "k2_not_bit_identical": sorted(k for k, v in k2.items() if v != "bit identical")}
