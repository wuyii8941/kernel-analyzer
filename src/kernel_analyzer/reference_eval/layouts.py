"""Distributed layouts from a captured TTGIR (locked Triton 3.6.0 lowering).

Used by the reference evaluator for two kinds of evidence that TTIR does not carry:

* the combination order of a ``tt.reduce`` (``reduce_layouts``): sequential within a thread in register order,
  butterfly over the lanes along the axis, butterfly over the warps along the axis.  The same model drives the
  bitwise emulator (``emulate.py``), where it is checked bit for bit against the device for float sums;
* which threads hold each element of a loaded or stored tensor (``access_layouts``, ``element_threads``), for the
  thread-level conflict check between a store and a later load in one program.

Only blocked and slice layouts are modelled; anything else is reported as unknown and the caller treats the
evidence as missing (never as a proof either way).
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

_LIST = r"\[([\d,\s]*)\]"


def parse_layouts(ttgir: str) -> dict:
    layouts = {}
    for m in re.finditer(r"^(#[\w]+)\s*=\s*#ttg\.blocked<\{([^}]*)\}>", ttgir, re.M):
        body = m.group(2)
        vals = {k: [int(v) for v in re.search(k + r"\s*=\s*" + _LIST, body).group(1).split(",") if v.strip()]
                for k in ("sizePerThread", "threadsPerWarp", "warpsPerCTA", "order")}
        layouts[m.group(1)] = {"kind": "blocked", **vals}
    for m in re.finditer(r"^(#[\w]+)\s*=\s*#ttg\.slice<\{dim\s*=\s*(\d+)\s*:\s*i32,\s*parent\s*=\s*(#[\w]+)\}>", ttgir,
                         re.M):
        layouts[m.group(1)] = {"kind": "slice", "dim": int(m.group(2)), "parent": m.group(3)}
    return layouts


def _axis_params(layouts: dict, name: str, axis: int) -> dict:
    """spt / tpw / wpc of the reduced axis (a slice layout reduces along its parent's remaining dims)."""

    lay = layouts[name]
    if lay["kind"] == "slice":
        parent = lay["parent"]
        dims = [d for d in range(len(layouts[parent]["sizePerThread"])) if d != lay["dim"]]
        return _axis_params(layouts, parent, dims[axis])
    return {"spt": lay["sizePerThread"][axis], "tpw": lay["threadsPerWarp"][axis], "wpc": lay["warpsPerCTA"][axis]}


def reduce_layouts(module, ttgir: str) -> dict:
    """node_id of each TTIR tt.reduce -> layout parameters of its reduced axis (matched in program order)."""

    layouts = parse_layouts(ttgir)
    found = []
    for m in re.finditer(r'"tt\.reduce"\(([^)]*)\)\s*<\{axis = (\d+) : i32\}>', ttgir):
        tail = ttgir[m.end():]
        sig = re.search(r"\}\)\s*:\s*\(tensor<([\dx]+)x\w+,\s*(#[\w]+)>", tail)
        shape = tuple(int(s) for s in sig.group(1).split("x") if s)
        layout = sig.group(2)
        # A reshape with allow_reorder keeps every value in its register: the combination order is
        # the register order of the source layout.
        src = m.group(1).strip()
        while True:
            rs = re.search(r"^\s*" + re.escape(src) + r"\s*=\s*tt\.reshape\s+(%[\w#]+)\s+allow_reorder[^:]*:\s*"
                           r"tensor<([\dx]+)x\w+,\s*(#[\w]+)>", ttgir, re.M)
            if rs is None or len(rs.group(2).split("x")) != len(shape):
                break
            src, layout = rs.group(1), rs.group(3)
        found.append((int(m.group(2)), shape, layout))
    reduces = [op for fn in module.funcs.values() for op in fn.walk() if op.name == "tt.reduce"]
    if len(reduces) != len(found):
        return {}
    out = {}
    for op, (axis, shape, layout) in zip(reduces, found):
        if layout in layouts:
            out[op.node_id] = {"axis": axis, "shape": shape, "layout": layout,
                               **_axis_params(layouts, layout, axis)}
    return out


# ------------------------------------------------------------------------------------------- loads and stores

_ACCESS = re.compile(r"\btt\.(load|store)\s+(%[\w#]+)[^\n]*?:\s*(?:\(\s*)?(tensor<([\dx]*)x?!tt\.ptr<[^>]*>,\s*(#[\w]+)>"
                     r"|!tt\.ptr<[^>]*>)")


def access_layouts(module, ttgir: str) -> dict:
    """node_id of each TTIR tt.load / tt.store -> {"shape", "layout"} of its pointer operand in the TTGIR, matched
    in program order per kind.  Scalars (a plain !tt.ptr) get layout None and shape ().  Empty when the counts
    differ (the TTGIR passes may have reordered, fused or removed accesses): the evidence is then missing."""

    found = {"load": [], "store": []}
    for m in _ACCESS.finditer(ttgir):
        kind = m.group(1)
        if m.group(5) is None:
            found[kind].append({"shape": (), "layout": None})
        else:
            dims = tuple(int(s) for s in m.group(4).split("x") if s)
            found[kind].append({"shape": dims, "layout": m.group(5)})
    out = {}
    for kind in ("load", "store"):
        ops = [op for fn in module.funcs.values() for op in fn.walk() if op.name == f"tt.{kind}"]
        if len(ops) != len(found[kind]):
            return {}
        for op, info in zip(ops, found[kind]):
            out[op.node_id] = info
    return out


def element_threads(layouts: dict, layout: Optional[str], shape: tuple, num_warps: int) -> Optional[np.ndarray]:
    """For a tensor of ``shape`` in a blocked layout: an int array (prod(shape), r) listing the linear thread ids
    holding each element (row-major element order; r = replication factor).  A scalar (layout None) is held by every
    thread of the CTA.  None when the layout is not a modelled blocked layout."""

    n_threads = 32 * num_warps
    if layout is None:
        return np.arange(n_threads, dtype=np.int64)[None, :]
    lay = layouts.get(layout)
    if lay is None or lay["kind"] != "blocked" or len(lay["sizePerThread"]) != len(shape):
        return None
    spt, tpw, wpc, order = lay["sizePerThread"], lay["threadsPerWarp"], lay["warpsPerCTA"], lay["order"]
    if int(np.prod(tpw)) != 32 or int(np.prod(wpc)) != num_warps:
        return None
    rank = len(shape)
    # linearize lane and warp coordinates along `order` (fastest-varying dim first)
    def linear(coords, sizes):
        lin = np.zeros_like(coords[0])
        stride = 1
        for d in order:
            lin = lin + coords[d] * stride
            stride *= sizes[d]
        return lin
    idx = np.indices(shape).reshape(rank, -1)
    # every (lane, warp) assignment that covers each element: the layout tile repeats along a dim when the tensor is
    # larger (registers) and is replicated across threads when the tensor is smaller
    reps = []
    for d in range(rank):
        cover = spt[d] * tpw[d] * wpc[d]
        reps.append(max(1, cover // shape[d]) if shape[d] < cover else 1)
    copies = []
    for k in np.ndindex(*reps):
        coords_lane, coords_warp = [], []
        for d in range(rank):
            e = idx[d] + k[d] * shape[d]  # replicated copies sit at offsets of the tensor size inside the tile
            t = (e // spt[d]) % (tpw[d] * wpc[d])
            coords_lane.append(t % tpw[d])
            coords_warp.append(t // tpw[d])
        lane = linear(coords_lane, tpw)
        warp = linear(coords_warp, wpc)
        copies.append(warp * 32 + lane)
    return np.stack(copies, axis=1)
