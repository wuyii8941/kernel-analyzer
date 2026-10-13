"""Distributed layouts from a captured TTGIR (locked Triton 3.6.0 lowering).

Used by the reference evaluator for two kinds of evidence that TTIR does not carry:

* the combination order of a ``tt.reduce`` (``reduce_layouts``): sequential within a thread in register order,
  butterfly over the lanes along the axis, butterfly over the warps along the axis.  The same model drives the
  bitwise emulator (``emulate.py``), where it is checked bit for bit against the device for float sums;
* which threads hold each element of a loaded or stored tensor (``access_layouts``, ``element_threads``), for the
  thread-level conflict check between a store and a later load in one program.

Blocked and slice layouts are modelled; for reductions also the tensor-core accumulator layout (nvidia_mma v2) and
linear layouts.  Anything else is reported as unknown and the caller treats the evidence as missing (never as a
proof either way).
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
    for m in re.finditer(r"^(#[\w]+)\s*=\s*#ttg\.nvidia_mma<\{([^}]*)\}>", ttgir, re.M):
        body = m.group(2)
        major = re.search(r"versionMajor\s*=\s*(\d+)", body)
        wpc = re.search(r"warpsPerCTA\s*=\s*" + _LIST, body)
        instr = re.search(r"instrShape\s*=\s*" + _LIST, body)
        if major and wpc and instr:
            layouts[m.group(1)] = {"kind": "mma", "version": int(major.group(1)),
                                   "warpsPerCTA": [int(v) for v in wpc.group(1).split(",") if v.strip()],
                                   "instrShape": [int(v) for v in instr.group(1).split(",") if v.strip()]}
    for m in re.finditer(r"^(#[\w]+)\s*=\s*#ttg\.linear<\{(.*)\}>\s*$", ttgir, re.M):
        bases = {}
        for key in ("register", "lane", "warp", "block"):
            mm = re.search(key + r"\s*=\s*\[((?:\[[^\]]*\],?\s*)*)\]", m.group(2))
            bases[key] = [tuple(int(x) for x in b.split(",") if x.strip())
                          for b in re.findall(r"\[([^\[\]]*)\]", mm.group(1))] if mm else []
        layouts[m.group(1)] = {"kind": "linear", **bases}
    return layouts


def linear_bases(layouts: dict, name: str, shape: tuple) -> Optional[tuple]:
    """(register, lane, warp) bases -- the coordinate offset each index bit selects -- of a linear or tensor-core
    accumulator (nvidia_mma v2) layout over ``shape``; None for other layouts or a replicated tensor.

    nvidia_mma v2 (sm_80 / sm_86, mma.sync m16n8): in a 16 x 8 tile, register bit 0 -> column + 1, bit 1 -> row + 8;
    lane bits -> column + 2, + 4, row + 1, + 2, + 4; warps along warpsPerCTA (dim 1 fastest); the CTA tile then repeats
    over the shape, columns first (each repetition a further register bit)."""
    lay = layouts.get(name)
    if lay is None or len(shape) != 2:
        return None
    if lay["kind"] == "linear":
        if lay.get("block"):
            return None
        reg, lane, warp = lay["register"], lay["lane"], lay["warp"]
    elif lay["kind"] == "mma" and lay["version"] == 2 and lay["instrShape"] == [16, 8]:
        reg = [(0, 1), (8, 0)]
        lane = [(0, 2), (0, 4), (1, 0), (2, 0), (4, 0)]
        wm, wn = lay["warpsPerCTA"]
        warp = [(0, 8 * (1 << i)) for i in range(int(np.log2(wn)))] + \
               [(16 * (1 << i), 0) for i in range(int(np.log2(wm)))]
        tile = (16 * wm, 8 * wn)
        if shape[0] % tile[0] or shape[1] % tile[1]:
            return None
        reg = reg + [(0, tile[1] * (1 << i)) for i in range(int(np.log2(shape[1] // tile[1])))] + \
            [(tile[0] * (1 << i), 0) for i in range(int(np.log2(shape[0] // tile[0])))]
    else:
        return None
    if any(b == (0, 0) for b in reg + lane + warp) or 2 ** len(reg + lane + warp) != shape[0] * shape[1]:
        return None                                       # broadcast / replicated: not modelled
    return reg, lane, warp


def linear_axis_params(bases: tuple, shape: tuple, axis: int) -> Optional[dict]:
    """Reduce parameters of a linear layout along ``axis``: every thread combines its registers along the axis in
    register order, then the lanes along the axis (butterfly, highest lane bit first), then the warps.  ``perm[p]`` is
    the axis index of position p = warp * R * T + lane * R + register of the 1-D model.  None when a basis mixes the
    axis with the other dimension (the order would then depend on the row)."""
    reg, lane, warp = bases
    other = 1 - axis
    groups = []
    for bs in (reg, lane, warp):
        along = [b for b in bs if b[axis] != 0]
        if any(b[axis] != 0 and b[other] != 0 for b in bs):
            return None
        groups.append(along)
    R, T, W = (1 << len(g) for g in groups)
    n = shape[axis]
    if R * T * W != n:
        return None
    perm = np.empty(n, dtype=np.int64)
    for w in range(W):
        for l in range(T):
            for r in range(R):
                idx = 0
                for bits, bs in ((r, groups[0]), (l, groups[1]), (w, groups[2])):
                    for i, b in enumerate(bs):
                        if bits >> i & 1:
                            idx ^= b[axis]
                perm[w * R * T + l * R + r] = idx
    return {"spt": R, "tpw": T, "wpc": W, "perm": perm}


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
        flat = None
        while True:
            rs = re.search(r"^\s*" + re.escape(src) + r"\s*=\s*tt\.reshape\s+(%[\w#]+)\s+allow_reorder[^:]*:\s*"
                           r"tensor<([\dx]+)x\w+,\s*(#[\w]+)>", ttgir, re.M)
            if rs is None:
                break
            src_shape = tuple(int(v) for v in rs.group(2).split("x") if v)
            if len(src_shape) != len(shape):
                # a flattening reshape into the reduced 1-D tensor: the values stay in the source layout's registers
                if len(shape) == 1:
                    flat = (src_shape, rs.group(3))
                break
            src, layout = rs.group(1), rs.group(3)
        found.append((int(m.group(2)), shape, layout, flat))
    reduces = [op for fn in module.funcs.values() for op in fn.walk() if op.name == "tt.reduce"]
    if len(reduces) != len(found):
        return {}
    out = {}
    for op, (axis, shape, layout, flat) in zip(reduces, found):
        if flat is not None:
            params = flattened_params(layouts, *flat)
            if params is not None:
                out[op.node_id] = {"axis": axis, "shape": shape, "layout": flat[1], **params}
                continue
        if layout in layouts and layouts[layout]["kind"] in ("mma", "linear"):
            bases = linear_bases(layouts, layout, shape)
            params = linear_axis_params(bases, shape, axis) if bases is not None else None
            if params is not None:
                out[op.node_id] = {"axis": axis, "shape": shape, "layout": layout, **params}
            continue
        if layout in layouts:
            try:
                params = _axis_params(layouts, layout, axis)
            except KeyError:                              # a slice of a layout that is not blocked: not modelled
                continue
            out[op.node_id] = {"axis": axis, "shape": shape, "layout": layout, **params}
    return out


def flattened_params(layouts: dict, src_shape: tuple, name: str) -> Optional[dict]:
    """1-D reduce parameters of a blocked-layout tensor flattened in its registers (allow_reorder reshape): every
    thread combines all its registers in register order (sizePerThread dims in ``order``, then repetitions), then the
    lanes and the warps.  ``perm[p]`` is the row-major source index of the value at position p of the 1-D model
    (p = warp * R * T + lane * R + register).  None for other layouts or a replicated tensor."""
    lay = layouts.get(name)
    if lay is None or lay["kind"] != "blocked" or len(lay["sizePerThread"]) != len(src_shape):
        return None
    spt, tpw, wpc, order = lay["sizePerThread"], lay["threadsPerWarp"], lay["warpsPerCTA"], lay["order"]
    tile = [s * t * w for s, t, w in zip(spt, tpw, wpc)]
    if any(n % t for n, t in zip(src_shape, tile)):
        return None
    reps = [n // t for n, t in zip(src_shape, tile)]
    idx = np.indices(src_shape).reshape(len(src_shape), -1)
    lane = np.zeros(idx.shape[1], dtype=np.int64)
    warp = np.zeros_like(lane)
    reg_in = np.zeros_like(lane)
    reg_rep = np.zeros_like(lane)
    sl = sw = sj = sr = 1
    for d in order:
        x = idx[d]
        rep, within = x // tile[d], x % tile[d]
        w, l, j = within // (spt[d] * tpw[d]), (within % (spt[d] * tpw[d])) // spt[d], within % spt[d]
        lane += l * sl
        warp += w * sw
        reg_in += j * sj
        reg_rep += rep * sr
        sl, sw, sj, sr = sl * tpw[d], sw * wpc[d], sj * spt[d], sr * reps[d]
    R = sj * sr
    reg = reg_in + sj * reg_rep
    pos = warp * (R * sl) + lane * R + reg
    perm = np.empty(idx.shape[1], dtype=np.int64)
    perm[pos] = np.arange(idx.shape[1])
    return {"spt": int(R), "tpw": int(sl), "wpc": int(sw), "perm": perm}


def scan_layouts(module, ttgir: str) -> dict:
    """node_id of each TTIR tt.scan -> axis, reverse flag, shape and layout parameters of the scanned axis (matched in
    program order; empty when the counts differ)."""

    layouts = parse_layouts(ttgir)
    found = []
    for m in re.finditer(r'"tt\.scan"\(([^)]*)\)\s*<\{axis = (\d+) : i32, reverse = (true|false)\}>', ttgir):
        sig = re.search(r"\}\)\s*:\s*\(tensor<([\dx]+)x\w+,\s*(#[\w]+)>", ttgir[m.end():])
        if sig is None:
            return {}
        shape = tuple(int(s) for s in sig.group(1).split("x") if s)
        found.append((int(m.group(2)), m.group(3) == "true", shape, sig.group(2), len(m.group(1).split(","))))
    scans = [op for fn in module.funcs.values() for op in fn.walk() if op.name == "tt.scan"]
    if len(scans) != len(found):
        return {}
    out = {}
    for op, (axis, reverse, shape, layout, n_operands) in zip(scans, found):
        lay = layouts.get(layout)
        if lay is None or lay["kind"] != "blocked":
            continue
        out[op.node_id] = {"axis": axis, "reverse": reverse, "shape": shape, "layout": layout,
                           "operands": n_operands, **_axis_params(layouts, layout, axis)}
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
