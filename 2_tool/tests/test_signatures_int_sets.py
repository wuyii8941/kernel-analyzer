"""Integer set targets for atomic return values (DSL v2 increment 14, rc3 02 5.3 / 10).  Compiled-only kernels (sm_86)
and an official Gluon TTGIR evaluated on synthetic captures, CPU only.  Every returned value an execution order can
produce is enumerated independently (all permutations of the programs) and must lie in the reference's interval;
misuses of a set (arithmetic, address, loop bound, later overwrite, an undecided branch, a later launch) must leave no
point and no set claim."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

triton = pytest.importorskip("triton")

import signature_harness as H  # noqa: E402
from test_signatures_structural import C, _kernel  # noqa: E402

CATS = ("positive", "boundary", "premise_violation")
_CACHE: dict = {}


def _compile(name, body, bufs, scalars):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    params = list(bufs) + list(scalars)
    mod = _kernel(name, body, params)
    sig = {k: H.SIG[d] for k, (d, _) in bufs.items()}
    sig.update({k: t for k, (t, _) in scalars.items()})
    sig["N"] = "constexpr"
    key = (name, body, tuple(sig.items()))
    if key not in _CACHE:
        ck = triton.compile(ASTSource(fn=mod.kernel, signature=sig, constexprs={"N": C}),
                            target=GPUTarget("cuda", 86, 32), options={"num_warps": 1})
        _CACHE[key] = {k: ck.asm[k] for k in ("ttir", "ttgir")}
    return dict(_CACHE[key])


def _launch(i, name, asm, bufs, scalars, grid, after=None):
    from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch
    args, base = [], 1 << 20
    for j, (k, (d, arr)) in enumerate(bufs.items()):
        arr = np.ascontiguousarray(np.asarray(arr, H.NP[d]))
        raw = arr.reshape(-1).view(np.uint8).copy()
        aft = raw.copy() if after is None or k not in after else \
            np.ascontiguousarray(np.asarray(after[k], H.NP[d])).reshape(-1).view(np.uint8).copy()
        args.append(CapturedArg(index=j, name=k, kind="tensor", constexpr=False, signature_type=H.SIG[d],
                                dtype=H.TORCH[d], shape=arr.shape, stride=None, element_size=arr.itemsize,
                                data_ptr=base * (j + 1), storage_ptr=base * (j + 1), storage_nbytes=raw.size,
                                storage_id=j, before=raw, after=aft))
    for k, (t, v) in scalars.items():
        args.append(CapturedArg(index=len(args), name=k, kind="int", constexpr=False, signature_type=t, value=v))
    args.append(CapturedArg(index=len(args), name="N", kind="int", constexpr=True, signature_type="constexpr", value=C))
    return CapturedLaunch(index=i, kernel_name=name, kernel_hash="", grid=grid, args=args, asm=asm, cubin_sha256=None,
                          metadata={}, libtriton_sha256=None)


def _run(name, body, bufs, grid, scalars=None, after=None):
    """The reference of one launch and the storage ident of each buffer."""
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence
    scalars = scalars or {}
    asm = _compile(name, body, bufs, scalars)
    ref = evaluate_sequence([_launch(0, name, asm, bufs, scalars, grid, after)]).launches[0]
    return ref, {k: (1 << 20) * (j + 1) for j, k in enumerate(bufs)}


def _orders(init, xs, op, width):
    """Every value each program's update can return, over all execution orders (signed representation)."""
    half, mask = 1 << (width - 1), (1 << width) - 1
    seen = [set() for _ in xs]
    for perm in itertools.permutations(range(len(xs))):
        acc = init
        for k in perm:
            seen[k].add(((acc + half) & mask) - half)
            acc = op(acc, xs[k]) & mask
    return seen


def _assert_hulls(ref, ident, seen):
    buf = ref.buffers[ident]
    cls = ref.element_classes(ident)
    for i, values in enumerate(seen):
        if not values:   # not checked here
            continue
        if len(values) == 1:
            assert buf.st[i] == H.ST_OK and int(buf.lo[i]) == next(iter(values)), i
            continue
        assert cls[i] == "set_target", (i, cls[i], ref.reasons)
        assert all(buf.iset_lo[i] <= v <= buf.iset_hi[i] for v in values), (i, values, buf.iset_lo[i], buf.iset_hi[i])
        assert buf.st[i] == H.ST_NE                              # inside the evaluator: never a value


RET = ("    pid = tl.program_id(0)\n    v = tl.load(x_ptr + pid)\n    old = tl.atomic_{op}(z, v)\n"
       "    tl.store(out + pid, old)\n")
OPS = {"add": lambda a, b: a + b, "max": max, "min": min, "xchg": lambda a, b: b, "and": lambda a, b: a & b,
       "or": lambda a, b: a | b, "xor": lambda a, b: a ^ b}


@pytest.mark.parametrize("category", CATS)
@pytest.mark.parametrize("dtype", ["int32", "int64", "uint32"])
def test_official_pattern_integer_add_returns_a_set_target(dtype, category):
    # test_atomic_rmw (official): 5 programs add 2^i into one address and store the old value
    width = 64 if dtype == "int64" else 32
    x = [1 << i for i in range(5)]
    z0 = 0
    if category == "boundary":
        z0 = (1 << (width - 1)) - 20                             # z0 + 31 leaves the signed range: wraps
    body = RET.replace("{op}", "add")
    if category == "premise_violation":
        body = body.replace("tl.store(out + pid, old)", "tl.store(out + pid, old + 1)")   # arithmetic on the set
    bufs = {"x_ptr": (dtype, x), "z": (dtype, [z0]), "out": (dtype, np.zeros(5))}
    ref, ids = _run(f"iset_add_{dtype}", body, bufs, (5, 1, 1))
    cls = ref.element_classes(ids["out"])
    if category == "boundary":
        # programs 0-3 may add 16 on top of z0: their hull leaves the signed range (not established); program 4
        # only ever sees 1 + 2 + 4 + 8 added: its hull fits
        assert (cls[:4] == "not_established").all() and any("not representable" in r for r in ref.reasons)
        _assert_hulls(ref, ids["out"], [set()] * 4 + [_orders(z0, x, OPS["add"], width)[4]])
        return
    if category == "premise_violation":
        assert (cls[:5] == "not_established").all() and any("used by arith.addi" in r for r in ref.reasons)
        return
    _assert_hulls(ref, ids["out"], _orders(z0, x, OPS["add"], width))
    assert any(r.startswith("set:") for r in ref.reasons)
    # the final value of the address is a point (the sum), not a set
    zb = ref.buffers[ids["z"]]
    assert zb.st[0] == H.ST_OK and int(zb.lo[0]) == 31


@pytest.mark.parametrize("category", CATS)
@pytest.mark.parametrize("op,dtype", [("max", "int32"), ("min", "int32"), ("xchg", "int32"), ("max", "uint32"),
                                      ("min", "uint32")])
def test_order_kinds_return_a_set_target(op, dtype, category):
    x = {"positive": [3, -7, 12, 5, -1], "boundary": [-(1 << 31), (1 << 31) - 1, 0, -1, 1],
         "premise_violation": [3, -7, 12, 5, -1]}[category]
    if dtype == "uint32":
        x = [v & 0xFFFFFFFF for v in x]
    width = 32
    z0 = 4
    body = RET.replace("{op}", op)
    if category == "premise_violation":   # the set as an address: the store may write any element of out
        body = body.replace("tl.store(out + pid, old)", "tl.store(out + (old & 3), pid)")
    bufs = {"x_ptr": (dtype, x), "z": (dtype, [z0]), "out": (dtype, np.zeros(5))}
    ref, ids = _run(f"iset_{op}_{dtype}", body, bufs, (5, 1, 1))
    if category == "premise_violation":
        assert not (ref.established(ids["out"])).any()
        return
    if dtype == "uint32" and op in ("max", "min"):     # unsigned order on the bit patterns
        fn = (lambda a, b: max(a, b)) if op == "max" else (lambda a, b: min(a, b))
    else:
        def fn(a, b, f=OPS[op]):
            s = lambda t: ((t + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
            return f(s(a), s(b))
    _assert_hulls(ref, ids["out"], _orders(z0, x, fn, width))


@pytest.mark.parametrize("category", CATS)
@pytest.mark.parametrize("op", ["and", "or", "xor"])
def test_bitwise_kinds_return_the_hull_of_the_reachable_values(op, category):
    n = 5
    x = [0b10110, 0b01101, 0b11000, 0b00111, 0b10001]
    if category == "boundary":
        n = 14                                                   # 13 other updates: 2^13 > 4096 reachable values
        x = [1 << i for i in range(n)]
    body = RET.replace("{op}", op)
    if category == "premise_violation":   # stored, then overwritten by an atomic update in the same program
        body = body + "    tl.atomic_add(out + pid, 1)\n"
    z0 = 0b11111 if op == "and" else 0
    bufs = {"x_ptr": ("int32", x), "z": ("int32", [z0]), "out": ("int32", np.zeros(n))}
    ref, ids = _run(f"iset_bit_{op}{'_n' if category == 'boundary' else ''}", body, bufs, (n, 1, 1))
    cls = ref.element_classes(ids["out"])
    if category == "premise_violation":
        assert not (cls[:n] == "set_target").any()
        return
    if category == "boundary":
        if op in ("xor", "or"):    # 13 distinct single bits from 0: every subset gives another value (8192)
            assert (cls[:n] == "not_established").all() and any("not representable" in r for r in ref.reasons)
        else:                      # and from 0b11111: at most 32 values are reachable
            # 14 programs are too many to permute: every order returns z0 AND some subset of the others' values
            seen = []
            for k in range(n):
                others = [v for j, v in enumerate(x) if j != k]
                vals = set()
                for mask in range(1 << len(others)):
                    acc = z0
                    for j, v in enumerate(others):
                        if mask >> j & 1:
                            acc &= v
                    vals.add(acc)
                seen.append(vals)
            _assert_hulls(ref, ids["out"], seen)
        return
    _assert_hulls(ref, ids["out"], _orders(z0, x, OPS[op], 32))


SEQ_WRITE = "    pid = tl.program_id(0)\n    tl.store(out + pid, pid + 100)\n"
SEQ_OTHER = "    pid = tl.program_id(0)\n    tl.store(q + pid, pid)\n"


@pytest.mark.parametrize("category", CATS)
def test_set_target_across_launches_and_overwrites(category):
    """Positive: a later launch that does not touch the buffer keeps the set target.  Boundary: the same program
    stores the set and then a point (the point stands).  Premise violation: a later launch overwrites the element."""
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence
    x = [1, 2, 4, 8, 16]
    bufs = {"x_ptr": ("int32", x), "z": ("int32", [0]), "out": ("int32", np.zeros(5)), "q": ("int32", np.zeros(5))}
    body = RET.replace("{op}", "add")
    if category == "boundary":
        body = body + "    tl.store(out + pid, pid + 7)\n"
    asm1 = _compile("iset_seq1" + category[:1], body, bufs, {})
    l1 = _launch(0, "k1", asm1, bufs, {}, (5, 1, 1))
    if category == "premise_violation":
        b2 = {"out": ("int32", np.zeros(5))}
        l2 = _launch(1, "k2", _compile("iset_seq_w", SEQ_WRITE, b2, {}), b2, {}, (5, 1, 1))
    else:
        b2 = {"q": ("int32", np.zeros(5))}
        l2 = _launch(1, "k2", _compile("iset_seq_o", SEQ_OTHER, b2, {}), b2, {}, (5, 1, 1))
        l2.args[0].storage_ptr = l2.args[0].data_ptr = 4 << 20   # the q storage of launch 1
    seq = evaluate_sequence([l1, l2])
    first, second = seq.launches
    out_id = 3 << 20
    if category == "boundary":
        buf = first.buffers[out_id]
        assert (first.element_classes(out_id)[:5] == "complete_composed").all()
        assert [int(v) for v in buf.lo[:5]] == [7, 8, 9, 10, 11]
        return
    assert (first.element_classes(out_id)[:5] == "set_target").all()
    if category == "positive":
        assert (second.element_classes(out_id)[:5] == "set_target").all()
    else:
        out_id2 = 1 << 20                                        # launch 2 binds out as its first storage
        buf = second.buffers[out_id2]
        assert buf.iset is None or not buf.iset[:5].any()
        assert [int(v) for v in buf.lo[:5]] == [100, 101, 102, 103, 104]


BRANCH = ("    pid = tl.program_id(0)\n    vf = tl.load(xf + pid)\n    of = tl.atomic_add(zf, vf)\n"
          "    v = tl.load(x_ptr + pid)\n    old = tl.atomic_add(z, v)\n    if of > THRESH:\n"
          "        tl.store(out + pid, old)\n")


@pytest.mark.parametrize("category", CATS)
def test_undecided_branch_keeps_a_set_target_only_if_both_branches_agree(category):
    """The float atomic return is itself a set (an interval): compared with 1.5 it is undecided, so both branches run
    and only one stores.  Positive / boundary: a threshold every order exceeds (decided)."""
    # boundary: every returned float lies in [0, 30] (enclosed slightly outward), so "> 31.0" is decided false
    thresh = {"positive": "-1.0", "boundary": "31.0", "premise_violation": "1.5"}[category]
    body = BRANCH.replace("THRESH", thresh)
    bufs = {"xf": ("fp32", [1.0, 2.0, 4.0, 8.0, 16.0]), "zf": ("fp32", [0.0]), "x_ptr": ("int32", [1, 2, 4, 8, 16]),
            "z": ("int32", [0]), "out": ("int32", np.zeros(5))}
    ref, ids = _run("iset_branch_" + category[:1], body, bufs, (5, 1, 1))
    cls = ref.element_classes(ids["out"])
    if category == "premise_violation":
        assert ref.rules.get("path.branch_union", 0) >= 1
        assert not (cls[:5] == "set_target").any()
        assert not ref.established(ids["out"])[:5].any()
        return
    if category == "boundary":
        assert ref.rules.get("path.branch_union", 0) == 0 and (cls[:5] == "not_written").all()
        return
    _assert_hulls(ref, ids["out"], _orders(0, [1, 2, 4, 8, 16], OPS["add"], 32))


def test_compare_counts_actual_values_against_the_set():
    x = [1, 2, 4, 8, 16]
    bufs = {"x_ptr": ("int32", x), "z": ("int32", [0]), "out": ("int32", np.zeros(5))}
    # an actual execution in program order returns 0, 1, 3, 7, 15; then one value outside its set
    ref, ids = _run("iset_cmp", RET.replace("{op}", "add"), bufs, (5, 1, 1),
                    after={"out": [0, 1, 3, 7, 15], "z": [31]})
    rep = ref.compare()["out"]
    assert rep["integer_in_set"] == 5 and rep["integer_outside_set"] == 0 and rep["classes"]["set_target"] == 5
    ref, ids = _run("iset_cmp", RET.replace("{op}", "add"), bufs, (5, 1, 1),
                    after={"out": [0, 1, 3, 7, 999], "z": [31]})
    assert ref.compare()["out"]["integer_outside_set"] == 1


@pytest.mark.parametrize("category", CATS)
def test_gluon_local_atomic_integer_return_is_a_set_target(category):
    """Official Gluon TTGIR (tests/data/gluon_ttgir/atomic_scatter_rmw_add.ttgir): each lane adds 1 into a shared row
    (initially 0) of its column; when both lanes of a column hit one row, each returns 0 or 1 depending on the order."""
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence
    ttgir = (Path(__file__).parent / "data" / "gluon_ttgir" / "atomic_scatter_rmw_add.ttgir").read_text()
    idx = {"positive": np.zeros((2, 16)), "boundary": np.full((2, 16), 31),       # collisions in row 0 / row 31
           "premise_violation": np.stack([np.zeros(16), np.ones(16)])}[category].astype(np.int32)  # no collision
    bufs = {"values_ptr": ("int32", np.zeros(32)), "indices_ptr": ("int32", idx), "mask_ptr": ("int8", np.ones(32)),
            "old_ptr": ("int32", np.zeros(32)), "final_ptr": ("int32", np.zeros(32 * 16))}
    ref = evaluate_sequence([_launch(0, "g", {"ttgir": ttgir}, bufs, {}, (1, 1, 1))]).launches[0]
    old_id = 4 << 20
    cls = ref.element_classes(old_id)
    buf = ref.buffers[old_id]
    if category == "premise_violation":
        assert (cls[:32] == "complete_composed").all() and (buf.lo[:32] == 0).all()
        return
    assert (cls[:32] == "set_target").all(), ref.reasons
    assert (buf.iset_lo[:32] <= 0).all() and (buf.iset_hi[:32] >= 1).all()
    assert any("local atomic return value" in r for r in ref.reasons)


LOOP = ("    pid = tl.program_id(0)\n    v = tl.load(x_ptr + pid)\n    old = tl.atomic_add(z + ZOFF, v)\n"
        "    acc = 0\n    for i in range(old):\n        acc += 1\n    tl.store(out + pid, acc)\n")


@pytest.mark.parametrize("category", CATS)
def test_set_as_loop_bound_is_not_evaluated(category):
    """Positive: one program per address, the returned value is a point and bounds the loop.  Boundary: a point bound
    of 0 (no iteration).  Premise violation: five programs share the address, the bound is a set: the program aborts
    with the reason instead of running the loop with the lower bound."""
    zoff = "0" if category == "premise_violation" else "pid"
    z = {"positive": [3, 1, 4, 1, 5], "boundary": [0, 0, 0, 0, 0], "premise_violation": [0, 0, 0, 0, 0]}[category]
    bufs = {"x_ptr": ("int32", [1, 2, 4, 8, 16]), "z": ("int32", z), "out": ("int32", np.zeros(5))}
    ref, ids = _run("iset_loop_" + category[:1], LOOP.replace("ZOFF", zoff), bufs, (5, 1, 1))
    buf = ref.buffers[ids["out"]]
    if category == "premise_violation":
        assert any("loop bound is an integer set target" in r for r in ref.aborted.values())
        assert not ref.established(ids["out"])[:5].any()
        return
    assert (buf.st[:5] == H.ST_OK).all() and [int(v) for v in buf.lo[:5]] == z
