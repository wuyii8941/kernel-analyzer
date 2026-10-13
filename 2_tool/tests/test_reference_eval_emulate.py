"""Bitwise emulation of captured launches and in-kernel localization (needs CUDA)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gmpy2")
pytest.importorskip("triton")

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402
from kernel_analyzer.reference_eval.emulate import HardwareOracle, _compare, emulate, localize, verify  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _capture(fn):
    rec = TritonLaunchRecorder()
    with rec:
        fn()
        torch.cuda.synchronize()
    return rec.launches[-1]


def test_oracle_renames_operands_before_the_result():
    text = """module {
  tt.func public @k(%y: f32) {
    %y_11 = math.rsqrt %y : f32 loc(#loc52)
    tt.return
  }
}"""
    op = next(o for o in parse_ttir(text).entry().walk() if o.name == "math.rsqrt")
    line = HardwareOracle.rewrite(op, {"%y": "%x0"}, "%y")
    assert line == "%y = math.rsqrt %x0 : tensor<1024xf32>"


@cuda
def test_softmax_and_layernorm_reproduce_bitwise():
    from scripts import reference_eval_kernels as k

    g = torch.Generator(device="cuda").manual_seed(5)
    rows = torch.randn(8, 100, device="cuda", generator=g) * 3
    softmax = _capture(lambda: k.softmax_rows[(8,)](rows, torch.empty_like(rows), 100, 100, BLOCK=128))
    w, b = torch.randn(100, device="cuda", generator=g), torch.randn(100, device="cuda", generator=g)
    layernorm = _capture(lambda: k.layernorm_rows[(8,)](rows, w, b, torch.empty_like(rows), 100, 1e-5, BLOCK=128))
    for launch in (softmax, layernorm):
        r = verify(launch)
        assert r["bitwise_reproduced"]
        assert all(v["emulated"] == v["written"] for v in r["buffers"].values())
        assert not r["lowering_choices"]["uncontracted_from_output"]
        assert not r["lowering_choices"]["swapped_from_output"]
    # layernorm exercises contraction into the reduction and the div.full -> sub contraction
    nodes = verify(layernorm)["nodes"]
    assert nodes.get("reduce_tree_fused") and nodes.get("oracle_group")


@cuda
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_tensor_core_dot_reproduces_bitwise(dtype):
    """tf32 / bf16 / f16 mma follows the tensor_core.py rule bit for bit, also with a masked K tail."""
    from scripts import reference_eval_kernels as k

    g = torch.Generator(device="cuda").manual_seed(11)
    dt = getattr(torch, dtype)
    A = torch.randn(64, 200, device="cuda", generator=g).to(dt)
    B = (torch.randn(200, 32, device="cuda", generator=g) * 0.1).to(dt)
    C = torch.empty(64, 32, device="cuda")
    launch = _capture(lambda: k.matmul[(2, 2)](A, B, C, 64, 32, 200, *A.stride(), *B.stride(), *C.stride(),
                                               BM=32, BN=16, BK=32, PRECISION="tf32"))
    r = verify(launch, ungrouped=[], swapped=[])
    assert r["bitwise_reproduced"]
    assert r["buffers"]["C"]["emulated"] == r["buffers"]["C"]["written"] == 64 * 32
    assert any(n.startswith("dot_tensor_core") for n in r["nodes"])


@cuda
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_overflowing_reduction_is_not_claimed(dtype):
    """A partial sum that overflows becomes inf (and inf - inf NaN) on the device: the emulator must not report a
    finite value beyond the format's range as emulated (composition study, C3 cells)."""
    from scripts import reference_eval_kernels as k  # noqa: F401
    import triton
    import triton.language as tl

    @triton.jit
    def rowsum(x, o, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(o + tl.program_id(0), tl.sum(tl.load(x + tl.program_id(0) * N + i), axis=0))

    dt = getattr(torch, dtype)
    big = float(torch.finfo(dt).max) * 0.75
    x = torch.randn(2, 128, device="cuda").to(dt)
    x[0, :8] = big            # row 0: partial sums overflow to +inf
    x[1, :4] = big
    x[1, 64:68] = -big        # row 1: +inf and -inf meet: NaN
    out = torch.empty(2, device="cuda", dtype=dt)
    launch = _capture(lambda: rowsum[(2,)](x, out, N=128))
    r = verify(launch, ungrouped=[], swapped=[])
    assert r["status"] != "mismatch", r["buffers"]
    assert r["buffers"]["o"]["mismatch"] == 0


@cuda
@pytest.mark.parametrize("dtype,num_warps", [("float32", 4), ("float16", 1), ("bfloat16", 8)])
def test_scan_reproduces_bitwise(dtype, num_warps):
    """A forward 1-D float cumsum follows the scan_order.py combination order bit for bit."""
    import triton
    import triton.language as tl

    @triton.jit
    def cumsum(x, o, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(o + tl.program_id(0) * N + i, tl.cumsum(tl.load(x + tl.program_id(0) * N + i), axis=0))

    dt = getattr(torch, dtype)
    g = torch.Generator(device="cuda").manual_seed(3)
    x = (torch.randn(3, 1024, device="cuda", generator=g) * 4).to(dt)
    out = torch.empty_like(x)
    launch = _capture(lambda: cumsum[(3,)](x, out, N=1024, num_warps=num_warps))
    r = verify(launch, ungrouped=[], swapped=[])
    assert r["bitwise_reproduced"]
    assert r["buffers"]["o"]["emulated"] == r["buffers"]["o"]["written"] == 3 * 1024
    assert r["nodes"].get("scan_order")


@cuda
def test_reverse_scan_is_not_emulable():
    import triton
    import triton.language as tl

    @triton.jit
    def rcumsum(x, o, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(o + i, tl.cumsum(tl.load(x + i), axis=0, reverse=True))

    x = torch.randn(256, device="cuda")
    out = torch.empty_like(x)
    launch = _capture(lambda: rcumsum[(1,)](x, out, N=256))
    r = verify(launch)
    assert r["buffers"]["o"]["emulated"] == 0
    assert any("scan" in n for n in r["not_emulable_nodes"])


def _subnormal_heavy(n, g):
    """f32 values of which half are subnormal (both signs), the rest ordinary."""
    tiny = torch.rand(n, device="cuda", generator=g) * 2.0 ** -126 * torch.sign(torch.randn(n, device="cuda", generator=g))
    big = torch.randn(n, device="cuda", generator=g)
    return torch.where(torch.arange(n, device="cuda") % 2 == 0, tiny, big)


@cuda
@pytest.mark.parametrize("reflect_ftz", [True, False])
def test_floor_ceil_and_rounding_libdevice_follow_reflect_ftz(reflect_ftz):
    """With enable_reflect_ftz the libdevice-lowered floor / ceil and the rounding-suffixed libdevice operations flush
    subnormal operands and results to signed zero (cvt.rmi.ftz, add.rn.ftz, ...); without it they do not."""
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as ld

    @triton.jit
    def k(x, y, o1, o2, o3, o4, N: tl.constexpr):
        i = tl.arange(0, N)
        a = tl.load(x + i)
        b = tl.load(y + i)
        tl.store(o1 + i, tl.floor(a))
        tl.store(o2 + i, tl.ceil(a))
        tl.store(o3 + i, ld.add_rn(a, b))
        tl.store(o4 + i, ld.mul_rz(a, b))

    g = torch.Generator(device="cuda").manual_seed(9)
    x, y = _subnormal_heavy(512, g), _subnormal_heavy(512, g)
    outs = [torch.empty_like(x) for _ in range(4)]
    launch = _capture(lambda: k[(1,)](x, y, *outs, N=512, enable_reflect_ftz=reflect_ftz))
    r = verify(launch, ungrouped=[], swapped=[])
    assert r["bitwise_reproduced"], r["buffers"]
    assert all(v["emulated"] == v["written"] for v in r["buffers"].values())


@cuda
def test_exact_zero_sum_under_round_down_is_negative_zero():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as ld

    @triton.jit
    def k(x, y, o, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(o + i, ld.add_rd(tl.load(x + i), tl.load(y + i)))

    x = torch.randn(128, device="cuda", dtype=torch.float64)
    out = torch.empty_like(x)
    launch = _capture(lambda: k[(1,)](x, -x, out, N=128))
    r = verify(launch, ungrouped=[], swapped=[])
    assert r["bitwise_reproduced"], r["buffers"]
    assert bool(torch.all(torch.signbit(out)))


@cuda
def test_libdevice_product_fused_into_every_add_with_nan_lanes():
    """fast_logf ends in an LLVM multiply that aggressive fusion folds into each add consuming it, whatever the use
    count; lanes holding NaN (log of a negative) must not push the other lanes off the grouped path."""
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as ld

    @triton.jit
    def k(x, o, N: tl.constexpr):
        i = tl.arange(0, N)
        v = ld.fast_logf(tl.load(x + i))
        w = v + v
        tl.store(o + i, w + v)

    g = torch.Generator(device="cuda").manual_seed(4)
    x = torch.randn(1024, device="cuda", generator=g) * 2.0 ** torch.randint(-60, 60, (1024,), device="cuda",
                                                                             generator=g)
    out = torch.empty_like(x)
    launch = _capture(lambda: k[(1,)](x, out, N=1024))
    r = verify(launch, ungrouped=[], swapped=[])
    assert r["bitwise_reproduced"], r["buffers"]
    assert r["buffers"]["o"]["nan_both"] > 0


@cuda
def test_libdevice_without_declared_semantics_runs_on_the_device():
    """A libdevice call the reference cannot form (no declared semantics) is still emulated: the device evaluates it on
    the emulated operands."""
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as ld

    @triton.jit
    def k(x, o, oi, N: tl.constexpr):
        i = tl.arange(0, N)
        v = tl.load(x + i)
        tl.store(o + i, ld.erfcinv(v * 0.5 + 1.0) * 2.0 + v)
        tl.store(oi + i, ld.brev(v.to(tl.int32)))

    x = torch.randn(256, device="cuda")
    out, outi = torch.empty_like(x), torch.empty(256, device="cuda", dtype=torch.int32)
    launch = _capture(lambda: k[(1,)](x, out, outi, N=256))
    from kernel_analyzer.reference_eval import emulate as E
    res, emu = E.emulate(launch, ungrouped=[], swapped=[])
    r = verify(launch, ungrouped=[], swapped=[])
    assert r["bitwise_reproduced"], r["buffers"]
    assert r["buffers"]["o"]["emulated"] == 256
    ib = next(b for b in res.buffers.values() if b.name == "oi")
    assert bool(np.all(ib.lo[ib.written] == ib.actual_after[ib.written]))


@cuda
@pytest.mark.parametrize("fn", ["div_rd", "sqrt_ru", "add_rz", "mul_rn"])
def test_f64_operations_are_exact_in_every_rounding_mode(fn):
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as ld

    @triton.jit
    def k(x, y, o, N: tl.constexpr, FN: tl.constexpr):
        i = tl.arange(0, N)
        a = tl.load(x + i)
        b = tl.load(y + i)
        if FN == 0:
            r = ld.div_rd(a, b)
        elif FN == 1:
            r = ld.sqrt_ru(tl.abs(a))
        elif FN == 2:
            r = ld.add_rz(a, b)
        else:
            r = ld.mul_rn(a, b) + a / b
        tl.store(o + i, r)

    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(512, device="cuda", dtype=torch.float64, generator=g) * 1e-300
    y = torch.randn(512, device="cuda", dtype=torch.float64, generator=g)
    out = torch.empty_like(x)
    code = ["div_rd", "sqrt_ru", "add_rz", "mul_rn"].index(fn)
    launch = _capture(lambda: k[(1,)](x, y, out, N=512, FN=code))
    r = verify(launch, ungrouped=[], swapped=[])
    assert r["bitwise_reproduced"], r["buffers"]
    assert r["buffers"]["o"]["emulated"] == 512


@cuda
def test_single_operation_reduce_and_scan_regions_follow_the_layout_order():
    """A product reduction and non-associative / integer scans: the lowering's order with the device evaluating the
    region's operation reproduces the device bit for bit."""
    import triton
    import triton.language as tl

    @triton.jit
    def sub_(a, b):
        return a - b

    @triton.jit
    def imax(a, b):
        return tl.maximum(a, b)

    @triton.jit
    def k(x, xi, o1, o2, o3, N: tl.constexpr):
        i = tl.arange(0, N)
        v = tl.load(x + i)
        tl.store(o1 + tl.program_id(0), tl.reduce(1.0 + 0.01 * v, 0, _mul))
        tl.store(o2 + i, tl.associative_scan(v, 0, sub_))
        tl.store(o3 + i, tl.associative_scan(tl.load(xi + i), 0, imax))

    @triton.jit
    def _mul(a, b):
        return a * b

    x = torch.randn(1024, device="cuda")
    xi = torch.randint(-1000, 1000, (1024,), device="cuda", dtype=torch.int32)
    o1 = torch.empty(1, device="cuda")
    o2 = torch.empty_like(x)
    o3 = torch.empty_like(xi)
    launch = _capture(lambda: k[(1,)](x, xi, o1, o2, o3, N=1024))
    r = verify(launch, ungrouped=[], swapped=[])
    assert r["bitwise_reproduced"], r["buffers"]
    assert r["buffers"]["o1"]["emulated"] == 1 and r["buffers"]["o2"]["emulated"] == 1024


@cuda
def test_reshaped_product_reduction_and_broadcast_quotient():
    """Two lowering facts found on library kernels (Liger group norm, sparsemax): a product reshaped with
    can_reorder still contracts into the sum's combine, and a scalar div.full broadcast into a vector subtraction is
    left uncontracted by ptxas -- read from the SASS of the in-memory capture."""
    import triton
    import triton.language as tl

    @triton.jit
    def k(x, o, ss, N: tl.constexpr):
        i = tl.arange(0, N)
        v = tl.load(x + tl.program_id(0) * N + i)
        sq = tl.sum(tl.reshape(v * v, (N,), can_reorder=True), axis=0)
        s = tl.sum(v, axis=0)
        tau = (s - 1.0) / sq
        tl.store(ss + tl.program_id(0), sq)
        tl.store(o + tl.program_id(0) * N + i, v - tau)

    x = torch.randn(16, 512, device="cuda")
    out = torch.empty_like(x)
    ss = torch.empty(16, device="cuda")
    launch = _capture(lambda: k[(16,)](x, out, ss, N=512))
    r = verify(launch, ungrouped=[], swapped=[])
    assert r["bitwise_reproduced"], r["buffers"]
    assert r["nodes"].get("reduce_tree_fused")


@cuda
def test_fp8_conversion_clamp_and_directed_fma_with_flushed_zeros():
    """sm_86's software fp8 e5m2 conversion rounds ties away from zero (a bf16 tie such as 0.5625 goes to 0.625, not
    to the even 0.5); tt.clampf's min / max have their own signed-zero rule; an fma rounding down whose operands flush
    to zero gives +0 (the outward interval of an exact zero must not round to the smallest subnormal)."""
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as ld

    @triton.jit
    def k(x, lo, hi, a, b, o8, oc, of, N: tl.constexpr):
        i = tl.arange(0, N)
        v = tl.load(x + i)
        tl.store(o8 + i, v.to(tl.float8e5))
        tl.store(oc + i, tl.clamp(v.to(tl.float32), tl.load(lo + i), tl.load(hi + i), propagate_nan=tl.PropagateNan.ALL))
        tl.store(of + i, ld.fma_rd(tl.load(a + i), tl.load(b + i), tl.load(a + i)))

    g = torch.Generator(device="cuda").manual_seed(8)
    x = torch.cat([torch.tensor([0.5625, -0.140625, -0.8125, 0.0, -0.0, 3.0], device="cuda"),
                   torch.randn(250, device="cuda", generator=g)]).to(torch.bfloat16)
    lo = -torch.rand(256, device="cuda", generator=g)
    lo[:6] = -0.0
    hi = torch.rand(256, device="cuda", generator=g)
    a = _subnormal_heavy(256, g)
    b = torch.randn(256, device="cuda", generator=g) * 1e30
    o8 = torch.empty(256, device="cuda", dtype=torch.float8_e5m2)
    oc, of = torch.empty(256, device="cuda"), torch.empty(256, device="cuda")
    launch = _capture(lambda: k[(1,)](x, lo, hi, a, b, o8, oc, of, N=256))
    r = verify(launch, ungrouped=[], swapped=[])
    assert r["bitwise_reproduced"], r["buffers"]
    assert float(o8[0].float()) == 0.625


@cuda
def test_large_integer_conversions_are_exact():
    """fptoui to u64 above 2^63 keeps its bit pattern, and an i64 beyond 2^53 converts to f64 / f32 with one exact
    rounding (the float64 path left an interval and a later f64 add claimed the wrong neighbour)."""
    import triton
    import triton.language as tl

    @triton.jit
    def k(x, n, o1, o2, o3, o4, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(o1 + i, tl.load(x + i).to(tl.uint64))
        tl.store(o4 + i, (tl.load(x + i) * 1.3e-17).to(tl.uint8))
        v = tl.load(n + i)
        d = v.to(tl.float64)
        tl.store(o2 + i, d + d)
        tl.store(o3 + i, v.to(tl.float32))

    g = torch.Generator(device="cuda").manual_seed(6)
    x = torch.rand(128, device="cuda", generator=g, dtype=torch.float64) * 1.8e19
    n = torch.randint(-2 ** 62, 2 ** 62, (128,), device="cuda", generator=g, dtype=torch.int64) * 2 + 1
    o1 = torch.empty(128, device="cuda", dtype=torch.int64)
    o2 = torch.empty(128, device="cuda", dtype=torch.float64)
    o3 = torch.empty(128, device="cuda", dtype=torch.float32)
    o4 = torch.empty(128, device="cuda", dtype=torch.uint8)
    launch = _capture(lambda: k[(1,)](x, n, o1, o2, o3, o4, N=128))
    from kernel_analyzer.reference_eval import emulate as E
    res, emu = E.emulate(launch, ungrouped=[], swapped=[])
    r = verify(launch, ungrouped=[], swapped=[])
    assert r["bitwise_reproduced"], r["buffers"]
    assert r["buffers"]["o2"]["emulated"] == 128 and r["buffers"]["o3"]["emulated"] == 128
    for name in ("o1", "o4"):  # u64 above 2^63, u8 above 127 (held as negative signed values)
        ib = next(b for b in res.buffers.values() if b.name == name)
        assert bool(np.all(ib.lo[ib.written] == ib.actual_after[ib.written])), name


def test_strided_buffer_operand_is_captured_as_its_view():
    """A wrapper with ``_base`` + shape / stride / offset (FlagGems' StridedBuffer) is captured as the equivalent view
    of its base tensor, so library kernels that pass such operands can be emulated (composition study corpus)."""
    from kernel_analyzer.reference_eval.capture import _unwrap

    class Strided:
        def __init__(self, base, shape, strides, offset):
            self._base, self.shape, self._strides, self.offset, self.dtype = base, shape, strides, offset, base.dtype

        def stride(self):
            return self._strides

        def data_ptr(self):
            return self._base.data_ptr() + self.offset * self._base.element_size()

    base = torch.arange(24, dtype=torch.float32)
    t, name = _unwrap(Strided(base, (3, 4), (8, 2), 1))
    assert name == "float32"
    assert torch.equal(t, torch.as_strided(base, (3, 4), (8, 2), 1))
    assert _unwrap(Strided(base, (3,), (-1,), 2)) == (None, None)


@cuda
def test_split_precision_dot_is_not_emulable():
    from scripts import reference_eval_kernels as k

    A = torch.randn(64, 48, device="cuda")
    B = torch.randn(48, 32, device="cuda")
    C = torch.empty(64, 32, device="cuda")
    launch = _capture(lambda: k.matmul[(2, 2)](A, B, C, 64, 32, 48, *A.stride(), *B.stride(), *C.stride(),
                                               BM=32, BN=16, BK=16, PRECISION="tf32x3"))
    r = verify(launch)
    assert r["buffers"]["C"]["emulated"] == 0
    assert any("tensor-core" in n for n in r["not_emulable_nodes"])


@cuda
def test_directed_rounding_node_is_localized():
    from scripts import mutation_kernels as mk

    x = torch.randn(2048, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    y = torch.empty_like(x)
    launch = _capture(lambda: mk.m_scale[(8,)](x, y, 0.1, 2048, BLOCK=256, MUT=1))
    assert verify(launch)["bitwise_reproduced"]
    rows = localize(launch)["nodes"]
    top = max(rows, key=lambda r: abs(r["buffers"]["Y"]["mean"]))
    assert "__nv_fmul_rd" in top["text"]
    b = top["buffers"]["Y"]
    assert b["mean"] < 0 and b["positive"] == 0


def test_comparison_uses_storage_bits():
    written = np.ones(3, dtype=bool)
    lo = np.array([-0.0, 1.0, 0.0])
    st = np.array([0, 0, 1], dtype=np.int8)  # the third element is emulated as NaN
    actual = np.array([0.0, 1.0, np.nan])
    bits = np.array([0.0, 1.0, np.nan], dtype=np.float32).view(np.uint32).astype(np.uint64)
    m = _compare(written, lo, lo, st, actual, "f32", bits)
    assert not m["bit_equal"][0] and m["value_equal"][0]  # -0 against +0: equal values, different bits
    assert m["bit_equal"][1]
    assert m["nan_both"][2] and not m["bit_equal"][2]  # NaN only as a class


@cuda
def test_atomic_kernel_is_not_established():
    from scripts import reference_eval_kernels as k

    x = torch.randn(1000, device="cuda")
    launch = _capture(lambda: k.atomic_accumulate[(4,)](x, torch.zeros(4, device="cuda"), torch.empty_like(x), 1000,
                                                        BLOCK=256, USE_OLD=False))
    r = verify(launch)
    assert r["status"] == "not_established" and not r["bitwise_reproduced"]


@cuda
def test_rounded_substitution_predicts_the_div_rn_kernel():
    """Model intervention (division node correctly rounded) against the real kernel change."""

    from scripts import mutation_kernels as mk

    x = torch.randn(16, 200, device="cuda", generator=torch.Generator(device="cuda").manual_seed(4)) * 3
    outs = {}
    for mut in (0, 1):
        y = torch.empty_like(x)
        outs[mut] = (_capture(lambda: mk.m_softmax[(16,)](x, y, 200, 200, BLOCK=256, MUT=mut)), y)
    launch = outs[0][0]
    div = next(o.node_id for o in parse_ttir(launch.asm["ttir"]).entry().walk() if o.name == "arith.divf")
    res, _ = emulate(launch, exact_nodes=[div], substitution="rn")
    buf = next(b for b in res.buffers.values() if b.name == "Y")
    real = outs[1][1].reshape(-1).double().cpu().numpy()
    assert np.array_equal(buf.lo[buf.written], real[buf.written[: real.size]])


def test_zero_sign_follows_ieee():
    from kernel_analyzer.reference_eval.emulate import _zero_sign
    from kernel_analyzer.reference_eval.ttir_eval import _ftv

    def tv(v):
        v = np.array(v, dtype=np.float64)
        return _ftv("f32", v, v.copy(), np.zeros(v.shape, dtype=np.int8), np.zeros(v.shape, dtype=bool), frozenset())

    out = tv([0.0, 0.0, 0.0, 0.0])
    a, b = tv([1.5, -0.0, -0.0, 0.0]), tv([-1.5, -0.0, 0.0, -0.0])
    r = _zero_sign(out, "add", [a, b])
    assert list(np.signbit(r.lo)) == [False, True, False, False]  # x + (-x) = +0; -0 + -0 = -0
    r = _zero_sign(out, "fma", [tv([-2.0, 0.0, -1.0, 2.0]), tv([0.0, 3.0, 0.0, 0.5]), tv([-0.0, -0.0, 0.0, -1.0])])
    assert list(np.signbit(r.lo)) == [True, False, False, False]  # (-2*0) + -0 = -0; 0*3 + -0 = +0; 2*0.5 - 1 = +0


def test_locations_link_ttir_ops_to_ptx_and_sass():
    from kernel_analyzer.reference_eval.emulate import (op_location, ptx_instructions_by_location,
                                                        sass_instructions_by_line, ttir_locations)

    ttir = """module {
  tt.func public @k(%a: f32) {
    %t = arith.mulf %a, %a : f32 loc(#loc3)
    tt.return
  }
}
#loc1 = loc("/x/kern.py":12:7)
#loc3 = loc("t"(#loc1))
"""
    op = next(o for o in parse_ttir(ttir).entry().walk() if o.name == "arith.mulf")
    loc = op_location(op, ttir_locations(ttir), {"%t": "    %t = arith.mulf %a, %a : f32 loc(#loc3)"})
    assert loc == ("kern.py", 12, 7)
    ptx = "\t.loc\t1 12 7\n\tmul.f32 \t%r1, %r2, %r2;\n\t.loc\t1 13 0\n\tadd.f32 \t%r3, %r1, %r4;\n\t.file\t1 \"/x/kern.py\"\n"
    assert ptx_instructions_by_location(ptx)[("kern.py", 12, 7)] == ["mul.f32"]
    sass = '\t//## File "/x/kern.py", line 13\n        /*0010*/                   FADD R3, R1, R4 ;\n'
    assert sass_instructions_by_line(sass)[("kern.py", 13)] == ["FADD"]


@cuda
def test_lanes_without_a_declared_value_make_the_status_partial():
    """An output with lanes the reference leaves without a value (out-of-range float-to-int: poison) is not bit
    identical even when every emulated lane agrees (composition-study review: partial emulation was reported as
    bit identical)."""
    import triton
    import triton.language as tl

    @triton.jit
    def round_trip(x_ptr, y_ptr, B: tl.constexpr):
        r = tl.arange(0, B)
        tl.store(y_ptr + r, tl.load(x_ptr + r).to(tl.int32).to(tl.float32))

    x = torch.linspace(-4, 4, 256, device="cuda")
    x[::16] = 1e12                                      # outside int32: no declared value
    launch = _capture(lambda: round_trip[(1,)](x, torch.empty_like(x), B=256))
    r = verify(launch, ungrouped=[], swapped=[])
    buf = r["buffers"]["y_ptr"]
    assert buf["not_emulated"] == 16 and buf["mismatch"] == 0
    assert r["status"] == "partial" and not r["bitwise_reproduced"]


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_low_precision_multiply_add_contraction(dtype):
    """a * b - c * d on one source line: LLVM folds one product and gives the standalone multiply the first product's
    column whichever it keeps, so the product choice is read from the PTX dataflow, not the location (f32 failed
    before); a bf16 standalone product is fma.rn.bf16 x, y, -0 on sm_86 (Liger's rope at bf16, found by the corrected
    composition corpus, failed before)."""
    import triton
    import triton.language as tl

    @triton.jit
    def rot(a_ptr, b_ptr, c_ptr, d_ptr, y_ptr, B: tl.constexpr):
        r = tl.arange(0, B)
        a, b = tl.load(a_ptr + r), tl.load(b_ptr + r)
        c, d = tl.load(c_ptr + r), tl.load(d_ptr + r)
        tl.store(y_ptr + r, a * b - c * d)

    g = torch.Generator(device="cuda").manual_seed(11)
    xs = [torch.randn(1024, device="cuda", generator=g).to(dtype) for _ in range(4)]
    launch = _capture(lambda: rot[(1,)](*xs, torch.empty_like(xs[0]), B=1024))
    r = verify(launch, ungrouped=[], swapped=[])
    assert r["bitwise_reproduced"], r["buffers"]


def test_ptx_dataflow_finds_bf16_products_and_their_operand_locations():
    from kernel_analyzer.reference_eval.emulate import ptx_dataflow_multiplies
    ptx = """
	.loc	1 9 19
	ld.global.b16 	%rs1, [%rd1];
	.loc	1 9 39
	ld.global.b16 	%rs2, [%rd2];
	.loc	1 10 19
	ld.global.b16 	%rs3, [%rd3];
	.loc	1 10 39
	ld.global.b16 	%rs4, [%rd4];
	mov.b16 	%rs16, 0x8000;
	.loc	1 11 28
	fma.rn.bf16 	%rs5, %rs3, %rs4, %rs16;
	neg.bf16 	%rs6, %rs5;
	.loc	1 11 32
	fma.rn.bf16 	%rs7, %rs1, %rs2, %rs6;
	.file	1 "/x/k.py"
"""
    muls = ptx_dataflow_multiplies(ptx)
    assert muls == [(frozenset({("k.py", 10, 19)}), frozenset({("k.py", 10, 39)}))]   # c * d, not the fused a * b


@cuda
@pytest.mark.parametrize("shape", [(4, 128), (8, 64)])
def test_reduce_of_a_flattening_reshape_follows_the_source_registers(shape):
    """tl.sum over tl.reshape(x2d, (n,), can_reorder=True): the values stay in the 2-D layout's registers, so the
    combination order is that layout's (FlagGems group_norm at f32, found by the corrected composition corpus: the
    1-D layout's order was used)."""
    import triton
    import triton.language as tl

    @triton.jit
    def flat_sum(x_ptr, y_ptr, R: tl.constexpr, C: tl.constexpr):
        r = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
        x = tl.load(x_ptr + tl.program_id(0) * R * C + r)
        tl.store(y_ptr + tl.program_id(0), tl.sum(tl.reshape(x, (R * C,), can_reorder=True), axis=0))

    R, C = shape
    x = torch.randn(32 * R * C, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5))
    launch = _capture(lambda: flat_sum[(32,)](x, torch.empty(32, device="cuda"), R=R, C=C))
    r = verify(launch, ungrouped=[], swapped=[])
    assert r["bitwise_reproduced"], r["buffers"]


@cuda
def test_product_folded_into_a_reduce_through_a_flattening_reshape():
    """tl.sum(tl.reshape(d * d, (n,), can_reorder=True)): the product stays in its register and LLVM folds it into
    the first combine, as for a same-shape reshape (FlagGems group_norm's variance at f32)."""
    import triton
    import triton.language as tl

    @triton.jit
    def flat_sq_sum(x_ptr, y_ptr, R: tl.constexpr, C: tl.constexpr):
        r = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
        x = tl.load(x_ptr + tl.program_id(0) * R * C + r)
        d = x - 0.25
        tl.store(y_ptr + tl.program_id(0), tl.sum(tl.reshape(d * d, (R * C,), can_reorder=True), axis=0))

    x = torch.randn(32 * 512, device="cuda", generator=torch.Generator(device="cuda").manual_seed(6))
    launch = _capture(lambda: flat_sq_sum[(32,)](x, torch.empty(32, device="cuda"), R=4, C=128))
    r = verify(launch, ungrouped=[], swapped=[])
    assert r["bitwise_reproduced"], r["buffers"]
