"""TTIR parsing, mapping coverage and the kernel-level reference on real launches."""

import glob
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gmpy2 = pytest.importorskip("gmpy2")
pytest.importorskip("triton")
from gmpy2 import mpq  # noqa: E402

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import (  # noqa: E402
    ST_NE,
    ST_UNDEF,
    KernelReferenceEvaluator,
    NumericMode,
)
from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage, registry_coverage  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
ND, RC = NumericMode.NUMERICAL_DIFFERENCE, NumericMode.ROUNDING_CHECK


def test_registry_of_locked_build_is_covered_exhaustively():
    report = registry_coverage()
    assert report["complete"], report["missing"]
    assert report["stale_table_entries"] == []


def test_corpus_kernels_parse_and_have_complete_instruction_coverage():
    files = sorted(glob.glob(str(ROOT / "results/reference_eval/ttir_corpus/*/*.ttir")))
    assert files
    for path in files:
        report = kernel_coverage(parse_ttir(Path(path).read_text()))
        assert report["complete"], (path, report["rejected"])


# ---------------------------------------------------------------------------
# GPU launches
# ---------------------------------------------------------------------------


def _kernels():
    from scripts import reference_eval_kernels

    return reference_eval_kernels


def _capture(fn):
    recorder = TritonLaunchRecorder()
    with recorder:
        fn()
        torch.cuda.synchronize()
    return recorder.launches[-1]


def _evaluate(launch, mode=ND, **kwargs):
    return KernelReferenceEvaluator(parse_ttir(launch.asm["ttir"]), mode=mode).evaluate(launch, **kwargs)


def _buffer(result, name):
    for ident, buf in result.buffers.items():
        if buf.name == name:
            return ident, buf
    raise KeyError(name)


def _inputs(n=1000, k=4, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(n, device="cuda", generator=g) for _ in range(k)]


def _ieee_workloads(k):
    n = 1000
    x, a, b, c = _inputs(n)
    yield "scale_masked", "Y", lambda: k.scale_masked[(4,)](x, torch.empty_like(x), n, 0.1, BLOCK=256)
    yield "sequential_sum4", "Y", lambda: k.sequential_sum4[(4,)](x, a, b, c, torch.empty_like(x), n, BLOCK=256)
    yield "conversions", "YBF", lambda: k.conversions[(4,)](
        x, torch.empty(n, device="cuda", dtype=torch.float16), torch.empty(n, device="cuda", dtype=torch.bfloat16),
        torch.empty(n, device="cuda", dtype=torch.float8_e5m2), torch.empty_like(x), n, BLOCK=256)
    yield "store_load_chain", "Y", lambda: k.store_load_chain[(4,)](x, torch.empty_like(x), torch.empty_like(x),
                                                                    n, BLOCK=256)
    yield "bit_level", "Y", lambda: k.bit_level[(4,)](x, torch.empty_like(x), n, BLOCK=256)
    yield "while_loop", "Y", lambda: k.while_loop[(4,)](x, torch.empty_like(x), n, BLOCK=256)
    yield "masked_copy", "Y", lambda: k.masked_copy[(1,)](x, torch.empty(128, device="cuda"),
                                                          torch.empty(128, device="cuda"), 100, BLOCK=128)


@cuda
@pytest.mark.parametrize("index", range(7))
def test_rounding_check_mode_reproduces_ieee_kernels_bit_for_bit(index):
    name, out, fn = list(_ieee_workloads(_kernels()))[index]
    result = _evaluate(_capture(fn), RC)
    assert not result.aborted
    ident, _ = _buffer(result, out)
    _, actual, lo, hi, st, _ = result.residual_arrays(ident)
    finite = st == 0
    assert finite.sum() > 0
    assert np.array_equal(lo[finite], hi[finite]), name
    assert np.array_equal(lo[finite], actual[finite]), name


@cuda
def test_fp_fusion_is_the_source_of_the_rounding_check_mismatch():
    k = _kernels()
    n = 1000
    a, b, c = _inputs(n, 3, seed=1)

    def run(fusion):
        return _evaluate(_capture(lambda: k.add_then_mul[(4,)](a, b, c, torch.empty_like(a), n, BLOCK=256,
                                                               enable_fp_fusion=fusion)), RC)

    for fusion, expect_mismatch in ((False, False), (True, True)):
        result = run(fusion)
        ident, _ = _buffer(result, "Y")
        _, actual, lo, _, _, _ = result.residual_arrays(ident)
        assert bool((lo != actual).any()) is expect_mismatch


@cuda
def test_numerical_difference_reference_contains_independent_exact_values():
    k = _kernels()
    n = 1000
    x, a, b, c = _inputs(n, seed=2)
    result = _evaluate(_capture(lambda: k.sequential_sum4[(4,)](x, a, b, c, torch.empty_like(x), n, BLOCK=256)))
    ident, _ = _buffer(result, "Y")
    _, actual, lo, hi, _, _ = result.residual_arrays(ident)
    cols = [t.cpu().double().numpy() for t in (x, a, b, c)]
    for i in range(0, n, 37):
        exact = sum(mpq(col[i]) for col in cols)
        assert mpq(lo[i]) <= exact <= mpq(hi[i])
    assert (lo == hi).all()  # exact sums of four FP32 values are representable in float64

    A = torch.randn(64, 48, device="cuda")
    B = torch.randn(48, 32, device="cuda")
    C = torch.empty(64, 32, device="cuda")
    result = _evaluate(_capture(lambda: k.matmul[(2, 2)](A, B, C, 64, 32, 48, *A.stride(), *B.stride(),
                                                         *C.stride(), BM=32, BN=16, BK=16, PRECISION="ieee")))
    ident, buf = _buffer(result, "C")
    An, Bn = A.cpu().double().numpy(), B.cpu().double().numpy()
    for i, j in [(0, 0), (5, 17), (63, 31), (40, 3)]:
        exact = sum(mpq(An[i, kk]) * mpq(Bn[kk, j]) for kk in range(48))
        assert mpq(buf.lo[i * 32 + j]) <= exact <= mpq(buf.hi[i * 32 + j])

    result = _evaluate(_capture(lambda: k.exp_sum_divide[(1,)](x, torch.empty_like(x), 200, BLOCK=256)))
    ident, buf = _buffer(result, "Y")
    xs = x[:200].cpu().double().numpy()
    with gmpy2.context(precision=400):
        exps = [gmpy2.exp(gmpy2.mpfr(v)) for v in xs]
        total = sum(exps)
        for i in (0, 50, 199):
            assert mpq(buf.lo[i]) <= mpq(exps[i] / total) <= mpq(buf.hi[i])


@cuda
def test_branch_is_decided_by_the_reference_value():
    k = _kernels()
    x = torch.full((16,), 2.0 ** 24, device="cuda")
    y = torch.empty_like(x)
    result = _evaluate(_capture(lambda: k.branch_after_rounding[(1,)](x, y, 2.0 ** 24, BLOCK=16)))
    assert y.eq(0.0).all()  # FP32: 2**24 + 1 == 2**24, so the device took the else branch
    ident, buf = _buffer(result, "Y")
    _, actual, lo, hi, st, cond = result.residual_arrays(ident)
    assert (lo == 1.0).all() and (hi == 1.0).all()
    assert (actual - lo == -1.0).all()
    assert (result.element_classes(ident)[buf.written] == "complete_composed").all()


@cuda
def test_masked_load_without_other_is_undefined_only_where_used():
    k = _kernels()
    x = torch.randn(128, device="cuda")
    result = _evaluate(_capture(lambda: k.masked_copy[(1,)](x, torch.empty_like(x), torch.empty_like(x), 100,
                                                            BLOCK=128)))
    iy, by = _buffer(result, "Y")
    iz, bz = _buffer(result, "Z")
    assert by.written.sum() == 100 and (by.st[by.written] == 0).all()
    assert (bz.st[:100] == 0).all() and (bz.st[100:128] == ST_UNDEF).all()


@cuda
def test_atomic_return_value_is_not_established_when_used():
    k = _kernels()
    n = 1000
    (x,) = _inputs(n, 1, seed=3)
    result = _evaluate(_capture(lambda: k.atomic_accumulate[(4,)](x, torch.zeros(4, device="cuda"),
                                                                  torch.empty_like(x), n, BLOCK=256, USE_OLD=True)))
    _, old = _buffer(result, "OLD")
    _, out = _buffer(result, "OUT")
    assert (old.st[old.written] == ST_NE).all()
    assert (out.st[out.written] == 0).all()
    exact = [sum(mpq(v) for v in x.cpu().double().numpy()[j::4]) for j in range(4)]
    for j in range(4):
        assert mpq(out.lo[j]) <= exact[j] <= mpq(out.hi[j])


@cuda
def test_pinning_a_load_to_its_captured_value_downgrades_the_output():
    k = _kernels()
    n = 1000
    (x,) = _inputs(n, 1, seed=4)
    launch = _capture(lambda: k.store_load_chain[(4,)](x, torch.empty_like(x), torch.empty_like(x), n, BLOCK=256))
    free = _evaluate(launch)
    pinned = _evaluate(launch, pin_loads=("%u",))
    iy, by = _buffer(free, "Y")
    xs = x.cpu().double().numpy()
    # (x + 2**24) - 2**24 is exactly x in the reals; float64 endpoints keep width 0
    # unless the intermediate needs more than 53 bits (tiny |x|), where it encloses x.
    assert ((by.lo[:n] <= xs) & (xs <= by.hi[:n])).all() and (by.cond[:n] == False).all()  # noqa: E712
    assert (by.lo[:n] == by.hi[:n]).mean() > 0.95
    _, actual, lo, _, _, cond = pinned.residual_arrays(_buffer(pinned, "Y")[0])
    assert cond.all()
    assert (pinned.element_classes(_buffer(pinned, "Y")[0])[_buffer(pinned, "Y")[1].written]
            == "conditional_local").all()
    assert np.array_equal(actual, lo)  # the upstream FP32 rounding is erased by the pinned load
    assert (by.lo[:n] != actual).any()


@cuda
def test_capture_package_replays_bit_for_bit_from_the_saved_binary(tmp_path):
    from kernel_analyzer.reference_eval.capture import check_replay, load_compiled, load_launch, save_launch

    k = _kernels()
    rows = torch.randn(8, 100, device="cuda")
    launch = _capture(lambda: k.softmax_rows[(8,)](rows, torch.empty_like(rows), 100, 100, BLOCK=128))
    assert check_replay(launch)["bitwise_reproduced"]
    save_launch(launch, tmp_path / "pkg")
    loaded = load_launch(tmp_path / "pkg")
    report = check_replay(loaded, load_compiled(tmp_path / "pkg"))
    assert report["bitwise_reproduced"] and report["replays_identical"]
    assert report["correspondence"]["ir_sha256"]["ttir"]


@cuda
def test_composed_reference_across_launches_with_windowed_capture():
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence

    k = _kernels()
    n, block, chunks = 1 << 16, 1024, 12
    blocks = [3, 17, 40, 63]
    window = np.concatenate([np.arange(b * block, (b + 1) * block) for b in blocks])
    acc = torch.zeros(n, device="cuda")
    reference_acc = torch.zeros(n, device="cuda")
    contributions = []
    recorder = TritonLaunchRecorder(window=lambda kernel, name, t: window)
    g = torch.Generator(device="cuda").manual_seed(7)
    with recorder:
        for c in range(chunks):
            contrib = torch.randn(n, device="cuda", generator=g) * (10.0 ** (c % 5 - 2))
            contributions.append(contrib.cpu().double().numpy()[window])
            k.accumulate[(n // block,)](acc, contrib, n, BLOCK=block)
            reference_acc += contrib
        torch.cuda.synchronize()
    assert torch.equal(acc, reference_acc)  # the Triton add is bitwise the torch in-place add
    sequence = evaluate_sequence(recorder.launches, programs_for=lambda launch: [(b, 0, 0) for b in blocks])
    final = sequence.launches[-1]
    ident = acc.untyped_storage().data_ptr()
    buf = sequence.memory[ident]
    assert (buf.st == 0).all() and not buf.cond.any()
    exact = [sum(mpq(col[i]) for col in contributions) for i in range(window.size)]
    assert all(mpq(buf.lo[i]) <= exact[i] <= mpq(buf.hi[i]) for i in range(0, window.size, 97))
    # The accumulator chain is never rewritten outside the launches; contributions are external inputs.
    assert all(e["buffer"] != "ACC" for e in sequence.external_writes)
    report = final.compare()["ACC"]
    assert report["classes"]["complete_composed"] == window.size
    assert report["residual_positive"] + report["residual_negative"] > 0


@cuda
def test_unsigned_storage_and_signed_registers_round_trip():
    k = _kernels()
    codes = torch.arange(256, dtype=torch.uint8, device="cuda")
    out = torch.empty_like(codes)
    signed = torch.empty(256, dtype=torch.int32, device="cuda")
    result = _evaluate(_capture(lambda: k.uint8_codes[(1,)](codes, out, signed, 256, BLOCK=256)))
    for name in ("OUT", "SIGNED"):
        report = result.compare()[name]
        assert report["integer_mismatches"] == 0 and report["integer_matches"] == 256
