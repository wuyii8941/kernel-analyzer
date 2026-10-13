"""Compositional bias prediction (reference_eval.bias_predict): local contracts against exact rounding, the mma
contract against the bit-exact tensor-core model, and one ex-ante prediction checked on the device."""

import numpy as np
import pytest

from kernel_analyzer.reference_eval import bias_predict as BP
from kernel_analyzer.reference_eval import intervals as iv
from kernel_analyzer.reference_eval import tensor_core


def test_ulp_sigbits_lowbit():
    assert BP.ulp_of(np.array([1.0, 1.5, 2.0]), "f32").tolist() == [2.0 ** -23, 2.0 ** -23, 2.0 ** -22]
    assert BP.ulp_of(np.array([1.0]), "bf16")[0] == 2.0 ** -7
    assert BP.ulp_of(np.array([2.0 ** -130]), "f32")[0] == 2.0 ** -149          # subnormal spacing
    assert BP.sigbits(np.array([0.0, 1.0, 1.5, 3.0 * 2 ** -40, 1 + 2 ** -52])).tolist() == [0, 1, 2, 2, 53]
    assert BP.lowbit_exp(np.array([1.0, 1.5, 6.0])).tolist() == [0, -1, 1]


def test_rn_and_directed_contracts_match_exact_rounding_means():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200000)
    mu, half, var = BP.rn_contract(x, "bf16")
    r = iv.round_nearest_even(x, "bf16")[0] - x
    assert np.array_equal(mu, r) and np.allclose(half, BP.ulp_of(x, "bf16") / 2)  # the rounding relation itself
    assert abs(r.mean()) < 3 * r.std() / np.sqrt(x.size)                        # RN of regular data: mean 0
    assert np.isclose(np.sqrt(var.mean()), r.std(), rtol=0.02)
    assert np.all(BP.rn_contract(x, "bf16", exact_inputs=True)[2] == 0)           # exact operands: determined
    mu, half, var = BP.directed_contract(x, "tf32", "rtz")
    t = tensor_core.round_sig(x, 11, "rz") - x
    sx = np.sign(x)
    assert np.isclose((sx * mu).mean(), (sx * t).mean(), rtol=0.01)              # truncation: -sign * ulp / 2
    # an operand that carries no more bits than the target is converted exactly
    xb = iv.round_nearest_even(x, "bf16")[0]
    mu, half, _ = BP.directed_contract(xb, "tf32", "rtz", BP.sigbits(xb))
    assert np.all(mu == 0) and np.all(half == 0)


def test_rounding_relation_at_estimated_points_carries_the_b2_term():
    """RN of smooth data to e5m2: the asymmetric cell at each power of two makes the residual negatively correlated
    with the value (composition study E4: -0.3%, where 'RN is unbiased' gives 0).  The rounding relation evaluated at
    estimates of the operating points (here off by a relative 1e-6, far below a cell) predicts it."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal(2_000_000)
    r = iv.round_nearest_even(x, "f8E5M2")[0] - x
    est = x * (1 + 1e-6 * rng.standard_normal(x.size))
    mu, _, _ = BP.rn_contract(est, "f8E5M2")
    measured, predicted = (r * x).sum() / (x * x).sum(), (mu * x).sum() / (x * x).sum()
    se = np.std(r * x) * np.sqrt(x.size) / (x * x).sum()
    assert measured < -3 * se
    assert abs(measured - predicted) <= 3 * se + 0.1 * abs(predicted), (measured, predicted, se)


def test_add_contract_absorption():
    one = np.ones(4)
    s = np.array([2.0 ** -25, -2.0 ** -25, 0.75 * 2 ** -23, 3 * 2.0 ** -23])
    mu, half, _ = BP.add_contract(one, s, "f32")
    exact = iv.round_nearest_even(one + s, "f32")[0] - (one + s)
    assert np.array_equal(mu[:3], exact[:3])                     # absorbed / rounded up / tie at a power of two
    assert mu[3] == 0 and half[3] == 0                                            # representable sum


@pytest.mark.parametrize("fmt,K", [("tf32", 256), ("bf16", 512), ("f16", 256)])
def test_mma_contract_predicts_the_bias_of_the_bit_exact_rule(fmt, K):
    """The contract (means from coarse features) and the bit-exact model (tensor_core.mma_chain) give the same bias
    <e, r> / <r, r> on random operands."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal((64, K))
    b = rng.standard_normal((K, 64))
    if fmt != "tf32":
        a = iv.round_nearest_even(a, fmt)[0]
        b = iv.round_nearest_even(b, fmt)[0]
    else:
        a = iv.round_nearest_even(a, "f32")[0]
        b = iv.round_nearest_even(b, "f32")[0]
    c = np.zeros((64, 64))
    r = a @ b
    model, covered = tensor_core.mma_chain(a, b, c, fmt)
    assert covered.all()
    e = model - r
    mu, half, var = BP.mma_contract(a, b, c, fmt, BP.sigbits(a), BP.sigbits(b))
    measured = (e * r).sum() / (r * r).sum()
    predicted = (mu * r).sum() / (r * r).sum()
    se = np.std((e - mu) * r) * np.sqrt(r.size) / (r * r).sum()
    assert abs(measured - predicted) <= 3 * se + 0.1 * abs(predicted), (measured, predicted, se)
    assert np.all(np.abs(e - mu) <= half + 1e-300)                               # first-order radius bounds each error


def test_sum_tree_contract_absorption_in_low_precision():
    """A sequential bf16 sum of many positive terms stagnates: the absorption rule predicts the mean deficit."""
    rng = np.random.default_rng(2)
    x = iv.round_nearest_even(rng.random((2000, 256)), "bf16")[0]

    def seq(v, comb):
        acc = v[..., 0]
        for j in range(1, v.shape[-1]):
            acc = comb(acc, v[..., j])
        return acc

    def rn(u, w):
        return iv.round_nearest_even(u + w, "bf16")[0]
    device = seq(x, rn)
    exact = x.sum(-1)
    mu, half, var = BP.sum_tree_contract(x, "bf16", seq)
    e = device - exact
    se = e.std() / np.sqrt(e.size)
    assert e.mean() < -3 * se
    assert abs(mu.mean() - e.mean()) <= 3 * se, (mu.mean(), e.mean(), se)
    # complete stagnation (the sum stops at 256 = 2^8): the rule on device-tracking partials follows it
    x = iv.round_nearest_even(rng.random((500, 1024)), "bf16")[0]
    e = seq(x, rn) - x.sum(-1)
    mu, _, _ = BP.sum_tree_contract(x, "bf16", seq)
    assert np.isclose(mu.mean(), e.mean(), rtol=0.01) and e.mean() < -200


torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_ex_ante_prediction_of_a_tf32_dot():
    pytest.importorskip("triton")
    import triton
    import triton.language as tl

    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder

    @triton.jit
    def dot_k(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        rk = tl.arange(0, K)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b))

    g = torch.Generator(device="cuda").manual_seed(3)
    a = torch.randn(64, 128, device="cuda", generator=g)
    b = torch.randn(128, 64, device="cuda", generator=g).bfloat16().float()      # weights with 8 significant bits
    c = torch.full((64, 64), 7.0, device="cuda")
    with TritonLaunchRecorder(execute=False) as rec:
        dot_k[(1,)](a, b, c, M=64, N=64, K=128)
    launch = rec.launches[0]
    assert not launch.executed and torch.all(c == 7.0)                            # nothing ran
    pred = BP.predict(launch)
    summary = BP.summarize(pred)
    assert summary["status"] == "predicted" and summary["sites"].get("mma_tf32")
    o = pred["outputs"]["c_ptr"]
    dot_k[(1,)](a, b, c, M=64, N=64, K=128)
    torch.cuda.synchronize()
    r = o["exact"].reshape(64, 64)
    e = c.double().cpu().numpy() - r
    measured = (e * r).sum() / (r * r).sum()
    predicted = summary["outputs"]["c_ptr"]["alignment"]
    assert -5e-4 < predicted < -2e-4                                              # the activation truncation alone
    assert abs(measured - predicted) < 0.1 * abs(predicted)
    assert np.all(np.abs(e - o["delta"].reshape(64, 64)) <= o["radius"].reshape(64, 64))


@cuda
def test_e5m2_software_conversion_matches_the_device():
    """The conversion function the predictor evaluates for f8E5M2 is the device's (f32 and f16 operands)."""
    g = torch.Generator(device="cuda").manual_seed(8)
    for dt in (torch.float32, torch.float16):
        x = (torch.randn(200000, device="cuda", generator=g) * 3).to(dt)
        x = torch.cat([x, torch.tensor([2.0 ** -15, 3 * 2.0 ** -17, 1.125, -1.125, 0.0], device="cuda", dtype=dt)])
        pytest.importorskip("triton")
        import triton
        import triton.language as tl

        @triton.jit
        def conv(x_ptr, y_ptr, n, B: tl.constexpr):
            r = tl.program_id(0) * B + tl.arange(0, B)
            m = r < n
            tl.store(y_ptr + r, tl.load(x_ptr + r, mask=m).to(tl.float8e5).to(tl.float32), mask=m)

        y = torch.empty(x.numel(), device="cuda")
        conv[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), B=1024)
        xd = x.double().cpu().numpy()
        ok = np.abs(xd) <= 57344
        model = BP.e5m2_software(xd, dt == torch.float32)
        assert np.array_equal(model[ok], y.double().cpu().numpy()[ok])


@cuda
def test_noise_pass_carries_operand_exactness():
    """A rounding chain fed by a rounded value: the noise pass must keep the variance of the later sites (v1 took
    every operand as exact in the noise pass and predicted almost no noise for an e5m2 chain)."""
    pytest.importorskip("triton")
    import triton
    import triton.language as tl

    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder

    @triton.jit
    def chain(x_ptr, y_ptr, B: tl.constexpr):
        r = tl.arange(0, B)
        v = tl.load(x_ptr + r) * 1.1
        v = v.to(tl.float8e5).to(tl.float32)
        tl.store(y_ptr + r, tl.sqrt(tl.abs(v) + 1.0))

    x = torch.randn(4096, device="cuda", generator=torch.Generator(device="cuda").manual_seed(9))
    y = torch.empty_like(x)
    with TritonLaunchRecorder(execute=False) as rec:
        chain[(1,)](x, y, B=4096)
    s = BP.summarize(BP.predict(rec.launches[0]))
    chain[(1,)](x, y, B=4096)
    g = np.sqrt(np.abs(x.double().cpu().numpy() * 1.1) + 1.0)
    measured = np.sqrt(np.mean(((y.double().cpu().numpy() - g) / g) ** 2))
    assert 0.5 < s["outputs"]["y_ptr"]["rms_relative_error"] / measured < 2.0


@cuda
def test_approximate_function_underflowing_to_zero_is_determined():
    """exp2 of a masked logit (x - 1e6, the causal mask of flash attention) underflows to 0 in f32: the residual is
    determined, not outside the characterized cells (the tutorial's causal attention abstained before)."""
    pytest.importorskip("triton")
    import triton
    import triton.language as tl

    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder

    @triton.jit
    def masked_exp2(x_ptr, y_ptr, B: tl.constexpr):
        r = tl.arange(0, B)
        x = tl.load(x_ptr + r)
        tl.store(y_ptr + r, tl.math.exp2(x + tl.where(r % 2 == 0, 0.0, -1.0e6)))

    x = torch.randn(4096, device="cuda", generator=torch.Generator(device="cuda").manual_seed(10))
    y = torch.empty_like(x)
    with TritonLaunchRecorder(execute=False) as rec:
        masked_exp2[(1,)](x, y, B=4096)
    s = BP.summarize(BP.predict(rec.launches[0]))
    assert s["status"] == "predicted", s["failures"]


def test_fma_chain_carries_the_rounded_accumulator():
    """1 + 2^-24 + 2^-24 + 2^-24 as an fp32 fma chain: each increment is a tie that rounds to even and is lost, so the
    device result is 1 and the error -3 * 2^-24.  The contract must follow the rounded accumulator, not the exact
    prefix (it predicted -1 * 2^-24; external review of c415b62)."""
    K = 128
    a = np.zeros((1, K))
    b = np.ones((K, 1))
    a[0, :4] = [1.0, 2.0 ** -24, 2.0 ** -24, 2.0 ** -24]
    mu, _, _ = BP.fma_chain_contract(a, b, np.zeros((1, 1)))
    assert mu[0, 0] == -3 * 2.0 ** -24
