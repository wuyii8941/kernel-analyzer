"""Derivative reference (forward mode on TTIR) and the dot-product adjoint check."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gmpy2 = pytest.importorskip("gmpy2")
pytest.importorskip("triton")
from gmpy2 import mpq  # noqa: E402

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _capture(fn):
    recorder = TritonLaunchRecorder()
    with recorder:
        fn()
        torch.cuda.synchronize()
    return recorder.launches[-1]


def _buffer(result, name):
    for buf in result.buffers.values():
        if buf.name == name:
            return buf
    raise KeyError(name)


def _kernels():
    from scripts import reference_eval_kernels

    return reference_eval_kernels


@cuda
def test_cube_jvp_is_exact_and_the_adjoint_check_does_not_false_alarm():
    k = _kernels()
    xs = torch.tensor([0.0, 0.75, -1.5, 3.0, 1e-3, -7.25], device="cuda")
    n = xs.numel()
    launch = _capture(lambda: k.cube[(1,)](xs, torch.empty_like(xs), n, BLOCK=8))
    u = np.array([1.0, -2.0, 0.5, 1.0, 3.0, -1.0])
    result = KernelReferenceEvaluator(parse_ttir(launch.asm["ttir"])).evaluate(launch, tangents={"X": u})
    y = _buffer(result, "Y")
    jlo, jhi = y.d[0][:n], y.d[1][:n]
    x = xs.cpu().double().numpy()
    v = np.array([1.0, 1.0, -1.0, 0.25, 2.0, 1.0])
    h = mpq(1, 64)
    for i in range(n):
        exact = 3 * mpq(x[i]) ** 2 * mpq(u[i])
        assert jlo[i] == jhi[i] and mpq(jlo[i]) == exact  # J u exact, width 0 (also at x = 0)
        backward = 3 * mpq(x[i]) ** 2 * mpq(v[i]) * mpq(u[i])  # <B(x, v), u> with B = 3 x^2 v
        assert backward - mpq(v[i]) * mpq(jlo[i]) == 0
        f = lambda t: t ** 3  # noqa: E731
        central = (f(mpq(x[i]) + h * mpq(u[i])) - f(mpq(x[i]) - h * mpq(u[i]))) / (2 * h)
        assert backward - mpq(v[i]) * central == -(h ** 2) * mpq(u[i]) ** 3 * mpq(v[i])


@cuda
def test_softmax_jvp_encloses_the_analytic_derivative():
    k = _kernels()
    n = 200
    x = torch.randn(n, device="cuda") * 2.0
    launch = _capture(lambda: k.softmax_exp[(1,)](x, torch.empty_like(x), n, BLOCK=256))
    rng = np.random.default_rng(0)
    u = np.zeros(n)
    u[:n] = rng.standard_normal(n)
    result = KernelReferenceEvaluator(parse_ttir(launch.asm["ttir"])).evaluate(launch, tangents={"X": u})
    y = _buffer(result, "Y")
    assert not result.aborted
    xd = x.cpu().double().numpy()
    with gmpy2.context(precision=300):
        e = [gmpy2.exp(gmpy2.mpfr(float(v))) for v in xd]
        s = sum(e)
        p = [mpq(t / s) for t in e]
    pu = sum(pi * mpq(ui) for pi, ui in zip(p, u))
    for i in range(0, n, 7):
        exact = p[i] * (mpq(u[i]) - pu)
        assert mpq(y.d[0][i]) <= exact <= mpq(y.d[1][i])
    assert np.max(y.d[1][:n] - y.d[0][:n]) < 1e-12
