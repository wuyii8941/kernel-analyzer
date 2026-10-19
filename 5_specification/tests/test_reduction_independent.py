"""Independent exact checks for the reviewed common reductions.

The expected forward values use Python Fraction loops over explicit indices.
The smooth polynomial VJPs are reconstructed from the scalar objective by an
exact symmetric difference. No REDUCE helper, broadcast-back helper, or VJP is
used to construct those expected gradients.
"""
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A
from kernel_spec.num import ends


def values(shape):
    out = np.empty(shape, dtype=object)
    for n, index in enumerate(np.ndindex(shape)):
        out[index] = Q((n % 7) - 3, 4)
    return out


def rows(a, axes):
    axes = tuple(sorted(axes))
    rest = tuple(i for i in range(a.ndim) if i not in axes)
    out_shape = tuple(a.shape[i] for i in rest)
    red_shape = tuple(a.shape[i] for i in axes)
    for oi in np.ndindex(out_shape):
        row = []
        for ri in np.ndindex(red_shape):
            index = [0] * a.ndim
            for k, axis in enumerate(rest):
                index[axis] = oi[k]
            for k, axis in enumerate(axes):
                index[axis] = ri[k]
            row.append(a[tuple(index)])
        yield oi, row


def reduce_exact(a, axes, op):
    rest = tuple(i for i in range(a.ndim) if i not in axes)
    out = np.empty(tuple(a.shape[i] for i in rest), dtype=object)
    for index, row in rows(a, axes):
        if op == "sum":
            out[index] = sum(row, Q(0))
        elif op == "mean":
            out[index] = sum(row, Q(0)) / len(row)
        elif op == "prod":
            value = Q(1)
            for item in row:
                value *= item
            out[index] = value
    return out


def cumsum_exact(a, axis):
    out = np.empty(a.shape, dtype=object)
    for index in np.ndindex(tuple(s for i, s in enumerate(a.shape) if i != axis)):
        total = Q(0)
        for j in range(a.shape[axis]):
            full = list(index)
            full.insert(axis, j)
            total += a[tuple(full)]
            out[tuple(full)] = total
    return out


def variance_exact(a, axes, correction):
    rest = tuple(i for i in range(a.ndim) if i not in axes)
    out = np.empty(tuple(a.shape[i] for i in rest), dtype=object)
    for index, row in rows(a, axes):
        denominator = len(row) - correction
        if denominator <= 0:
            raise ZeroDivisionError
        mean = sum(row, Q(0)) / len(row)
        out[index] = sum(((item - mean) ** 2 for item in row), Q(0)) / denominator
    return out


def exact_gradient(forward, x, upstream):
    def objective(arg):
        output = forward(arg)
        return sum((output[i] * upstream[i] for i in np.ndindex(output.shape)), Q(0))
    result = np.empty(x.shape, dtype=object)
    for index in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += 1
        minus[index] -= 1
        result[index] = (objective(plus) - objective(minus)) / 2
    return result


def assert_tensor(test, actual, expected):
    test.assertEqual(actual.shape, expected.shape)
    for index in np.ndindex(expected.shape):
        test.assertEqual(actual.a[index].v, expected[index], index)


def as_mp(value):
    import mpmath as mp
    if isinstance(value, Q):
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def assert_tensor_encloses(test, actual, expected):
    test.assertEqual(actual.shape, expected.shape)
    for index in np.ndindex(expected.shape):
        lo, hi = ends(actual.a[index].v)
        wanted = as_mp(expected[index])
        test.assertLessEqual(as_mp(lo), wanted, index)
        test.assertGreaterEqual(as_mp(hi), wanted, index)


def standard_deviation_expected(a, axes, correction):
    import mpmath as mp
    variance = variance_exact(a, axes, correction)
    out = np.empty(variance.shape, dtype=object)
    with mp.workdps(100):
        for index in np.ndindex(variance.shape):
            out[index] = mp.sqrt(as_mp(variance[index]))
    return out


def standard_deviation_gradient_expected(a, axes, correction, upstream):
    import mpmath as mp
    rest = tuple(i for i in range(a.ndim) if i not in axes)
    out = np.empty(a.shape, dtype=object)
    with mp.workdps(100):
        for output_index, row in rows(a, axes):
            denominator = len(row) - correction
            mean = sum(row, Q(0)) / len(row)
            variance = sum(((item - mean) ** 2 for item in row), Q(0)) / denominator
            standard_deviation = mp.sqrt(as_mp(variance))
            for reduced_index in np.ndindex(tuple(a.shape[i] for i in axes)):
                full_index = [0] * a.ndim
                for k, axis in enumerate(rest):
                    full_index[axis] = output_index[k]
                for k, axis in enumerate(axes):
                    full_index[axis] = reduced_index[k]
                full_index = tuple(full_index)
                out[full_index] = (as_mp(upstream[output_index])
                                   * as_mp(a[full_index] - mean)
                                   / (denominator * standard_deviation))
    return out


def vector_two_norm_expected(a, axes):
    import mpmath as mp
    shape = tuple(a.shape[i] for i in range(a.ndim) if i not in axes)
    out = np.empty(shape, dtype=object)
    with mp.workdps(100):
        for output_index, row in rows(a, axes):
            out[output_index] = mp.sqrt(sum((as_mp(item) ** 2 for item in row), mp.mpf(0)))
    return out


def vector_two_norm_gradient_expected(a, axes, upstream):
    import mpmath as mp
    rest = tuple(i for i in range(a.ndim) if i not in axes)
    out = np.empty(a.shape, dtype=object)
    with mp.workdps(100):
        for output_index, row in rows(a, axes):
            norm = mp.sqrt(sum((as_mp(item) ** 2 for item in row), mp.mpf(0)))
            for reduced_index in np.ndindex(tuple(a.shape[i] for i in axes)):
                full_index = [0] * a.ndim
                for k, axis in enumerate(rest):
                    full_index[axis] = output_index[k]
                for k, axis in enumerate(axes):
                    full_index[axis] = reduced_index[k]
                full_index = tuple(full_index)
                out[full_index] = (as_mp(upstream[output_index])
                                   * as_mp(a[full_index]) / norm)
    return out


class IndependentReductions(unittest.TestCase):
    def test_forward_sum_mean_product_and_cumsum(self):
        x = values((2, 3, 2))
        for axes in ((0,), (1,), (0, 2)):
            for name, fn in (("sum", A.sum_), ("mean", A.mean)):
                with self.subTest(name=name, axes=axes):
                    expected = reduce_exact(x, axes, name)
                    assert_tensor(self, fn(A.tensor(x, "f64"), list(axes)), expected)
        assert_tensor(self, A.prod(A.tensor(x, "f64"), 1), reduce_exact(x, (1,), "prod"))
        for axis in range(x.ndim):
            assert_tensor(self, A.cumsum(A.tensor(x, "f64"), axis), cumsum_exact(x, axis))

    def test_exact_vjps_from_independent_objectives(self):
        x = values((2, 3))
        for name, forward, vjp in (
                ("sum", lambda z: reduce_exact(z, (1,), "sum"), A.vjp_sum),
                ("mean", lambda z: reduce_exact(z, (1,), "mean"), A.vjp_mean),
                ("prod", lambda z: reduce_exact(z, (1,), "prod"), A.vjp_prod)):
            upstream = values((2,))
            expected = exact_gradient(forward, x, upstream)
            with self.subTest(name=name):
                actual = vjp(A.tensor(upstream, "f64"), A.tensor(x, "f64"), 1)
                assert_tensor(self, actual, expected)
        upstream = values(x.shape)
        expected = exact_gradient(lambda z: cumsum_exact(z, 1), x, upstream)
        assert_tensor(self, A.vjp_cumsum(A.tensor(upstream, "f64"), A.tensor(x, "f64"), 1), expected)

    def test_variance_and_vjp_follow_the_independent_corrected_formula(self):
        x = values((2, 3, 2))
        for axes in ((1,), (0, 2)):
            for correction in (0, 1):
                expected = variance_exact(x, axes, correction)
                upstream = values(expected.shape)
                gradient = exact_gradient(
                    lambda z: variance_exact(z, axes, correction), x, upstream)
                with self.subTest(axes=axes, correction=correction):
                    assert_tensor(
                        self,
                        A.var(A.tensor(x, "f64"), list(axes), correction),
                        expected,
                    )
                    assert_tensor(
                        self,
                        A.vjp_var(
                            A.tensor(upstream, "f64"), A.tensor(x, "f64"),
                            list(axes), correction,
                        ),
                        gradient,
                    )
        constant = A.tensor([[2, 2, 2]], "f64")
        self.assertEqual(A.plain(A.var(constant, 1, 1)), [0])
        self.assertEqual(A.plain(A.vjp_var(A.tensor([7], "f64"), constant, 1, 1)),
                         [[0, 0, 0]])

    def test_vector_two_norm_and_vjp_follow_the_independent_formula(self):
        x = values((2, 3, 2))
        for axes in ((1,), (0, 2)):
            expected = vector_two_norm_expected(x, axes)
            upstream = values(expected.shape)
            gradient = vector_two_norm_gradient_expected(x, axes, upstream)
            with self.subTest(axes=axes):
                assert_tensor_encloses(
                    self,
                    A.linalg_vector_norm(A.tensor(x, "f64"), 2, list(axes)),
                    expected,
                )
                assert_tensor_encloses(
                    self,
                    A.vjp_linalg_vector_norm(
                        A.tensor(upstream, "f64"), A.tensor(x, "f64"),
                        2, list(axes),
                    ),
                    gradient,
                )

    def test_vector_two_norm_matches_pytorch_210_on_finite_float64_cases(self):
        import torch
        self.assertEqual(torch.__version__, "2.10.0+cu128")
        for shape in ((2, 3), (2, 3, 2)):
            raw = values(shape)
            native = torch.tensor(np.asarray(raw, dtype=float), dtype=torch.float64,
                                  requires_grad=True)
            dims = (0, -1, tuple(range(1, len(shape))))
            for dim in dims:
                for keepdim in (False, True):
                    native.grad = None
                    output = torch.linalg.vector_norm(native, ord=2, dim=dim,
                                                      keepdim=keepdim)
                    upstream = torch.arange(1, output.numel() + 1,
                                            dtype=torch.float64).reshape(output.shape)
                    output.backward(upstream)
                    selected = (dim,) if isinstance(dim, int) else dim
                    axes = tuple(axis % len(shape) for axis in selected)
                    expected = vector_two_norm_expected(raw, axes)
                    upstream_expected = np.asarray(upstream.detach().numpy(), dtype=object)
                    if keepdim:
                        upstream_expected = np.squeeze(upstream_expected, axis=axes)
                        expected = expected.reshape(
                            tuple(1 if i in axes else shape[i] for i in range(len(shape))))
                    gradient = vector_two_norm_gradient_expected(
                        raw, axes, upstream_expected
                    )
                    with self.subTest(shape=shape, dim=dim, keepdim=keepdim):
                        np.testing.assert_allclose(output.detach().numpy(),
                                                   np.asarray(expected, dtype=float),
                                                   rtol=0, atol=2e-15)
                        np.testing.assert_allclose(native.grad.numpy(),
                                                   np.asarray(gradient, dtype=float),
                                                   rtol=0, atol=2e-15)

    def test_variance_formula_matches_pytorch_210_on_finite_float64_cases(self):
        import torch
        self.assertEqual(torch.__version__, "2.10.0+cu128")
        for shape in ((2, 3), (2, 3, 2)):
            raw = values(shape)
            native = torch.tensor(np.asarray(raw, dtype=float), dtype=torch.float64,
                                  requires_grad=True)
            for dim in (0, -1, tuple(range(1, len(shape)))):
                for correction in (0, 1):
                    for keepdim in (False, True):
                        native.grad = None
                        output = torch.var(native, dim=dim, correction=correction,
                                           keepdim=keepdim)
                        upstream = torch.arange(1, output.numel() + 1,
                                                dtype=torch.float64).reshape(output.shape)
                        output.backward(upstream)
                        dims = (dim,) if isinstance(dim, int) else dim
                        axes = tuple(d % len(shape) for d in dims)
                        exact_squeezed = variance_exact(raw, axes, correction)
                        upstream_exact = np.asarray(upstream.detach().numpy(), dtype=object)
                        if keepdim:
                            upstream_exact = np.squeeze(upstream_exact, axis=axes)
                            exact = exact_squeezed.reshape(
                                tuple(1 if i in axes else shape[i] for i in range(len(shape))))
                        else:
                            exact = exact_squeezed
                        expected_grad = exact_gradient(
                            lambda z: variance_exact(z, axes, correction),
                            raw,
                            upstream_exact,
                        )
                        with self.subTest(shape=shape, dim=dim, correction=correction,
                                          keepdim=keepdim):
                            np.testing.assert_allclose(output.detach().numpy(),
                                                       np.asarray(exact, dtype=float),
                                                       rtol=0, atol=2e-15)
                            np.testing.assert_allclose(native.grad.numpy(),
                                                       np.asarray(expected_grad, dtype=float),
                                                       rtol=0, atol=2e-15)

    def test_standard_deviation_and_vjp_follow_the_independent_formula(self):
        x = values((2, 3, 2))
        for axes in ((1,), (0, 2)):
            for correction in (0, 1):
                expected = standard_deviation_expected(x, axes, correction)
                upstream = values(expected.shape)
                gradient = standard_deviation_gradient_expected(
                    x, axes, correction, upstream
                )
                with self.subTest(axes=axes, correction=correction):
                    assert_tensor_encloses(
                        self,
                        A.std(A.tensor(x, "f64"), list(axes), correction),
                        expected,
                    )
                    assert_tensor_encloses(
                        self,
                        A.vjp_std(
                            A.tensor(upstream, "f64"), A.tensor(x, "f64"),
                            list(axes), correction,
                        ),
                        gradient,
                    )

    def test_standard_deviation_formula_matches_pytorch_210_on_positive_slices(self):
        import torch
        self.assertEqual(torch.__version__, "2.10.0+cu128")
        for shape in ((2, 3), (2, 3, 2)):
            raw = values(shape)
            native = torch.tensor(np.asarray(raw, dtype=float), dtype=torch.float64,
                                  requires_grad=True)
            for dim in (0, -1, tuple(range(1, len(shape)))):
                for correction in (0, 1):
                    for keepdim in (False, True):
                        native.grad = None
                        output = torch.std(native, dim=dim, correction=correction,
                                           keepdim=keepdim)
                        upstream = torch.arange(1, output.numel() + 1,
                                                dtype=torch.float64).reshape(output.shape)
                        output.backward(upstream)
                        dims = (dim,) if isinstance(dim, int) else dim
                        axes = tuple(d % len(shape) for d in dims)
                        exact_squeezed = standard_deviation_expected(raw, axes, correction)
                        upstream_expected = np.asarray(upstream.detach().numpy(), dtype=object)
                        if keepdim:
                            upstream_expected = np.squeeze(upstream_expected, axis=axes)
                            exact = exact_squeezed.reshape(
                                tuple(1 if i in axes else shape[i] for i in range(len(shape))))
                        else:
                            exact = exact_squeezed
                        expected_grad = standard_deviation_gradient_expected(
                            raw, axes, correction, upstream_expected
                        )
                        with self.subTest(shape=shape, dim=dim, correction=correction,
                                          keepdim=keepdim):
                            np.testing.assert_allclose(output.detach().numpy(),
                                                       np.asarray(exact, dtype=float),
                                                       rtol=0, atol=2e-15)
                            np.testing.assert_allclose(native.grad.numpy(),
                                                       np.asarray(expected_grad, dtype=float),
                                                       rtol=0, atol=2e-15)

    def test_var_mean_and_combined_vjp_use_both_complete_outputs(self):
        x = values((2, 3, 2))
        for axes in ((1,), (0, 2)):
            for correction in (0, 1):
                expected_var = variance_exact(x, axes, correction)
                expected_mean = reduce_exact(x, axes, "mean")
                g_var = values(expected_var.shape)
                g_mean = values(expected_mean.shape) + Q(3, 4)
                expected_gradient = exact_gradient(
                    lambda z: (
                        variance_exact(z, axes, correction) * g_var
                        + reduce_exact(z, axes, "mean") * g_mean
                    ),
                    x,
                    np.full(expected_var.shape, Q(1), dtype=object),
                )
                with self.subTest(axes=axes, correction=correction):
                    actual_var, actual_mean = A.var_mean(
                        A.tensor(x, "f64"), list(axes), correction
                    )
                    assert_tensor(self, actual_var, expected_var)
                    assert_tensor(self, actual_mean, expected_mean)
                    assert_tensor(
                        self,
                        A.vjp_var_mean(
                            A.tensor(g_var, "f64"), A.tensor(g_mean, "f64"),
                            A.tensor(x, "f64"), list(axes), correction,
                        ),
                        expected_gradient,
                    )

    def test_var_mean_matches_pytorch_210_for_two_outputs_and_vjp(self):
        import torch
        self.assertEqual(torch.__version__, "2.10.0+cu128")
        for shape in ((2, 3), (2, 3, 2)):
            raw = values(shape)
            native = torch.tensor(np.asarray(raw, dtype=float), dtype=torch.float64,
                                  requires_grad=True)
            for dim in (0, -1, tuple(range(1, len(shape)))):
                for correction in (0, 1):
                    for keepdim in (False, True):
                        native.grad = None
                        var_output, mean_output = torch.var_mean(
                            native, dim=dim, correction=correction, keepdim=keepdim
                        )
                        g_var = torch.arange(1, var_output.numel() + 1,
                                             dtype=torch.float64).reshape(var_output.shape)
                        g_mean = torch.arange(2, mean_output.numel() + 2,
                                              dtype=torch.float64).reshape(mean_output.shape)
                        (var_output * g_var + mean_output * g_mean).sum().backward()
                        dims = (dim,) if isinstance(dim, int) else dim
                        axes = tuple(d % len(shape) for d in dims)
                        expected_var_squeezed = variance_exact(raw, axes, correction)
                        expected_mean_squeezed = reduce_exact(raw, axes, "mean")
                        g_var_expected = np.asarray(g_var.detach().numpy(), dtype=object)
                        g_mean_expected = np.asarray(g_mean.detach().numpy(), dtype=object)
                        if keepdim:
                            g_var_expected = np.squeeze(g_var_expected, axis=axes)
                            g_mean_expected = np.squeeze(g_mean_expected, axis=axes)
                            output_shape = tuple(
                                1 if i in axes else shape[i] for i in range(len(shape))
                            )
                            expected_var = expected_var_squeezed.reshape(output_shape)
                            expected_mean = expected_mean_squeezed.reshape(output_shape)
                        else:
                            expected_var = expected_var_squeezed
                            expected_mean = expected_mean_squeezed
                        expected_gradient = exact_gradient(
                            lambda z: (
                                variance_exact(z, axes, correction) * g_var_expected
                                + reduce_exact(z, axes, "mean") * g_mean_expected
                            ),
                            raw,
                            np.full(expected_var_squeezed.shape, Q(1), dtype=object),
                        )
                        with self.subTest(shape=shape, dim=dim, correction=correction,
                                          keepdim=keepdim):
                            np.testing.assert_allclose(var_output.detach().numpy(),
                                                       np.asarray(expected_var, dtype=float),
                                                       rtol=0, atol=2e-15)
                            np.testing.assert_allclose(mean_output.detach().numpy(),
                                                       np.asarray(expected_mean, dtype=float),
                                                       rtol=0, atol=2e-15)
                            np.testing.assert_allclose(native.grad.numpy(),
                                                       np.asarray(expected_gradient, dtype=float),
                                                       rtol=0, atol=2e-15)

    def test_extrema_ties_follow_the_retained_public_rule(self):
        x = A.tensor([[1, 3, 3], [2, -1, -1]], "f64")
        self.assertEqual(A.plain(A.argmax(x, 1)), [1, 0])
        self.assertEqual(A.plain(A.argmin(x, 1)), [0, 1])
        self.assertEqual(A.plain(A.vjp_amax(A.tensor([2, 4], "f64"), x, 1)),
                         [[0, 1, 1], [4, 0, 0]])
        self.assertEqual(A.plain(A.vjp_amin(A.tensor([2, 4], "f64"), x, 1)),
                         [[2, 0, 0], [0, 2, 2]])
        self.assertEqual(A.plain(A.vjp_max_dim(A.tensor([2, 4], "f64"), x, 1)),
                         [[0, 2, 0], [4, 0, 0]])
        self.assertEqual(A.plain(A.vjp_min_dim(A.tensor([2, 4], "f64"), x, 1)),
                         [[2, 0, 0], [0, 4, 0]])

    def test_equal_logsumexp_slice_has_exact_vjp_and_encloses_log_count(self):
        x = A.tensor([[0, 0, 0, 0]], "f64")
        value = A.logsumexp(x, 1).a[0]
        import mpmath as mp
        lo, hi = ends(value.v)
        expected = mp.log(4)
        self.assertLessEqual(mp.mpf(lo.numerator) / lo.denominator, expected)
        self.assertGreaterEqual(mp.mpf(hi.numerator) / hi.denominator, expected)
        gradient = A.vjp_logsumexp(A.tensor([3], "f64"), x, 1)
        for item in gradient.a.flat:
            lo, hi = ends(item.v)
            self.assertLessEqual(lo, Q(3, 4))
            self.assertGreaterEqual(hi, Q(3, 4))


if __name__ == "__main__":
    unittest.main()
