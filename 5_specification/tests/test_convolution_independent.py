"""Independent exact reconstruction of the reviewed convolution profile.

The forward function below directly enumerates the documented cross-correlation
with Fraction arithmetic.  The scalar objective is affine in each input,
weight and bias coordinate, so the symmetric unit difference is its exact
partial derivative rather than a numerical approximation.
"""
from fractions import Fraction as Q

import numpy as np

from kernel_spec import aten as A, registry
from kernel_spec.num import evaluate


def values(shape, offset=0):
    result = np.empty(shape, dtype=object)
    for number, index in enumerate(np.ndindex(shape)):
        result[index] = Q((number + offset) % 5 - 2, 4)
    return result


def direct_convolution(x, weight, bias, stride, padding, dilation, groups):
    spatial = x.ndim - 2
    output_spatial = tuple(
        (x.shape[2 + axis] + 2 * padding[axis]
         - dilation[axis] * (weight.shape[2 + axis] - 1) - 1) // stride[axis] + 1
        for axis in range(spatial)
    )
    result = np.empty((x.shape[0], weight.shape[0], *output_spatial), dtype=object)
    outputs_per_group = weight.shape[0] // groups
    inputs_per_group = weight.shape[1]
    for batch in range(x.shape[0]):
        for output_channel in range(weight.shape[0]):
            group = output_channel // outputs_per_group
            for output_index in np.ndindex(output_spatial):
                total = bias[output_channel]
                for local_input in range(inputs_per_group):
                    input_channel = group * inputs_per_group + local_input
                    for kernel_index in np.ndindex(weight.shape[2:]):
                        source = tuple(
                            output_index[axis] * stride[axis]
                            + kernel_index[axis] * dilation[axis] - padding[axis]
                            for axis in range(spatial)
                        )
                        if all(0 <= source[axis] < x.shape[2 + axis] for axis in range(spatial)):
                            total += x[(batch, input_channel, *source)] * weight[
                                (output_channel, local_input, *kernel_index)]
                result[(batch, output_channel, *output_index)] = total
    return result


def exact_derivatives(forward, operands, upstream):
    def objective(args):
        output = forward(*args)
        return sum((output[index] * upstream[index] for index in np.ndindex(output.shape)), Q(0))

    derivatives = []
    for operand_number, operand in enumerate(operands):
        result = np.empty(operand.shape, dtype=object)
        for index in np.ndindex(operand.shape):
            plus = [value.copy() for value in operands]
            minus = [value.copy() for value in operands]
            plus[operand_number][index] += 1
            minus[operand_number][index] -= 1
            result[index] = (objective(plus) - objective(minus)) / 2
        derivatives.append(result)
    return derivatives


def assert_tensor_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == "f64"
    for index in np.ndindex(expected.shape):
        assert actual.a[index].v == expected[index]


def test_forward_and_vjp_match_independent_one_and_two_dimensional_formulas():
    cases = [
        ((1, 2, 5), (2, 2, 2), [2], [1], [2], 1),
        ((1, 2, 5), (2, 1, 2), [1], [1], [1], 2),
        ((1, 2, 3, 4), (2, 2, 2, 2), [1, 2], [1, 0], [1, 1], 1),
        ((1, 2, 3, 3), (2, 1, 2, 2), [1, 1], [0, 1], [1, 1], 2),
    ]
    for x_shape, weight_shape, stride, padding, dilation, groups in cases:
        x = values(x_shape)
        weight = values(weight_shape, 1)
        bias = values((weight_shape[0],), 2)
        forward = lambda a, w, b: direct_convolution(a, w, b, stride, padding, dilation, groups)
        expected = forward(x, weight, bias)
        actual = A.convolution(A.tensor(x, "f64"), A.tensor(weight, "f64"), A.tensor(bias, "f64"),
                               stride, padding, dilation, False, [0] * len(stride), groups)
        assert_tensor_equal(actual, expected)

        upstream = values(expected.shape, 3)
        derivatives = exact_derivatives(forward, [x, weight, bias], upstream)
        gradients = A.convolution_backward(
            A.tensor(upstream, "f64"), A.tensor(x, "f64"), A.tensor(weight, "f64"),
            [weight_shape[0]], stride, padding, dilation, False, [0] * len(stride), groups,
            [True, True, True],
        )
        for gradient, derivative in zip(gradients, derivatives):
            assert_tensor_equal(gradient, derivative)


def tensor(values_, shape, dtype="f32"):
    return {"t": dtype, "v": values_, "shape": shape}


def test_complete_backward_output_preserves_null_positions_and_metadata():
    adapter = registry.load("CONV").adapter
    call = {"op": "aten.convolution_backward", "args": [
        tensor([[[1, 1, 1]]], [1, 1, 3]), tensor([[[1, 2, 3]]], [1, 1, 3]),
        tensor([[[2]]], [1, 1, 1]), None, [1], [0], [1], False, [0], 1,
        [True, False, True],
    ]}
    output = [tensor([[[2, 2, 2]]], [1, 1, 3]), None, tensor([3], [1])]
    assert adapter.check_output("real", call, {"out": output})
    verdict, _, labels = registry.check_output("CONV", "real", call, {"out": output})
    assert verdict == "consistent"
    assert labels["semantic_verdict"] == "not_established"
    for bad in (
        output[:2],
        [output[0], tensor([[[6]]], [1, 1, 1]), output[2]],
        [output[0], None, tensor([3], [1], "f64")],
        [tensor([[[2, 2, 2]]], [1, 3]), None, output[2]],
    ):
        assert not adapter.check_output("real", call, {"out": bad})


def test_profile_refuses_calls_outside_reviewed_schema_and_domain():
    base = [tensor([[[1]]], [1, 1, 1]), tensor([[[2]]], [1, 1, 1]), None,
            [1], [0], [1], False, [0], 1]
    cases = [
        {"op": "aten.convolution", "args": [tensor([[[1]]], [1, 1, 1], "i64"),
                                               tensor([[[2]]], [1, 1, 1], "i64"), *base[2:]]},
        {"op": "aten.convolution", "args": [tensor([[["inf"]]], [1, 1, 1]), *base[1:]]},
        {"op": "aten.convolution", "args": [tensor([1], [1]), *base[1:]]},
        {"op": "aten.convolution", "args": [tensor([[[[[1]]]]], [1, 1, 1, 1, 1]),
                                               tensor([[[[[2]]]]], [1, 1, 1, 1, 1]), None,
                                               [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1]},
        {"op": "aten.convolution_backward", "args": [tensor([[[1]]], [1, 1, 1]),
            tensor([[[1]]], [1, 1, 1]), tensor([[[2]]], [1, 1, 1]), None,
            [1], [0], [1], False, [0], 1]},
    ]
    expected = ["unsupported", "unsupported", "invalid_input", "unsupported", "invalid_input"]
    reference = registry.load("CONV").adapter.reference
    for call, outcome in zip(cases, expected):
        assert evaluate(reference, "real", call).outcome == outcome
