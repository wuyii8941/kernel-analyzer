"""Adapter for the reviewed finite max_pool2d values scope in contracts/POOL.yaml."""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec import ieee
from kernel_spec.aten_adapter import decode, make_output_checker, make_reference
from kernel_spec.num import Invalid, Unsupported, Result
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "POOL.yaml").read_text())["covers"]
PROFILE = ("finite_max2d_values",)
FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}


def _raw_dtype(value):
    if not isinstance(value, dict) or "t" not in value:
        return None
    try:
        return A.dt(value["t"])
    except Invalid:
        if "complex" in str(value["t"]).lower():
            raise Unsupported("complex tensors are outside the reviewed POOL profile")
        raise


def _raw_leaves(value):
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        else:
            yield item


def _source_tensor(value, name):
    dtype = _raw_dtype(value)
    if dtype is not None and dtype not in FLOAT_DTYPES:
        raise Unsupported(f"{name} dtype {dtype!r} is outside the finite floating POOL profile")
    if isinstance(value, dict) and "v" in value:
        for raw in _raw_leaves(value["v"]):
            if isinstance(raw, (bool, np.bool_)):
                raise Invalid(f"{name} floating values cannot be Boolean literals")
            if isinstance(raw, str):
                spelling = raw.strip().lower()
                if spelling in {"nan", "inf", "+inf", "-inf"}:
                    raise Unsupported(f"{name} contains a special value outside the reviewed profile")
                try:
                    if spelling.startswith("-") and Fraction(spelling) == 0:
                        raise Unsupported("signed-zero distinctions are outside the reviewed POOL profile")
                except ValueError:
                    pass
            if isinstance(raw, (float, np.floating)):
                if not math.isfinite(float(raw)):
                    raise Unsupported(f"{name} contains a special value outside the reviewed profile")
                if raw == 0 and math.copysign(1.0, float(raw)) < 0:
                    raise Unsupported("signed-zero distinctions are outside the reviewed POOL profile")
    tensor = decode(value)
    if not isinstance(tensor, A.Ten):
        raise Invalid(f"{name} must be a tensor")
    if tensor.dtype not in FLOAT_DTYPES:
        raise Unsupported(f"{name} must use f16, bf16, f32 or f64")
    if tensor.a.ndim not in (3, 4) or any(size == 0 for size in tensor.shape):
        raise Unsupported("POOL requires a positive rank-3 or rank-4 tensor")
    for item in tensor.a.flat:
        if not isinstance(item, R) or not item.exact:
            raise Invalid(f"{name} values must be exact real numbers")
        if not item.finite:
            raise Unsupported(f"{name} contains a special value outside the reviewed profile")
        stored = ieee.round_to(item, tensor.dtype)
        if not stored.exact or stored.v != item.v:
            raise Invalid(f"{item!r} is not exactly representable as {tensor.dtype}")
    return tensor


def _decode_argument(op, name, value, arguments):
    return _source_tensor(value, name) if name == "a" else decode(value)


def _pair(value, name, minimum):
    if isinstance(value, (bool, np.bool_)):
        raise Invalid(f"{name} must be an integer or a pair of integers")
    if isinstance(value, (int, np.integer)):
        pair = (int(value), int(value))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)) for item in value):
            raise Invalid(f"{name} must contain integers")
        pair = tuple(int(item) for item in value)
    else:
        raise Invalid(f"{name} must be an integer or a pair of integers")
    if any(item < minimum for item in pair):
        raise Invalid(f"{name} components must be at least {minimum}")
    return pair


def _flag(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise Invalid(f"{name} must be boolean")
    return bool(value)


def _max_pool2d_values(a, kernel_size, stride=None, padding=0, dilation=1,
                       ceil_mode=False, return_indices=False, *, mode="max"):
    kernel = _pair(kernel_size, "kernel_size", 1)
    if stride is None or stride == [] or stride == ():
        stride_pair = (1, 1) if mode == "default_stride_one" else kernel
    else:
        stride_pair = _pair(stride, "stride", 1)
    padding_pair = _pair(padding, "padding", 0)
    if padding_pair != (0, 0):
        raise Unsupported("nonzero padding is outside the reviewed POOL profile")
    dilation_pair = _pair(dilation, "dilation", 1)
    if mode == "ignore_dilation":
        dilation_pair = (1, 1)
    if _flag(ceil_mode, "ceil_mode"):
        raise Unsupported("ceil_mode=true is outside the reviewed POOL profile")
    if _flag(return_indices, "return_indices"):
        raise Unsupported("returned indices are outside the reviewed POOL profile")
    if mode == "swap_kernel":
        kernel = kernel[::-1]

    height, width = a.shape[-2:]
    effective_h = dilation_pair[0] * (kernel[0] - 1) + 1
    effective_w = dilation_pair[1] * (kernel[1] - 1) + 1
    out_h = (height - effective_h) // stride_pair[0] + 1
    out_w = (width - effective_w) // stride_pair[1] + 1
    if out_h < 1 or out_w < 1:
        raise Invalid("max_pool2d output dimensions must be positive")
    out = np.empty(a.shape[:-2] + (out_h, out_w), dtype=object)
    for lead in np.ndindex(a.shape[:-2]):
        for oh in range(out_h):
            for ow in range(out_w):
                positions = [(oh * stride_pair[0] + mh * dilation_pair[0],
                              ow * stride_pair[1] + mw * dilation_pair[1])
                             for mh in range(kernel[0]) for mw in range(kernel[1])]
                if mode == "omit_last":
                    positions = positions[:-1]
                values = [a.a[lead + position] for position in positions]
                if not values:
                    raise Invalid("pooling window contains no selected input")
                if mode == "first":
                    selected = values[0]
                elif mode == "min":
                    selected = values[0]
                    for value in values[1:]:
                        if value < selected:
                            selected = value
                else:
                    selected = values[0]
                    for value in values[1:]:
                        if value > selected:
                            selected = value
                out[lead + (oh, ow)] = selected
    return A.Ten(out, a.dtype)


def _aten_max_pool2d(a, kernel_size, stride=(), padding=0, dilation=1, ceil_mode=False):
    return _max_pool2d_values(a, kernel_size, stride, padding, dilation, ceil_mode, False)


def _functional_max_pool2d(a, kernel_size, stride=None, padding=0, dilation=1,
                           ceil_mode=False, return_indices=False):
    return _max_pool2d_values(a, kernel_size, stride, padding, dilation, ceil_mode, return_indices)


OPS = {
    "aten.max_pool2d": _aten_max_pool2d,
    "torch.nn.functional.max_pool2d": _functional_max_pool2d,
}
_max_reference = make_reference(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument)
_max_check_output = make_output_checker(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=_decode_argument)

AVG_PROFILE = 'finite_avg2d_autograd'
MAX_AUTOGRAD_PROFILE = 'torch210_unique_max2d_autograd'
AVG_KEYS = {'op', 'x', 'upstream', 'kernel_size', 'stride', 'padding', 'ceil_mode',
            'count_include_pad', 'divisor_override'}
MAX_AUTOGRAD_KEYS = {'op', 'x', 'upstream', 'kernel_size', 'stride', 'padding',
                     'dilation', 'ceil_mode', 'return_indices'}


def _avg_pair(value, name, minimum):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = (value[0], value[0])
    return _pair(value, name, minimum)


def _average(inp, *, divisor_reading='R2'):
    from kernel_spec import e2_pool
    if not isinstance(inp, dict) or set(inp) != AVG_KEYS:
        raise Invalid('average-pooling task requires all declared inputs/options and no extra fields')
    if inp['op'] != 'avg_pool2d':
        raise Unsupported('this profile selects only two-dimensional average pooling with its input VJP')
    for name in ('x', 'upstream'):
        if not isinstance(inp[name], dict) or set(inp[name]) != {'t', 'v', 'shape'}:
            raise Invalid(f'{name} requires a complete physical tensor record')
    x, upstream = _source_tensor(inp['x'], 'x'), _source_tensor(inp['upstream'], 'upstream')
    if x.dtype != upstream.dtype:
        raise Invalid('upstream must have the input dtype')
    kernel = _avg_pair(inp['kernel_size'], 'kernel_size', 1)
    stride = kernel if inp['stride'] is None or inp['stride'] == [] or inp['stride'] == () else _avg_pair(inp['stride'], 'stride', 1)
    padding = _avg_pair(inp['padding'], 'padding', 0)
    ceil = _flag(inp['ceil_mode'], 'ceil_mode')
    include = _flag(inp['count_include_pad'], 'count_include_pad')
    override = inp['divisor_override']
    if override is not None:
        if isinstance(override, (bool, np.bool_)) or not isinstance(override, (int, np.integer)):
            raise Invalid('divisor_override must be an integer')
        override = int(override)
        if override == 0:
            raise Invalid('divisor_override must be nonzero')
    # Reuse the existing exact window evaluator only after the narrower public
    # input contract has been validated. Its average path has no argmax result.
    fractions = lambda t: np.vectorize(lambda item: item.v, otypes=[object])(t.a).tolist()
    fields = e2_pool.evaluate('real', {'op': 'avg_pool2d', 'x': fractions(x),
        'upstream': fractions(upstream), 'kernel': list(kernel), 'stride': list(stride),
        'padding': list(padding), 'ceil_mode': ceil, 'count_include_pad': include,
        'divisor_override': override}, divisor_reading=divisor_reading)
    if set(fields) != {'out', 'grad', 'argmax'} or fields['argmax'] != []:
        raise ValueError('unexpected output of the average-pooling evaluator')
    return Result('value', {k: fields[k] for k in ('out', 'grad')})


def _pooled_length(length, kernel, stride, padding, dilation, ceil):
    effective = dilation * (kernel - 1) + 1
    numerator = length + 2 * padding - effective
    result = ((numerator + stride - 1) // stride if ceil else numerator // stride) + 1
    # PyTorch removes a ceil-mode window whose first sample lies wholly in the
    # right/bottom padding. This is the correction stated on the public page.
    if ceil and result > 0 and (result - 1) * stride >= length + padding:
        result -= 1
    if result < 1:
        raise Invalid('max_pool2d output dimensions must be positive')
    return result


def _max_autograd(inp, *, mode='correct'):
    if not isinstance(inp, dict) or set(inp) != MAX_AUTOGRAD_KEYS:
        raise Invalid('max-pooling autograd task requires all declared inputs/options and no extra fields')
    if inp['op'] != 'torch.nn.functional.max_pool2d':
        raise Unsupported('this profile selects the public functional two-dimensional max-pooling task')
    for name in ('x', 'upstream'):
        if not isinstance(inp[name], dict) or set(inp[name]) != {'t', 'v', 'shape'}:
            raise Invalid(f'{name} requires a complete physical tensor record')
    x, upstream = _source_tensor(inp['x'], 'x'), _source_tensor(inp['upstream'], 'upstream')
    if x.dtype != upstream.dtype:
        raise Invalid('upstream must have the input dtype')
    kernel = _pair(inp['kernel_size'], 'kernel_size', 1)
    stride = kernel if inp['stride'] is None or inp['stride'] == [] or inp['stride'] == () else _pair(inp['stride'], 'stride', 1)
    padding = _pair(inp['padding'], 'padding', 0)
    dilation = _pair(inp['dilation'], 'dilation', 1)
    ceil = _flag(inp['ceil_mode'], 'ceil_mode')
    if not _flag(inp['return_indices'], 'return_indices'):
        raise Unsupported('the selected complete task requires return_indices=true')
    if any(2 * pad > size for pad, size in zip(padding, kernel)):
        raise Invalid('padding must be at most half the corresponding kernel size')

    height, width = x.shape[-2:]
    shape_dilation = dilation
    shape_ceil = False if mode == 'floor_shape' else ceil
    out_h = _pooled_length(height, kernel[0], stride[0], padding[0], shape_dilation[0], shape_ceil)
    out_w = _pooled_length(width, kernel[1], stride[1], padding[1], shape_dilation[1], shape_ceil)
    expected_shape = x.shape[:-2] + (out_h, out_w)
    if upstream.shape != expected_shape:
        raise Invalid(f'upstream shape must be {expected_shape}')

    out = np.empty(expected_shape, dtype=object)
    indices = np.empty(expected_shape, dtype=object)
    grad = np.empty(x.shape, dtype=object)
    grad.fill(R(0))
    sample_dilation = (1, 1) if mode == 'ignore_dilation' else dilation
    for lead in np.ndindex(x.shape[:-2]):
        for oh in range(out_h):
            for ow in range(out_w):
                top = oh * stride[0] - (0 if mode == 'ignore_padding' else padding[0])
                left = ow * stride[1] - (0 if mode == 'ignore_padding' else padding[1])
                positions = []
                for kh in range(kernel[0]):
                    ih = top + kh * sample_dilation[0]
                    for kw in range(kernel[1]):
                        iw = left + kw * sample_dilation[1]
                        if 0 <= ih < height and 0 <= iw < width:
                            positions.append((ih, iw))
                if not positions:
                    raise Unsupported('a selected window has no real input coordinate')
                best = x.a[lead + positions[0]]
                for position in positions[1:]:
                    value = x.a[lead + position]
                    if value > best:
                        best = value
                selected = [position for position in positions if x.a[lead + position].same(best)]
                if len(selected) != 1:
                    raise Unsupported('tied maximum: the selected task has no unique ordinary derivative')
                ih, iw = selected[0]
                output_position = lead + (oh, ow)
                out[output_position] = best
                indices[output_position] = iw * height + ih if mode == 'column_index' else ih * width + iw
                contribution = upstream.a[output_position]
                input_position = lead + (ih, iw)
                grad[input_position] = contribution if mode == 'overwrite_grad' else grad[input_position] + contribution
    exact_values = lambda array: np.vectorize(lambda value: value.v, otypes=[object])(array).tolist()
    fields = {'out': exact_values(out), 'indices': indices.tolist(), 'grad': exact_values(grad)}
    if mode == 'omit_indices':
        fields.pop('indices')
    return Result('value', fields)


def reference(profile, inp):
    if profile == AVG_PROFILE:
        return _average(inp)
    if profile == MAX_AUTOGRAD_PROFILE:
        return _max_autograd(inp)
    return _max_reference(profile, inp)


def check_output(profile, inp, observed):
    if profile not in (AVG_PROFILE, MAX_AUTOGRAD_PROFILE):
        return _max_check_output(profile, inp, observed)
    from kernel_spec.compare import check_enclosed_output
    expected = (_average(inp) if profile == AVG_PROFILE else _max_autograd(inp)).fields
    if not isinstance(observed, dict) or set(observed) != set(expected):
        return False
    candidate = {}
    for key, wanted in expected.items():
        raw = observed[key]
        if not isinstance(raw, dict) or set(raw) != {'t', 'v', 'shape'}:
            return False
        try:
            shape = tuple(A._layout_sizes(raw['shape'], 'observed shape'))
            value = decode(raw)
            wanted_dtype = 'i64' if key == 'indices' else _raw_dtype(inp['x'])
            if value.dtype != wanted_dtype or shape != value.shape or value.shape != np.shape(wanted):
                return False
            for item in value.a.flat:
                if key == 'indices':
                    if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                        return False
                else:
                    if not isinstance(item, R) or not item.exact or not item.finite:
                        return False
                    rounded = ieee.round_to(item, value.dtype)
                    if not rounded.exact or rounded.v != item.v:
                        return False
            candidate[key] = value.a.tolist()
        except (Invalid, Unsupported):
            return False
    return check_enclosed_output(expected, candidate)


def _mutant_aten(mode):
    def operation(a, kernel_size, stride=(), padding=0, dilation=1, ceil_mode=False):
        return _max_pool2d_values(a, kernel_size, stride, padding, dilation, ceil_mode, False, mode=mode)
    return make_reference(COVERS, extra={**OPS, "aten.max_pool2d": operation}, profiles=PROFILE,
                          argument_decoder=_decode_argument)


def _mutant_functional(mode):
    def operation(a, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
        return _max_pool2d_values(a, kernel_size, stride, padding, dilation, ceil_mode, return_indices, mode=mode)
    return make_reference(COVERS, extra={**OPS, "torch.nn.functional.max_pool2d": operation}, profiles=PROFILE,
                          argument_decoder=_decode_argument)


MUTANTS = {
    "P2.M1": _mutant_aten("min"),
    "P2.M2": _mutant_aten("first"),
    "P2.M3": _mutant_aten("ignore_dilation"),
    "P2.M4": _mutant_functional("default_stride_one"),
    "P2.M5": _mutant_aten("swap_kernel"),
    "P2.M6": _mutant_aten("omit_last"),
}


def _avg_mutant(mode):
    def mutant(profile, inp):
        if profile != AVG_PROFILE:
            raise Unsupported('average-pooling mutant needs its selected profile')
        if mode == 'fixed_kernel':
            return _average(inp, divisor_reading='R1')
        if mode == 'backward_fixed_kernel':
            correct = _average(inp).fields
            wrong = _average(inp, divisor_reading='R1').fields
            return Result('value', {'out': correct['out'], 'grad': wrong['grad']})
        if mode == 'ignore_override':
            return _average({**inp, 'divisor_override': None})
        if mode == 'include_pad':
            return _average({**inp, 'count_include_pad': True})
        if mode == 'omit_gradient':
            return Result('value', {'out': _average(inp).fields['out']})
        raise ValueError(mode)
    return mutant


MUTANTS.update({f'PA.M{i}': _avg_mutant(mode) for i, mode in enumerate(
    ['fixed_kernel', 'backward_fixed_kernel', 'ignore_override', 'include_pad', 'omit_gradient'], 1)})


def _max_autograd_mutant(mode):
    def mutant(profile, inp):
        if profile != MAX_AUTOGRAD_PROFILE:
            raise Unsupported('max-pooling autograd mutant needs its selected profile')
        return _max_autograd(inp, mode=mode)
    return mutant


MUTANTS.update({f'PM.M{i}': _max_autograd_mutant(mode) for i, mode in enumerate(
    ['floor_shape', 'ignore_padding', 'column_index', 'overwrite_grad', 'ignore_dilation', 'omit_indices'], 1)})
READINGS = {}
