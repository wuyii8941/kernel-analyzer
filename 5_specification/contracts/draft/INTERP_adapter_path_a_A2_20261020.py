"""Reference for the selected PyTorch 2.10 legacy-nearest 2D interpolation task and the ATen overloads
upsample_nearest2d / upsample_nearest2d_backward (output_size with optional per-dimension scales)."""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec import ieee
from kernel_spec.aten_adapter import decode, make_output_checker, make_reference
from kernel_spec.compare import check_enclosed_output
from kernel_spec.num import Invalid, Unsupported, Result
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / 'INTERP.yaml').read_text())['covers']
PROFILE = 'torch210_nearest2d_size_autograd'
KEYS = {'op', 'x', 'upstream', 'size', 'scale_factor', 'mode', 'align_corners',
        'recompute_scale_factor', 'antialias'}
FLOAT_DTYPES = {'f16', 'bf16', 'f32', 'f64'}


def _raw_dtype(value):
    if not isinstance(value, dict) or 't' not in value:
        return None
    try:
        return A.dt(value['t'])
    except Invalid:
        if 'complex' in str(value['t']).lower():
            raise Unsupported('complex tensors are outside the selected interpolation profile')
        raise


def _leaves(value):
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        else:
            yield item


def _physical(value, name):
    if not isinstance(value, dict) or set(value) != {'t', 'v', 'shape'}:
        raise Invalid(f'{name} requires a complete physical tensor record')
    dtype = _raw_dtype(value)
    if dtype not in FLOAT_DTYPES:
        raise Unsupported(f'{name} must use f16, bf16, f32 or f64')
    for raw in _leaves(value['v']):
        if isinstance(raw, (bool, np.bool_)):
            raise Invalid(f'{name} floating values cannot be Boolean literals')
        if isinstance(raw, str):
            spelling = raw.strip().lower()
            if spelling in {'nan', 'inf', '+inf', '-inf'}:
                raise Unsupported(f'{name} contains a special value')
            try:
                if spelling.startswith('-') and Fraction(spelling) == 0:
                    raise Unsupported('signed-zero distinctions are outside the selected profile')
            except ValueError:
                pass
        if isinstance(raw, (float, np.floating)):
            if not math.isfinite(float(raw)):
                raise Unsupported(f'{name} contains a special value')
            if raw == 0 and math.copysign(1.0, float(raw)) < 0:
                raise Unsupported('signed-zero distinctions are outside the selected profile')
    tensor = decode(value)
    if not isinstance(tensor, A.Ten):
        raise Invalid(f'{name} must be a tensor')
    if tensor.a.ndim != 4 or any(length <= 0 for length in tensor.shape):
        raise Unsupported('the selected task requires a positive rank-4 tensor')
    for item in tensor.a.flat:
        if not isinstance(item, R) or not item.exact:
            raise Invalid(f'{name} values must be exact real numbers')
        if not item.finite:
            raise Unsupported(f'{name} contains a special value')
        stored = ieee.round_to(item, tensor.dtype)
        if not stored.exact or stored.v != item.v:
            raise Invalid(f'{item!r} is not exactly representable as {tensor.dtype}')
    return tensor


def _size(value):
    if isinstance(value, (bool, np.bool_)):
        raise Invalid('size must be a positive integer or pair')
    if isinstance(value, (int, np.integer)):
        result = (int(value), int(value))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)) for item in value):
            raise Invalid('size must contain integers')
        result = tuple(int(item) for item in value)
    else:
        raise Invalid('size must be a positive integer or pair')
    if any(item <= 0 for item in result):
        raise Invalid('size components must be positive')
    return result


def _evaluate(inp, *, mode='correct'):
    if not isinstance(inp, dict) or set(inp) != KEYS:
        raise Invalid('interpolation task requires every declared field and no extra fields')
    if inp['op'] != 'torch.nn.functional.interpolate':
        raise Unsupported('this profile selects the public functional interpolation task')
    if inp['scale_factor'] is not None:
        raise Unsupported('scale-factor interpolation is a different task')
    if inp['mode'] != 'nearest':
        raise Unsupported('this profile selects PyTorch 2.10 legacy nearest mode')
    if inp['align_corners'] is not None:
        raise Invalid('align_corners must be None for nearest mode')
    recompute = inp['recompute_scale_factor']
    if recompute is not None and not isinstance(recompute, (bool, np.bool_)):
        raise Invalid('recompute_scale_factor must be None or boolean')
    if recompute:
        raise Invalid('recompute_scale_factor=true is invalid with explicit size')
    if not isinstance(inp['antialias'], (bool, np.bool_)):
        raise Invalid('antialias must be boolean')
    if inp['antialias']:
        raise Invalid('antialias=true is invalid for nearest mode')
    x, upstream = _physical(inp['x'], 'x'), _physical(inp['upstream'], 'upstream')
    if x.dtype != upstream.dtype:
        raise Invalid('upstream must have the input dtype')
    out_h, out_w = _size(inp['size'])
    expected_shape = x.shape[:2] + (out_h, out_w)
    if upstream.shape != expected_shape:
        raise Invalid(f'upstream shape must be {expected_shape}')

    height, width = x.shape[-2:]
    out = np.empty(expected_shape, dtype=object)
    grad = np.empty(x.shape, dtype=object)
    grad.fill(R(0))

    def coordinate(output_index, input_length, output_length, other_input, other_output):
        if mode == 'nearest_exact':
            return min(((2 * output_index + 1) * input_length) // (2 * output_length), input_length - 1)
        if mode == 'round':
            return min((2 * output_index * input_length + output_length) // (2 * output_length), input_length - 1)
        if mode == 'swap_scales':
            return min(output_index * other_input // other_output, input_length - 1)
        return output_index * input_length // output_length

    for n, c, oh, ow in np.ndindex(expected_shape):
        ih = coordinate(oh, height, out_h, width, out_w)
        iw = coordinate(ow, width, out_w, height, out_h)
        out[n, c, oh, ow] = x.a[n, c, ih, iw]
        if mode == 'overwrite_grad':
            grad[n, c, ih, iw] = upstream.a[n, c, oh, ow]
        else:
            grad[n, c, ih, iw] = grad[n, c, ih, iw] + upstream.a[n, c, oh, ow]
    exact = lambda array: np.vectorize(lambda value: value.v, otypes=[object])(array).tolist()
    fields = {'out': exact(out), 'grad': exact(grad)}
    if mode == 'omit_gradient':
        fields.pop('grad')
    return Result('value', fields)


ATEN_PROFILE = 'torch210_aten_upsample_nearest2d'
ATEN_COVERS = ['aten.upsample_nearest2d', 'aten.upsample_nearest2d_backward']


def _scale(value, name):
    """None, or the exact rational value of the received positive finite scale (IN.ATEN_SCALES)."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, Fraction, np.integer, np.floating)):
        raise Invalid(f'{name} must be None or a floating scale')
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise Unsupported(f'non-finite {name} is outside the selected profile')
    exact = Fraction(value)
    if exact <= 0:
        raise Unsupported(f'nonpositive {name} is outside the selected profile')
    return exact


def _representable(value, dtype):
    stored = ieee.round_to(R(value), dtype)
    return stored.exact and stored.v == value


def _source_indices(input_length, output_length, scale, name, *, mode='correct'):
    """IN.ATEN_SCALES: input index floor(o*q) with q = 1/scale when a scale is given, otherwise input/output."""
    if mode == 'ignore_scale':
        scale = None
    if scale is None:
        ratio = Fraction(input_length, output_length)
    else:
        ratio = scale if mode == 'multiply_scale' else 1 / scale
        # The retained decomposition evaluates isize/(isize*scale) in binary64 and (float32 o)*scale; the exact
        # reading is used only where every one of these quantities is exactly representable.
        if not (_representable(input_length * scale, 'f64') and _representable(ratio, 'f32')
                and all(_representable(o * ratio, 'f32') for o in range(output_length))):
            raise Unsupported(f'{name}={float(scale)!r}: the source index arithmetic is not exact; '
                              'its floating rounding is outside the selected profile')
    offset = Fraction(1, 2) if mode == 'nearest_exact' else 0
    indices = [math.floor((o + offset) * ratio) for o in range(output_length)]
    if any(index >= input_length for index in indices):
        raise Unsupported(f'{name} selects an input index beyond the input; the retained source leaves it undefined')
    return indices


def _aten_size(output_size, scales_h, scales_w):
    if output_size is None or isinstance(scales_h, (list, tuple)):
        raise Unsupported('the upsample_nearest2d.vec overload is outside the selected profile')
    return _size(output_size), _scale(scales_h, 'scales_h'), _scale(scales_w, 'scales_w')


def _tensor_argument(op, name, value, arguments):
    if name in ('a', 'grad_output'):
        return _physical(value, name)
    return decode(value)


def _aten_forward(a, output_size, scales_h=None, scales_w=None, *, mode='correct'):
    (out_h, out_w), sh, sw = _aten_size(output_size, scales_h, scales_w)
    if mode == 'swap':
        sh, sw = sw, sh
    height, width = a.shape[-2:]
    rows = _source_indices(height, out_h, sh, 'scales_h', mode=mode)
    cols = _source_indices(width, out_w, sw, 'scales_w', mode=mode)
    out = np.empty(a.shape[:2] + (out_h, out_w), dtype=object)
    for n, c, oh, ow in np.ndindex(out.shape):
        out[n, c, oh, ow] = a.a[n, c, rows[oh], cols[ow]]
    return A.Ten(out, a.dtype)


def _aten_backward(grad_output, output_size, input_size, scales_h=None, scales_w=None, *, mode='correct'):
    (out_h, out_w), sh, sw = _aten_size(output_size, scales_h, scales_w)
    input_shape = A._layout_sizes(input_size, 'input_size')
    if len(input_shape) != 4 or any(length <= 0 for length in input_shape):
        raise Invalid('input_size must contain four positive sizes')
    expected = tuple(input_shape[:2]) + (out_h, out_w)
    if grad_output.shape != expected:
        raise Invalid(f'grad_output shape must be {expected}')
    rows = _source_indices(input_shape[2], out_h, sh, 'scales_h')
    cols = _source_indices(input_shape[3], out_w, sw, 'scales_w')
    grad = np.empty(tuple(input_shape), dtype=object)
    grad.fill(R(0))
    for n, c, oh, ow in np.ndindex(expected):
        target = (n, c, rows[oh], cols[ow])
        contribution = grad_output.a[n, c, oh, ow]
        grad[target] = contribution if mode == 'overwrite_grad' else grad[target] + contribution
    return A.Ten(grad, grad_output.dtype)


def _forward_op(a, output_size, scales_h=None, scales_w=None):
    """IN.ATEN: aten.upsample_nearest2d."""
    return _aten_forward(a, output_size, scales_h, scales_w)


def _backward_op(grad_output, output_size, input_size, scales_h=None, scales_w=None):
    """IN.ATEN: aten.upsample_nearest2d_backward, the VJP of the forward with the same index rule."""
    return _aten_backward(grad_output, output_size, input_size, scales_h, scales_w)


ATEN_OPS = {'aten.upsample_nearest2d': _forward_op, 'aten.upsample_nearest2d_backward': _backward_op}
_aten_reference = make_reference(ATEN_COVERS, extra=ATEN_OPS, profiles=(ATEN_PROFILE,),
                                 argument_decoder=_tensor_argument)
_aten_check_output = make_output_checker(ATEN_COVERS, extra=ATEN_OPS, profiles=(ATEN_PROFILE,),
                                         argument_decoder=_tensor_argument)


def reference(profile, inp):
    if profile == ATEN_PROFILE:
        return _aten_reference(profile, inp)
    if profile != PROFILE:
        raise Unsupported('unknown INTERP profile')
    if isinstance(inp, dict) and isinstance(inp.get('op'), str) and inp['op'].startswith('aten.'):
        raise Unsupported(f'profile {PROFILE} selects the public functional task record, not an ATen call')
    return _evaluate(inp)


def check_output(profile, inp, observed):
    if profile == ATEN_PROFILE:
        return _aten_check_output(profile, inp, observed)
    if profile != PROFILE:
        return False
    expected = _evaluate(inp).fields
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
            if value.dtype != _raw_dtype(inp['x']) or shape != value.shape or value.shape != np.shape(wanted):
                return False
            for item in value.a.flat:
                if not isinstance(item, R) or not item.exact or not item.finite:
                    return False
                rounded = ieee.round_to(item, value.dtype)
                if not rounded.exact or rounded.v != item.v:
                    return False
            candidate[key] = value.a.tolist()
        except (Invalid, Unsupported):
            return False
    return check_enclosed_output(expected, candidate)


def _mutant(mode):
    def mutant(profile, inp):
        if profile != PROFILE:
            raise Unsupported('INTERP mutant needs its selected profile')
        return _evaluate(inp, mode=mode)
    return mutant


MUTANTS = {f'IN.M{i}': _mutant(mode) for i, mode in enumerate(
    ['nearest_exact', 'round', 'swap_scales', 'overwrite_grad', 'omit_gradient'], 1)}


def _aten_mutant(mode):
    def forward(a, output_size, scales_h=None, scales_w=None):
        return _aten_forward(a, output_size, scales_h, scales_w, mode=mode)

    def backward(grad_output, output_size, input_size, scales_h=None, scales_w=None):
        return _aten_backward(grad_output, output_size, input_size, scales_h, scales_w, mode=mode)
    return make_reference(ATEN_COVERS, extra={'aten.upsample_nearest2d': forward,
                                              'aten.upsample_nearest2d_backward': backward},
                          profiles=(ATEN_PROFILE,), argument_decoder=_tensor_argument)


MUTANTS.update({f'IN.M{i}': _aten_mutant(mode) for i, mode in enumerate(
    ['ignore_scale', 'multiply_scale', 'swap', 'overwrite_grad', 'nearest_exact'], 6)})
READINGS = {}
