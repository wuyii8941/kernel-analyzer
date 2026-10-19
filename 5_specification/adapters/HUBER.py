"""Physical float32, unweighted public SmoothL1/Huber tasks required by external calls."""
from fractions import Fraction
import math
import yaml

from kernel_spec import aten as A, ieee, regression_loss as L
from kernel_spec.aten_adapter import decode, make_reference, make_output_checker
from kernel_spec.num import Invalid, Unsupported
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / 'HUBER.yaml').read_text())['covers']
PROFILE = ('torch210_unweighted_f32',)
OPS = {'aten.smooth_l1_loss': L.smooth_l1_loss, 'aten.huber_loss': L.huber_loss,
       'vjp.aten.smooth_l1_loss': L.vjp_smooth_l1_loss, 'vjp.aten.huber_loss': L.vjp_huber_loss}


def argument(op, name, value, raw):
    if name in ('a', 'target', 'grad'):
        if not isinstance(value, dict) or 't' not in value or 'v' not in value:
            raise Invalid('loss data must be physical tensor records')
        if A.dt(value['t']) != 'f32':
            raise Unsupported('reviewed loss task requires float32')
        tensor = decode(value)
        for element in tensor.a.flat:
            if not isinstance(element, R):
                raise Invalid('loss data must be real')
            if not element.finite or not element.exact:
                raise Unsupported('loss data must be finite point values')
            rounded = ieee.round_to(element, 'f32')
            if not rounded.exact or rounded.v != element.v:
                raise Invalid('loss input is not exactly representable as float32')
        return tensor
    if name in ('beta', 'delta'):
        if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
            raise Invalid('loss threshold must be a finite numeric scalar')
        if isinstance(value, float) and math.isnan(value):
            raise Invalid('NaN threshold is not a valid task parameter')
        b = R(value)
        if not b.finite:
            raise Unsupported('infinite threshold is outside the reviewed task')
        rounded = ieee.round_to(b, 'f64')
        if not rounded.exact or rounded.v != b.v:
            raise Invalid('threshold must equal its received Python binary64 value')
        return b
    return decode(value)


reference = make_reference(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=argument)
check_output = make_output_checker(COVERS, extra=OPS, profiles=PROFILE, argument_decoder=argument)


def wrong_scale(a, target, reduction='mean', beta=1.):
    return L.huber_loss(a, target, reduction, beta)


def wrong_mean(a, target, reduction='mean', delta=1.):
    return L.huber_loss(a, target, 'sum', delta)


def wrong_target(grad, a, target, reduction='mean', delta=1.):
    dx, _ = L.vjp_huber_loss(grad, a, target, reduction, delta)
    return [dx, dx]


def missing_target(grad, a, target, reduction='mean', delta=1.):
    return L.vjp_huber_loss(grad, a, target, reduction, delta)[:1]


def wrong_zero(grad, a, target, reduction='mean', beta=1.):
    if R(beta).v == 0:
        d = A.sub(a, target)
        direction = A.where(A.ge(d, 0), 1, -1)
        dx = A.mul(grad, direction)
        if reduction == 'mean':
            dx = A.mul(dx, Fraction(1, a.a.size))
        return [dx, A.neg(dx)]
    return L.vjp_smooth_l1_loss(grad, a, target, reduction, beta)


def wrong_upstream(grad, a, target, reduction='mean', delta=1.):
    return L.vjp_huber_loss(A.full_like(grad, 1), a, target, reduction, delta)


def mutant(op, fn):
    return make_reference(COVERS, extra={**OPS, op: fn}, profiles=PROFILE, argument_decoder=argument)


MUTANTS = {
    'HB.M1': mutant('aten.smooth_l1_loss', wrong_scale),
    'HB.M2': mutant('aten.huber_loss', wrong_mean),
    'HB.M3': mutant('vjp.aten.huber_loss', wrong_target),
    'HB.M4': mutant('vjp.aten.huber_loss', missing_target),
    'HB.M5': mutant('vjp.aten.smooth_l1_loss', wrong_zero),
    'HB.M6': mutant('vjp.aten.huber_loss', wrong_upstream),
}
READINGS = {}
