"""Exact piecewise SmoothL1 / Huber tasks, preserving complete two-input VJPs."""
from fractions import Fraction

from . import aten as A
from .real import R
from .num import Invalid, Unsupported


def inputs(a, target, reduction, threshold, smooth):
    a, target = A.as_ten(a), A.as_ten(target)
    if a.shape != target.shape or a.dtype != target.dtype:
        raise Invalid('loss inputs must have identical shape and dtype')
    if not a.a.size:
        raise Unsupported('empty losses are outside the reviewed task')
    if reduction not in ('none', 'sum', 'mean'):
        raise Invalid('reduction must be none, sum or mean')
    b = R(threshold)
    if not b.finite or not b.exact:
        raise Unsupported('loss threshold must be finite and exact')
    if b.v < 0 or (not smooth and b.v == 0):
        raise Invalid('SmoothL1 beta must be nonnegative; Huber delta must be positive')
    return a, target, b


def elementwise(a, target, reduction, threshold, smooth):
    a, target, b = inputs(a, target, reduction, threshold, smooth)
    d = A.sub(a, target)
    z = A.abs_(d)
    if smooth and b.v == 0:
        return z, A.sign(d)
    quadratic = A.mul(A.mul(d, d), Fraction(1, 2))
    linear = A.sub(z, b / 2)
    if smooth:
        value = A.where(A.lt(z, b), A.div(quadratic, b), linear)
        derivative = A.where(A.lt(z, b), A.div(d, b), A.sign(d))
    else:
        value = A.where(A.lt(z, b), quadratic, A.mul(linear, b))
        derivative = A.where(A.lt(z, b), d, A.mul(A.sign(d), b))
    return value, derivative


def forward(a, target, reduction, threshold, smooth):
    value, _ = elementwise(a, target, reduction, threshold, smooth)
    return value if reduction == 'none' else A.sum_(value) if reduction == 'sum' else A.mean(value)


def backward(grad, a, target, reduction, threshold, smooth):
    a, target, _ = inputs(a, target, reduction, threshold, smooth)
    grad = A.as_ten(grad)
    expected = a.shape if reduction == 'none' else ()
    if grad.shape != expected or grad.dtype != a.dtype:
        raise Invalid('upstream must have the full forward output shape and input dtype')
    _, derivative = elementwise(a, target, reduction, threshold, smooth)
    dx = A.mul(grad, derivative)
    if reduction == 'mean':
        dx = A.mul(dx, Fraction(1, a.a.size))
    return [dx, A.neg(dx)]


def smooth_l1_loss(a, target, reduction='mean', beta=1.):
    return forward(a, target, reduction, beta, True)


def vjp_smooth_l1_loss(grad, a, target, reduction='mean', beta=1.):
    return backward(grad, a, target, reduction, beta, True)


def huber_loss(a, target, reduction='mean', delta=1.):
    return forward(a, target, reduction, delta, False)


def vjp_huber_loss(grad, a, target, reduction='mean', delta=1.):
    return backward(grad, a, target, reduction, delta, False)
