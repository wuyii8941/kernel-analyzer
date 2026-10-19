"""Independent finite-input real reference for row softmax and mean NLL.

Used as evidence for an explicitly selected task, not automatic kernel binding.
For each row m=max(x), l=log(sum(exp(x-m))). The mean loss is
sum(m+l-x[target]) / number of nonignored labels. All rows' m and l are returned.
Repeated exact input values are counted, so each distinct exponential is
evaluated once per row. Integer multiplicities preserve the real expression.
"""
from fractions import Fraction

import flint
import numpy as np


def _ball(x):
    value = Fraction(float(x))
    return flint.arb(value.numerator) / flint.arb(value.denominator)


def bounds(value):
    # Rational endpoint serialization, never a rounded decimal display string.
    lo, hi = value.lower().fmpq(), value.upper().fmpq()
    return (Fraction(int(lo.p), int(lo.q)), Fraction(int(hi.p), int(hi.q)))


def evaluate_region(graph, inputs, *, logits_input, labels_input, precision=256,
                    max_elements=10_000_000, max_unique=200_000):
    """Evaluate only after the entire selected graph matches the declared formula.

    The float32 conversion in the accepted graph must preserve every input
    value. Other cast profiles need their own explicit treatment.
    """
    from kernel_spec.num import Invalid, NotEstablished
    from softmax_task import require_region_formula
    if set(inputs) != {logits_input, labels_input}:
        raise Invalid("the selected region requires exactly its two declared inputs")
    x, labels = np.asarray(inputs[logits_input]), np.asarray(inputs[labels_input])
    if type(max_elements) is not int or max_elements <= 0:
        raise Invalid("max_elements must be a positive integer")
    if x.size > max_elements:
        raise NotEstablished("softmax input element budget exceeded")
    roles = require_region_formula(graph, logits_input=logits_input, labels_input=labels_input,
                                   logits_shape=x.shape)
    if labels.shape != (1, x.shape[0] + 1):
        raise Invalid("label input does not match the declared slice and row count")
    if x.dtype not in (np.dtype('float16'), np.dtype('float32'), np.dtype('float64')) or not np.isfinite(x).all():
        raise NotEstablished("selected task requires finite floating logits")
    with np.errstate(over="ignore", invalid="ignore"):
        if not np.array_equal(x.astype(np.float32).astype(np.float64), x):
            raise NotEstablished("float32 conversion is not exact for these inputs")
    result = evaluate(x, labels[0, 1:], ignore_index=-100, precision=precision,
                      max_elements=max_elements, max_unique=max_unique)
    result["output_nodes"] = roles
    return result


def evaluate(logits, labels, *, ignore_index=-100, precision=256, max_elements=10_000_000, max_unique=200_000):
    from kernel_spec.num import Invalid, NotEstablished, Undefined
    for name, value in (("precision", precision), ("max_elements", max_elements), ("max_unique", max_unique)):
        if type(value) is not int or value <= 0:
            raise Invalid(f"{name} must be a positive integer")
    x, y = np.asarray(logits), np.asarray(labels)
    if (x.ndim != 2 or not all(x.shape) or x.dtype not in (np.dtype('float16'), np.dtype('float32'), np.dtype('float64'))
            or y.shape != (x.shape[0],) or y.dtype.kind not in "iu" or type(ignore_index) is not int):
        raise Invalid("expected a nonempty floating matrix and one integer label per row")
    if x.size > max_elements:
        raise NotEstablished("softmax input element budget exceeded")
    if not np.isfinite(x).all():
        raise NotEstablished("this independent reference requires finite logits")
    valid = y != ignore_index
    if any(int(v) < 0 or int(v) >= x.shape[1] for v in y[valid]):
        raise Invalid("label is outside the class dimension")
    count = int(np.count_nonzero(valid))
    if count == 0:
        raise Undefined("mean loss has no nonignored labels")
    old = flint.ctx.prec
    flint.ctx.prec = precision
    try:
        maxima, logs, total, unique = [], [], flint.arb(0), 0
        for row, label, include in zip(x, y, valid):
            values, counts = np.unique(row, return_counts=True)
            unique += len(values)
            if unique > max_unique:
                raise NotEstablished("softmax distinct-exponential budget exceeded")
            maximum = _ball(values[-1])
            mass = flint.arb(0)
            for value, multiplicity in zip(values, counts):
                mass += (_ball(value) - maximum).exp() * int(multiplicity)
            log_mass = mass.log()
            if not log_mass.is_finite():
                raise NotEstablished("precision did not establish a finite log mass")
            maxima.append(bounds(maximum))
            logs.append(bounds(log_mass))
            if include:
                total += maximum + log_mass - _ball(row[int(label)])
        return {"maxima": maxima, "log_mass": logs, "mean_loss": bounds(total / count),
                "count": count, "distinct_exponentials": unique, "input_elements": int(x.size),
                "precision": precision}
    finally:
        flint.ctx.prec = old
