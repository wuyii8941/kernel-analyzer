"""Reviewed strict-real state transition for one public PyTorch 2.10 RMSprop parameter."""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from kernel_spec.compare import check_enclosed_output
from kernel_spec.e1_optim import flag, grad_or_none, hyper, state_dict
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer, to_iv, vector


PROFILE = "torch210_dense_real"
REQUIRED = {
    "state", "gradient", "lr", "alpha", "eps", "weight_decay", "momentum", "centered", "maximize",
}
OPTIONAL = {"foreach", "capturable", "differentiable"}
FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}
INTEGER_DTYPES = {"i8", "i16", "i32", "i64", *FLOAT_DTYPES}


def _strategy(inp, key):
    value = inp.get(key)
    if value is not None and type(value) is not bool:
        raise Invalid(f"{key} must be true, false or null")
    return value


def _inputs(profile, inp):
    if profile != PROFILE:
        raise Unsupported(f"profile {profile!r}")
    if not isinstance(inp, dict) or any(not isinstance(key, str) for key in inp):
        raise Invalid("RMSprop input must be a mapping with string field names")
    missing = REQUIRED - set(inp)
    if missing:
        raise Invalid(f"RMSprop input is missing {sorted(missing)}")
    extra = set(inp) - REQUIRED - OPTIONAL
    if extra:
        raise Unsupported(f"RMSprop fields outside the selected public task: {sorted(extra)}")

    centered, maximize = flag(inp, "centered"), flag(inp, "maximize")
    _strategy(inp, "foreach")
    for key in ("capturable", "differentiable"):
        value = inp.get(key, False)
        if type(value) is not bool:
            raise Invalid(f"{key} must be a boolean")
        if value:
            raise Unsupported(f"{key}=True is outside the reviewed RMSprop profile")
    hp = hyper(inp, ("lr", "alpha", "eps", "weight_decay", "momentum"),
               nonneg=("lr", "alpha", "eps", "weight_decay", "momentum"))

    st = state_dict(inp, {"theta", "v", "gave", "b", "t"})
    theta = vector(st["theta"], "theta")
    if not theta:
        raise Invalid("empty parameter")
    dimension = len(theta)
    v, gave, buffer = (vector(st[name], name) for name in ("v", "gave", "b"))
    if st["t"] is None:
        if v or gave or buffer:
            raise Invalid("an uninitialized RMSprop state must use v=[], gave=[], b=[] and t=null")
        step = None
    else:
        step = integer(st["t"], "t", lo=0)
        if len(v) != dimension:
            raise Invalid("an initialized RMSprop state needs a complete square-average vector")
        if len(gave) != (dimension if centered else 0):
            raise Invalid("gave must be complete exactly when centered is enabled")
        if len(buffer) != (dimension if hp["momentum"] > 0 else 0):
            raise Invalid("b must be complete exactly when momentum is positive")
        if any(value < 0 for value in v):
            raise Invalid("negative square average")
    gradient = grad_or_none(inp, dimension)
    return theta, v, gave, buffer, step, gradient, hp, centered, maximize


def _initialized(theta, v, gave, buffer, step, centered, momentum):
    if step is not None:
        return v, gave, buffer, step
    zeros = [Fraction(0)] * len(theta)
    return list(zeros), list(zeros) if centered else [], list(zeros) if momentum > 0 else [], 0


def _transition(theta, v, gave, buffer, step, gradient, hp, centered, maximize):
    if gradient is None:
        return v, gave, buffer, step, None, None
    alpha, wd, momentum = hp["alpha"], hp["weight_decay"], hp["momentum"]
    v, gave, buffer, step0 = _initialized(theta, v, gave, buffer, step, centered, momentum)
    processed = [(-received if maximize else received) + wd * old
                 for old, received in zip(theta, gradient)]
    v1 = [alpha * old + (1 - alpha) * g * g for old, g in zip(v, processed)]
    gave1 = [alpha * old + (1 - alpha) * g for old, g in zip(gave, processed)] if centered else []
    adjusted = [new - avg * avg for new, avg in zip(v1, gave1)] if centered else list(v1)
    if any(value < 0 for value in adjusted):
        raise Undefined("negative centered square estimate")
    if hp["eps"] == 0 and any(value == 0 for value in adjusted):
        raise Undefined("sqrt(adjusted square average) + eps is zero")
    return v1, gave1, buffer, step0 + 1, processed, adjusted


def reference(profile, inp):
    theta, v, gave, buffer, step, gradient, hp, centered, maximize = _inputs(profile, inp)
    if gradient is None:
        return Result("value", {"theta": theta, "v": v, "gave": gave, "b": buffer, "t": step})
    v1, gave1, buffer0, step1, processed, adjusted = _transition(
        theta, v, gave, buffer, step, gradient, hp, centered, maximize)
    buffer1, theta1 = [], []
    for index, (old, g, square) in enumerate(zip(theta, processed, adjusted)):
        denominator = iv.sqrt(to_iv(square)) + to_iv(hp["eps"])
        quotient = to_iv(g) / denominator
        if hp["momentum"] > 0:
            new_buffer = to_iv(hp["momentum"] * buffer0[index]) + quotient
            update = to_iv(hp["lr"]) * new_buffer
            buffer1.append(new_buffer)
        else:
            update = to_iv(hp["lr"]) * quotient
        theta1.append(to_iv(old) - update)
    return Result("value", {"theta": theta1, "v": v1, "gave": gave1, "b": buffer1, "t": step1})


def _observed_vector(value, name, length):
    if isinstance(value, dict):
        if set(value) != {"t", "v", "shape"} or value.get("t") not in FLOAT_DTYPES \
                or value.get("shape") != [length]:
            raise Invalid(f"observed {name} tensor record has the wrong dtype or shape")
        value = value["v"]
    return vector(value, f"observed {name}", length)


def _observed_t(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if set(value) != {"t", "v", "shape"} or value.get("t") not in INTEGER_DTYPES \
                or value.get("shape") != []:
            raise Invalid("observed t tensor record has the wrong dtype or shape")
        value = value["v"]
    return integer(value, "observed t", lo=0)


def check_output(profile, inp, observed):
    """Check every state field and the two square-root update relations without a tolerance."""
    theta, v, gave, buffer, step, gradient, hp, centered, maximize = _inputs(profile, inp)
    if not isinstance(observed, dict) or set(observed) != {"theta", "v", "gave", "b", "t"}:
        return False
    active = gradient is not None or step is not None
    try:
        xs = _observed_vector(observed["theta"], "theta", len(theta))
        vs = _observed_vector(observed["v"], "v", len(theta) if active else 0)
        avgs = _observed_vector(observed["gave"], "gave", len(theta) if active and centered else 0)
        buffers = _observed_vector(
            observed["b"], "b", len(theta) if active and hp["momentum"] > 0 else 0)
        observed_step = _observed_t(observed["t"])
    except Invalid:
        return False
    if gradient is None:
        return check_enclosed_output(
            {"theta": theta, "v": v, "gave": gave, "b": buffer, "t": step},
            {"theta": xs, "v": vs, "gave": avgs, "b": buffers, "t": observed_step},
        )

    v1, gave1, buffer0, step1, processed, adjusted = _transition(
        theta, v, gave, buffer, step, gradient, hp, centered, maximize)
    if (vs, avgs, observed_step) != (v1, gave1, step1):
        return False
    if hp["momentum"] > 0:
        for old_theta, new_theta, old_buffer, new_buffer, g, square in zip(
                theta, xs, buffer0, buffers, processed, adjusted):
            if new_theta != old_theta - hp["lr"] * new_buffer:
                return False
            quotient = new_buffer - hp["momentum"] * old_buffer
            if quotient == 0:
                if g != 0:
                    return False
            else:
                root = g / quotient - hp["eps"]
                if root < 0 or root * root != square:
                    return False
    else:
        if buffers:
            return False
        for old_theta, new_theta, g, square in zip(theta, xs, processed, adjusted):
            delta = old_theta - new_theta
            if delta == 0:
                if hp["lr"] * g != 0:
                    return False
            else:
                root = hp["lr"] * g / delta - hp["eps"]
                if root < 0 or root * root != square:
                    return False
    return True


def _mutated(profile, inp, mode):
    theta, v, gave, buffer, step, gradient, hp, centered, maximize = _inputs(profile, inp)
    if gradient is None:
        if mode != "none_as_zero":
            return Result("value", {"theta": theta, "v": v, "gave": gave, "b": buffer, "t": step})
        gradient = [Fraction(0)] * len(theta)
    alpha, eps, wd, momentum, lr = (hp[key] for key in ("alpha", "eps", "weight_decay", "momentum", "lr"))
    v, gave, buffer, step0 = _initialized(theta, v, gave, buffer, step, centered, momentum)
    theta1, v1, gave1, buffer1 = [], [], [], []
    for index, (old, received) in enumerate(zip(theta, gradient)):
        if mode == "maximize_after_decay":
            g = received + wd * old
            g = -g if maximize else g
        elif mode == "decay_after_statistics":
            g = -received if maximize else received
        else:
            g = (-received if maximize else received) + wd * old
        new_v = alpha * v[index] + (1 - alpha) * g * g
        new_avg = alpha * gave[index] + (1 - alpha) * g if centered else None
        if centered and mode == "centered_old_gave":
            square = new_v - gave[index] * gave[index]
        elif centered and mode == "centered_ignored":
            square = new_v
        elif centered and mode == "centered_no_square":
            square = new_v - new_avg
        elif centered:
            square = new_v - new_avg * new_avg
        else:
            square = new_v
        if square < 0 or (square == 0 and eps == 0):
            raise Undefined("denominator undefined")
        den = iv.sqrt(to_iv(square + eps)) if mode == "eps_inside_sqrt" \
            else iv.sqrt(to_iv(square)) + to_iv(eps)
        update_g = g + wd * old if mode == "decay_after_statistics" else g
        quotient = to_iv(update_g) / den
        if momentum > 0:
            new_buffer = to_iv(momentum * buffer[index]) + (to_iv(lr) * quotient if mode == "lr_in_buffer" else quotient)
            update = new_buffer if mode == "lr_in_buffer" else to_iv(lr) * new_buffer
            buffer1.append(new_buffer)
        else:
            update = to_iv(lr) * quotient
        theta1.append(to_iv(old) - update)
        v1.append(new_v)
        if centered:
            gave1.append(new_avg)
    step1 = step0 + 1
    if mode == "step_not_counted":
        step1 = step0
    if mode == "state_not_stored":
        v1, gave1, buffer1, step1 = v, gave, buffer, step0
    if mode == "fictitious_gave" and not centered:
        gave1 = [Fraction(0)] * len(theta)
    if mode == "fictitious_buffer" and momentum == 0:
        buffer1 = [Fraction(0)] * len(theta)
    return Result("value", {"theta": theta1, "v": v1, "gave": gave1, "b": buffer1, "t": step1})


def _mutant(mode):
    return lambda profile, inp: _mutated(profile, inp, mode)


MUTANTS = {
    "RM.M1": _mutant("eps_inside_sqrt"),
    "RM.M2": _mutant("centered_old_gave"),
    "RM.M3": _mutant("centered_ignored"),
    "RM.M4": _mutant("maximize_after_decay"),
    "RM.M5": _mutant("lr_in_buffer"),
    "RM.M6": _mutant("none_as_zero"),
    "RM.M7": _mutant("centered_no_square"),
    "RM.M8": _mutant("step_not_counted"),
    "RM.M9": _mutant("fictitious_gave"),
    "RM.M10": _mutant("fictitious_buffer"),
    "RM.M11": _mutant("state_not_stored"),
    "RM.M12": _mutant("decay_after_statistics"),
}
READINGS = {}
