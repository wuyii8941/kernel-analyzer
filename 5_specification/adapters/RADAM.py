"""Reviewed strict-real state transition for one public PyTorch 2.10 RAdam parameter."""
from __future__ import annotations

from decimal import Clamped, Overflow, Subnormal, Underflow
from fractions import Fraction

from mpmath import iv

from kernel_spec.compare import check_enclosed_output
from kernel_spec.e1_optim import flag, grad_or_none, hyper, state_dict
from kernel_spec.num import Invalid, NotEstablished, Result, Undefined, Unsupported, integer, to_iv, vector


PROFILE = "torch210_dense_real"
REQUIRED = {
    "state", "gradient", "lr", "beta1", "beta2", "eps", "weight_decay",
    "maximize", "decoupled_weight_decay",
}
OPTIONAL = {"foreach", "capturable", "differentiable"}
FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}
INTEGER_DTYPES = {"i8", "i16", "i32", "i64", *FLOAT_DTYPES}
MAX_EXACT_STEP = 10000


def _strategy(inp, key):
    value = inp.get(key)
    if value is not None and type(value) is not bool:
        raise Invalid(f"{key} must be true, false or null")
    return value


def _inputs(profile, inp):
    if profile != PROFILE:
        raise Unsupported(f"profile {profile!r}")
    if not isinstance(inp, dict) or any(not isinstance(key, str) for key in inp):
        raise Invalid("RAdam input must be a mapping with string field names")
    missing = REQUIRED - set(inp)
    if missing:
        raise Invalid(f"RAdam input is missing {sorted(missing)}")
    extra = set(inp) - REQUIRED - OPTIONAL
    if extra:
        raise Unsupported(f"RAdam fields outside the selected public task: {sorted(extra)}")

    maximize = flag(inp, "maximize")
    decoupled = flag(inp, "decoupled_weight_decay")
    _strategy(inp, "foreach")
    for key in ("capturable", "differentiable"):
        value = inp.get(key, False)
        if type(value) is not bool:
            raise Invalid(f"{key} must be a boolean")
        if value:
            raise Unsupported(f"{key}=True is outside the reviewed RAdam profile")

    st = state_dict(inp, {"theta", "m", "v", "t"})
    theta = vector(st["theta"], "theta")
    if not theta:
        raise Invalid("empty parameter")
    dimension = len(theta)
    m, v = vector(st["m"], "m"), vector(st["v"], "v")
    if st["t"] is None:
        if m or v:
            raise Invalid("an uninitialized RAdam state must use m=[], v=[] and t=null")
        step = None
    else:
        step = integer(st["t"], "t", lo=0)
        if len(m) != dimension or len(v) != dimension:
            raise Invalid("an initialized RAdam state needs complete m and v vectors")
        if any(value < 0 for value in v):
            raise Invalid("negative second moment")

    gradient = grad_or_none(inp, dimension)
    hp = hyper(inp, ("lr", "beta1", "beta2", "eps", "weight_decay"),
               nonneg=("lr", "eps", "weight_decay"))
    if not (0 <= hp["beta1"] < 1 and 0 <= hp["beta2"] < 1):
        raise Invalid("betas must be in [0, 1)")
    return theta, m, v, step, gradient, hp, maximize, decoupled


def _initialized(theta, m, v, step):
    if step is not None:
        return m, v, step
    zeros = [Fraction(0)] * len(theta)
    return list(zeros), list(zeros), 0


def _state_transition(theta, m, v, step, gradient, hp, maximize, decoupled):
    if gradient is None:
        return theta, m, v, step, None
    m, v, step0 = _initialized(theta, m, v, step)
    b1, b2, lr, wd = (hp[key] for key in ("beta1", "beta2", "lr", "weight_decay"))
    received = [(-value if maximize else value) for value in gradient]
    if decoupled:
        bases = [old * (1 - lr * wd) for old in theta]
        processed = received
    else:
        bases = list(theta)
        processed = [value + wd * old for old, value in zip(theta, received)]
    m1 = [b1 * old + (1 - b1) * value for old, value in zip(m, processed)]
    v1 = [b2 * old + (1 - b2) * value * value for old, value in zip(v, processed)]
    return bases, m1, v1, step0 + 1, processed


def _step_scalars(beta1, beta2, step):
    if step > MAX_EXACT_STEP:
        raise NotEstablished("exact RAdam step exponent exceeds the declared reference budget")
    correction1 = 1 - beta1 ** step
    correction2 = 1 - beta2 ** step
    rho_inf = 2 / (1 - beta2) - 1
    rho = rho_inf - 2 * step * beta2 ** step / correction2
    if rho > 4:
        rect2 = (rho - 4) * (rho - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho)
        if rect2 <= 0:
            raise NotEstablished("RAdam rectification factor is not positive in the selected branch")
    else:
        rect2 = None
    return correction1, correction2, rho, rect2


def reference(profile, inp):
    theta, m, v, step, gradient, hp, maximize, decoupled = _inputs(profile, inp)
    bases, m1, v1, step1, _ = _state_transition(theta, m, v, step, gradient, hp, maximize, decoupled)
    if gradient is None:
        return Result("value", {"theta": theta, "m": m, "v": v, "t": step})

    correction1, correction2, rho, rect2 = _step_scalars(hp["beta1"], hp["beta2"], step1)
    if rho > 5 and hp["eps"] == 0 and any(value == 0 for value in v1):
        raise Undefined("sqrt(second moment) + eps is zero in the rectified branch")
    try:
        theta1 = []
        for base, moment, second in zip(bases, m1, v1):
            first = to_iv(hp["lr"] * moment / correction1)
            if rho > 5:
                scale = iv.sqrt(to_iv(rect2)) * iv.sqrt(to_iv(correction2))
                update = first * scale / (iv.sqrt(to_iv(second)) + to_iv(hp["eps"]))
                theta1.append(to_iv(base) - update)
            else:
                theta1.append(base - hp["lr"] * moment / correction1)
    except (Overflow, Underflow, Subnormal, Clamped) as exc:
        raise NotEstablished(f"RAdam reference could not enclose the result: {type(exc).__name__}: {exc}") from exc
    return Result("value", {"theta": theta1, "m": m1, "v": v1, "t": step1})


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


def _rectified_coordinate_matches(base, new, moment, second, lr, eps, correction1, correction2, rect2):
    """Check d*(sqrt(v)+eps)=a*sqrt(q) exactly, including its signs."""
    delta = base - new
    first = lr * moment / correction1
    q = rect2 * correction2
    if delta == 0:
        return first == 0
    if first == 0 or delta * first <= 0:
        return False
    if eps == 0:
        return delta * delta * second == first * first * q
    # Squaring d*sqrt(v) = a*sqrt(q) - d*eps also requires both
    # sides to have d's sign; otherwise the conjugate root is accepted.
    if first * first * q < delta * delta * eps * eps:
        return False
    isolated = first * first * q + delta * delta * eps * eps - delta * delta * second
    return isolated > 0 and isolated * isolated == 4 * first * first * delta * delta * eps * eps * q


def check_output(profile, inp, observed):
    """Check every public state field and the selected RAdam update relation without a tolerance."""
    theta, m, v, step, gradient, hp, maximize, decoupled = _inputs(profile, inp)
    if not isinstance(observed, dict) or set(observed) != {"theta", "m", "v", "t"}:
        return False
    active = gradient is not None or step is not None
    try:
        xs = _observed_vector(observed["theta"], "theta", len(theta))
        ms = _observed_vector(observed["m"], "m", len(theta) if active else 0)
        vs = _observed_vector(observed["v"], "v", len(theta) if active else 0)
        observed_step = _observed_t(observed["t"])
    except Invalid:
        return False
    if gradient is None:
        return check_enclosed_output(
            {"theta": theta, "m": m, "v": v, "t": step},
            {"theta": xs, "m": ms, "v": vs, "t": observed_step},
        )

    bases, m1, v1, step1, _ = _state_transition(theta, m, v, step, gradient, hp, maximize, decoupled)
    if (ms, vs, observed_step) != (m1, v1, step1):
        return False
    correction1, correction2, rho, rect2 = _step_scalars(hp["beta1"], hp["beta2"], step1)
    if rho > 5 and hp["eps"] == 0 and any(value == 0 for value in v1):
        raise Undefined("sqrt(second moment) + eps is zero in the rectified branch")
    for base, new, moment, second in zip(bases, xs, m1, v1):
        if rho > 5:
            if not _rectified_coordinate_matches(
                    base, new, moment, second, hp["lr"], hp["eps"],
                    correction1, correction2, rect2):
                return False
        elif new != base - hp["lr"] * moment / correction1:
            return False
    return True


def _mutated(profile, inp, mode):
    theta, m, v, step, gradient, hp, maximize, decoupled = _inputs(profile, inp)
    if gradient is None:
        if mode != "none_as_zero":
            return Result("value", {"theta": theta, "m": m, "v": v, "t": step})
        gradient = [Fraction(0)] * len(theta)
    if mode == "decoupled_swapped":
        decoupled = not decoupled
    m, v, step0 = _initialized(theta, m, v, step)
    b1, b2, lr, eps, wd = (hp[key] for key in ("beta1", "beta2", "lr", "eps", "weight_decay"))
    step1 = step0 + 1
    correction_step = step0 if mode == "old_step" else step1
    if correction_step == 0:
        raise Undefined("zero correction step")
    correction1, correction2, rho, rect2 = _step_scalars(b1, b2, correction_step)
    threshold = 4 if mode == "threshold_four" else 5
    rectified = rho > threshold
    if mode == "always_rectified":
        rectified = True
        if rect2 is None:
            rect2 = Fraction(1)
    out_m, out_v, out_theta = [], [], []
    for index, (old, received) in enumerate(zip(theta, gradient)):
        if mode == "maximize_after_decay":
            processed = received + (0 if decoupled else wd * old)
            processed = -processed if maximize else processed
        else:
            processed = -received if maximize else received
            if not decoupled and mode != "decay_after_moments":
                processed += wd * old
        base = old * (1 - lr * wd) if decoupled else old
        moment = b1 * m[index] + (1 - b1) * processed
        second = b2 * v[index] + (1 - b2) * processed * processed
        update_moment = moment + wd * old if mode == "decay_after_moments" and not decoupled else moment
        first = update_moment if mode == "no_first_correction" else update_moment / correction1
        if rectified:
            if second == 0 and eps == 0:
                raise Undefined("zero denominator")
            den = iv.sqrt(to_iv(second + eps)) if mode == "eps_inside_sqrt" \
                else iv.sqrt(to_iv(second)) + to_iv(eps)
            beta_scale = R_ONE if mode == "no_second_correction" else iv.sqrt(to_iv(correction2))
            rect_scale = iv.sqrt(to_iv(rect2 if rect2 is not None else Fraction(1)))
            update = to_iv(lr * first) * rect_scale * beta_scale / den
        else:
            update = to_iv(lr * first)
        out_theta.append(to_iv(base) - update)
        out_m.append(moment)
        out_v.append(second)
    if mode == "state_not_stored":
        out_m, out_v, step1 = m, v, step0
    return Result("value", {"theta": out_theta, "m": out_m, "v": out_v, "t": step1})


R_ONE = to_iv(Fraction(1))


def _mutant(mode):
    return lambda profile, inp: _mutated(profile, inp, mode)


MUTANTS = {
    "RA.M1": _mutant("threshold_four"),
    "RA.M2": _mutant("always_rectified"),
    "RA.M3": _mutant("no_first_correction"),
    "RA.M4": _mutant("no_second_correction"),
    "RA.M5": _mutant("eps_inside_sqrt"),
    "RA.M6": _mutant("decoupled_swapped"),
    "RA.M7": _mutant("maximize_after_decay"),
    "RA.M8": _mutant("none_as_zero"),
    "RA.M9": _mutant("old_step"),
    "RA.M10": _mutant("state_not_stored"),
    "RA.M11": _mutant("decay_after_moments"),
}
READINGS = {}
