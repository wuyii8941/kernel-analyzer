"""Reviewed strict-real state transition for one public PyTorch 2.10 Adam parameter."""
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
    "maximize", "amsgrad", "decoupled_weight_decay",
}
OPTIONAL = {"foreach", "fused", "capturable", "differentiable"}
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
        raise Invalid("Adam input must be a mapping with string field names")
    missing = REQUIRED - set(inp)
    if missing:
        raise Invalid(f"Adam input is missing {sorted(missing)}")
    extra = set(inp) - REQUIRED - OPTIONAL
    if extra:
        raise Unsupported(f"Adam fields outside the selected public task: {sorted(extra)}")

    maximize, amsgrad = flag(inp, "maximize"), flag(inp, "amsgrad")
    if flag(inp, "decoupled_weight_decay"):
        raise Unsupported("decoupled_weight_decay=True is the AdamW task")
    foreach, fused = _strategy(inp, "foreach"), _strategy(inp, "fused")
    if foreach is True and fused is True:
        raise Invalid("foreach and fused cannot both be true")
    for key in ("capturable", "differentiable"):
        value = inp.get(key, False)
        if type(value) is not bool:
            raise Invalid(f"{key} must be a boolean")
        if value:
            raise Unsupported(f"{key}=True is outside the reviewed Adam profile")

    st = state_dict(inp, {"theta", "m", "v", "vmax", "t"})
    theta = vector(st["theta"], "theta")
    if not theta:
        raise Invalid("empty parameter")
    dimension = len(theta)
    m, v, vmax = (vector(st[name], name) for name in ("m", "v", "vmax"))
    if st["t"] is None:
        if m or v or vmax:
            raise Invalid("an uninitialized Adam state must use m=[], v=[], vmax=[] and t=null")
        t = None
    else:
        t = integer(st["t"], "t", lo=0)
        if len(m) != dimension or len(v) != dimension:
            raise Invalid("an initialized Adam state needs complete m and v vectors")
        expected_vmax = dimension if amsgrad else 0
        if len(vmax) != expected_vmax:
            raise Invalid("vmax must be complete exactly when amsgrad is enabled")
        if any(value < 0 for value in v + vmax):
            raise Invalid("negative second moment or AMSGrad maximum")

    gradient = grad_or_none(inp, dimension)
    hp = hyper(inp, ("lr", "beta1", "beta2", "eps", "weight_decay"),
               nonneg=("lr", "eps", "weight_decay"))
    if not (0 <= hp["beta1"] < 1 and 0 <= hp["beta2"] < 1):
        raise Invalid("betas must be in [0, 1)")
    return theta, m, v, vmax, t, gradient, hp, maximize, amsgrad


def _initialized(theta, m, v, vmax, t, amsgrad):
    if t is not None:
        return m, v, vmax, t
    zeros = [Fraction(0)] * len(theta)
    return list(zeros), list(zeros), list(zeros) if amsgrad else [], 0


def _state_transition(theta, m, v, vmax, t, gradient, hp, maximize, amsgrad):
    if gradient is None:
        return m, v, vmax, t, None
    m, v, vmax, t0 = _initialized(theta, m, v, vmax, t, amsgrad)
    b1, b2, wd = hp["beta1"], hp["beta2"], hp["weight_decay"]
    processed = [(-received if maximize else received) + wd * old
                 for old, received in zip(theta, gradient)]
    m1 = [b1 * old + (1 - b1) * g for old, g in zip(m, processed)]
    v1 = [b2 * old + (1 - b2) * g * g for old, g in zip(v, processed)]
    vmax1 = [max(old, new) for old, new in zip(vmax, v1)] if amsgrad else []
    return m1, v1, vmax1, t0 + 1, processed


def reference(profile, inp):
    theta, m, v, vmax, t, gradient, hp, maximize, amsgrad = _inputs(profile, inp)
    m1, v1, vmax1, t1, _ = _state_transition(theta, m, v, vmax, t, gradient, hp, maximize, amsgrad)
    if gradient is None:
        return Result("value", {"theta": theta, "m": m, "v": v, "vmax": vmax, "t": t})

    selected = vmax1 if amsgrad else v1
    if hp["eps"] == 0 and any(value == 0 for value in selected):
        raise Undefined("sqrt(corrected second moment) + eps is zero")
    try:
        bc1 = 1 - to_iv(hp["beta1"]) ** t1
        bc2 = 1 - to_iv(hp["beta2"]) ** t1
        theta1 = []
        for old, moment, second in zip(theta, m1, selected):
            mhat = to_iv(moment) / bc1
            vhat = to_iv(second) / bc2
            theta1.append(to_iv(old) - to_iv(hp["lr"]) * mhat / (iv.sqrt(vhat) + to_iv(hp["eps"])))
    except (Overflow, Underflow, Subnormal, Clamped) as exc:
        raise NotEstablished(f"Adam reference could not enclose the result: {type(exc).__name__}: {exc}") from exc
    return Result("value", {"theta": theta1, "m": m1, "v": v1, "vmax": vmax1, "t": t1})


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
    """Check all state fields; verify theta through an exact square-root relation."""
    theta, m, v, vmax, t, gradient, hp, maximize, amsgrad = _inputs(profile, inp)
    if not isinstance(observed, dict) or set(observed) != {"theta", "m", "v", "vmax", "t"}:
        return False
    active = gradient is not None or t is not None
    try:
        xs = _observed_vector(observed["theta"], "theta", len(theta))
        ms = _observed_vector(observed["m"], "m", len(theta) if active else 0)
        vs = _observed_vector(observed["v"], "v", len(theta) if active else 0)
        xms = _observed_vector(observed["vmax"], "vmax", len(theta) if active and amsgrad else 0)
        ts = _observed_t(observed["t"])
    except Invalid:
        return False
    if gradient is None:
        return check_enclosed_output(
            {"theta": theta, "m": m, "v": v, "vmax": vmax, "t": t},
            {"theta": xs, "m": ms, "v": vs, "vmax": xms, "t": ts},
        )

    m1, v1, vmax1, t1, _ = _state_transition(theta, m, v, vmax, t, gradient, hp, maximize, amsgrad)
    if (ms, vs, xms, ts) != (m1, v1, vmax1, t1):
        return False
    selected = vmax1 if amsgrad else v1
    if hp["eps"] == 0 and any(value == 0 for value in selected):
        raise Undefined("sqrt(corrected second moment) + eps is zero")
    if t1 > 10000:
        raise NotEstablished("exact candidate checking exceeded the declared step-exponent budget")
    bc1, bc2 = 1 - hp["beta1"] ** t1, 1 - hp["beta2"] ** t1
    for old, new, moment, second in zip(theta, xs, m1, selected):
        mhat, vhat = moment / bc1, second / bc2
        delta = old - new
        if delta == 0:
            if hp["lr"] * mhat != 0:
                return False
        else:
            root = hp["lr"] * mhat / delta - hp["eps"]
            if root < 0 or root * root != vhat:
                return False
    return True


def _mutated(profile, inp, mode):
    theta, m, v, vmax, t, gradient, hp, maximize, amsgrad = _inputs(profile, inp)
    if gradient is None:
        if mode != "none_as_zero":
            return Result("value", {"theta": theta, "m": m, "v": v, "vmax": vmax, "t": t})
        gradient = [Fraction(0)] * len(theta)
    m, v, vmax, t0 = _initialized(theta, m, v, vmax, t, amsgrad)
    b1, b2, lr, eps, wd = (hp[key] for key in ("beta1", "beta2", "lr", "eps", "weight_decay"))
    t1 = t0 + 1
    out, m1, v1, vmax1 = [], [], [], []
    for index, (old, received) in enumerate(zip(theta, gradient)):
        if mode == "maximize_after_decay":
            g = received + wd * old
            g = -g if maximize else g
        elif mode == "maximize_at_update":
            g = received + wd * old
        elif mode == "decoupled_decay":
            g = -received if maximize else received
        else:
            g = (-received if maximize else received) + wd * old
        mn = b1 * m[index] + (1 - b1) * g
        vn = b2 * v[index] + (1 - b2) * g * g
        correction_step = t0 if mode == "old_step_correction" else t1
        bc1, bc2 = 1 - b1 ** correction_step, 1 - b2 ** correction_step
        if mode == "no_bias_correction":
            bc1 = bc2 = Fraction(1)
        xn = max(vmax[index], vn) if amsgrad else None
        if amsgrad and mode == "amsgrad_corrected_max":
            xn = max(vmax[index], vn / bc2)
            selected = xn
        elif amsgrad and mode != "amsgrad_ignored":
            selected = xn / bc2
        else:
            selected = vn / bc2
        if eps == 0 and selected == 0:
            raise Undefined("zero denominator")
        den = iv.sqrt(to_iv(selected + eps)) if mode == "eps_inside_sqrt" \
            else iv.sqrt(to_iv(selected)) + to_iv(eps)
        step = to_iv(lr * (mn / bc1)) / den
        base = to_iv(old * (1 - lr * wd)) if mode == "decoupled_decay" else to_iv(old)
        new = base + step if mode == "maximize_at_update" and maximize else base - step
        out.append(new)
        m1.append(mn)
        v1.append(vn)
        if amsgrad:
            vmax1.append(vmax[index] if mode == "vmax_not_stored" else xn)

    if mode == "state_not_stored":
        m1, v1, vmax1, t1 = m, v, vmax, t0
    if mode == "fictitious_vmax" and not amsgrad:
        vmax1 = [Fraction(0)] * len(theta)
    return Result("value", {"theta": out, "m": m1, "v": v1, "vmax": vmax1, "t": t1})


def _mutant(mode):
    return lambda profile, inp: _mutated(profile, inp, mode)


MUTANTS = {
    "AD.M1": _mutant("no_bias_correction"),
    "AD.M2": _mutant("eps_inside_sqrt"),
    "AD.M3": _mutant("amsgrad_ignored"),
    "AD.M4": _mutant("amsgrad_corrected_max"),
    "AD.M5": _mutant("vmax_not_stored"),
    "AD.M6": _mutant("maximize_after_decay"),
    "AD.M7": _mutant("maximize_at_update"),
    "AD.M8": _mutant("decoupled_decay"),
    "AD.M9": _mutant("none_as_zero"),
    "AD.M10": _mutant("old_step_correction"),
    "AD.M11": _mutant("state_not_stored"),
    "AD.M12": _mutant("fictitious_vmax"),
}
READINGS = {}
