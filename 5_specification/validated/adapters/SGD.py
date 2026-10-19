"""Reviewed exact-real state transition for one public PyTorch 2.10 SGD parameter."""
from __future__ import annotations

from fractions import Fraction

from kernel_spec.compare import check_enclosed_output
from kernel_spec.e1_optim import flag, grad_or_none, hyper, state_dict
from kernel_spec.num import Invalid, Result, Unsupported, vector


PROFILE = "torch210"
REQUIRED = {
    "state", "gradient", "lr", "momentum", "dampening", "weight_decay", "nesterov", "maximize",
}
OPTIONAL = {"foreach", "fused", "differentiable"}
FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}


def _strategy(inp, key):
    value = inp.get(key)
    if value is not None and type(value) is not bool:
        raise Invalid(f"{key} must be true, false or null")
    return value


def _inputs(profile, inp, *, enforce_nesterov=True):
    if profile != PROFILE:
        raise Unsupported(f"profile {profile!r}")
    if not isinstance(inp, dict) or any(not isinstance(key, str) for key in inp):
        raise Invalid("SGD input must be a mapping with string field names")
    missing = REQUIRED - set(inp)
    if missing:
        raise Invalid(f"SGD input is missing {sorted(missing)}")
    extra = set(inp) - REQUIRED - OPTIONAL
    if extra:
        raise Unsupported(f"SGD fields outside the selected public task: {sorted(extra)}")

    maximize, nesterov = flag(inp, "maximize"), flag(inp, "nesterov")
    foreach, fused = _strategy(inp, "foreach"), _strategy(inp, "fused")
    differentiable = inp.get("differentiable", False)
    if type(differentiable) is not bool:
        raise Invalid("differentiable must be a boolean")
    if differentiable:
        raise Unsupported("differentiable=True is outside the reviewed SGD profile")
    if foreach is True and fused is True:
        raise Invalid("foreach and fused cannot both be true")

    st = state_dict(inp, {"theta", "b"})
    theta = vector(st["theta"], "theta")
    if not theta:
        raise Invalid("empty parameter")
    b = vector(st["b"], "b")
    if len(b) not in (0, len(theta)):
        raise Invalid("momentum buffer must be [] or have the parameter length")
    gradient = grad_or_none(inp, len(theta))
    hp = hyper(inp, ("lr", "momentum", "dampening", "weight_decay"),
               nonneg=("lr", "momentum", "weight_decay"))
    if enforce_nesterov and nesterov and (hp["momentum"] <= 0 or hp["dampening"] != 0):
        raise Invalid("nesterov requires momentum > 0 and dampening = 0")
    return theta, b, gradient, hp, maximize, nesterov


def _correct_fields(theta, b, gradient, hp, maximize, nesterov):
    if gradient is None:
        return {"theta": theta, "b": b}
    lr, mu, tau, wd = (hp[key] for key in ("lr", "momentum", "dampening", "weight_decay"))
    out, b_new = [], []
    for index, (old, received) in enumerate(zip(theta, gradient)):
        processed = (-received if maximize else received) + wd * old
        if mu != 0:
            new_buffer = processed if not b else mu * b[index] + (1 - tau) * processed
            b_new.append(new_buffer)
            direction = processed + mu * new_buffer if nesterov else new_buffer
        else:
            direction = processed
        out.append(old - lr * direction)
    return {"theta": out, "b": b_new if mu != 0 else b}


def reference(profile, inp):
    return Result("value", _correct_fields(*_inputs(profile, inp)))


def _observed_vector(value, name, length):
    if isinstance(value, dict):
        if set(value) != {"t", "v", "shape"}:
            raise Invalid(f"observed {name} tensor record must contain exactly t, v and shape")
        if value["t"] not in FLOAT_DTYPES:
            raise Invalid(f"observed {name} has an unsupported dtype")
        if value["shape"] != [length]:
            raise Invalid(f"observed {name} has the wrong shape")
        value = value["v"]
    return vector(value, f"observed {name}", length)


def check_output(profile, inp, observed):
    expected = reference(profile, inp).fields
    if not isinstance(observed, dict) or set(observed) != {"theta", "b"}:
        return False
    try:
        theta = _observed_vector(observed["theta"], "theta", len(expected["theta"]))
        b = _observed_vector(observed["b"], "b", len(expected["b"]))
    except Invalid:
        return False
    return check_enclosed_output(expected, {"theta": theta, "b": b})


# Each mutant implements one explicitly listed alternative reading without calling the reference.
def _mutated(profile, inp, mode):
    theta, b, gradient, hp, maximize, nesterov = _inputs(
        profile, inp, enforce_nesterov=mode != "nesterov_unchecked")
    lr, mu, tau, wd = (hp[key] for key in ("lr", "momentum", "dampening", "weight_decay"))
    if gradient is None:
        if mode != "none_as_zero":
            return Result("value", {"theta": theta, "b": b})
        gradient = [Fraction(0)] * len(theta)

    out, b_new = [], []
    for index, (old, received) in enumerate(zip(theta, gradient)):
        if mode == "maximize_after_decay":
            processed = received + wd * old
            processed = -processed if maximize else processed
        elif mode in {"maximize_at_update", "decay_after_momentum"}:
            processed = -received if (maximize and mode == "decay_after_momentum") else received
        else:
            processed = (-received if maximize else received) + wd * old

        if mu != 0:
            if mode == "lr_in_buffer":
                new_buffer = (mu * b[index] if b else 0) + lr * (1 - tau) * processed
                out.append(old - new_buffer)
                b_new.append(new_buffer)
                continue
            if not b:
                new_buffer = (1 - tau) * processed if mode == "dampening_first_step" else processed
            else:
                applied_tau = Fraction(0) if mode == "ignore_warm_dampening" else tau
                new_buffer = mu * b[index] + (1 - applied_tau) * processed
            b_new.append(new_buffer)
            if nesterov and mode != "nesterov_ignored":
                direction = processed + mu * new_buffer
            else:
                direction = new_buffer
        else:
            direction = processed

        if mode == "decay_after_momentum":
            direction += wd * old
        if mode == "maximize_at_update" and maximize:
            out.append(old + lr * direction)
        else:
            out.append(old - lr * direction)

    if mode == "buffer_not_stored":
        b_new = b
    elif mu == 0:
        b_new = [] if mode == "discard_buffer_no_momentum" else b
    return Result("value", {"theta": out, "b": b_new})


def _mutant(mode):
    return lambda profile, inp: _mutated(profile, inp, mode)


MUTANTS = {
    "SG.M1": _mutant("maximize_after_decay"),
    "SG.M2": _mutant("dampening_first_step"),
    "SG.M3": _mutant("nesterov_ignored"),
    "SG.M4": _mutant("maximize_at_update"),
    "SG.M5": _mutant("none_as_zero"),
    "SG.M6": _mutant("nesterov_unchecked"),
    "SG.M7": _mutant("buffer_not_stored"),
    "SG.M8": _mutant("lr_in_buffer"),
    "SG.M9": _mutant("ignore_warm_dampening"),
    "SG.M10": _mutant("decay_after_momentum"),
    "SG.M11": _mutant("discard_buffer_no_momentum"),
}
READINGS = {}
