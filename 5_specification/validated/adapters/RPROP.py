"""Reviewed exact state transition for one public PyTorch 2.10 Rprop parameter."""
from __future__ import annotations

from fractions import Fraction

from kernel_spec.compare import check_enclosed_output
from kernel_spec.e1_optim import flag, state_dict
from kernel_spec.num import Invalid, Result, Unsupported, integer, rational


PROFILE = "torch210_dense_real"
REQUIRED = {
    "state", "gradient", "dtype", "lr", "etaminus", "etaplus", "step_min", "step_max", "maximize",
}
OPTIONAL = {"foreach", "capturable", "differentiable"}
FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}
STEP_DTYPES = {"i8", "i16", "i32", "i64", *FLOAT_DTYPES}


def _shape(value, what):
    if not isinstance(value, list):
        return ()
    if not value:
        raise Invalid(f"{what}: empty dimension")
    children = [_shape(item, what) for item in value]
    if any(child != children[0] for child in children):
        raise Invalid(f"{what}: ragged tensor")
    return (len(value),) + children[0]


def _map(function, value):
    return [_map(function, item) for item in value] if isinstance(value, list) else function(value)


def _read(value, shape, what):
    actual = _shape(value, what)
    if actual != tuple(shape):
        raise Invalid(f"{what}: shape {actual}, expected {tuple(shape)}")
    return _map(lambda item: rational(item, what), value)


def _flat(value):
    return [item for child in value for item in _flat(child)] if isinstance(value, list) else [value]


def _rebuild(values, shape):
    iterator = iter(values)

    def build(sizes):
        if len(sizes) == 1:
            return [next(iterator) for _ in range(sizes[0])]
        return [build(sizes[1:]) for _ in range(sizes[0])]

    return build(tuple(shape))


def _zeros(shape):
    return _rebuild([Fraction(0)] * _size(shape), shape)


def _full(shape, value):
    return _rebuild([value] * _size(shape), shape)


def _size(shape):
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _strategy(inp):
    value = inp.get("foreach")
    if value is not None and type(value) is not bool:
        raise Invalid("foreach must be true, false or null")
    return value


def _inputs(profile, inp):
    if profile != PROFILE:
        raise Unsupported(f"profile {profile!r}")
    if not isinstance(inp, dict) or any(not isinstance(key, str) for key in inp):
        raise Invalid("Rprop input must be a mapping with string field names")
    missing = REQUIRED - set(inp)
    if missing:
        raise Invalid(f"Rprop input is missing {sorted(missing)}")
    extra = set(inp) - REQUIRED - OPTIONAL
    if extra:
        raise Unsupported(f"Rprop fields outside the selected public task: {sorted(extra)}")
    _strategy(inp)
    maximize = flag(inp, "maximize")
    for option in ("capturable", "differentiable"):
        value = inp.get(option, False)
        if type(value) is not bool:
            raise Invalid(f"{option} must be a boolean")
        if value:
            raise Unsupported(f"{option}=True is outside the reviewed Rprop profile")

    dtype = inp["dtype"]
    if dtype not in FLOAT_DTYPES:
        raise Unsupported("the reviewed Rprop profile requires dtype f16, bf16, f32 or f64")
    lr = rational(inp["lr"], "lr")
    etaminus = rational(inp["etaminus"], "etaminus")
    etaplus = rational(inp["etaplus"], "etaplus")
    step_min = rational(inp["step_min"], "step_min")
    step_max = rational(inp["step_max"], "step_max")
    if lr < 0:
        raise Invalid("lr must be >= 0")
    if not (0 < etaminus < 1 < etaplus):
        raise Invalid("etas must satisfy 0 < etaminus < 1 < etaplus")
    if not (0 <= step_min <= step_max):
        raise Unsupported("the reviewed Rprop profile requires 0 <= step_min <= step_max")

    state = state_dict(inp, {"theta", "prev", "eta", "t"})
    shape = _shape(state["theta"], "theta")
    if not shape:
        raise Invalid("a scalar parameter is outside the reviewed profile")
    theta = _read(state["theta"], shape, "theta")
    if state["t"] is None:
        if state["prev"] != [] or state["eta"] != []:
            raise Invalid("an uninitialized Rprop state must use prev=eta=[] and t=null")
        prev, eta, step = [], [], None
    else:
        step = integer(state["t"], "t", lo=0)
        prev = _read(state["prev"], shape, "prev")
        eta = _read(state["eta"], shape, "eta")
        if any(value < 0 for value in _flat(eta)):
            raise Invalid("Rprop step_size state must be nonnegative")
    gradient_value = inp.get("gradient", "missing")
    if gradient_value == "missing":
        raise Invalid("missing gradient (use null for a parameter without gradient)")
    gradient = None if gradient_value is None else _read(gradient_value, shape, "gradient")
    hp = {"lr": lr, "etaminus": etaminus, "etaplus": etaplus,
          "step_min": step_min, "step_max": step_max}
    return shape, theta, prev, eta, step, gradient, hp, maximize, dtype


def _sign(value):
    return (value > 0) - (value < 0)


def _evaluate(profile, inp, mode=None):
    shape, theta, prev, eta, step, gradient, hp, maximize, _ = _inputs(profile, inp)
    if gradient is None and mode != "none_as_zero":
        return Result("value", {"theta": theta, "prev": prev, "eta": eta, "t": step})
    if step is None:
        prev = _zeros(shape)
        eta = _full(shape, Fraction(0) if mode == "cold_eta_zero" else hp["lr"])
        step = 0
    if gradient is None:
        gradient = _zeros(shape)

    theta_values, prev_values, eta_values, gradient_values = map(_flat, (theta, prev, eta, gradient))
    theta_new, prev_new, eta_new = [], [], []
    for parameter, previous, old_eta, received in zip(theta_values, prev_values, eta_values, gradient_values):
        processed = received
        if maximize and mode != "maximize_at_update":
            processed = -processed
        direction = _sign(previous * processed)
        if mode == "zero_as_increase" and direction == 0:
            direction = 1
        factor = hp["etaplus"] if direction > 0 else hp["etaminus"] if direction < 0 else Fraction(1)
        candidate_eta = old_eta * factor
        if mode == "no_clamp" or (mode == "box_zero_unclamped" and direction == 0):
            new_eta = candidate_eta
        else:
            new_eta = min(max(candidate_eta, hp["step_min"]), hp["step_max"])
        effective = processed
        if direction < 0 and mode != "no_zeroing":
            effective = Fraction(0)
        amount = new_eta * effective if mode == "magnitude_not_sign" else new_eta * _sign(effective)
        if maximize and mode == "maximize_at_update":
            theta_new.append(parameter + amount)
        else:
            theta_new.append(parameter - amount)
        prev_new.append(processed if mode == "prev_not_zeroed" else effective)
        eta_new.append(new_eta)
    step_new = step if mode == "step_not_counted" else step + 1
    return Result("value", {
        "theta": _rebuild(theta_new, shape), "prev": _rebuild(prev_new, shape),
        "eta": _rebuild(eta_new, shape), "t": step_new,
    })


def reference(profile, inp):
    return _evaluate(profile, inp)


def _observed_tensor(value, name, shape, dtype):
    if isinstance(value, dict):
        if set(value) != {"t", "v", "shape"} or value.get("t") != dtype \
                or value.get("shape") != list(shape):
            raise Invalid(f"observed {name} tensor record has the wrong dtype or shape")
        value = value["v"]
    return _read(value, shape, f"observed {name}")


def _observed_step(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if set(value) != {"t", "v", "shape"} or value.get("t") not in STEP_DTYPES \
                or value.get("shape") != []:
            raise Invalid("observed t tensor record has the wrong dtype or shape")
        value = value["v"]
    return integer(value, "observed t", lo=0)


def check_output(profile, inp, observed):
    shape, *_, dtype = _inputs(profile, inp)
    if not isinstance(observed, dict) or set(observed) != {"theta", "prev", "eta", "t"}:
        return False
    try:
        candidate = {"theta": _observed_tensor(observed["theta"], "theta", shape, dtype),
                     "t": _observed_step(observed["t"])}
        for name in ("prev", "eta"):
            candidate[name] = [] if observed[name] == [] else \
                _observed_tensor(observed[name], name, shape, dtype)
    except Invalid:
        return False
    return check_enclosed_output(reference(profile, inp).fields, candidate)


def _mutant(mode):
    return lambda profile, inp: _evaluate(profile, inp, mode)


MUTANTS = {
    "RP.M1": _mutant("prev_not_zeroed"),
    "RP.M2": _mutant("no_zeroing"),
    "RP.M3": _mutant("zero_as_increase"),
    "RP.M4": _mutant("no_clamp"),
    "RP.M5": _mutant("magnitude_not_sign"),
    "RP.M6": _mutant("maximize_at_update"),
    "RP.M7": _mutant("none_as_zero"),
    "RP.M8": _mutant("step_not_counted"),
    "RP.M9": _mutant("box_zero_unclamped"),
    "RP.M10": _mutant("cold_eta_zero"),
}
READINGS = {}
