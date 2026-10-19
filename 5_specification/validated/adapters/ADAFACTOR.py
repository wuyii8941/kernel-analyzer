"""Reviewed strict-real state transition for one public PyTorch 2.10 Adafactor parameter."""
from __future__ import annotations

from fractions import Fraction

from kernel_spec.compare import check_enclosed_output
from kernel_spec.e1_optim import flag, hyper, state_dict
from kernel_spec.num import Invalid, NotEstablished, Result, Undefined, Unsupported, integer, rational
from kernel_spec.real import R, max_, min_, sqrt


VECTOR = "torch210_vector_real"
FACTORED = "torch210_factored_real"
PROFILES = {VECTOR, FACTORED}
REQUIRED = {
    "state", "gradient", "dtype", "lr", "beta2_decay", "eps1", "eps2", "d",
    "weight_decay", "maximize",
}
OPTIONAL = {"foreach"}
FLOAT_DTYPES = {"f16", "bf16", "f32", "f64"}
STEP_DTYPES = {"i8", "i16", "i32", "i64", *FLOAT_DTYPES}
FINFO_EPS = {
    "f16": Fraction(1, 2**10),
    "bf16": Fraction(1, 2**7),
    "f32": Fraction(1, 2**23),
    "f64": Fraction(1, 2**52),
}
MAX_EXACT_STEP = 10000


def shape_of(value, what):
    if not isinstance(value, list):
        return ()
    if not value:
        raise Invalid(f"{what}: empty dimension")
    children = [shape_of(item, what) for item in value]
    if any(child != children[0] for child in children):
        raise Invalid(f"{what}: ragged tensor")
    return (len(value),) + children[0]


def _map(function, value):
    return [_map(function, item) for item in value] if isinstance(value, list) else function(value)


def _read(value, shape, what):
    actual = shape_of(value, what)
    if actual != tuple(shape):
        raise Invalid(f"{what}: shape {actual}, expected {tuple(shape)}")
    return _map(lambda item: rational(item, what), value)


def _flat(value):
    return [item for child in value for item in _flat(child)] if isinstance(value, list) else [value]


def _matrices(value):
    """Return trailing two-dimensional matrices in leading-index order."""
    return [value] if not isinstance(value[0][0], list) else [matrix for child in value for matrix in _matrices(child)]


def _rebuild(values, shape):
    iterator = iter(values)

    def build(sizes):
        if len(sizes) == 1:
            return [next(iterator) for _ in range(sizes[0])]
        return [build(sizes[1:]) for _ in range(sizes[0])]

    return build(tuple(shape))


def _zeros(shape):
    if len(shape) == 1:
        return [Fraction(0) for _ in range(shape[0])]
    return [_zeros(shape[1:]) for _ in range(shape[0])]


def _mean(values):
    if all(isinstance(value, Fraction) for value in values):
        return sum(values, Fraction(0)) / len(values)
    return sum((R(value) for value in values), R(0)) / R(len(values))


def _rms(values):
    return sqrt(R(_mean([value * value for value in values])))


def _strategy(inp):
    value = inp.get("foreach")
    if value is not None and type(value) is not bool:
        raise Invalid("foreach must be true, false or null")
    return value


def _inputs(profile, inp):
    if profile not in PROFILES:
        raise Unsupported(f"profile {profile!r}")
    if not isinstance(inp, dict) or any(not isinstance(key, str) for key in inp):
        raise Invalid("Adafactor input must be a mapping with string field names")
    missing = REQUIRED - set(inp)
    if missing:
        raise Invalid(f"Adafactor input is missing {sorted(missing)}")
    extra = set(inp) - REQUIRED - OPTIONAL
    if extra:
        raise Unsupported(f"Adafactor fields outside the selected public task: {sorted(extra)}")
    _strategy(inp)
    maximize = flag(inp, "maximize")

    dtype = inp["dtype"]
    if dtype not in FLOAT_DTYPES:
        raise Unsupported("the reviewed Adafactor profiles require dtype f16, bf16, f32 or f64")
    eps1 = FINFO_EPS[dtype] if inp["eps1"] is None else rational(inp["eps1"], "eps1")
    hp = hyper(inp, ("lr", "beta2_decay", "eps2", "d", "weight_decay"),
               nonneg=("lr", "eps2", "weight_decay"))
    hp["eps1"] = eps1
    if eps1 < 0:
        raise Invalid("eps1 must be null or >= 0")
    if hp["beta2_decay"] > 0:
        raise Invalid("beta2_decay must be <= 0")
    if hp["d"] < 1:
        raise Invalid("d must be >= 1")

    state = state_dict(inp, {"theta", "row", "col", "var", "t"})
    shape = shape_of(state["theta"], "theta")
    if not shape:
        raise Invalid("a scalar parameter is outside the reviewed profiles")
    if profile == VECTOR and len(shape) != 1:
        raise Invalid("the vector profile requires a rank-1 parameter")
    if profile == FACTORED and len(shape) < 2:
        raise Invalid("the factored profile requires a parameter of rank at least 2")
    theta = _read(state["theta"], shape, "theta")

    if state["t"] is None:
        if state["row"] != [] or state["col"] != [] or state["var"] != []:
            raise Invalid("an uninitialized Adafactor state must use row=col=var=[] and t=null")
        row, col, var, step = [], [], [], None
    else:
        step = integer(state["t"], "t", lo=0)
        if profile == VECTOR:
            if state["row"] != [] or state["col"] != []:
                raise Invalid("a vector Adafactor state must use row=col=[]")
            row, col = [], []
            var = _read(state["var"], shape, "var")
            factors = _flat(var)
        else:
            if state["var"] != []:
                raise Invalid("a factored Adafactor state must use var=[]")
            row_shape = shape[:-1] + (1,)
            col_shape = shape[:-2] + (1, shape[-1])
            row = _read(state["row"], row_shape, "row")
            col = _read(state["col"], col_shape, "col")
            var = []
            factors = _flat(row) + _flat(col)
        if any(value < 0 for value in factors):
            raise Invalid("Adafactor variance state must be nonnegative")

    gradient_value = inp.get("gradient", "missing")
    if gradient_value == "missing":
        raise Invalid("missing gradient (use null for a parameter without gradient)")
    gradient = None if gradient_value is None else _read(gradient_value, shape, "gradient")
    return shape, theta, row, col, var, step, gradient, hp, maximize


def _one_minus_beta(step, decay):
    if step > MAX_EXACT_STEP:
        raise NotEstablished("exact Adafactor step power exceeds the declared reference budget")
    if step == 1 or decay == 0:
        return Fraction(1)
    if decay.denominator == 1:
        return Fraction(step) ** decay.numerator
    return R(step) ** R(decay)


def _lerp(old, target, weight):
    if isinstance(weight, Fraction):
        return old + weight * (target - old)
    return R(old) + R(weight) * (R(target) - R(old))


def _initialized(profile, shape, row, col, var, step):
    if step is not None:
        return row, col, var, step
    if profile == VECTOR:
        return [], [], _zeros(shape), 0
    return _zeros(shape[:-1] + (1,)), _zeros(shape[:-2] + (1, shape[-1])), [], 0


def _selected_fields(profile, theta, row, col, var, step):
    if profile == VECTOR:
        return {"theta": theta, "var": var, "t": step}
    return {"theta": theta, "row": row, "col": col, "t": step}


def _reference(profile, inp, mode=None):
    shape, theta, row, col, var, step, gradient, hp, maximize = _inputs(profile, inp)
    if gradient is None and mode != "none_as_zero":
        return Result("value", _selected_fields(profile, theta, row, col, var, step))
    row, col, var, step0 = _initialized(profile, shape, row, col, var, step)
    if gradient is None:
        gradient = _zeros(shape)
    if maximize and mode != "maximize_ignored":
        gradient = _map(lambda value: -value, gradient)

    step1 = step0 + 1
    beta_step = step0 if mode == "old_step_in_beta" else step1
    if beta_step == 0:
        raise Undefined("the old-step Adafactor reading attempts zero to a non-positive power")
    weight = _one_minus_beta(beta_step, hp["beta2_decay"])
    rho_cap = Fraction(1, 100) if mode == "paper_rho_cap" else hp["lr"]
    rho = min_(R(rho_cap), R(1) / sqrt(R(step1)))
    theta_values = _flat(theta)
    decayed = [value * (1 - hp["lr"] * hp["weight_decay"]) for value in theta_values]
    alpha_values = decayed if mode == "alpha_after_decay" else theta_values
    alpha = max_(R(hp["eps2"]), _rms(alpha_values)) * rho
    eps1 = hp["eps1"]

    if profile == VECTOR:
        targets = [value * value + (eps1 if mode == "eps1_added" else 0) for value in gradient]
        var_new = [_lerp(old, target, weight) for old, target in zip(var, targets)]
        updates = []
        for received, estimate in zip(gradient, var_new):
            floor = R(estimate) if mode == "eps1_added" else max_(R(estimate), R(eps1 * eps1))
            if floor.sign() == 0:
                raise Undefined("Adafactor update has zero variance and zero eps1")
            updates.append(R(received) / sqrt(floor))
        row_new, col_new = [], []
    else:
        gradients = _matrices(gradient)
        row_matrices = _matrices(row)
        col_matrices = _matrices(col)
        original_batches = len(gradients)
        if mode == "flatten_leading" and original_batches > 1:
            gradients = [[line for matrix in gradients for line in matrix]]
            row_matrices = [[[value] for value in _flat(row)]]
            col_matrices = [[list(_matrices(col)[0][0])]]

        row_batches, col_batches, updates = [], [], []
        for grad_matrix, row_matrix, col_matrix in zip(gradients, row_matrices, col_matrices):
            rows, columns = len(grad_matrix), len(grad_matrix[0])
            row_targets = []
            for i in range(rows):
                values = [grad_matrix[i][j] ** 2 for j in range(columns)]
                reduction = sum(values, Fraction(0)) if mode == "sums_in_state" else _mean(values)
                row_targets.append(reduction + (eps1 if mode == "eps1_added" else 0))
            col_targets = []
            for j in range(columns):
                values = [grad_matrix[i][j] ** 2 for i in range(rows)]
                reduction = sum(values, Fraction(0)) if mode == "sums_in_state" else _mean(values)
                col_targets.append(reduction + (eps1 if mode == "eps1_added" else 0))
            row_batch = [_lerp(row_matrix[i][0], row_targets[i], weight) for i in range(rows)]
            col_batch = [_lerp(col_matrix[0][j], col_targets[j], weight) for j in range(columns)]
            row_batches.append(row_batch)
            col_batches.append(col_batch)

            if mode in {"sums_in_state", "sum_denominator"}:
                normalizer = sum((R(value) for value in row_batch), R(0))
            else:
                normalizer = R(_mean(row_batch))
            if mode != "eps1_added":
                normalizer = max_(normalizer, R(eps1))
            if normalizer.sign() == 0:
                raise Undefined("Adafactor factored variance has a zero normalizer")
            for i in range(rows):
                for j in range(columns):
                    estimate = R(row_batch[i]) * R(col_batch[j]) / normalizer
                    floor = estimate if mode == "eps1_added" else max_(estimate, R(eps1 * eps1))
                    if floor.sign() == 0:
                        raise Undefined("Adafactor update has zero variance and zero eps1")
                    updates.append(R(grad_matrix[i][j]) / sqrt(floor))

        if mode == "flatten_leading" and original_batches > 1:
            col_batches = [col_batches[0] for _ in range(original_batches)]
        row_shape = shape[:-1] + (1,)
        col_shape = shape[:-2] + (1, shape[-1])
        row_values = [value for batch in row_batches for value in batch]
        col_values = [value for batch in col_batches for value in batch]
        row_new = _rebuild(row_values, row_shape)
        col_new = _rebuild(col_values, col_shape)
        var_new = []

    scale = R(1) if mode == "no_clipping" else max_(R(1), _rms(updates) / R(hp["d"]))
    bases = theta_values if mode == "decay_lost" else decayed
    theta_new = _rebuild([R(base) - alpha * update / scale for base, update in zip(bases, updates)], shape)
    if mode == "squeezed_state" and profile == FACTORED and len(shape) == 2:
        row_new = [line[0] for line in row_new]
        col_new = col_new[0]
    return Result("value", _selected_fields(profile, theta_new, row_new, col_new, var_new, step1))


def reference(profile, inp):
    return _reference(profile, inp)


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
            raise Invalid("observed t tensor record has the wrong shape")
        value = value["v"]
    return integer(value, "observed t", lo=0)


def check_output(profile, inp, observed):
    """Require the complete public state; nonpoint formula enclosures remain not established."""
    shape, *_ = _inputs(profile, inp)
    expected_keys = {"theta", "var", "t"} if profile == VECTOR else {"theta", "row", "col", "t"}
    if not isinstance(observed, dict) or set(observed) != expected_keys:
        return False
    try:
        dtype = inp["dtype"]
        candidate = {"theta": _observed_tensor(observed["theta"], "theta", shape, dtype),
                     "t": _observed_step(observed["t"])}
        if profile == VECTOR:
            candidate["var"] = [] if observed["var"] == [] else \
                _observed_tensor(observed["var"], "var", shape, dtype)
        else:
            row_shape = shape[:-1] + (1,)
            col_shape = shape[:-2] + (1, shape[-1])
            candidate["row"] = [] if observed["row"] == [] else \
                _observed_tensor(observed["row"], "row", row_shape, dtype)
            candidate["col"] = [] if observed["col"] == [] else \
                _observed_tensor(observed["col"], "col", col_shape, dtype)
    except Invalid:
        return False
    return check_enclosed_output(reference(profile, inp).fields, candidate)


def _mutant(mode):
    return lambda profile, inp: _reference(profile, inp, mode)


MUTANTS = {
    "AF.M1": _mutant("sums_in_state"),
    "AF.M2": _mutant("alpha_after_decay"),
    "AF.M3": _mutant("decay_lost"),
    "AF.M4": _mutant("eps1_added"),
    "AF.M5": _mutant("no_clipping"),
    "AF.M6": _mutant("paper_rho_cap"),
    "AF.M7": _mutant("old_step_in_beta"),
    "AF.M8": _mutant("maximize_ignored"),
    "AF.M9": _mutant("none_as_zero"),
    "AF.M10": _mutant("flatten_leading"),
    "AF.M11": _mutant("sum_denominator"),
    "AF.M12": _mutant("squeezed_state"),
}
READINGS = {}
