"""Reviewed Mixtral top-2 routing and supplied-expert combination."""
from __future__ import annotations

import math

import numpy as np

from kernel_spec import d1_attention as A
from kernel_spec import ieee
from kernel_spec import real as rl
from kernel_spec.compare import check_enclosed_output
from kernel_spec.num import Invalid, Result, Unsupported, require_fields
from kernel_spec.real import R


PROFILE = "mixtral_top2_real"
REQUIRED = {"router_logits", "expert_outputs", "dtype"}
DTYPES = {"f16", "bf16", "f32", "f64"}


def _walk(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _physical_tensor(value, rank, name, dtype):
    for item in _walk(value):
        if isinstance(item, (bool, np.bool_)):
            raise Invalid(f"{name}: Boolean is not a floating value")
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            raise Unsupported(f"{name}: special values are outside the reviewed Mixtral profile")
        if isinstance(item, str) and item.strip().lower() in {"inf", "+inf", "-inf", "nan", "+nan", "-nan"}:
            raise Unsupported(f"{name}: special values are outside the reviewed Mixtral profile")
    tensor, shape = A.tensor(value, rank, name)
    for item in _walk(tensor):
        rounded = ieee.round_to(R(item), dtype)
        if not rounded.exact or rounded.v != item:
            raise Invalid(f"{name}: value {item!r} is not exactly representable as {dtype}")
    return tensor, shape


def _parse(profile, inp):
    if profile != PROFILE:
        raise Unsupported(f"profile {profile!r}")
    require_fields(inp, REQUIRED, "MOE")
    extra = set(inp) - REQUIRED
    if extra:
        raise Unsupported(f"fields outside the selected Mixtral routing task: {sorted(extra)}")
    dtype = inp["dtype"]
    if dtype not in DTYPES:
        raise Unsupported("the reviewed Mixtral profile requires dtype f16, bf16, f32 or f64")
    logits, ls = _physical_tensor(inp["router_logits"], 2, "router_logits", dtype)
    experts, ys = _physical_tensor(inp["expert_outputs"], 3, "expert_outputs", dtype)
    tokens, count = ls
    if tokens <= 0 or count < 2:
        raise Invalid("router_logits must have shape [T,E] with T > 0 and E >= 2")
    if ys[:2] != ls or ys[2] <= 0:
        raise Invalid("expert_outputs must have shape [T,E,D] with positive D")
    for token, row in enumerate(logits):
        if len(set(row)) != count:
            raise Unsupported(f"token {token}: tied router logits are outside the reviewed profile")
    return {"logits": logits, "experts": experts, "T": tokens, "E": count, "D": ys[2], "dtype": dtype}


def _plain(value):
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value.v if isinstance(value, R) else value


def _softmax(logits, indices):
    maximum = max(logits[index] for index in indices)
    exponentials = [rl.exp(R(logits[index] - maximum)) for index in indices]
    total = R(0)
    for value in exponentials:
        total = total + value
    return [value / total for value in exponentials]


def _evaluate(profile, inp, mode=None):
    parsed = _parse(profile, inp)
    selected, weights, outputs = [], [], []
    for token in range(parsed["T"]):
        logits = parsed["logits"][token]
        order = sorted(range(parsed["E"]), key=lambda expert: logits[expert], reverse=(mode != "bottom_two"))
        chosen = order[:2]
        active = chosen[:1] if mode == "top_one" else chosen
        mask = [expert in active for expert in range(parsed["E"])]
        dense = [R(0) for _ in range(parsed["E"])]

        if mode in {"raw_selected", "all_experts"}:
            all_weights = _softmax(logits, list(range(parsed["E"])))
            if mode == "all_experts":
                combine = list(range(parsed["E"]))
                for expert, value in enumerate(all_weights):
                    dense[expert] = value
            else:
                combine = active
                for expert in active:
                    dense[expert] = all_weights[expert]
        elif mode == "uniform":
            combine = active
            for expert in active:
                dense[expert] = R(1) / R(len(active))
        else:
            combine = active
            chosen_weights = _softmax(logits, active)
            if mode == "swap_weights" and len(chosen_weights) == 2:
                chosen_weights = list(reversed(chosen_weights))
            for expert, value in zip(active, chosen_weights):
                dense[expert] = value

        result = []
        for coordinate in range(parsed["D"]):
            value = R(0)
            for expert in combine:
                value = value + dense[expert] * R(parsed["experts"][token][expert][coordinate])
            result.append(value)
        selected.append(mask)
        weights.append(dense)
        outputs.append(result)
    return Result("value", {"selected": selected, "weights": _plain(weights), "out": _plain(outputs)})


def reference(profile, inp):
    return _evaluate(profile, inp)


def _shape(value, name):
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    children = [_shape(item, name) for item in value]
    if any(child != children[0] for child in children):
        raise Invalid(f"observed {name} is ragged")
    return (len(value),) + children[0]


def _observed_selected(value, shape):
    if isinstance(value, dict):
        if set(value) != {"t", "v", "shape"} or value.get("t") != "b8" or value.get("shape") != list(shape):
            raise Invalid("observed selected tensor record has the wrong dtype or shape")
        value = value["v"]
    if _shape(value, "selected") != tuple(shape) or any(type(item) is not bool for item in _walk(value)):
        raise Invalid("observed selected must be a complete Boolean tensor")
    return [list(row) for row in value]


def _observed_numeric(value, shape, dtype, name, rank):
    if isinstance(value, dict):
        if set(value) != {"t", "v", "shape"} or value.get("t") != dtype or value.get("shape") != list(shape):
            raise Invalid(f"observed {name} tensor record has the wrong dtype or shape")
        value = value["v"]
    tensor, actual = A.tensor(value, rank, f"observed {name}")
    if actual != tuple(shape):
        raise Invalid(f"observed {name} shape {actual}, expected {tuple(shape)}")
    return tensor


def check_output(profile, inp, observed):
    try:
        parsed = _parse(profile, inp)
        if not isinstance(observed, dict) or set(observed) != {"selected", "weights", "out"}:
            return False
        candidate = {
            "selected": _observed_selected(observed["selected"], (parsed["T"], parsed["E"])),
            "weights": _observed_numeric(
                observed["weights"], (parsed["T"], parsed["E"]), parsed["dtype"], "weights", 2),
            "out": _observed_numeric(observed["out"], (parsed["T"], parsed["D"]), parsed["dtype"], "out", 2),
        }
    except Invalid:
        return False
    return check_enclosed_output(reference(profile, inp).fields, candidate)


def _mutant(mode):
    return lambda profile, inp: _evaluate(profile, inp, mode)


MUTANTS = {
    "MO.M1": _mutant("raw_selected"),
    "MO.M2": _mutant("bottom_two"),
    "MO.M3": _mutant("top_one"),
    "MO.M4": _mutant("uniform"),
    "MO.M5": _mutant("all_experts"),
    "MO.M6": _mutant("swap_weights"),
}
READINGS = {}
