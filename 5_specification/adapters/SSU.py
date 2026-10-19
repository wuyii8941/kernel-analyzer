"""mamba-ssm 2.3.2.post1 selective-state-update task reference."""
from __future__ import annotations

import math

import numpy as np

from kernel_spec import d2_common as U
from kernel_spec import ieee
from kernel_spec import real as rl
from kernel_spec.compare import check_enclosed_output
from kernel_spec.num import Invalid, NotEstablished, Result, Unsupported
from kernel_spec.real import R


PROFILE = "mamba232_real"
KEYS = {
    "state", "x", "dt", "A", "B", "C", "D", "z", "dt_bias",
    "dt_softplus", "state_batch_indices", "dtype",
}


def _walk(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _physical(value, rank, name):
    for item in _walk(value):
        if isinstance(item, (bool, np.bool_)):
            raise Invalid(f"{name}: Boolean is not a floating value")
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            raise Unsupported(f"{name}: special values are outside the reviewed profile")
        if isinstance(item, str) and item.strip().lower() in {"inf", "+inf", "-inf", "nan", "+nan", "-nan"}:
            raise Unsupported(f"{name}: special values are outside the reviewed profile")
    tensor, shape = U.tensor(value, rank, name)
    for item in _walk(tensor):
        rounded = ieee.round_to(item, "f32")
        if not rounded.exact or rounded.v != item.v:
            raise Invalid(f"{name}: value {item.v!r} is not exactly representable as float32")
    return tensor, shape


def parse(profile, inp):
    if profile != PROFILE:
        raise Unsupported(f"profile {profile!r}")
    U.check_keys(inp, KEYS, required=KEYS)
    if inp["dtype"] != "f32":
        raise Unsupported("the reviewed mamba-ssm profile requires float32 tensors")

    state_shape = U.shape_of(inp["state"], "state")
    heads = len(state_shape) == 4
    if len(state_shape) not in (3, 4):
        raise Invalid("state must be (slots, dim, dstate) or (slots, nheads, dim, dstate)")

    def unsq_shape(shape, key):
        if key == "state":
            return (shape[0], 1) + shape[1:]
        if key in ("x", "dt", "z", "B", "C"):
            return (shape[0], 1, shape[1])
        return (1,) + shape

    unsqueeze = {
        "state": lambda tensor: [[row] for row in tensor],
        "x": lambda tensor: [[row] for row in tensor],
        "dt": lambda tensor: [[row] for row in tensor],
        "z": lambda tensor: [[row] for row in tensor],
        "A": lambda tensor: [tensor],
        "B": lambda tensor: [[row] for row in tensor],
        "C": lambda tensor: [[row] for row in tensor],
        "D": lambda tensor: [tensor],
        "dt_bias": lambda tensor: [tensor],
    }
    ranks4 = {"state": 4, "x": 3, "dt": 3, "z": 3, "A": 3, "B": 3, "C": 3, "D": 2, "dt_bias": 2}
    values, shapes = {}, {}
    for key, rank4 in ranks4.items():
        value = inp[key]
        if value is None:
            values[key], shapes[key] = None, None
            continue
        tensor, shape = _physical(value, rank4 if heads else rank4 - 1, key)
        if not heads:
            tensor, shape = unsqueeze[key](tensor), unsq_shape(shape, key)
        values[key], shapes[key] = tensor, shape

    for key in ("state", "x", "dt", "A", "B", "C"):
        if values[key] is None:
            raise Invalid(f"{key} is required")
    if values["D"] is None or values["dt_bias"] is None:
        raise Unsupported(
            "mamba-ssm 2.3.2.post1 cannot launch this wrapper with D or dt_bias absent"
        )
    slots, nheads, dim, dstate = shapes["state"]
    batch = shapes["x"][0]
    if min(slots, nheads, dim, dstate, batch) <= 0:
        raise Invalid("state and input dimensions must be positive")
    if shapes["x"] != (batch, nheads, dim) or shapes["dt"] != shapes["x"]:
        raise Invalid("x and dt must match (batch, nheads, dim)")
    if shapes["A"] != (nheads, dim, dstate):
        raise Invalid("A must be (nheads, dim, dstate)")
    ngroups = shapes["B"][1]
    if ngroups < 1 or nheads % ngroups:
        raise Invalid("nheads must be divisible by ngroups")
    if shapes["B"] != (batch, ngroups, dstate) or shapes["C"] != shapes["B"]:
        raise Invalid("B and C must be (batch, ngroups, dstate)")
    for key, expected in (("D", (nheads, dim)), ("dt_bias", (nheads, dim)), ("z", (batch, nheads, dim))):
        if values[key] is not None and shapes[key] != expected:
            raise Invalid(f"{key} has shape {shapes[key]}, expected {expected}")

    indices = inp["state_batch_indices"]
    if indices is not None:
        indices, index_shape = U.ints(indices, 1, "state_batch_indices")
        if index_shape != (batch,) or any(index >= slots for index in indices):
            raise Invalid("state_batch_indices must have one in-range entry per input row")
        active = [index for index in indices if index >= 0]
        if len(set(active)) != len(active):
            raise NotEstablished("concurrent rows address the same state slot; no update order is declared")
    elif slots != batch:
        raise Invalid("without state_batch_indices the state slot count must equal the input batch")

    return {
        "v": values, "slots": slots, "nheads": nheads, "dim": dim, "dstate": dstate,
        "batch": batch, "ngroups": ngroups, "indices": indices,
        "softplus": U.flag(inp, "dt_softplus"), "heads": heads,
        "y_shape": (batch, nheads, dim) if heads else (batch, dim),
        "state_shape": state_shape,
    }


def _finish(parsed, y, state):
    if not parsed["heads"]:
        y = [row[0] for row in y]
        state = [slot[0] for slot in state]
    return Result("value", {"out": {"y": U.plain(y), "state": U.plain(state)}})


def _softplus(value):
    # torch.nn.functional.softplus(beta=1, threshold=20) uses the linear branch
    # only when beta*x is strictly greater than the threshold.
    return value if value > R(20) else rl.softplus(value)


def reference(profile, inp):
    parsed = parse(profile, inp)
    values = parsed["v"]
    nheads, dim, dstate = parsed["nheads"], parsed["dim"], parsed["dstate"]
    repeats = nheads // parsed["ngroups"]
    state = [[[row[:] for row in head] for head in slot] for slot in values["state"]]
    y = [[[None] * dim for _ in range(nheads)] for _ in range(parsed["batch"])]
    for batch_index in range(parsed["batch"]):
        slot = parsed["indices"][batch_index] if parsed["indices"] is not None else batch_index
        if slot < 0:
            y[batch_index] = [[R(0)] * dim for _ in range(nheads)]
            continue
        for head in range(nheads):
            group = head // repeats
            for channel in range(dim):
                delta = values["dt"][batch_index][head][channel]
                if values["dt_bias"] is not None:
                    delta = delta + values["dt_bias"][head][channel]
                if parsed["softplus"]:
                    delta = _softplus(delta)
                output = R(0)
                for coordinate in range(dstate):
                    decay = rl.exp(delta * values["A"][head][channel][coordinate])
                    incoming = delta * values["B"][batch_index][group][coordinate]
                    state[slot][head][channel][coordinate] = (
                        state[slot][head][channel][coordinate] * decay
                        + incoming * values["x"][batch_index][head][channel]
                    )
                    output = output + state[slot][head][channel][coordinate] * values["C"][batch_index][group][coordinate]
                if values["D"] is not None:
                    output = output + values["x"][batch_index][head][channel] * values["D"][head][channel]
                if values["z"] is not None:
                    gate = values["z"][batch_index][head][channel]
                    output = output * gate * rl.sigmoid(gate)
                y[batch_index][head][channel] = output
    return _finish(parsed, y, state)


def _observed(value, shape, name):
    if isinstance(value, dict):
        if set(value) != {"t", "v", "shape"} or value.get("t") != "f32" or value.get("shape") != list(shape):
            raise Invalid(f"observed {name} tensor record has the wrong dtype or shape")
        value = value["v"]
    tensor, actual = U.tensor(value, len(shape), f"observed {name}")
    if actual != tuple(shape):
        raise Invalid(f"observed {name} shape {actual}, expected {tuple(shape)}")
    return tensor


def check_output(profile, inp, observed):
    try:
        parsed = parse(profile, inp)
        if not isinstance(observed, dict) or set(observed) != {"out"}:
            return False
        out = observed["out"]
        if not isinstance(out, dict) or set(out) != {"y", "state"}:
            return False
        candidate = {"out": {
            "y": _observed(out["y"], parsed["y_shape"], "y"),
            "state": _observed(out["state"], parsed["state_shape"], "state"),
        }}
    except Invalid:
        return False
    return check_enclosed_output(reference(profile, inp).fields, candidate)


# Independent element loops with one deliberately wrong interpretation.
def _loop(profile, inp, mode):
    parsed = parse(profile, inp)
    values = parsed["v"]
    nheads, dim, dstate = parsed["nheads"], parsed["dim"], parsed["dstate"]
    old = {(slot, head, channel, coordinate): values["state"][slot][head][channel][coordinate]
           for slot in range(parsed["slots"]) for head in range(nheads)
           for channel in range(dim) for coordinate in range(dstate)}
    new = dict(old)
    y = {}
    for batch_index in range(parsed["batch"]):
        slot = parsed["indices"][batch_index] if parsed["indices"] is not None else batch_index
        if slot < 0 and mode != "padding_processed":
            for head in range(nheads):
                for channel in range(dim):
                    y[(batch_index, head, channel)] = R(0)
            continue
        slot = max(slot, 0) if mode == "padding_processed" else slot
        for head in range(nheads):
            group = head % parsed["ngroups"] if mode == "groups_modulo" else head // (nheads // parsed["ngroups"])
            for channel in range(dim):
                delta = values["dt"][batch_index][head][channel]
                if values["dt_bias"] is not None and mode != "dt_bias_ignored":
                    delta = delta + values["dt_bias"][head][channel]
                if parsed["softplus"] and mode != "softplus_ignored":
                    delta = _softplus(delta)
                output = R(0)
                for coordinate in range(dstate):
                    coefficient = values["A"][head][channel][coordinate]
                    decay = rl.exp(coefficient) if mode == "decay_without_dt" else rl.exp(coefficient * delta)
                    incoming = values["B"][batch_index][group][coordinate]
                    if mode != "input_without_dt":
                        incoming = incoming * delta
                    new[(slot, head, channel, coordinate)] = (
                        old[(slot, head, channel, coordinate)] * decay
                        + incoming * values["x"][batch_index][head][channel]
                    )
                    source = old if mode == "output_from_old_state" else new
                    output = output + values["C"][batch_index][group][coordinate] * source[(slot, head, channel, coordinate)]
                if values["D"] is not None and mode not in ("skip_D_omitted", "gate_before_D"):
                    output = output + values["D"][head][channel] * values["x"][batch_index][head][channel]
                if values["z"] is not None:
                    gate = values["z"][batch_index][head][channel]
                    output = output * gate if mode == "gate_without_sigmoid" else output * gate * rl.sigmoid(gate)
                if mode == "gate_before_D" and values["D"] is not None:
                    output = output + values["D"][head][channel] * values["x"][batch_index][head][channel]
                y[(batch_index, head, channel)] = output
        if mode == "state_not_written":
            new = dict(old)
    ys = [[[y[(batch_index, head, channel)] for channel in range(dim)] for head in range(nheads)]
          for batch_index in range(parsed["batch"])]
    states = [[[[new[(slot, head, channel, coordinate)] for coordinate in range(dstate)]
                for channel in range(dim)] for head in range(nheads)] for slot in range(parsed["slots"])]
    return _finish(parsed, ys, states)


def _mutant(mode):
    return lambda profile, inp: _loop(profile, inp, mode)


MUTANTS = {
    "SS.M1": _mutant("dt_bias_ignored"),
    "SS.M2": _mutant("skip_D_omitted"),
    "SS.M3": _mutant("output_from_old_state"),
    "SS.M4": _mutant("decay_without_dt"),
    "SS.M5": _mutant("input_without_dt"),
    "SS.M6": _mutant("softplus_ignored"),
    "SS.M7": _mutant("gate_without_sigmoid"),
    "SS.M8": _mutant("gate_before_D"),
    "SS.M9": _mutant("groups_modulo"),
    "SS.M10": _mutant("padding_processed"),
    "SS.M11": _mutant("state_not_written"),
}
READINGS = {}
