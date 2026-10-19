"""Reviewed PyTorch 2.10 Triton BSR softmax values core."""
from __future__ import annotations

import math

import numpy as np

from kernel_spec import d1_attention as A
from kernel_spec import ieee
from kernel_spec import real as rl
from kernel_spec.compare import check_enclosed_output
from kernel_spec.num import Invalid, Result, Unsupported, require_fields
from kernel_spec.real import R


PROFILE = "torch210_bsr_softmax_values_real"
REQUIRED = {"shape", "blocksize", "crow", "col", "values", "dtype"}
DTYPES = {"f16", "bf16", "f32"}


def _walk(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _integer_vector(value, name):
    if not isinstance(value, (list, tuple)):
        raise Invalid(f"{name} must be a list of integers")
    result = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise Invalid(f"{name} must contain only integers")
        result.append(int(item))
    return result


def _physical_values(value, nnz, rows, columns, dtype):
    if nnz == 0:
        if value != []:
            raise Invalid("values must be empty when there are no specified blocks")
        return []
    for item in _walk(value):
        if isinstance(item, (bool, np.bool_)):
            raise Invalid("values: Boolean is not a floating value")
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            raise Unsupported("special values are outside the reviewed BSR softmax profile")
        if isinstance(item, str) and item.strip().lower() in {"inf", "+inf", "-inf", "nan", "+nan", "-nan"}:
            raise Unsupported("special values are outside the reviewed BSR softmax profile")
    tensor, actual = A.tensor(value, 3, "values")
    if actual != (nnz, rows, columns):
        raise Invalid(f"values shape {actual}, expected {(nnz, rows, columns)}")
    for item in _walk(tensor):
        rounded = ieee.round_to(R(item), dtype)
        if not rounded.exact or rounded.v != item:
            raise Invalid(f"values: value {item!r} is not exactly representable as {dtype}")
    return tensor


def _parse(profile, inp):
    if profile != PROFILE:
        raise Unsupported(f"profile {profile!r}")
    require_fields(inp, REQUIRED, "BSR")
    extra = set(inp) - REQUIRED
    if extra:
        raise Unsupported(f"fields outside the selected BSR softmax values task: {sorted(extra)}")
    dtype = inp["dtype"]
    if dtype not in DTYPES:
        raise Unsupported("the reviewed PyTorch 2.10 BSR softmax task requires dtype f16, bf16 or f32")
    shape = _integer_vector(inp["shape"], "shape")
    blocksize = _integer_vector(inp["blocksize"], "blocksize")
    if len(shape) != 2 or len(blocksize) != 2:
        raise Invalid("shape and blocksize must each contain two integers")
    matrix_rows, matrix_columns = shape
    block_rows, block_columns = blocksize
    if min(shape + blocksize) <= 0 or matrix_rows % block_rows or matrix_columns % block_columns:
        raise Invalid("shape must be positive and divisible by positive blocksize")
    crow = _integer_vector(inp["crow"], "crow")
    col = _integer_vector(inp["col"], "col")
    nnz = len(col)
    block_row_count = matrix_rows // block_rows
    block_column_count = matrix_columns // block_columns
    if (len(crow) != block_row_count + 1 or not crow or crow[0] != 0 or crow[-1] != nnz
            or any(left > right for left, right in zip(crow, crow[1:]))):
        raise Invalid("crow must contain block-row offsets from zero through the number of specified blocks")
    for block_row in range(block_row_count):
        columns = col[crow[block_row]:crow[block_row + 1]]
        if any(column < 0 or column >= block_column_count for column in columns):
            raise Invalid("column block index is outside the matrix shape")
        if any(left >= right for left, right in zip(columns, columns[1:])):
            raise Invalid("column block indices must be strictly increasing within each block row")
    values = _physical_values(inp["values"], nnz, block_rows, block_columns, dtype)
    return {
        "M": matrix_rows, "N": matrix_columns, "R": block_rows, "C": block_columns,
        "crow": crow, "col": col, "values": values, "nnz": nnz, "dtype": dtype,
    }


def _plain(value):
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value.v if isinstance(value, R) else value


def _normalised(xs):
    maximum = max(xs)
    exponentials = [rl.exp(R(value - maximum)) for value in xs]
    total = R(0)
    for value in exponentials:
        total = total + value
    return [value / total for value in exponentials]


def _evaluate(profile, inp, mode=None):
    parsed = _parse(profile, inp)
    if parsed["nnz"] == 0:
        return Result("value", {"out": []})
    out = [[[R(0) for _ in range(parsed["C"])] for _ in range(parsed["R"])]
           for _ in range(parsed["nnz"])]
    for block_row in range(parsed["M"] // parsed["R"]):
        entries = list(range(parsed["crow"][block_row], parsed["crow"][block_row + 1]))
        if not entries:
            continue
        if mode == "blockrow_all":
            positions = [(entry, row, column) for entry in entries
                         for row in range(parsed["R"]) for column in range(parsed["C"])]
            weights = _normalised([parsed["values"][entry][row][column] for entry, row, column in positions])
            for position, weight in zip(positions, weights):
                entry, row, column = position
                out[entry][row][column] = weight
            continue
        for row in range(parsed["R"]):
            if mode == "per_block":
                for entry in entries:
                    weights = _normalised(parsed["values"][entry][row])
                    for column, weight in enumerate(weights):
                        out[entry][row][column] = weight
                continue
            if mode == "per_column":
                for column in range(parsed["C"]):
                    weights = _normalised([parsed["values"][entry][row][column] for entry in entries])
                    for entry, weight in zip(entries, weights):
                        out[entry][row][column] = weight
                continue
            positions = [(entry, column) for entry in entries for column in range(parsed["C"])]
            xs = [parsed["values"][entry][row][column] for entry, column in positions]
            if mode == "dense_unspecified":
                xs = xs + [R(0)] * (parsed["N"] - len(xs))
                weights = _normalised(xs)[:len(positions)]
            elif mode == "zeros_excluded":
                kept = [(position, value) for position, value in zip(positions, xs) if value != 0]
                weights = _normalised([value for _, value in kept]) if kept else []
                for (entry, column), weight in zip((position for position, _ in kept), weights):
                    out[entry][row][column] = weight
                continue
            elif mode == "sigmoid":
                weights = [rl.exp(R(value)) / (R(1) + rl.exp(R(value))) for value in xs]
            else:
                weights = _normalised(xs)
            for (entry, column), weight in zip(positions, weights):
                out[entry][row][column] = weight
    return Result("value", {"out": _plain(out)})


def reference(profile, inp):
    return _evaluate(profile, inp)


def _observed(value, shape, dtype):
    if isinstance(value, dict):
        if set(value) != {"t", "v", "shape"} or value.get("t") != dtype or value.get("shape") != list(shape):
            raise Invalid("observed out tensor record has the wrong dtype or shape")
        value = value["v"]
    if shape[0] == 0:
        if value != []:
            raise Invalid("observed out must be the empty value buffer")
        return []
    tensor, actual = A.tensor(value, 3, "observed out")
    if actual != shape:
        raise Invalid(f"observed out shape {actual}, expected {shape}")
    return tensor


def check_output(profile, inp, observed):
    try:
        parsed = _parse(profile, inp)
        if not isinstance(observed, dict) or set(observed) != {"out"}:
            return False
        candidate = {"out": _observed(
            observed["out"], (parsed["nnz"], parsed["R"], parsed["C"]), parsed["dtype"])}
    except Invalid:
        return False
    return check_enclosed_output(reference(profile, inp).fields, candidate)


def _mutant(mode):
    return lambda profile, inp: _evaluate(profile, inp, mode)


MUTANTS = {
    "BS.M1": _mutant("per_block"),
    "BS.M2": _mutant("dense_unspecified"),
    "BS.M3": _mutant("zeros_excluded"),
    "BS.M4": _mutant("blockrow_all"),
    "BS.M5": _mutant("per_column"),
    "BS.M6": _mutant("sigmoid"),
}
READINGS = {}
