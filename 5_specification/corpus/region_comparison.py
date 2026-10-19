"""Complete G-versus-f interval comparison for the selected softmax region.

These are conditional mathematical comparisons, not device/kernel certification.
"""
import math
from fractions import Fraction

import numpy as np

from kernel_spec.num import NotEstablished


def validate_sequence(launches, result):
    if len(result.launches) != len(launches) or result.external_writes:
        raise NotEstablished(f"composed reference is incomplete or resets memory from device intermediates: {result.external_writes}")
    summaries = []
    for launch, ref in zip(launches, result.launches):
        grid = launch.grid
        if (len(grid) != 3 or any(type(n) is not int or n <= 0 for n in grid)
                or len(ref.programs) != math.prod(grid) or len(set(ref.programs)) != len(ref.programs)
                or any(len(p) != 3 or any(type(i) is not int or not 0 <= i < n for i, n in zip(p, grid))
                       for p in ref.programs)):
            raise NotEstablished("composed reference did not evaluate every program in the full grid")
        if ref.mode != "numerical_difference" or ref.aborted or ref.reasons or ref.notes:
            raise NotEstablished(f"launch {launch.index}: mode={ref.mode}; aborted={ref.aborted}; "
                                 f"reasons={ref.reasons}; notes={ref.notes}")
        summaries.append({"launch": launch.index, "programs": len(ref.programs)})
    if any(row.get("evidence") != "bytes and version counters" for row in result.carry_overs):
        raise NotEstablished("reference carry-over lacks captured version-counter evidence")
    return summaries


def compare_outputs(memory, outputs, reference):
    mapping = reference["output_nodes"]
    if set(mapping) != {"mean_loss", "maxima", "log_mass", "count"} or len(set(mapping.values())) != 4:
        raise NotEstablished("the selected reference must account for all four distinct output roles")
    if {item["return"]["node"] for item in outputs} != set(mapping.values()):
        raise NotEstablished("computed region outputs and selected mathematical outputs disagree")
    roles = {node: role for role, node in mapping.items()}
    results = []
    for item in outputs:
        arg = item["arg"]
        role = roles[item["return"]["node"]]
        target = ([(reference[role], reference[role])] if role == "count" else
                  [reference[role]] if role == "mean_loss" else reference[role])
        if len(target) != math.prod(arg.shape):
            raise NotEstablished("mathematical reference omits output coordinates")
        buf = memory.get(arg.storage_ptr)
        if (buf is None or buf.index is not None or buf.dtype != arg.dtype or buf.kind != "f"
                or buf.hi is None):
            raise NotEstablished("complete floating output reference storage is unavailable")
        if type(arg.element_size) is not int or arg.element_size <= 0 or arg.storage_nbytes % arg.element_size:
            raise NotEstablished("output element size or storage length is invalid")
        arrays = [buf.lo, buf.hi, buf.st, buf.cond, buf.written]
        if any(np.ndim(x) != 1 or len(x) != arg.storage_nbytes // arg.element_size for x in arrays):
            raise NotEstablished("reference storage shape disagrees with the capture")
        if len(arg.shape) != len(arg.stride) or any(type(n) is not int or n < 0 for n in (*arg.shape, *arg.stride)):
            raise NotEstablished("output layout is unsupported")
        offset = arg.data_ptr - arg.storage_ptr
        if offset < 0 or offset % arg.element_size:
            raise NotEstablished("output storage offset is invalid")
        rows = []
        for coordinate, interval in zip(np.ndindex(arg.shape), target):
            index = offset // arg.element_size + sum(i * s for i, s in zip(coordinate, arg.stride))
            if not 0 <= index < len(buf.lo):
                raise NotEstablished("output coordinate is outside reference storage")
            if (buf.st[index] != 0 or not buf.written[index] or buf.cond[index]
                    or buf.premise is not None and buf.premise[index]
                    or buf.iset is not None and buf.iset[index]):
                raise NotEstablished("output coordinate is unwritten, nonfinite, conditional or depends on an unproved premise")
            if not math.isfinite(float(buf.lo[index])) or not math.isfinite(float(buf.hi[index])):
                raise NotEstablished("output interval is not finite")
            lo, hi = Fraction(float(buf.lo[index])), Fraction(float(buf.hi[index]))
            f_lo, f_hi = map(Fraction, interval)
            if lo > hi or f_lo > f_hi:
                raise RuntimeError("inverted reference interval")
            lower, upper = lo - f_hi, hi - f_lo
            relation = "exact_equal" if lower == upper == 0 else "zero_excluded" if lower > 0 or upper < 0 else "zero_enclosed"
            rows.append({"coordinate": list(coordinate), "g_lower": lo, "g_upper": hi,
                         "f_lower": f_lo, "f_upper": f_hi, "difference_lower": lower,
                         "difference_upper": upper, "relation": relation})
        results.append({"return": item["return"], "role": role, "elements": len(rows), "rows": rows})
    return results


def evaluate_and_compare(launches, outputs, reference, max_programs=2048):
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence
    if type(max_programs) is not int or max_programs <= 0:
        raise ValueError("max_programs must be a positive integer")
    if any(len(x.grid) != 3 or any(type(n) is not int or n <= 0 for n in x.grid) for x in launches):
        raise NotEstablished("invalid program grid")
    programs = sum(math.prod(x.grid) for x in launches)
    if programs > max_programs:
        raise NotEstablished("full composed reference exceeds the program budget")
    result = evaluate_sequence(launches)
    summaries = validate_sequence(launches, result)
    compared = compare_outputs(result.memory, outputs, reference)
    return {"mode": "numerical_difference", "launches": summaries, "carry_overs": result.carry_overs,
            "external_writes": result.external_writes, "outputs": compared,
            "semantic_verdict": "not_established",
            "limitation": "conditional G-f comparison; complete kernel/task and device-effect certification remain required"}
