"""Verbatim function transcriptions fetched again at 1aee15e.

Source: ttir_eval.py lines 4423-4610 (GitHub connector turn854file0),
measure.py lines 153-190 (turn855file0). This file is NOT a checkout.
Use --repo to verify each executable function AST against a local checkout.
"""
from __future__ import annotations

def _int_op(name, op, args) -> TV:
    out_elem = op.result_types[0].elem
    width = INT_WIDTH[out_elem]
    st = _merge_status(*args)
    cond = _merge_cond(*args)
    reasons = _merge_reasons(*args)
    if out_elem == "i1" and name in ("andi", "ori", "xori"):
        a, b = (x.lo.astype(np.int8) for x in args)
        if name == "andi":
            v = np.where((a == 0) | (b == 0), 0, np.where((a == 1) & (b == 1), 1, MAYBE))
        elif name == "ori":
            v = np.where((a == 1) | (b == 1), 1, np.where((a == 0) & (b == 0), 0, MAYBE))
        else:
            v = np.where((a == MAYBE) | (b == MAYBE), MAYBE, a ^ b)
        return TV("b", "i1", v.astype(np.int8), None, None, st, cond, reasons)
    if name in ("extsi", "extui", "trunci"):
        x = args[0]
        in_w = INT_WIDTH[x.elem]
        vals = x.lo.astype(np.int64)
        if x.kind == "b":
            if (vals == MAYBE).any():
                st = np.where(vals == MAYBE, ST_NE, st).astype(np.int8)
            vals = np.where(vals == MAYBE, 0, vals)
            if name == "extsi":
                vals = -vals
        elif name == "extui":
            vals = _unsigned(vals, in_w).astype(np.int64)
        v = _wrap(vals, width)
        kind = "b" if out_elem == "i1" else "i"
        if kind == "b":
            v = (v & 1).astype(np.int8)
        return TV(kind, out_elem, v, None, None, st, cond, reasons)
    vals = [x.lo.astype(np.int64) for x in args]
    with np.errstate(all="ignore"):
        if name == "addi":
            v = vals[0] + vals[1]
        elif name == "subi":
            v = vals[0] - vals[1]
        elif name == "muli":
            v = vals[0] * vals[1]
        elif name in ("divsi", "remsi", "ceildivsi", "floordivsi"):
            a, b = vals
            zero = (b == 0) | ((a == -(1 << (width - 1))) & (b == -1))  # division by zero, INT_MIN / -1
            st = np.where(zero, ST_NE, st).astype(np.int8)
            bb = np.where(zero, 1, b)
            q = np.abs(a) // np.abs(bb) * np.where((a >= 0) == (bb >= 0), 1, -1)
            if name == "divsi":
                v = q
            elif name == "remsi":
                v = a - q * bb
            elif name == "ceildivsi":
                v = -((-a) // bb)
            else:
                v = a // bb
        elif name in ("divui", "remui", "ceildivui"):
            a, b = (_unsigned(x, width).astype(np.int64) for x in vals)
            zero = b == 0
            st = np.where(zero, ST_NE, st).astype(np.int8)
            bb = np.where(zero, 1, b)
            v = a // bb if name == "divui" else (a % bb if name == "remui" else -((-a) // bb))
        elif name == "andi":
            v = vals[0] & vals[1]
        elif name == "ori":
            v = vals[0] | vals[1]
        elif name == "xori":
            v = vals[0] ^ vals[1]
        elif name in ("shli", "shrsi", "shrui"):
            a, s = vals
            bad = (s < 0) | (s >= width)
            st = np.where(bad, ST_NE, st).astype(np.int8)
            s = np.where(bad, 0, s)
            if name == "shli":
                v = a << s
            elif name == "shrsi":
                v = a >> s
            else:
                v = (_unsigned(a, width).astype(np.int64) if width < 64 else a) >> s
        elif name in ("maxsi", "minsi"):
            v = (np.maximum if name == "maxsi" else np.minimum)(vals[0], vals[1])
        elif name in ("maxui", "minui"):
            ua, ub = (_unsigned(x, width).astype(np.int64) for x in vals)
            v = (np.maximum if name == "maxui" else np.minimum)(ua, ub)
        elif name == "mulhiui":
            if width == 32:
                ua, ub = (_unsigned(x, 32).astype(np.uint64) for x in vals)
                v = ((ua * ub) >> np.uint64(32)).astype(np.int64)
            elif width == 64:  # DSL v2 increment 3: the high 64 bits of the exact 128-bit product (Python integers)
                ua, ub = (_unsigned(x, 64) for x in vals)
                ua, ub = np.broadcast_arrays(ua, ub)
                hi = [(int(p) * int(q)) >> 64 for p, q in zip(ua.reshape(-1).tolist(), ub.reshape(-1).tolist())]
                v = np.array(hi, dtype=np.uint64).reshape(ua.shape).view(np.int64)
            else:
                raise ProgramAbort(f"{op.node_id}: mulhiui width {width}")
        elif name == "absi":
            v = np.abs(vals[0])
        else:
            raise ProgramAbort(f"{op.node_id}: integer op {name}")
    if name in ("muli", "andi"):
        # An established exact zero absorbs the other operand: x * 0 = x & 0 = 0 for every integer x, so a lane
        # whose other operand is undefined (e.g. a masked load without `other`) is still determined.
        a, b = args
        zero = ((a.st == ST_OK) & (vals[0] == 0)) | ((b.st == ST_OK) & (vals[1] == 0))
        if zero.any():
            st = np.where(zero, ST_OK, st).astype(np.int8)
            v = np.where(zero, 0, v)
    return TV("i", out_elem, _wrap(v, width), None, None, st, cond, reasons)


def expand(decl: dict) -> dict:
    """The full comparison declaration (written into the report; never changed during confirmation)."""

    miss = missing_items(decl)
    if miss:
        raise MissingDeclaration(miss)
    units = dict(DEFAULT_UNITS, **(decl.get("units") or {}))
    exp = {"call": decl["call"], "inputs": decl["inputs"], "compare": decl["compare"], "budget": decl["budget"],
           "rule_classes": copy.deepcopy(DEFAULT_RULE_CLASSES), "multiplicity": "Holm within each rule class",
           "units": units, "alpha": decl.get("alpha", 0.05),
           "reference_classes_allowed": ALLOWED_REFERENCE_CLASSES,
           "resolution": dict(DEFAULT_RESOLUTION, **(decl.get("resolution") or {})),
           "factors": decl.get("factors") or {},
           "factor_levels": _factor_levels(decl["inputs"], decl.get("factors")), "versions": _versions(),
           "error_budget": decl.get("error_budget"),
           "magnitude_bound": decl.get("magnitude_bound"),
           "bounded_route": "recorded only: the entry does not apply the declared error budget (bounded_mean_test is "
                            "used by the calibration); approximate route (endpoint-conservative t) for every output",
           "mixed_sources": "outputs reading a non-Triton intermediate: numerical difference only, no semantic verdict",
           "statistics": "frozen endpoint-conservative t per rule (analysis._summarize) + contract_v3 cannot-judge "
                         "rules (S0 = 2, N0 = 64, n_min = 16)",
           "reference_scope": "call-level complete only when every non-Triton upstream value is a copy of the declared "
                              "inputs; otherwise kernel-level (upstream values captured) and not counted as complete "
                              "for the call"}
    body = json.dumps({k: decl[k] for k in ("call", "inputs", "compare", "budget")}, sort_keys=True, default=str)
    exp["declaration_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    exp["_base_dir"] = decl.get("_base_dir", ".")
    return exp


def _execution_status(rows, launch_info, r_exec) -> dict:
    """Execution validity of one output over the units (DSL v2 rc3 02 8.7): repeated launches and race findings."""
    race = sorted({r.split("@")[0] for p in rows for r in (p["reasons"] or {}) if "execution race" in r})
    unknown = sorted({r.split("@")[0] for p in rows for r in (p["reasons"] or {}) if "execution validity" in r})
    differ, reps_done, inputs_differ = [], 0, False
    for p in rows:
        w = p["written"]
        reps_done = max(reps_done, len(p["k_reps"]))
        inputs_differ |= p["repeat_inputs_differ"]
        if any(not np.array_equal(k[w], p["k"][w], equal_nan=True) for k in p["k_reps"] if k.shape == p["k"].shape):
            differ.append(p["seed"])
    atomics = any(li.get("float_atomics") for li in (launch_info or []))
    out = {"launches_per_input": 1 + reps_done, "units_with_different_repeats": len(differ),
           "units": len(rows), "repeat_inputs_not_reproducible": inputs_differ, "race_findings": race,
           "unknown_validity_findings": unknown, "float_atomics": atomics}
    if race:
        out.update(status="execution race found by the reference: statistics withheld", statistics="withheld")
    elif differ and atomics and not inputs_differ:
        out.update(status=f"atomic execution randomness: residuals averaged within the input over {1 + reps_done} "
                          "launches", statistics="within-input mean")
    elif differ:
        out.update(status="repeated launches differ without an identified cause: execution validity not established "
                          "(diagnosis needed)" + ("; the regenerated inputs differ" if inputs_differ else ""),
                   statistics="withheld")
    elif reps_done == 0:
        out.update(status="not repeated", statistics="per launch")
    else:
        out.update(status="repeated launches bitwise identical", statistics="per launch")
    return out
