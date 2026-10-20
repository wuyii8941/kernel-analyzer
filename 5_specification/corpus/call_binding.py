#!/usr/bin/env python3
"""Check a public-call boundary against one or more specification tasks.

This module checks explicit mappings.  It does not infer a task from a kernel
name or from numerical agreement.  A drafted mapping can support development,
but only a retained independent review of the mapping and an admitted contract
make the reference usable by a later bias experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

SPEC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPEC))

from kernel_spec import bulk, registry  # noqa: E402
from kernel_spec.compare import validate_reference_values  # noqa: E402
from kernel_spec.num import IV, Result, ends, evaluate  # noqa: E402
from kernel_spec.real import R  # noqa: E402
from kernel_spec.schema import profile_outputs  # noqa: E402


class BindingError(ValueError):
    pass


_MISSING = object()


def _tokens(pointer):
    if not isinstance(pointer, str) or not pointer.startswith("/") or pointer == "/":
        raise BindingError("mapping paths must be nonempty JSON pointers")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _received_input_root(source):
    parts = _tokens(source)
    if len(parts) >= 3 and parts[0] == "tensor_arguments" and parts[2] == "before":
        root = "/".join(("", *parts[:3]))
        if source != root:
            raise BindingError(f"task input source {source!r} maps only part of received tensor {root!r}")
        return root
    if len(parts) >= 2 and parts[0] == "scalars":
        root = "/".join(("", *parts[:2]))
        if source != root:
            raise BindingError(f"task input source {source!r} maps only part of received scalar value {root!r}")
        return root
    raise BindingError(f"task input source {source!r} is not a received scalar or tensor pre-state")


def _get(value, pointer):
    for part in _tokens(pointer):
        if isinstance(value, dict):
            if part not in value:
                raise BindingError(f"source path {pointer!r} is absent")
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise BindingError(f"source path {pointer!r} does not follow the observation structure")
    return value


def _new_container(next_part):
    return [] if next_part.isdigit() else {}


def _set(root, pointer, value):
    parts = _tokens(pointer)
    node = root
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        if isinstance(node, dict):
            if last:
                if part in node:
                    raise BindingError(f"mapping target {pointer!r} is repeated or overlaps another mapping")
                node[part] = value
            else:
                if part not in node:
                    node[part] = _new_container(parts[index + 1])
                elif not isinstance(node[part], (dict, list)):
                    raise BindingError(f"mapping target {pointer!r} overlaps another mapping")
                node = node[part]
        elif isinstance(node, list) and part.isdigit():
            position = int(part)
            while len(node) <= position:
                node.append(_MISSING)
            if last:
                if node[position] is not _MISSING:
                    raise BindingError(f"mapping target {pointer!r} is repeated or overlaps another mapping")
                node[position] = value
            else:
                if node[position] is _MISSING:
                    node[position] = _new_container(parts[index + 1])
                elif not isinstance(node[position], (dict, list)):
                    raise BindingError(f"mapping target {pointer!r} overlaps another mapping")
                node = node[position]
        else:
            raise BindingError(f"mapping target {pointer!r} has an invalid list position")


def _same_record_values(left, right):
    """Equality of two tensor records in which NaN equals NaN (a list comparison treats NaN as unequal to itself,
    which declared untouched NaN inputs changed). Other values compare with ==, as before."""
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right) == {"t", "v", "shape"}:
        a, b = _numeric_array(left["v"]), _numeric_array(right["v"])
        if a is not None and b is not None and a.dtype.kind == b.dtype.kind:
            return left["t"] == right["t"] and left["shape"] == right["shape"] and a.shape == b.shape and \
                bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_record_values(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_record_values(x, y) for x, y in zip(left, right))
    if isinstance(left, float) and isinstance(right, float) and left != left and right != right:
        return True
    return left == right


def _complete(value):
    if value is _MISSING:
        return False
    if isinstance(value, dict) and set(value) == {"t", "v", "shape"}:
        return True       # a tensor record is a leaf: holes come only from list targets built by _set
    if isinstance(value, dict):
        return all(_complete(v) for v in value.values())
    if isinstance(value, list):
        return all(_complete(v) for v in value)
    return True


def _tensor_record(value):
    if not isinstance(value, dict) or set(value) != {"t", "v", "shape"}:
        raise BindingError("tensor observations must contain exactly t, v and shape")
    if not isinstance(value["t"], str) or not isinstance(value["shape"], list):
        raise BindingError("tensor dtype and shape are malformed")
    shape = value["shape"]
    if any(type(n) is not int or n < 0 for n in shape):
        raise BindingError("tensor shape must contain nonnegative integers")

    def has_shape(x, expected):
        if not expected:
            return not isinstance(x, list)
        if not isinstance(x, list) or len(x) != expected[0]:
            return False
        # An empty dimension has no children from which later dimensions can be
        # reconstructed.  The explicit shape remains authoritative there.
        return expected[0] == 0 or all(has_shape(child, expected[1:]) for child in x)

    if not (_numeric_shape(value["v"]) == tuple(shape) and 0 not in shape) and not has_shape(value["v"], shape):
        raise BindingError("tensor values do not have the declared shape")
    return value


def _numeric_array(values):
    """The nested value list as one numeric NumPy array, or None (ragged, empty, or non-numeric leaves). A fast path
    only: every caller falls back to the element-wise check when this returns None."""
    try:
        arr = np.asarray(values)
    except (ValueError, TypeError):
        return None
    return arr if arr.dtype.kind in "fbiu" and arr.size else None


def _numeric_shape(values):
    arr = _numeric_array(values)
    return None if arr is None else tuple(arr.shape)


def _convert(value, conversion):
    if conversion in (None, "identity"):
        return value
    if conversion == "tensor_record":
        return _tensor_record(value)
    if conversion == "tensor_values":
        return _tensor_record(value)["v"]
    raise BindingError(f"unknown mapping conversion {conversion!r}")


def _mapped(mappings, observation):
    if not isinstance(mappings, list) or not mappings:
        raise BindingError("each task needs nonempty input and output mappings")
    result, sources, by_target = {}, [], {}
    for item in mappings:
        if not isinstance(item, dict) or not isinstance(item.get("target"), str):
            raise BindingError("each mapping needs a target path")
        target = item["target"]
        has_source, has_literal = "source" in item, "literal" in item
        if has_source == has_literal:
            raise BindingError(f"mapping {target!r} needs exactly one source or literal")
        if has_source:
            source = item["source"]
            value = _get(observation, source)
            sources.append(source)
        else:
            if not isinstance(item.get("basis"), str) or not item["basis"].strip():
                raise BindingError(f"literal mapping {target!r} needs a nonblank task or API basis")
            value = item["literal"]
        value = _convert(value, item.get("conversion"))
        _set(result, target, value)
        by_target[target] = item
    if not _complete(result):
        raise BindingError("a mapping leaves holes in a list")
    return result, sources, by_target


def _plain(value):
    return value.v if isinstance(value, R) else value


def _leaves(value):
    value = _plain(value)
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from _leaves(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def _is_bulk(value):
    return hasattr(value, "bulk_bounds")


def _bulk_observed(observed):
    return np.asarray(observed, dtype=np.float64).ravel()


def _bulk_difference(observed, reference):
    """fixed_sample_difference for an array reference (kernel_spec.bulk): the same counts, with the coordinate sums
    computed exactly from the binary64 values (the reference side is its directed-rounding enclosure)."""
    point = _bulk_observed(observed)
    lo, hi = (a.ravel() for a in reference.bulk_bounds())
    if point.size != lo.size:
        return {"status": "shape_mismatch", "observed": int(point.size), "reference": int(lo.size)}
    ok = np.isfinite(point)
    n = int(ok.sum())
    if not n:
        return {"status": "no_numeric_coordinates", "coordinates": int(point.size)}
    p, l, h = point[ok], lo[ok], hi[ok]
    sp, sl, sh = (_exact_sum(x) for x in (p, l, h))
    positive, negative = int((p > h).sum()), int((p < l).sum())
    return {"status": "computed", "coordinates": n, "positive": positive, "negative": negative,
            "contains_zero": n - positive - negative, "exact_equal": int(((p == l) & (p == h)).sum()),
            "mean_error_interval": [(sp - sh) / n, (sp - sl) / n],
            "meaning": "fixed-sample coordinate mean of K-f; not a population bias estimate"}


def _exact_sum(arr):
    nums, exp = bulk._ints(np.ascontiguousarray(arr, dtype=np.float64))
    return Fraction(sum(nums)) * Fraction(2) ** exp


def _bulk_membership(observed, reference):
    """check_enclosed_output for an array reference: False if a coordinate is outside (or not finite), True if every
    coordinate equals a point reference, otherwise not established."""
    point = _bulk_observed(observed)
    lo, hi = (a.ravel() for a in reference.bulk_bounds())
    if point.size != lo.size:
        return False
    if not np.all(np.isfinite(point)):
        return False
    equal = (lo == hi) & (point == lo)
    if reference.exact:
        return bool(np.all(equal))
    if np.any((point < lo) | (point > hi)):
        return False
    return True if np.all(equal) else None


def _reference(contract, cid, profile, inputs):
    """The reference result and the evaluator that produced it: kernel_spec.bulk where it accepts the call, otherwise
    the contract adapter."""
    accepted = {}

    def run():
        got = bulk.evaluate(cid, profile, inputs)
        accepted["result"] = got
        return got if got is not None else Result("value", {})
    result = evaluate(run)
    if "result" not in accepted:                  # bulk raised: the adapter's own refusal for that condition
        return result, bulk.VERSION
    if accepted["result"] is None:
        return evaluate(contract.adapter.reference, profile, inputs), "adapter"
    return result, bulk.VERSION


def _same_structure(observed, reference):
    """Whether every promised field and sequence position is present.

    Numeric leaves may differ in representation (for example, a float against
    an exact rational or interval), so only containers are compared here.
    """
    observed, reference = _plain(observed), _plain(reference)
    if _is_bulk(reference):
        if isinstance(observed, dict) and set(observed) == {"t", "v", "shape"}:
            return list(observed["shape"]) == list(reference.shape)
        return isinstance(observed, list) and np.asarray(observed, dtype=np.float64).shape == reference.shape
    if isinstance(observed, dict) and set(observed) == {"t", "v", "shape"} and not isinstance(reference, dict):
        return _same_structure(_tensor_record(observed)["v"], reference)
    if isinstance(reference, dict):
        return isinstance(observed, dict) and set(observed) == set(reference) and all(
            _same_structure(observed[key], reference[key]) for key in reference)
    if isinstance(reference, (list, tuple)):
        return isinstance(observed, (list, tuple)) and len(observed) == len(reference) and all(
            _same_structure(left, right) for left, right in zip(observed, reference))
    return not isinstance(observed, (dict, list, tuple))


def _numeric_observation(observed, reference):
    """Strip observation-only tensor metadata when the contract returns values."""
    if isinstance(observed, dict) and set(observed) == {"t", "v", "shape"} and not isinstance(reference, dict):
        return _tensor_record(observed)["v"]
    return observed


def fixed_sample_difference(observed, reference):
    """Coordinate mean of K-f for one observation; never a population claim."""
    if _is_bulk(reference):
        return _bulk_difference(observed, reference)
    actual, target = list(_leaves(observed)), list(_leaves(reference))
    if len(actual) != len(target):
        return {"status": "shape_mismatch", "observed": len(actual), "reference": len(target)}
    lower = upper = Fraction(0)
    positive = negative = contains_zero = exact_equal = numeric = 0
    for candidate, expected in zip(actual, target):
        expected = _plain(expected)
        if isinstance(candidate, bool) or isinstance(expected, bool):
            continue
        if not isinstance(candidate, (int, float, Fraction)) or not math.isfinite(float(candidate)):
            continue
        if not isinstance(expected, (int, Fraction, IV)) or isinstance(expected, bool):
            continue
        point = Fraction(candidate)
        lo, hi = ends(expected)
        dlo, dhi = point - hi, point - lo
        lower += dlo
        upper += dhi
        numeric += 1
        if dlo > 0:
            positive += 1
        elif dhi < 0:
            negative += 1
        else:
            contains_zero += 1
            exact_equal += int(dlo == dhi == 0)
    if not numeric:
        return {"status": "no_numeric_coordinates", "coordinates": len(actual)}
    return {"status": "computed", "coordinates": numeric, "positive": positive, "negative": negative,
            "contains_zero": contains_zero, "exact_equal": exact_equal,
            "mean_error_interval": [lower / numeric, upper / numeric],
            "meaning": "fixed-sample coordinate mean of K-f; not a population bias estimate"}


def _plan_review(plan, plan_path, reviewed_root):
    if plan.get("status") != "reviewed":
        return False, ["binding plan is drafted"]
    unresolved = plan.get("unresolved", [])
    if not isinstance(unresolved, list) or any(not isinstance(issue, str) or not issue.strip()
                                                for issue in unresolved):
        return False, ["binding plan unresolved issues must be a list of nonblank statements"]
    if unresolved:
        return False, [f"unresolved binding issue: {issue}" for issue in unresolved]
    if plan_path is None:
        return False, ["a reviewed binding plan must be loaded from a file"]
    try:
        if _load(plan_path) != plan:
            return False, ["the in-memory binding plan differs from the current plan file"]
    except (BindingError, OSError, ValueError) as error:
        return False, [f"cannot read the current binding plan: {error}"]
    reviewed_root = SPEC / "validated" / "bindings" if reviewed_root is None else Path(reviewed_root)
    reviewed = reviewed_root / f"{plan.get('id')}.json"
    if not reviewed.is_file():
        return False, ["independently reviewed binding plan is not retained"]
    if reviewed.read_bytes() != Path(plan_path).read_bytes():
        return False, ["current binding plan differs from the independently reviewed file"]
    return True, []


def _program_reference(plan):
    """The program-value model selected by a reviewed concrete binding.

    The default removes operation rounding so K-G measures numerical effects.
    A task whose operation is itself a declared conversion may instead select
    the existing rounding-check evaluator.  The selection needs a reviewable
    reason in the binding plan; it never changes the complete K-f comparison.
    """
    selected = plan.get("program_reference")
    if selected is None:
        return "numerical_difference"
    if not isinstance(selected, dict) or set(selected) != {"mode", "basis"}:
        raise BindingError("program_reference must contain exactly mode and basis")
    if selected.get("mode") not in ("numerical_difference", "rounding_check"):
        raise BindingError("program_reference mode must be numerical_difference or rounding_check")
    if not isinstance(selected.get("basis"), str) or not selected["basis"].strip():
        raise BindingError("program_reference needs a nonblank binding basis")
    return selected["mode"]


def inspect(plan, observation, *, plan_path=None, reviewed_root=None, _reference_results=None):
    report = {"candidate_status": "not_established", "binding_status": "not_established",
              "reference_status": "not_evaluated", "reference_usable_for_bias_measurement": False,
              "bias_verdict": "not_established", "semantic_verdict": "not_established", "tasks": []}
    try:
        if not isinstance(plan, dict) or plan.get("schema") != "triton-call-binding-v1":
            raise BindingError("unsupported binding-plan schema")
        if not isinstance(plan.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9._-]+", plan["id"]):
            raise BindingError("binding plan needs an id")
        if plan.get("status") not in ("drafted", "reviewed"):
            raise BindingError("binding plan status must be drafted or reviewed")
        subject = plan.get("subject")
        if not isinstance(subject, dict) or subject != observation.get("subject"):
            raise BindingError("observed library, version, callable or boundary differs from the binding plan")
        if plan.get("launch_scope") != "all_triton_launches_within_call":
            raise BindingError("binding plan must require all Triton launches within the call")
        report["program_reference_mode"] = _program_reference(plan)
        launches = observation.get("launches")
        if observation.get("launch_scope_complete") is not True or not isinstance(launches, list) or not launches \
                or any(not isinstance(name, str) or not name for name in launches):
            raise BindingError("complete nonempty Triton launch observation is required")
        tasks = plan.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise BindingError("binding plan needs at least one specification task")
        task_ids = [task.get("id") for task in tasks if isinstance(task, dict)]
        if len(task_ids) != len(tasks) or any(not isinstance(x, str) or not x for x in task_ids) \
                or len(set(task_ids)) != len(task_ids):
            raise BindingError("task ids must be nonempty and unique")
        public = observation.get("public_outputs")
        if not isinstance(public, list) or any(not isinstance(p, str) for p in public) or len(set(public)) != len(public):
            raise BindingError("public output paths must be an explicit unique list")

        conditioned = plan.get("conditioned_on_returns", [])
        if not isinstance(conditioned, list) or any(
                not isinstance(c, dict) or set(c) != {"source", "basis"} or not isinstance(c["source"], str)
                or not c["source"].startswith("/returns/") or not isinstance(c["basis"], str) or not c["basis"].strip()
                for c in conditioned):
            raise BindingError("conditioned_on_returns must list {source: /returns/..., basis} entries with a basis")
        conditioned_sources = {c["source"] for c in conditioned}
        if len(conditioned_sources) != len(conditioned):
            raise BindingError("a conditioning return is declared more than once")

        all_input_sources, all_output_sources, task_runtime = [], [], {}
        for task in tasks:
            cid, profile = task.get("contract"), task.get("profile")
            statement = task.get("task_statement")
            if not all(isinstance(x, str) and x.strip() for x in (cid, profile, statement)):
                raise BindingError(f"task {task.get('id')!r} lacks contract, profile or task statement")
            contract = registry.load(cid)
            if profile not in contract.data.get("profiles", {}):
                raise BindingError(f"task {task['id']}: unknown profile {profile!r}")
            inputs, input_sources, _ = _mapped(task.get("input_map"), observation)
            outputs, output_sources, output_by_target = _mapped(task.get("output_map"), observation)
            required = profile_outputs(contract.data, profile)
            if set(outputs) != set(required):
                raise BindingError(f"task {task['id']}: observed fields {sorted(outputs)} do not match {sorted(required)}")
            all_input_sources.extend(input_sources)
            all_output_sources.extend(output_sources)
            result, reference_backend = _reference(contract, cid, profile, inputs)
            if _reference_results is not None:
                if not isinstance(_reference_results, dict):
                    raise BindingError("internal reference result sink must be a mapping")
                _reference_results[task["id"]] = {"result": result, "inputs": inputs}
            if result.outcome == "value":
                validate_reference_values(result.fields, required)
                differences = {}
                for name, output in required.items():
                    observed_value = _numeric_observation(outputs[name], result.fields[name])
                    if output.get("guarantee") == "allowed_set":
                        differences[name] = {
                            "status": "not_applicable_allowed_set",
                            "meaning": "allowed-result membership is checked as a whole; a coordinate K-f mean is undefined",
                        }
                    else:
                        if not _same_structure(observed_value, result.fields[name]):
                            raise BindingError(
                                f"task {task['id']}: observed output {name!r} omits or adds promised fields or positions")
                    if output.get("guarantee") == "allowed_set":
                        pass
                    elif output.get("kind") in ("boolean", "discrete"):
                        differences[name] = {
                            "status": "not_applicable_discrete",
                            "meaning": "discrete-result membership is checked as a whole; a coordinate K-f mean is undefined",
                        }
                    else:
                        differences[name] = fixed_sample_difference(observed_value, result.fields[name])
                comparison = "not_established"
                checker = getattr(contract.adapter, "check_output", None)
                bulk_fields = [name for name in required if _is_bulk(result.fields[name])]
                if bulk_fields:
                    states = [_bulk_membership(_numeric_observation(outputs[name], result.fields[name]),
                                               result.fields[name]) for name in bulk_fields]
                    states += [None for name in required if name not in bulk_fields]
                    comparison = "outside_declared_result" if False in states else \
                        "not_established" if None in states else "inside_declared_result"
                elif checker is not None:
                    checked = evaluate(lambda: Result("value", {"matches": checker(profile, inputs, outputs)}))
                    if checked.outcome == "value" and type(checked.fields.get("matches")) is bool:
                        comparison = "inside_declared_result" if checked.fields["matches"] else \
                            "outside_declared_result"
                    else:
                        comparison = checked.outcome
            else:
                differences, comparison = None, "not_established"
            usable, reasons, labels = registry.activation(
                contract, profile, task_bindings={profile: statement},
                allow_draft=contract.data["review"]["status"] == "drafted")
            row = {"id": task["id"], "contract": cid, "profile": profile, "reference_backend": reference_backend,
                   "inputs": inputs, "observed_outputs": outputs,
                   "reference_outcome": result.outcome, "reference_note": result.note,
                   "fixed_call_against_f": comparison,
                   "fixed_call_meaning": ("membership of this one observed call in the contract result; "
                                          "outside membership can be ordinary finite-precision error and is not a "
                                          "population bias or semantic verdict"),
                   "fixed_sample_difference": differences,
                   "specification_admitted": usable, "admission_reasons": reasons, "labels": labels}
            report["tasks"].append(row)
            task_runtime[task["id"]] = {"task": task, "outputs": outputs, "output_by_target": output_by_target,
                                        "reference": result, "admitted": usable}

        unused_conditions = conditioned_sources - set(all_input_sources)
        if unused_conditions:
            raise BindingError(f"declared conditioning return {sorted(unused_conditions)} is not used by any task input")
        uncompared = conditioned_sources - set(all_output_sources)
        if uncompared:
            raise BindingError(f"conditioning return {sorted(uncompared)} must itself be a compared output")
        if conditioned_sources:
            report["conditioned_on_returns"] = sorted(conditioned_sources)
        if set(public) != set(all_output_sources):
            missing = sorted(set(public) - set(all_output_sources))
            extra = sorted(set(all_output_sources) - set(public))
            raise BindingError(f"public outputs and mapped outputs differ; missing={missing}, extra={extra}")
        if len(all_output_sources) != len(set(all_output_sources)):
            raise BindingError("one public output is mapped more than once")

        arguments = observation.get("tensor_arguments")
        scalars = observation.get("scalars", {})
        state_map = plan.get("state_map")
        if not isinstance(arguments, dict) or not isinstance(scalars, dict) or not isinstance(state_map, list):
            raise BindingError("tensor arguments, scalar inputs and the tensor state map must be explicit")
        by_argument = {}
        for state in state_map:
            if not isinstance(state, dict) or not isinstance(state.get("argument"), str):
                raise BindingError("every state entry needs an argument name")
            if state["argument"] in by_argument:
                raise BindingError("one tensor argument has multiple state entries")
            by_argument[state["argument"]] = state
        if set(arguments) != set(by_argument):
            raise BindingError("every tensor argument must have exactly one state entry")
        for name, states in arguments.items():
            if not isinstance(states, dict) or set(states) != {"before", "after"}:
                raise BindingError(f"tensor argument {name!r} needs complete before and after observations")
            before, after = _tensor_record(states["before"]), _tensor_record(states["after"])
            if before["t"] != after["t"] or before["shape"] != after["shape"]:
                raise BindingError(f"tensor argument {name!r} changed dtype or shape")
            state = by_argument[name]
            if state.get("effect") == "unchanged":
                if not _same_record_values(before, after):
                    raise BindingError(f"tensor argument {name!r} changed but the plan declares it unchanged")
            elif state.get("effect") == "output":
                tid, target = state.get("task"), state.get("target")
                if tid not in task_runtime or not isinstance(target, str):
                    raise BindingError(f"tensor argument {name!r} has an invalid post-state target")
                mapping = task_runtime[tid]["output_by_target"].get(target)
                expected_source = f"/tensor_arguments/{name}/after"
                if mapping is None or mapping.get("source") != expected_source:
                    raise BindingError(f"tensor argument {name!r} post-state is not mapped to the declared output")
            else:
                raise BindingError(f"tensor argument {name!r} has an unknown state effect")

        observed_inputs = {f"/tensor_arguments/{name}/before" for name in arguments}
        observed_inputs.update(f"/scalars/{name}" for name in scalars)

        mapped_inputs = {_received_input_root(source) for source in all_input_sources
                         if source not in conditioned_sources}
        unused_inputs = plan.get("unused_inputs", [])
        if not isinstance(unused_inputs, list):
            raise BindingError("unused_inputs must be a list")
        unused = set()
        for item in unused_inputs:
            if not isinstance(item, dict) or not isinstance(item.get("source"), str) or not isinstance(
                    item.get("basis"), str) or not item["basis"].strip():
                raise BindingError("every unused input needs a source and a nonblank task or API basis")
            source = item["source"]
            if source in unused:
                raise BindingError(f"unused input {source!r} is repeated")
            if source not in observed_inputs:
                raise BindingError(f"unused input {source!r} is not an observed top-level input")
            unused.add(source)
        overlap = mapped_inputs & unused
        if overlap:
            raise BindingError(f"inputs cannot be both mapped and unused: {sorted(overlap)}")
        accounted = mapped_inputs | unused
        if accounted != observed_inputs:
            missing = sorted(observed_inputs - accounted)
            extra = sorted(accounted - observed_inputs)
            raise BindingError(f"received inputs are not completely accounted for; missing={missing}, extra={extra}")

        report["candidate_status"] = "explicit contract/profile candidate"
        reviewed, review_reasons = _plan_review(plan, plan_path, reviewed_root)
        report["binding_review_reasons"] = review_reasons
        report["binding_status"] = "reviewed" if reviewed else "development_checked" if plan["status"] == "drafted" else "not_established"
        outcomes = {row["reference_outcome"] for row in report["tasks"]}
        report["reference_status"] = "value" if outcomes == {"value"} else ",".join(sorted(outcomes))
        report["reference_usable_for_bias_measurement"] = bool(
            reviewed and all(row["specification_admitted"] and row["reference_outcome"] == "value"
                             for row in report["tasks"]))
        report["reason_bias_not_established"] = (
            "this is one fixed call; a bias conclusion also needs an admitted reference, a reviewed binding, "
            "and repeated observations from a declared input distribution")
        report["launches"] = launches
    except BindingError as error:
        report["binding_status"] = "rejected"
        report["reason"] = str(error)
    except Exception as error:  # programming or dependency failures are not binding evidence
        report["status"] = "implementation_error"
        report["reason"] = f"{type(error).__name__}: {error}"
    return report


def _load(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise BindingError(f"duplicate JSON key {key!r}")
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("--out must be a new file")
    try:
        report = inspect(_load(args.plan), _load(args.observation), plan_path=args.plan)
    except (BindingError, OSError, ValueError) as error:
        report = {"candidate_status": "not_established", "binding_status": "rejected",
                  "reference_status": "not_evaluated", "bias_verdict": "not_established",
                  "semantic_verdict": "not_established", "reason": str(error)}

    def encode(value):
        if isinstance(value, Fraction):
            return str(value)
        raise TypeError(type(value).__name__)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, default=encode)
        stream.write("\n")
    print(report["binding_status"])
    return 0 if report["binding_status"] in ("development_checked", "reviewed") else 2 \
        if report.get("status") != "implementation_error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
