"""Admitted contract references for the measurement entry.

This module is the narrow bridge between ``measure`` and ``5_specification``.  It never accepts an arbitrary Python
reference callback.  The independently retained binding plan fixes the measured call, every contract task and the
contract-output-to-measured-output mapping.  The current call wrapper itself must also equal the retained readable
file.  Every sampled call is checked again with its actual inputs, outputs, state changes and recorded Triton launches.
"""
from __future__ import annotations

import importlib.util
import importlib.metadata
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch


class ContractBindingError(ValueError):
    pass


_DTYPES = {
    torch.float64: "f64", torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "f8e4m3", torch.float8_e5m2: "f8e5m2",
    torch.int64: "i64", torch.int32: "i32", torch.int16: "i16", torch.int8: "i8", torch.uint8: "u8",
    torch.bool: "b8",
}


def _backend(specification_root: Path):
    root = specification_root.resolve()
    module_path = root / "corpus" / "call_binding.py"
    if not module_path.is_file() or not (root / "kernel_spec" / "registry.py").is_file():
        raise ContractBindingError(f"no specification runtime at {root}")
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("_kernel_analyzer_call_binding", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    if Path(module.registry.ROOT).resolve() != root:
        raise ContractBindingError("a different kernel_spec package is already loaded in this process")
    return module


def _call_file(call_ref: str, base_dir: Path) -> Path:
    if not isinstance(call_ref, str) or ":" not in call_ref:
        raise ContractBindingError("the measured call must be a file.py:function reference")
    module, _ = call_ref.rsplit(":", 1)
    if not module.endswith(".py"):
        raise ContractBindingError("contract mode requires a readable file.py:function call wrapper")
    path = Path(module)
    if not path.is_absolute():
        path = base_dir / path
        if not path.is_file():
            path = Path(module)
    if not path.is_file():
        raise ContractBindingError(f"measured call wrapper is absent: {path}")
    return path.resolve()


def _tensor_record(tensor: torch.Tensor) -> dict:
    dtype = _DTYPES.get(tensor.dtype)
    if dtype is None:
        raise ContractBindingError(f"tensor dtype {tensor.dtype} is not representable by the contract input format")
    value = tensor.detach().cpu()
    return {"t": dtype, "v": value.tolist(), "shape": list(value.shape)}


def _record(value):
    if torch.is_tensor(value):
        return _tensor_record(value)
    if isinstance(value, dict):
        return {key: _record(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_record(item) for item in value]
    if isinstance(value, tuple):
        return [_record(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ContractBindingError(f"call value {type(value).__name__} is not representable in a binding observation")


def _contains_tensor(value):
    if torch.is_tensor(value):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def _same_view(left, right) -> bool:
    return torch.is_tensor(left) and torch.is_tensor(right) and \
        left.untyped_storage()._cdata == right.untyped_storage()._cdata and \
        (left.storage_offset(), tuple(left.shape), tuple(left.stride()), left.dtype) == \
        (right.storage_offset(), tuple(right.shape), tuple(right.stride()), right.dtype)


def _endpoints(value, backend):
    """One strict contract value as the narrowest containing binary64 interval."""
    value = value.v if type(value).__name__ == "R" else value
    if isinstance(value, str) and value in ("+inf", "-inf"):
        point = math.inf if value == "+inf" else -math.inf
        return point, point
    lo, hi = backend.ends(value)

    def outward(q, direction):
        point = float(q)
        if not math.isfinite(point):
            return point
        exact = Fraction.from_float(point)
        if direction < 0 and exact > q:
            return float(np.nextafter(point, -math.inf))
        if direction > 0 and exact < q:
            return float(np.nextafter(point, math.inf))
        return point

    return outward(lo, -1), outward(hi, 1)


def _arrays(value, backend):
    """Nested exact values/enclosures -> rectangular binary64 lower and upper arrays."""
    if isinstance(value, dict):
        raise ContractBindingError("a measured target must select a tensor/scalar leaf, not a mapping")
    if isinstance(value, (list, tuple)):
        pairs = [_arrays(item, backend) for item in value]
        try:
            return np.asarray([item[0] for item in pairs], dtype=np.float64), \
                np.asarray([item[1] for item in pairs], dtype=np.float64)
        except ValueError as error:
            raise ContractBindingError(f"contract output is ragged: {error}") from error
    lo, hi = _endpoints(value, backend)
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def _boolean_arrays(value):
    """Exact Boolean contract values in the tool's explicit 0/1 comparison frame.

    The contract is still checked as Boolean data.  The conversion happens only
    after that check so the measurement layer can count signed truth-value
    differences and exact mismatches without accepting arbitrary numeric values
    as Boolean references.
    """
    if isinstance(value, dict):
        raise ContractBindingError("a measured target must select a tensor/scalar leaf, not a mapping")
    if isinstance(value, (list, tuple)):
        pairs = [_boolean_arrays(item) for item in value]
        try:
            points = np.asarray([item[0] for item in pairs], dtype=np.float64)
        except ValueError as error:
            raise ContractBindingError(f"contract Boolean output is ragged: {error}") from error
        return points, points.copy()
    if type(value) is not bool:
        raise ContractBindingError("an exact Boolean contract output contains a non-Boolean value")
    point = np.asarray(1.0 if value else 0.0, dtype=np.float64)
    return point, point.copy()


class ContractReference:
    """Validate one retained binding plan and evaluate it for every measured call."""

    def __init__(self, *, plan_path, specification_root, call_ref, base_dir, measure_names,
                 backend=None, reviewed_root=None):
        root = Path(specification_root)
        if not root.is_absolute():
            root = Path(base_dir) / root
        self.root = root.resolve()
        self.backend = backend or _backend(self.root)
        self.plan_path = Path(plan_path)
        if not self.plan_path.is_absolute():
            self.plan_path = (Path(base_dir) / self.plan_path).resolve()
        self.plan = self.backend._load(self.plan_path)
        self.id = self.plan.get("id")
        self.call_ref = call_ref
        self.base_dir = Path(base_dir)
        self.call_path = None
        self.reviewed_root = Path(reviewed_root) if reviewed_root is not None else self.root / "validated"
        self.measure_names = list(measure_names)
        self._measure = {}
        self.output_kinds = {}
        self.program_reference_mode = None
        self._validate()

    def _validate(self):
        if self.plan.get("status") != "reviewed":
            raise ContractBindingError("contract measurement requires a reviewed binding plan")
        subject = self.plan.get("subject")
        if not isinstance(subject, dict) or subject.get("measure_call") != self.call_ref:
            raise ContractBindingError("binding plan subject does not name the measured call wrapper exactly")
        package = subject.get("runtime_package")
        if not isinstance(package, dict) or set(package) != {"distribution", "version"} or not all(
                isinstance(package.get(key), str) and package[key] for key in ("distribution", "version")):
            raise ContractBindingError("reviewed binding subject needs an exact installed runtime package and version")
        try:
            observed_version = importlib.metadata.version(package["distribution"])
        except importlib.metadata.PackageNotFoundError as error:
            raise ContractBindingError(
                f"bound runtime package {package['distribution']!r} is not installed") from error
        if observed_version != package["version"]:
            raise ContractBindingError(
                f"bound runtime package version is {observed_version}, expected {package['version']}")
        self.call_path = _call_file(self.call_ref, self.base_dir)
        reviewed, reasons = self.backend._plan_review(
            self.plan, self.plan_path, self.reviewed_root / "bindings")
        if not reviewed:
            raise ContractBindingError("binding plan is not admitted: " + "; ".join(reasons))
        retained_call = self.reviewed_root / "calls" / f"{self.id}.py"
        if not retained_call.is_file():
            raise ContractBindingError("independently reviewed call wrapper is not retained")
        if retained_call.read_bytes() != self.call_path.read_bytes():
            raise ContractBindingError("current call wrapper differs from the independently reviewed file")
        try:
            self.program_reference_mode = self.backend._program_reference(self.plan)
        except self.backend.BindingError as error:
            raise ContractBindingError(str(error)) from error

        outputs = []
        for task in self.plan.get("tasks") or []:
            cid, profile, statement = task.get("contract"), task.get("profile"), task.get("task_statement")
            contract = self.backend.registry.load(cid)
            admitted, reasons, _ = self.backend.registry.activation(
                contract, profile, task_bindings={profile: statement}, allow_draft=False)
            if not admitted:
                raise ContractBindingError(f"task {task.get('id')}: specification is not admitted: " +
                                           "; ".join(reasons))
            declared = self.backend.profile_outputs(contract.data, profile)
            for item in task.get("output_map") or []:
                target, name = item.get("target"), item.get("measure_output")
                if not isinstance(target, str) or not isinstance(name, str) or not name:
                    raise ContractBindingError("every bound contract output needs a measure_output in the reviewed plan")
                top = self.backend._tokens(target)[0]
                info = declared.get(top)
                if info is None:
                    raise ContractBindingError(f"task {task.get('id')}: measured target {target!r} is not declared")
                kind = info.get("kind")
                guarantee = info.get("guarantee")
                if kind == "boolean":
                    if guarantee != "exact":
                        raise ContractBindingError(
                            f"task {task.get('id')}: {target} needs an exact Boolean guarantee")
                elif kind == "discrete" or guarantee not in ("exact", "rigorous_enclosure"):
                    raise ContractBindingError(
                        f"task {task.get('id')}: {target} is neither a strict numeric nor exact Boolean target")
                if name in self._measure:
                    raise ContractBindingError(f"measured output {name!r} is mapped more than once")
                source = item.get("source")
                if source != f"/returns/{name}" and not (isinstance(source, str) and
                                                          source.startswith("/tensor_arguments/") and
                                                          source.endswith("/after")):
                    raise ContractBindingError(
                        f"measured output {name!r} must bind to its return or a declared tensor post-state")
                self._measure[name] = {"task": task, "target": target, "source": source, "kind": kind}
                self.output_kinds[name] = "boolean" if kind == "boolean" else "numeric"
                outputs.append(name)
        if set(outputs) != set(self.measure_names) or len(outputs) != len(self.measure_names):
            raise ContractBindingError("reviewed binding measure_output fields must equal compare.measure")

    def _observation(self, inp, outs, launches, snapshot):
        if not isinstance(inp, dict) or not all(isinstance(key, str) for key in inp):
            raise ContractBindingError("contract mode requires top-level named call inputs")
        if not isinstance(outs, dict) or set(outs) != set(self.measure_names):
            raise ContractBindingError("call outputs must equal the reviewed measured-output names")
        tensors, scalars = {}, {}
        for name, value in inp.items():
            if torch.is_tensor(value):
                tensors[name] = {"before": _tensor_record(snapshot.pre_call_tensor(name)),
                                 "after": _tensor_record(value)}
            else:
                if _contains_tensor(value):
                    raise ContractBindingError(
                        f"input {name!r} contains a nested tensor; contract mode needs each tensor as a named input")
                scalars[name] = _record(value)
        returns = {name: _record(value) for name, value in outs.items()}
        for name, item in self._measure.items():
            source = item["source"]
            if source.startswith("/tensor_arguments/"):
                argument = self.backend._tokens(source)[1]
                if argument not in inp or not _same_view(outs[name], inp[argument]):
                    raise ContractBindingError(
                        f"measured output {name!r} is not the declared post-state tensor view {argument!r}")
        public = [item["source"] for item in self._measure.values()]
        return {"subject": self.plan["subject"], "launch_scope_complete": True,
                "launches": [launch.kernel_name for launch in launches],
                "tensor_arguments": tensors, "scalars": scalars, "returns": returns,
                "public_outputs": public}

    def evaluate(self, inp, outs, launches, snapshot, binding_context=None):
        if binding_context is None:
            binding_context = {}
        if not isinstance(binding_context, dict):
            raise ContractBindingError("contract binding context must be a mapping")
        hidden = binding_context.get("unaccounted_sources_by_output", {})
        if not isinstance(hidden, dict) or any(not isinstance(name, str) or not isinstance(sources, list)
                                               or any(not isinstance(source, str) for source in sources)
                                               for name, sources in hidden.items()):
            raise ContractBindingError("unaccounted output sources must map output names to source descriptions")
        affected = {name: sources for name, sources in hidden.items() if sources}
        if affected:
            details = "; ".join(f"{name}: {', '.join(sources)}" for name, sources in sorted(affected.items()))
            enclosed = binding_context.get("dispatch_calls_with_untraced_internals", [])
            if enclosed:
                if not isinstance(enclosed, list) or any(
                        not isinstance(item, dict) or not isinstance(item.get("operator"), str)
                        or type(item.get("first_launch")) is not int
                        or type(item.get("stop_launch")) is not int
                        or not 0 <= item["first_launch"] < item["stop_launch"] for item in enclosed):
                    raise ContractBindingError("invalid dispatch visibility diagnostic")
                spans = ", ".join(
                    f"{item['operator']} (Triton launches [{item['first_launch']}, {item['stop_launch']}))"
                    for item in enclosed)
                details += "; ATen producer trace has untraced internals in: " + spans
            raise ContractBindingError(
                "public output depends on tensor state not accounted for by the binding plan: " + details)
        observation = self._observation(inp, outs, launches, snapshot)
        reference_results = {}
        report = self.backend.inspect(self.plan, observation, plan_path=self.plan_path,
                                      reviewed_root=self.reviewed_root / "bindings",
                                      _reference_results=reference_results)
        if report.get("binding_status") != "reviewed" or not report.get("reference_usable_for_bias_measurement"):
            raise ContractBindingError("sampled call did not retain an admitted specification binding: " +
                                       str(report.get("reason") or report.get("binding_review_reasons") or "unknown"))
        by_task = {row["id"]: row for row in report["tasks"]}
        specs = {}
        for name, item in self._measure.items():
            row = by_task[item["task"]["id"]]
            result = reference_results[row["id"]]["result"]
            if result.outcome != "value":
                raise ContractBindingError(f"admitted reference changed outcome to {result.outcome}: {result.note}")
            value = self.backend._get(result.fields, item["target"])
            lo, hi = (_boolean_arrays(value) if item["kind"] == "boolean"
                      else _arrays(value, self.backend))
            actual = outs[name]
            if not torch.is_tensor(actual):
                actual_shape = tuple(np.asarray(actual).shape)
            else:
                actual_shape = tuple(actual.shape)
            if tuple(lo.shape) != actual_shape or tuple(hi.shape) != actual_shape:
                raise ContractBindingError(
                    f"reference target {item['target']} shape {tuple(lo.shape)} differs from output {name} {actual_shape}")
            specs[name] = (lo, hi)
        summary = {"binding_id": self.id, "binding_status": report["binding_status"],
                   "reference_status": report["reference_status"],
                   "reference_usable_for_bias_measurement": True,
                   "program_reference_mode": self.program_reference_mode,
                   "output_kinds": dict(self.output_kinds),
                   "tasks": [{"id": row["id"], "contract": row["contract"], "profile": row["profile"]}
                             for row in report["tasks"]]}
        return specs, summary
