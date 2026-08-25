"""Bounded online FP32-storage replay for every generated Triton invocation."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import linecache
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch


def tensor_runtime_contract(value: torch.Tensor) -> dict[str, Any]:
    """Value-blind runtime metadata needed to identify an implementation ABI."""

    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "device_type": value.device.type,
        "layout": str(value.layout),
        "storage_offset": int(value.storage_offset()),
    }

from forkcert.directional_error_sketch import (
    SCHEMA_VERSION as DIRECTIONAL_SKETCH_SCHEMA_VERSION,
    fixed_flat_coordinate_indices,
)
from scripts.typed_triton_reference import (
    collect_programs,
    collect_program_variants,
    compile_fp32_pointer_kernels,
    compile_fp32_pointer_kernel_variants,
)


def runtime_signature(kernel: Any) -> list[tuple[str, Any]]:
    return [
        (str(name), value)
        for name, value in kernel.triton_meta["signature"].items()
        if value != "constexpr"
    ]


def storage_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        storage.data_ptr(),
        storage.nbytes(),
        tensor.dtype,
    )


def promoted_pointer_arguments(args: Sequence[Any]) -> tuple[list[Any], list[torch.Tensor]]:
    """Clone complete storages once and recreate every pointer view exactly."""

    bases: dict[tuple[Any, ...], torch.Tensor] = {}
    physical_dtypes: dict[tuple[Any, ...], torch.dtype] = {}
    promoted: list[Any] = []
    pointer_values: list[torch.Tensor] = []
    for value in args:
        if not isinstance(value, torch.Tensor):
            promoted.append(value)
            continue
        storage = value.untyped_storage()
        physical = (
            value.device.type, value.device.index,
            storage.data_ptr(), storage.nbytes(),
        )
        previous_dtype = physical_dtypes.setdefault(physical, value.dtype)
        if previous_dtype != value.dtype:
            raise RuntimeError(
                "cross-dtype aliases cannot be exactly promoted to FP32 storage: "
                f"{previous_dtype} vs {value.dtype}"
            )
        key = storage_key(value)
        base = bases.get(key)
        if base is None:
            storage_numel = value.untyped_storage().nbytes() // value.element_size()
            raw = value.as_strided((storage_numel,), (1,), 0).detach()
            dtype = torch.float32 if value.is_floating_point() else value.dtype
            base = raw.to(dtype=dtype, copy=True)
            bases[key] = base
        view = base.as_strided(value.shape, value.stride(), value.storage_offset())
        promoted.append(view)
        pointer_values.append(view)
    return promoted, pointer_values


def validate_compiled_triton_replay_abi(
    kernel: Any, args: Sequence[Any], promoted_args: Sequence[Any],
) -> None:
    """Reject dtype promotion through a binary whose pointer ABI is frozen.

    An Inductor Triton wrapper's ``triton_meta.signature`` fixes each pointer
    element type. Passing an FP32 allocation to a ``*bf16`` compiled pointer
    does not create an FP32 implementation of the program; it reinterprets the
    bytes under the old ABI.
    """

    signature = runtime_signature(kernel)
    pointer_annotations = [
        (name, str(annotation)) for name, annotation in signature
        if str(annotation).startswith("*")
    ]
    original_tensors = [value for value in args if isinstance(value, torch.Tensor)]
    promoted_tensors = [value for value in promoted_args if isinstance(value, torch.Tensor)]
    if len(pointer_annotations) != len(original_tensors) or len(original_tensors) != len(promoted_tensors):
        raise RuntimeError("compiled Triton replay pointer count mismatch")
    changed = [
        {
            "pointer": name, "compiled_abi": annotation,
            "original_dtype": str(original.dtype), "promoted_dtype": str(promoted.dtype),
        }
        for (name, annotation), original, promoted
        in zip(pointer_annotations, original_tensors, promoted_tensors)
        if original.dtype != promoted.dtype
    ]
    if changed:
        raise RuntimeError(
            "INVALID_REFERENCE_ABI: FP32 storage cannot be passed to a compiled "
            f"Triton pointer signature without regenerating the program: {changed[:4]}"
        )


def validate_typed_triton_reference_abi(
    kernel: Any, promoted_args: Sequence[Any],
) -> None:
    """Require every compiled pointer ABI to match its physical tensor dtype."""

    dtype_abis = {
        torch.bfloat16: "*bf16", torch.float16: "*fp16",
        torch.float32: "*fp32", torch.float64: "*fp64",
        torch.int64: "*i64", torch.int32: "*i32",
        torch.int16: "*i16", torch.int8: "*i8", torch.uint8: "*u8",
        torch.bool: "*i1",
    }
    pointer_annotations = [
        (name, str(annotation)) for name, annotation in runtime_signature(kernel)
        if str(annotation).startswith("*")
    ]
    tensors = [value for value in promoted_args if isinstance(value, torch.Tensor)]
    if len(pointer_annotations) != len(tensors):
        raise RuntimeError("typed Triton reference pointer count mismatch")
    mismatches = []
    for (name, annotation), tensor in zip(pointer_annotations, tensors):
        expected = dtype_abis.get(tensor.dtype)
        if expected is None or annotation != expected:
            mismatches.append({
                "pointer": name, "compiled_abi": annotation,
                "tensor_dtype": str(tensor.dtype), "expected_abi": expected,
            })
    if mismatches:
        raise RuntimeError(f"TYPED_REFERENCE_ABI_MISMATCH: {mismatches[:4]}")


def nonfinite_aware_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    sample_size: int,
    metric_chunk_elements: int,
    retain_sampled_values: bool = True,
) -> dict[str, Any]:
    """Compare finite values while explicitly accounting for nonfinite geometry."""

    if candidate.shape != reference.shape:
        raise ValueError("candidate/reference shape mismatch")
    numel = int(candidate.numel())
    candidate_flat = candidate.detach().reshape(-1)
    reference_flat = reference.detach().reshape(-1)
    if numel:
        positions = fixed_flat_coordinate_indices(numel, sample_size=sample_size)
        left = candidate_flat[positions.to(candidate.device)].double().cpu()
        right = reference_flat[positions.to(reference.device)].double().cpu()
        left_finite, right_finite = torch.isfinite(left), torch.isfinite(right)
        sampled_matching = (
            (torch.isnan(left) & torch.isnan(right))
            | (torch.isposinf(left) & torch.isposinf(right))
            | (torch.isneginf(left) & torch.isneginf(right))
        )
        delta = torch.where(
            left_finite & right_finite, left - right,
            torch.where(sampled_matching, torch.zeros_like(left), torch.full_like(left, float("nan"))),
        )
        sketch = {
            "schema_version": DIRECTIONAL_SKETCH_SCHEMA_VERSION,
            "selection_rule": "EVENLY_SPACED_FLAT_POSITIONS_FIXED_BEFORE_READING_VALUES",
            "sample_size": int(positions.numel()),
            "tensor_numel": numel,
            "flat_coordinate_indices": positions.tolist(),
            "signed_delta_values": delta.tolist(),
            "all_sampled_values_finite": bool(torch.isfinite(delta).all()),
            "candidate_values_used_to_select_coordinates": False,
        }
        if retain_sampled_values:
            sketch["candidate_values"] = left.tolist()
            sketch["reference_values"] = right.tolist()
    else:
        sketch = {
            "schema_version": DIRECTIONAL_SKETCH_SCHEMA_VERSION,
            "selection_rule": "EMPTY_ENDPOINT", "sample_size": 0,
            "tensor_numel": 0, "flat_coordinate_indices": [],
            "candidate_values": [], "reference_values": [], "signed_delta_values": [],
            "all_sampled_values_finite": True,
            "candidate_values_used_to_select_coordinates": False,
        }
    nonzero = matching_nan = matching_posinf = matching_neginf = mismatch = 0
    signed_sum = squared_sum = max_abs = reference_squared_sum = 0.0
    candidate_all_finite = reference_all_finite = True
    for start in range(0, numel, metric_chunk_elements):
        stop = min(start + metric_chunk_elements, numel)
        left = candidate_flat[start:stop].to(torch.float32, copy=True)
        right = reference_flat[start:stop].to(torch.float32, copy=True)
        left_finite, right_finite = torch.isfinite(left), torch.isfinite(right)
        left_all_finite = bool(left_finite.all().cpu())
        right_all_finite = bool(right_finite.all().cpu())
        candidate_all_finite &= left_all_finite
        reference_all_finite &= right_all_finite
        # Natural model endpoints overwhelmingly take this branch.  Avoid
        # constructing and reducing four additional nonfinite masks when both
        # operands are already known finite; the arithmetic reductions and
        # their order remain identical to the general path below.
        if left_all_finite and right_all_finite:
            reference_squared_sum += float(
                right.square().sum(dtype=torch.float64).cpu()
            )
            left.sub_(right)
            nonzero += int(torch.count_nonzero(left).cpu())
            signed_sum += float(left.sum(dtype=torch.float64).cpu())
            squared_sum += float(left.square().sum(dtype=torch.float64).cpu())
            if left.numel():
                max_abs = max(max_abs, float(left.abs().max().cpu()))
            continue
        both_nonfinite = ~left_finite & ~right_finite
        same_nan = both_nonfinite & torch.isnan(left) & torch.isnan(right)
        same_pos = both_nonfinite & torch.isposinf(left) & torch.isposinf(right)
        same_neg = both_nonfinite & torch.isneginf(left) & torch.isneginf(right)
        same_nonfinite = same_nan | same_pos | same_neg
        mismatch_mask = (~left_finite | ~right_finite) & ~same_nonfinite
        matching_nan += int(same_nan.sum().cpu())
        matching_posinf += int(same_pos.sum().cpu())
        matching_neginf += int(same_neg.sum().cpu())
        mismatch += int(mismatch_mask.sum().cpu())
        left.masked_fill_(~left_finite, 0.0)
        right.masked_fill_(~right_finite, 0.0)
        reference_squared_sum += float(right.square().sum(dtype=torch.float64).cpu())
        left.sub_(right)
        nonzero += int(torch.count_nonzero(left).cpu())
        signed_sum += float(left.sum(dtype=torch.float64).cpu())
        squared_sum += float(left.square().sum(dtype=torch.float64).cpu())
        if left.numel():
            max_abs = max(max_abs, float(left.abs().max().cpu()))
    divisor = max(numel, 1)
    return {
        "schema_version": "kernel-analyzer.nonfinite-aware-streaming-error.v1",
        "exact": nonzero == 0 and mismatch == 0,
        "nonzero_elements": nonzero,
        "nonzero_fraction": nonzero / divisor,
        "signed_mean": signed_sum / divisor,
        "rms": (squared_sum / divisor) ** 0.5,
        "max_abs": max_abs,
        "reference_rms": (reference_squared_sum / divisor) ** 0.5,
        "candidate_finite": candidate_all_finite,
        "reference_finite": reference_all_finite,
        "original_candidate_all_finite": candidate_all_finite,
        "original_reference_all_finite": reference_all_finite,
        "full_value_scan": True,
        "metric_accumulation_dtype": "torch.float64",
        "metric_chunk_elements": metric_chunk_elements,
        "directional_error_sketch": sketch,
        "matching_nan": matching_nan,
        "matching_posinf": matching_posinf,
        "matching_neginf": matching_neginf,
        "nonfinite_mismatch": mismatch,
        "nonfinite_geometry_exact": mismatch == 0,
        "finite_value_metrics_use_zero_sentinel_for_all_nonfinite": True,
    }


class GeneratedFP32Observer:
    def __init__(
        self,
        *,
        modules: Iterable[Any],
        campaign_rows: Sequence[Mapping[str, Any]],
        sample_size: int = 64,
        metric_chunk_elements: int = 1_048_576,
        repair_targets: Mapping[str, Sequence[str]] | None = None,
        allow_unlisted_calls: bool = False,
        raw_capture_dir: Path | None = None,
    ) -> None:
        self.modules = list(modules)
        self.sample_size = sample_size
        self.metric_chunk_elements = metric_chunk_elements
        self.repair_targets = {
            str(region_id): {str(name) for name in names}
            for region_id, names in (repair_targets or {}).items()
        }
        self.allow_unlisted_calls = allow_unlisted_calls
        self.raw_capture_dir = Path(raw_capture_dir) if raw_capture_dir is not None else None
        if self.raw_capture_dir is not None:
            self.raw_capture_dir.mkdir(parents=True, exist_ok=True)
        by_symbol: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in campaign_rows:
            by_symbol[str(row["symbol"])].append(row)
        self.rows_by_symbol = dict(by_symbol)
        self.exact_callsite_mode = all(
            all(key in row for key in ("source_path", "source_line", "source_line_sha256"))
            for row in campaign_rows
        )
        self.rows_by_callsite: dict[tuple[str, str, int, str], Mapping[str, Any]] = {}
        self.runtime_to_captured_path: dict[str, str] = {}
        if self.exact_callsite_mode:
            for row in campaign_rows:
                key = (
                    str(row["symbol"]), str(row["source_path"]),
                    int(row["source_line"]), str(row["source_line_sha256"]),
                )
                if key in self.rows_by_callsite:
                    raise ValueError(f"ambiguous Triton static callsite: {key}")
                self.rows_by_callsite[key] = row
            captured_by_ordinal: dict[int, str] = {}
            for row in campaign_rows:
                match = re.search(r"(?:^|/)model__(\d+)_", str(row["source_path"]))
                if match is not None:
                    captured_by_ordinal[int(match.group(1))] = str(row["source_path"])
            wrappers = []
            for module in self.modules:
                source = Path(module.__file__).resolve()
                if re.search(
                    r"# AOT ID: \['\d+_(?:forward|backward|inference)'\]",
                    source.read_text(errors="ignore")[:512],
                ):
                    wrappers.append(module)
            if captured_by_ordinal and max(captured_by_ordinal) >= len(wrappers):
                raise RuntimeError("replay compiled fewer wrappers than the Triton campaign")
            self.runtime_to_captured_path = {
                str(Path(module.__file__).resolve()): captured_by_ordinal[ordinal]
                for ordinal, module in enumerate(wrappers) if ordinal in captured_by_ordinal
            }
        expected_programs: dict[str, str] = {}
        expected_program_variants: set[tuple[str, str]] = set()
        for symbol, rows in self.rows_by_symbol.items():
            hashes = {str(row["embedded_program_sha256"]) for row in rows}
            expected_program_variants.update((symbol, value) for value in hashes)
            if not self.exact_callsite_mode and len(hashes) != 1:
                raise ValueError(f"campaign symbol has multiple programs: {symbol}")
            if len(hashes) == 1:
                expected_programs[symbol] = next(iter(hashes))
        self.expected_programs = expected_programs
        self.expected_program_variants = expected_program_variants
        self.expected_ids = {
            str(row["region_id"]) for rows in by_symbol.values() for row in rows
        }
        if len(self.expected_ids) != sum(map(len, by_symbol.values())):
            raise ValueError("campaign region IDs are not unique")
        self.counts: dict[str, int] = {}
        self.callsite_counts: dict[tuple[str, str, int, str], int] = defaultdict(int)
        self.records: list[dict[str, Any]] = []
        self.restores: list[tuple[Any, bool, Any]] = []
        self.reference_kernels: dict[str, Any] = {}
        self.reference_metadata: dict[str, dict[str, Any]] = {}

    def validate_program_identity(self) -> None:
        """Fail closed unless warmed modules embed the frozen Triton sources."""

        if self.exact_callsite_mode:
            observed = set(collect_program_variants(self.modules))
            missing = self.expected_program_variants - observed
            if missing:
                raise RuntimeError(
                    f"warmed Triton program variants differ from frozen campaign: {sorted(missing)}"
                )
            return

        observed = {
            symbol: hashlib.sha256(source.encode()).hexdigest()
            for symbol, source in collect_programs(self.modules).items()
        }
        missing = set(self.expected_programs) - set(observed)
        changed = {
            symbol: {"expected": expected, "observed": observed.get(symbol)}
            for symbol, expected in self.expected_programs.items()
            if observed.get(symbol) != expected
        }
        if missing or changed:
            preview = dict(list(sorted(changed.items()))[:8])
            raise RuntimeError(
                "warmed Triton program differs from frozen campaign: "
                f"missing={sorted(missing)[:8]} changed={preview}"
            )

    def discover(self) -> list[tuple[str, Any]]:
        found, seen = [], set()
        for module in self.modules:
            for symbol, value in vars(module).items():
                if symbol not in self.rows_by_symbol or id(value) in seen:
                    continue
                if callable(getattr(value, "run", None)) and hasattr(value, "triton_meta"):
                    seen.add(id(value))
                    found.append((symbol, value))
        return sorted(found, key=lambda item: (item[0], id(item[1])))

    def __enter__(self) -> "GeneratedFP32Observer":
        self.validate_program_identity()
        kernels = self.discover()
        missing = set(self.rows_by_symbol) - {symbol for symbol, _ in kernels}
        if missing:
            raise RuntimeError(f"campaign symbols absent from warmed modules: {sorted(missing)}")
        if self.exact_callsite_mode:
            self.reference_kernels, self.reference_metadata = (
                compile_fp32_pointer_kernel_variants(
                    self.modules, self.expected_program_variants,
                )
            )
        else:
            self.reference_kernels, self.reference_metadata = compile_fp32_pointer_kernels(
                self.modules, expected_program_sha256=self.expected_programs,
            )
        for symbol, kernel in kernels:
            had_run = "run" in vars(kernel)
            previous = vars(kernel).get("run")
            original = kernel.run

            def wrapped(
                *args: Any,
                _symbol: str = symbol,
                _kernel: Any = kernel,
                _original: Callable[..., Any] = original,
                **kwargs: Any,
            ) -> Any:
                index = self.counts.get(_symbol, 0)
                self.counts[_symbol] = index + 1
                if self.exact_callsite_mode:
                    caller = sys._getframe(1)
                    filename = str(Path(caller.f_code.co_filename).resolve())
                    line = int(caller.f_lineno)
                    source = linecache.getline(filename, line).strip()
                    source_digest = hashlib.sha256(source.encode()).hexdigest()
                    captured_path = self.runtime_to_captured_path.get(filename)
                    key = (_symbol, captured_path, line, source_digest)
                    row = self.rows_by_callsite.get(key)
                    if row is None:
                        if self.allow_unlisted_calls:
                            return _original(*args, **kwargs)
                        raise RuntimeError(f"runtime Triton call outside exact campaign: {key}")
                    callsite_index = self.callsite_counts[key]
                    self.callsite_counts[key] += 1
                else:
                    rows = self.rows_by_symbol[_symbol]
                    if index >= len(rows):
                        raise RuntimeError(f"runtime invocation outside campaign: {_symbol}:{index}")
                    row = rows[index]
                    callsite_index = index
                program_digest = str(row["embedded_program_sha256"])
                reference = self.reference_kernels[
                    (_symbol, program_digest) if self.exact_callsite_mode else _symbol
                ]
                pointer_names = [
                    name for name, annotation in runtime_signature(_kernel)
                    if str(annotation).startswith("*")
                ]
                tensor_args = [value for value in args if isinstance(value, torch.Tensor)]
                if len(pointer_names) != len(tensor_args):
                    raise RuntimeError(f"pointer signature mismatch: {_symbol}")
                promoted_args, promoted_tensors = promoted_pointer_arguments(args)
                validate_typed_triton_reference_abi(reference, promoted_args)
                promoted_pointers = dict(zip(pointer_names, promoted_tensors))
                result = _original(*args, **kwargs)
                reference.run(*promoted_args, **kwargs)
                output_names = [str(name) for name in row["output_names"]]
                candidate_pointers = dict(zip(pointer_names, tensor_args))
                missing_outputs = set(output_names) - set(candidate_pointers)
                if missing_outputs:
                    raise RuntimeError(f"runtime outputs absent: {_symbol}:{sorted(missing_outputs)}")
                metrics = {
                    name: nonfinite_aware_metrics(
                        candidate_pointers[name], promoted_pointers[name],
                        sample_size=self.sample_size,
                        metric_chunk_elements=self.metric_chunk_elements,
                        retain_sampled_values=False,
                    )
                    for name in output_names
                }
                # The promoted reference can differ in FP32 while becoming
                # bit-identical after it is cast back to the endpoint dtype.
                # Such a row has no executable same-dtype repair at this
                # boundary and must not be confused with a failed parameter
                # carrier binding.  Record both comparisons before mutating
                # the candidate output.
                same_dtype_metrics = {
                    name: nonfinite_aware_metrics(
                        candidate_pointers[name],
                        promoted_pointers[name].to(dtype=candidate_pointers[name].dtype),
                        sample_size=self.sample_size,
                        metric_chunk_elements=self.metric_chunk_elements,
                        retain_sampled_values=False,
                    )
                    for name in output_names
                }
                raw_capture = {}
                if self.raw_capture_dir is not None:
                    capture_id = f"{len(self.records):06d}_{row['region_id']}"
                    capture_dir = self.raw_capture_dir / capture_id
                    capture_dir.mkdir(parents=True, exist_ok=True)
                    for name in output_names:
                        candidate_path = capture_dir / f"{name}.candidate.pt"
                        reference_path = capture_dir / f"{name}.reference.pt"
                        torch.save(candidate_pointers[name].detach().cpu(), candidate_path)
                        torch.save(promoted_pointers[name].detach().cpu(), reference_path)
                        raw_capture[name] = {
                            "candidate": str(candidate_path),
                            "reference": str(reference_path),
                            "candidate_dtype": str(candidate_pointers[name].dtype),
                            "reference_dtype": str(promoted_pointers[name].dtype),
                            "shape": list(candidate_pointers[name].shape),
                        }
                repaired = []
                target_names = self.repair_targets.get(str(row["region_id"]), set())
                unknown_targets = target_names - set(output_names)
                if unknown_targets:
                    raise RuntimeError(
                        f"repair endpoint absent from target region: {sorted(unknown_targets)}"
                    )
                with torch.no_grad():
                    for name in sorted(target_names):
                        candidate_pointers[name].copy_(promoted_pointers[name])
                        repaired.append(name)
                self.records.append({
                    "region_id": row["region_id"],
                    "phase": row["phase"],
                    "symbol": _symbol,
                    "invocation_index": index,
                    "runtime_invocation_ordinal": len(self.records),
                    "callsite_execution_ordinal": callsite_index,
                    "reference_role": "PRECISION_ONLY_GENERATED_PROGRAM_COUNTERFACTUAL",
                    "reference_abi": "INDEPENDENT_RECOMPILED_FLOATING_POINTER_FP32",
                    "runtime_pointer_contracts": {
                        name: tensor_runtime_contract(candidate_pointers[name])
                        for name in pointer_names
                    },
                    "typed_reference_program_sha256": self.reference_metadata[
                        (f"{_symbol}@{program_digest}" if self.exact_callsite_mode else _symbol)
                    ][
                        "typed_program_sha256"
                    ],
                    "endpoint_metrics": metrics,
                    "same_dtype_repair_metrics": same_dtype_metrics,
                    "raw_capture": raw_capture,
                    "repaired_endpoints": repaired,
                })
                return result

            self.restores.append((kernel, had_run, previous))
            kernel.run = wrapped
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for kernel, had_run, previous in reversed(self.restores):
            if had_run:
                kernel.run = previous
            else:
                delattr(kernel, "run")
        self.restores.clear()
        self.reference_kernels.clear()

    def summary(self) -> dict[str, Any]:
        observed = {row["region_id"] for row in self.records}
        nonfinite_exact = all(
            endpoint["nonfinite_geometry_exact"]
            for row in self.records for endpoint in row["endpoint_metrics"].values()
        )
        missing = sorted(self.expected_ids - observed)
        complete = self.exact_callsite_mode or (
            observed == self.expected_ids and len(self.records) == len(self.expected_ids)
        )
        return {
            "status": "COMPLETE_ALL_TRITON_FP32_REPLAY" if complete else "UNRESOLVED",
            "denominator": {
                "expected_triton_invocations": len(self.expected_ids),
                "observed_triton_invocations": len(self.records),
                "static_calls_executed_in_measured_step": len(observed),
                "static_calls_not_executed_in_measured_step": len(missing),
                "nonfinite_geometry_exact_records": sum(
                    all(m["nonfinite_geometry_exact"] for m in r["endpoint_metrics"].values())
                    for r in self.records
                ),
                "records_with_nonfinite_mismatch": sum(
                    any(not m["nonfinite_geometry_exact"] for m in r["endpoint_metrics"].values())
                    for r in self.records
                ),
            },
            "nonfinite_geometry_exact": nonfinite_exact,
            "typed_reference_programs": self.reference_metadata,
            "missing_region_ids": missing,
            "records": self.records,
        }
