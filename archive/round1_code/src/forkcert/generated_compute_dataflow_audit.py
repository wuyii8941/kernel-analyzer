"""Exact pointer dataflow for every frozen generated compute invocation.

This audit deliberately does not infer semantic identity from a generated
symbol, source-node name, tensor shape, module path, or launch ordinal.  It
parses the actual ``Runner.call`` AST and the embedded Triton program:

* Triton pointer arguments are classified by actual ``tl.load``/``tl.store``
  use in that program;
* external-library outputs come from the explicit ``out=`` argument;
* direct ATen mutation is represented as an in-place read/write boundary;
* producer/consumer edges are created only by the same concrete Python tensor
  variable appearing at the two callsites.

The result is an implementation dataflow witness.  It is necessary for an
exact forward-plus-VJP binding, but does not grant mathematical or numerical
correctness by itself.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from forkcert.generated_runtime_call_completeness_audit import (
    _function_name,
    _runner_call,
    _source_segment,
    validate_generated_runtime_call_completeness_audit,
)
from forkcert.inductor_direct_runtime_call_inventory import (
    validate_inductor_direct_runtime_call_inventory,
)


SCHEMA_VERSION = "forkcert.generated-compute-dataflow-audit.v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def _phase(path: Path) -> str:
    name = path.parent.name.lower()
    if "forward" in name:
        return "FORWARD"
    if "backward" in name:
        return "BACKWARD"
    raise ValueError(f"generated source phase is unresolved: {path}")


def _assigned_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        result: set[str] = set()
        for child in target.elts:
            result.update(_assigned_names(child))
        return result
    return set()


def _runner_abi_names(method: ast.FunctionDef) -> set[str]:
    rows = []
    for statement in method.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.value, ast.Name)
            and statement.value.id == "args"
        ):
            rows.append(_assigned_names(statement.targets[0]))
    if len(rows) != 1 or not rows[0]:
        raise ValueError("generated Runner.call ABI unpack is not unique")
    return rows[0]


def _names(node: ast.AST) -> set[str]:
    return {
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)
    }


def _is_statically_empty_allocation(call: ast.Call) -> bool:
    """Recognize generated empty allocations whose shape has zero elements."""

    try:
        function = _function_name(call.func)
    except ValueError:
        return False
    if function != "empty_strided_cuda" or not call.args:
        return False
    shape = call.args[0]
    return isinstance(shape, (ast.Tuple, ast.List)) and any(
        isinstance(element, ast.Constant)
        and isinstance(element.value, int)
        and element.value == 0
        for element in shape.elts
    )


def _triton_access_modes(
    tree: ast.Module,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for statement in tree.body:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
            or not isinstance(statement.value, ast.Call)
        ):
            continue
        function = _function_name(statement.value.func)
        if function != "async_compile.triton":
            continue
        symbol = statement.targets[0].id
        if (
            len(statement.value.args) < 2
            or not isinstance(statement.value.args[1], ast.Constant)
            or not isinstance(statement.value.args[1].value, str)
        ):
            raise ValueError(f"{symbol} embedded Triton source is absent")
        embedded = ast.parse(statement.value.args[1].value)
        definitions = [
            node
            for node in embedded.body
            if isinstance(node, ast.FunctionDef) and node.name == symbol
        ]
        if len(definitions) != 1:
            raise ValueError(f"{symbol} Triton definition is not unique")
        definition = definitions[0]
        parameters = [argument.arg for argument in definition.args.args]
        pointer_parameter_set = {
            name for name in parameters if "ptr" in name
        }
        load_lines: dict[str, list[int]] = defaultdict(list)
        store_lines: dict[str, list[int]] = defaultdict(list)
        atomic_pointers: set[str] = set()
        for call in ast.walk(definition):
            if not isinstance(call, ast.Call) or not call.args:
                continue
            try:
                called = _function_name(call.func)
            except ValueError:
                continue
            if called not in {"tl.load", "tl.store", "tl.atomic_add"}:
                continue
            pointer_names = _names(call.args[0]) & pointer_parameter_set
            if len(pointer_names) != 1:
                raise ValueError(
                    f"{symbol} {called} pointer identity is not unique"
                )
            pointer = next(iter(pointer_names))
            if called == "tl.load":
                load_lines[pointer].append(int(call.lineno))
            elif called == "tl.store":
                store_lines[pointer].append(int(call.lineno))
            else:
                # Atomic add is a true read-modify-write endpoint even when
                # there is no separate tl.load/tl.store in the source.
                load_lines[pointer].append(int(call.lineno))
                store_lines[pointer].append(int(call.lineno))
                atomic_pointers.add(pointer)
        loaded = set(load_lines)
        stored = set(store_lines)
        if not stored:
            raise ValueError(f"{symbol} has no stored pointer endpoint")
        externally_loaded = {
            pointer
            for pointer in loaded
            if pointer not in stored
            or pointer in atomic_pointers
            or min(load_lines[pointer]) < min(store_lines[pointer])
        }
        internally_reloaded_outputs = loaded - externally_loaded
        pointer_parameters = loaded | stored
        result[symbol] = {
            "formal_parameters": parameters,
            "loaded_pointer_parameters": sorted(externally_loaded),
            "all_syntactically_loaded_pointer_parameters": sorted(loaded),
            "stored_pointer_parameters": sorted(stored),
            "read_write_pointer_parameters": sorted(
                externally_loaded & stored
            ),
            "atomic_read_write_pointer_parameters": sorted(atomic_pointers),
            "internally_reloaded_output_pointer_parameters": sorted(
                internally_reloaded_outputs
            ),
            "pointer_parameters": sorted(pointer_parameters),
            "embedded_program_sha256": hashlib.sha256(
                statement.value.args[1].value.encode()
            ).hexdigest(),
        }
    return result


def _expected_compute(
    inventory: Mapping[str, Any],
    direct: Mapping[str, Any],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    result = {}
    for row in [*inventory["regions"], *direct["rows"]]:
        function = (
            f"{row['symbol']}.run"
            if row["kind"] == "TRITON"
            else (
                f"extern_kernels.{row['symbol']}"
                if row["kind"] == "EXTERN"
                else row.get("runtime_function", f"aten.{row['symbol']}")
            )
        )
        key = (
            str(row["source_path"]),
            int(row["source_line"]),
            function,
        )
        if key in result:
            raise ValueError("generated compute callsite is duplicated")
        result[key] = row
    return result


def _argument_tensor_names(
    expression: ast.AST,
    *,
    tensor_names: set[str],
) -> list[str]:
    return sorted(_names(expression) & tensor_names)


def _single_tensor_name(
    expression: ast.AST,
    *,
    tensor_names: set[str],
    context: str,
) -> str:
    names = _argument_tensor_names(
        expression, tensor_names=tensor_names
    )
    if len(names) != 1:
        raise ValueError(f"{context} tensor identity is not unique: {names}")
    return names[0]


def _storage_root(
    name: str,
    aliases: Mapping[str, str],
) -> str:
    visited = set()
    current = name
    while current in aliases:
        if current in visited:
            raise ValueError("generated tensor alias cycle")
        visited.add(current)
        current = aliases[current]
    return current


def _call_boundary(
    *,
    node: ast.Call,
    function: str,
    kind: str,
    symbol: str,
    tensor_names: set[str],
    triton_modes: Mapping[str, Mapping[str, Any]],
    assigned_output_names: list[str],
) -> tuple[list[str], list[str], dict[str, Any]]:
    if kind == "TRITON":
        mode = triton_modes.get(symbol)
        if mode is None:
            raise ValueError(f"{symbol} Triton access modes are absent")
        parameters = list(mode["formal_parameters"])
        if len(node.args) > len(parameters):
            raise ValueError(f"{symbol} has too many positional arguments")
        actual_by_formal = dict(zip(parameters, node.args))
        pointer_actuals = set(mode["pointer_parameters"])
        if not pointer_actuals <= set(actual_by_formal):
            raise ValueError(f"{symbol} pointer actual is absent")
        loaded = []
        stored = []
        binding = {}
        for formal in mode["pointer_parameters"]:
            actual = actual_by_formal[formal]
            tensor = _single_tensor_name(
                actual,
                tensor_names=tensor_names,
                context=f"{symbol}.{formal}",
            )
            binding[formal] = {
                "tensor_variable": tensor,
                "actual_expression": ast.unparse(actual),
                "loaded": formal in mode["loaded_pointer_parameters"],
                "stored": formal in mode["stored_pointer_parameters"],
            }
            if formal in mode["loaded_pointer_parameters"]:
                loaded.append(tensor)
            if formal in mode["stored_pointer_parameters"]:
                stored.append(tensor)
        return sorted(set(loaded)), sorted(set(stored)), {
            "boundary_source": "EMBEDDED_TRITON_LOAD_STORE_AST",
            "formal_to_actual_pointer_binding": binding,
            "embedded_program_sha256": mode["embedded_program_sha256"],
        }

    if kind == "EXTERN":
        output_keywords = [
            keyword
            for keyword in node.keywords
            if keyword.arg == "out"
        ]
        if len(output_keywords) == 1:
            output = _single_tensor_name(
                output_keywords[0].value,
                tensor_names=tensor_names,
                context=f"{function}.out",
            )
            boundary_source = "EXPLICIT_EXTERNAL_OUT_KEYWORD_AST"
            output_expression = ast.unparse(output_keywords[0].value)
        elif not output_keywords and len(assigned_output_names) == 1:
            output = assigned_output_names[0]
            boundary_source = "EXPLICIT_EXTERNAL_ASSIGNED_RETURN_AST"
            output_expression = output
        else:
            raise ValueError(f"{function} output identity is not unique")
        inputs = []
        for position, argument in enumerate(node.args):
            names = _argument_tensor_names(
                argument, tensor_names=tensor_names
            )
            if len(names) != 1:
                raise ValueError(
                    f"{function}.arg{position} tensor identity differs: {names}"
                )
            inputs.extend(names)
        return sorted(set(inputs)), [output], {
            "boundary_source": boundary_source,
            "output_expression": output_expression,
        }

    if kind == "DIRECT_ATEN":
        if symbol != "index_put_" or len(node.args) != 4:
            raise ValueError(
                f"unsupported direct generated mutation: {symbol}"
            )
        target = _single_tensor_name(
            node.args[0],
            tensor_names=tensor_names,
            context=f"{function}.mutated_target",
        )
        inputs = []
        for argument in node.args[:3]:
            inputs.extend(
                _argument_tensor_names(
                    argument, tensor_names=tensor_names
                )
            )
        return sorted(set(inputs)), [target], {
            "boundary_source": "DIRECT_ATEN_INPLACE_SCHEMA_AND_CALL_AST",
            "mutated_target": target,
            "accumulate_expression": ast.unparse(node.args[3]),
        }

    if kind == "DIRECT_TENSOR_METHOD":
        if symbol != "copy_" or len(node.args) < 1 or not isinstance(node.func, ast.Attribute):
            raise ValueError(f"unsupported direct tensor method: {function}")
        target = _single_tensor_name(
            node.func.value, tensor_names=tensor_names,
            context=f"{function}.mutated_target",
        )
        inputs = _argument_tensor_names(node.args[0], tensor_names=tensor_names)
        if len(inputs) != 1:
            raise ValueError(f"{function} source tensor identity is not unique")
        return inputs, [target], {
            "boundary_source": "DIRECT_TENSOR_COPY_METHOD_AST",
            "mutated_target": target,
            "source_expression": ast.unparse(node.args[0]),
        }

    if kind == "DIRECT_TORCH_OP":
        if not assigned_output_names:
            raise ValueError(f"{function} assigned return is absent")
        inputs = []
        for argument in node.args:
            inputs.extend(_argument_tensor_names(argument, tensor_names=tensor_names))
        return sorted(set(inputs)), assigned_output_names, {
            "boundary_source": "DIRECT_TORCH_OP_ASSIGNED_RETURN_AST",
            "assigned_return_variables": assigned_output_names,
            "overload": function,
        }

    raise ValueError(f"unsupported generated compute kind: {kind}")


def build_generated_compute_dataflow_audit(
    *,
    trace_dir: Path,
    base_generated_inventory: Mapping[str, Any],
    direct_runtime_inventory: Mapping[str, Any],
    runtime_call_audit: Mapping[str, Any],
) -> dict[str, Any]:
    trace_dir = trace_dir.resolve()
    inventory = base_generated_inventory.get(
        "inventory", base_generated_inventory
    )
    if inventory.get("status") != "COMPLETE_GENERATED_REGION_SCHEDULE":
        raise ValueError("base generated inventory is incomplete")
    validate_inductor_direct_runtime_call_inventory(
        direct_runtime_inventory
    )
    validate_generated_runtime_call_completeness_audit(runtime_call_audit)
    if (
        direct_runtime_inventory["bindings"]["base_inventory_sha256"]
        != inventory["inventory_sha256"]
        or runtime_call_audit["bindings"][
            "base_generated_inventory_sha256"
        ]
        != inventory["inventory_sha256"]
    ):
        raise ValueError("generated inventory bindings differ")

    expected = _expected_compute(inventory, direct_runtime_inventory)
    rows = []
    producer_by_tensor: dict[tuple[str, str], str] = {}
    transitive_boundary_by_region: dict[str, set[str]] = {}
    transitive_region_by_region: dict[str, set[str]] = {}
    runner_abi: dict[str, list[str]] = {}
    observed: set[tuple[str, int, str]] = set()
    all_triton_symbols: set[tuple[str, str]] = set()

    for path in sorted(trace_dir.rglob("output_code.py")):
        relative = str(path.relative_to(trace_dir))
        phase = _phase(path)
        source = path.read_text(encoding="utf-8", errors="strict")
        lines = source.splitlines()
        lines_with_endings = source.splitlines(keepends=True)
        tree = ast.parse(source, filename=str(path))
        method = _runner_call(tree)
        abi_names = _runner_abi_names(method)
        runner_abi[phase] = sorted(abi_names)
        allocation_names: set[str] = set()
        zero_allocation_names: set[str] = set()
        assigned_call_outputs: dict[int, list[str]] = {}
        for node in ast.walk(method):
            if not isinstance(node, ast.Assign):
                continue
            if (
                (
                    isinstance(node.value, ast.Call)
                    and _function_name(node.value.func)
                    in {"empty_strided_cuda", "reinterpret_tensor"}
                )
                or isinstance(node.value, ast.Name)
                or (
                    isinstance(node.value, ast.Subscript)
                    and isinstance(node.value.value, ast.Name)
                )
            ):
                for target in node.targets:
                    allocation_names.update(_assigned_names(target))
                    if isinstance(node.value, ast.Call) and _is_statically_empty_allocation(node.value):
                        zero_allocation_names.update(_assigned_names(target))
            if isinstance(node.value, ast.Call):
                try:
                    assigned_function = _function_name(node.value.func)
                except ValueError:
                    assigned_function = ""
                if assigned_function.startswith("extern_kernels."):
                    names = sorted(set().union(*(_assigned_names(target) for target in node.targets)))
                    allocation_names.update(names)
                    assigned_call_outputs[id(node.value)] = names
                elif assigned_function.startswith("torch.ops.aten."):
                    names = sorted(set().union(*(_assigned_names(target) for target in node.targets)))
                    allocation_names.update(names)
                    assigned_call_outputs[id(node.value)] = names
        tensor_names = abi_names | allocation_names
        aliases: dict[str, str] = {}
        for node in ast.walk(method):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            if (
                isinstance(node.value, ast.Call)
                and _function_name(node.value.func) == "reinterpret_tensor"
                and node.value.args
            ):
                source_expression = node.value.args[0]
            elif (
                isinstance(node.value, ast.Name)
                and node.value.id != "args"
            ):
                source_expression = node.value
            elif (
                isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Name)
            ):
                source_expression = node.value.value
            else:
                continue
            targets = _assigned_names(node.targets[0])
            sources = _names(source_expression) & tensor_names
            if len(targets) != 1 or len(sources) != 1:
                raise ValueError(
                    f"generated tensor alias is ambiguous: {ast.unparse(node)} "
                    f"targets={sorted(targets)} sources={sorted(sources)}"
                )
            aliases[next(iter(targets))] = next(iter(sources))
        triton_modes = _triton_access_modes(tree)
        all_triton_symbols.update(
            (phase, symbol) for symbol in triton_modes
        )
        calls = sorted(
            (
                node
                for node in ast.walk(method)
                if isinstance(node, ast.Call)
            ),
            key=lambda node: (
                int(node.lineno),
                int(node.col_offset),
                int(getattr(node, "end_lineno", node.lineno)),
            ),
        )
        for node in calls:
            function = _function_name(node.func)
            key = (relative, int(node.lineno), function)
            expected_row = expected.get(key)
            if expected_row is None:
                continue
            observed.add(key)
            region_id = str(expected_row["region_id"])
            kind = str(expected_row["kind"])
            symbol = str(expected_row["symbol"])
            inputs, outputs, boundary_witness = _call_boundary(
                node=node,
                function=function,
                kind=kind,
                symbol=symbol,
                tensor_names=tensor_names,
                triton_modes=triton_modes,
                assigned_output_names=assigned_call_outputs.get(id(node), []),
            )
            direct_edges = []
            boundary_inputs = []
            zero_allocation_inputs = []
            transitive_boundaries: set[str] = set()
            transitive_regions: set[str] = set()
            for tensor in inputs:
                storage_root = _storage_root(tensor, aliases)
                producer = producer_by_tensor.get((phase, storage_root))
                if producer is None:
                    if storage_root in zero_allocation_names:
                        zero_allocation_inputs.append(storage_root)
                        transitive_boundaries.add(storage_root)
                    elif storage_root not in abi_names:
                        raise ValueError(
                            f"{region_id} input {tensor} (storage "
                            f"{storage_root}) has no producer or ABI"
                        )
                    boundary_inputs.append(storage_root)
                    transitive_boundaries.add(storage_root)
                else:
                    direct_edges.append(
                        {
                            "tensor_variable": tensor,
                            "storage_root_variable": storage_root,
                            "producer_region_id": producer,
                        }
                    )
                    transitive_regions.add(producer)
                    transitive_regions.update(
                        transitive_region_by_region[producer]
                    )
                    transitive_boundaries.update(
                        transitive_boundary_by_region[producer]
                    )
            previous_writers = []
            for tensor in outputs:
                storage_root = _storage_root(tensor, aliases)
                previous = producer_by_tensor.get((phase, storage_root))
                if previous is not None:
                    previous_writers.append(
                        {
                            "tensor_variable": tensor,
                            "storage_root_variable": storage_root,
                            "previous_writer_region_id": previous,
                            "also_loaded_by_this_invocation": tensor in inputs,
                        }
                    )
                producer_by_tensor[(phase, storage_root)] = region_id
            transitive_boundary_by_region[region_id] = transitive_boundaries
            transitive_region_by_region[region_id] = transitive_regions
            expression = _source_segment(lines_with_endings, node)
            source_line = lines[int(node.lineno) - 1].strip()
            rows.append(
                {
                    "region_id": region_id,
                    "phase": phase,
                    "kind": kind,
                    "symbol": symbol,
                    "source_path": relative,
                    "source_line": int(node.lineno),
                    "source_line_sha256": hashlib.sha256(
                        source_line.encode()
                    ).hexdigest(),
                    "call_expression": expression,
                    "input_tensor_variables": inputs,
                    "input_storage_root_variables": {
                        tensor: _storage_root(tensor, aliases)
                        for tensor in inputs
                    },
                    "output_tensor_variables": outputs,
                    "output_storage_root_variables": {
                        tensor: _storage_root(tensor, aliases)
                        for tensor in outputs
                    },
                    "direct_boundary_input_variables": sorted(
                        boundary_inputs
                    ),
                    "direct_zero_numel_allocation_input_variables": sorted(
                        zero_allocation_inputs
                    ),
                    "direct_producer_edges": sorted(
                        direct_edges,
                        key=lambda row: (
                            row["tensor_variable"],
                            row["producer_region_id"],
                        ),
                    ),
                    "previous_storage_writers": sorted(
                        previous_writers,
                        key=lambda row: (
                            row["tensor_variable"],
                            row["previous_writer_region_id"],
                        ),
                    ),
                    "transitive_runner_abi_input_variables": sorted(
                        transitive_boundaries
                    ),
                    "transitive_upstream_compute_region_ids": sorted(
                        transitive_regions
                    ),
                    "boundary_witness": boundary_witness,
                    "binding_status": "EXACT_GENERATED_POINTER_DATAFLOW",
                    "forward_vjp_semantic_identity_granted": False,
                    "candidate_correctness_granted": False,
                }
            )

    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        raise ValueError(f"generated compute callsites unobserved: {missing[:3]}")
    region_ids = [str(row["region_id"]) for row in rows]
    expected_compute_invocations = len(expected)
    if (
        len(region_ids) != expected_compute_invocations
        or len(region_ids) != len(set(region_ids))
    ):
        raise ValueError("generated compute dataflow denominator differs")
    successor_edges: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for edge in row["direct_producer_edges"]:
            successor_edges[edge["producer_region_id"]].append(
                {
                    "tensor_variable": edge["tensor_variable"],
                    "consumer_region_id": row["region_id"],
                }
            )
    for row in rows:
        row["direct_consumer_edges"] = sorted(
            successor_edges[row["region_id"]],
            key=lambda edge: (
                edge["tensor_variable"],
                edge["consumer_region_id"],
            ),
        )

    phase_counts = Counter(row["phase"] for row in rows)
    kind_counts = Counter(row["kind"] for row in rows)
    edge_count = sum(len(row["direct_producer_edges"]) for row in rows)
    gates = {
        "all_expected_compute_invocations_retained": (
            len(rows) == expected_compute_invocations
        ),
        "all_triton_boundaries_from_embedded_load_store_ast": all(
            row["boundary_witness"]["boundary_source"]
            == "EMBEDDED_TRITON_LOAD_STORE_AST"
            for row in rows
            if row["kind"] == "TRITON"
        ),
        "all_external_outputs_from_explicit_out_ast": all(
            row["boundary_witness"]["boundary_source"]
            in {"EXPLICIT_EXTERNAL_OUT_KEYWORD_AST", "EXPLICIT_EXTERNAL_ASSIGNED_RETURN_AST"}
            for row in rows
            if row["kind"] == "EXTERN"
        ),
        "all_direct_mutations_explicit": all(
            row["boundary_witness"]["boundary_source"]
            in {
                "DIRECT_ATEN_INPLACE_SCHEMA_AND_CALL_AST",
                "DIRECT_TENSOR_COPY_METHOD_AST",
                "DIRECT_TORCH_OP_ASSIGNED_RETURN_AST",
            }
            for row in rows
            if row["kind"] in {"DIRECT_ATEN", "DIRECT_TENSOR_METHOD", "DIRECT_TORCH_OP"}
        ),
        # Retain the v1 gate name for artifact compatibility.  A statically
        # zero-numel allocation is an exact structural boundary: it carries
        # no values and therefore needs no data producer.
        "every_input_has_exact_producer_or_runner_abi_boundary": True,
        "symbol_name_shape_module_and_ordinal_pairing_used": False,
        "forward_vjp_semantic_identity_granted": False,
        "candidate_correctness_granted": False,
        "property_generalization_allowed": False,
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_GENERATED_COMPUTE_POINTER_DATAFLOW",
        "trace_dir": str(trace_dir),
        "bindings": {
            "base_generated_inventory_sha256": str(
                inventory["inventory_sha256"]
            ),
            "direct_runtime_inventory_sha256": str(
                direct_runtime_inventory["inventory_sha256"]
            ),
            "runtime_call_audit_sha256": str(
                runtime_call_audit["audit_sha256"]
            ),
        },
        "denominator": {
            "compute_invocations": len(rows),
            "expected_compute_invocations": expected_compute_invocations,
            "kind_counts": dict(sorted(kind_counts.items())),
            "phase_counts": dict(sorted(phase_counts.items())),
            "direct_tensor_producer_edges": edge_count,
            "unique_phase_triton_programs": len(all_triton_symbols),
            "runner_abi_input_counts": {
                phase: len(names)
                for phase, names in sorted(runner_abi.items())
            },
        },
        "runner_abi_input_variables": runner_abi,
        "gates": gates,
        "rows": rows,
        "claim_boundary": {
            "supported": (
                "exact per-invocation generated input/output pointer identity "
                "and producer/consumer dataflow for all frozen compute calls"
            ),
            "not_supported": [
                "AOT or eager semantic identity from pointer dataflow alone",
                "candidate numerical correctness",
                "cross-configuration identity",
                "property generalization",
            ],
        },
    }
    payload["audit_sha256"] = _digest(payload)
    return payload


def validate_generated_compute_dataflow_audit(
    artifact: Mapping[str, Any],
) -> None:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("generated compute dataflow schema differs")
    if artifact.get("status") != (
        "COMPLETE_GENERATED_COMPUTE_POINTER_DATAFLOW"
    ):
        raise ValueError("generated compute dataflow audit is incomplete")
    payload = dict(artifact)
    observed = str(payload.pop("audit_sha256", ""))
    if observed != _digest(payload):
        raise ValueError("generated compute dataflow audit digest differs")
    expected = artifact["denominator"].get(
        "expected_compute_invocations", 1447
    )
    if artifact["denominator"]["compute_invocations"] != expected:
        raise ValueError("generated compute dataflow denominator differs")
    retained_gate = (
        "all_expected_compute_invocations_retained"
        if "all_expected_compute_invocations_retained" in artifact["gates"]
        else "all_1447_compute_invocations_retained"
    )
    true_gates = {
        retained_gate,
        "all_triton_boundaries_from_embedded_load_store_ast",
        "all_external_outputs_from_explicit_out_ast",
        "all_direct_mutations_explicit",
        "every_input_has_exact_producer_or_runner_abi_boundary",
    }
    if not all(artifact["gates"][key] for key in true_gates):
        failed = sorted(key for key in true_gates if not artifact["gates"][key])
        raise ValueError(f"generated compute dataflow evidence gate failed: {failed}")
    false_gates = set(artifact["gates"]) - true_gates
    if any(artifact["gates"][key] for key in false_gates):
        raise ValueError("generated compute dataflow audit overclaims")
