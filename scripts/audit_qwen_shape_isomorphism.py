#!/usr/bin/env python3
"""Prove seq64/128/256 Qwen eager and AOT programs are shape-parametric isomorphs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OLD_STRONG = ROOT / "results/coverage/qwen_full_invocation_inventory_strong.json.gz"
DOSSIER = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/atomic_forward_vjp_mathematical_dossier_v4.json"


def load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_parametric(left: Any, middle: Any, right: Any, path: str) -> tuple[int, int]:
    """Accept constants, device renaming, and affine/quadratic sequence functions."""
    if isinstance(left, dict) and isinstance(middle, dict) and isinstance(right, dict):
        if left.keys() != middle.keys() or left.keys() != right.keys():
            raise RuntimeError(f"mapping keys changed at {path}")
        constant = parametric = 0
        for key in left:
            c, p = validate_parametric(left[key], middle[key], right[key], f"{path}.{key}")
            constant += c; parametric += p
        return constant, parametric
    if isinstance(left, (list, tuple)) and isinstance(middle, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(middle) or len(left) != len(right):
            raise RuntimeError(f"sequence length changed at {path}")
        constant = parametric = 0
        for index, values in enumerate(zip(left, middle, right)):
            c, p = validate_parametric(*values, f"{path}[{index}]")
            constant += c; parametric += p
        return constant, parametric
    if left == middle == right:
        return 1, 0
    if all(isinstance(value, str) and value.startswith("cuda:") for value in (left, middle, right)):
        return 0, 1
    if all(isinstance(value, str) and re.fullmatch(r"-?\d+", value) for value in (left, middle, right)):
        return validate_parametric(
            int(left), int(middle), int(right), path
        )
    if all(isinstance(value, int) and not isinstance(value, bool) for value in (left, middle, right)):
        first = middle - left
        second = right - middle
        # For sequence lengths 64, 128 and 256, affine f(S) has second=2*first;
        # f(S)=a*S^2+b has second=4*first.  These cover every observed shape,
        # stride, range endpoint and flattened attention extent.
        if second in (2 * first, 4 * first):
            return 0, 1
    raise RuntimeError(
        f"non-parametric difference at {path}: {left!r}, {middle!r}, {right!r}"
    )


def old_aot_nodes() -> dict[str, dict[str, Any]]:
    dossier = load(DOSSIER)
    rows: dict[str, dict[str, Any]] = {}
    for unit in dossier["rows"]:
        forward = unit["forward"]
        rows[f"FORWARD:{forward['name']}"] = forward
        for node in unit["actual_local_vjp"]["program_nodes"]:
            rows[f"BACKWARD:{node['name']}"] = node
    for unit in dossier["backward_only_auxiliary_rows"]:
        node = unit["actual_node"]
        rows[f"BACKWARD:{node['name']}"] = node
    return rows


def captured_aot_nodes(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        f"{graph['phase']}:{node['name']}": node
        for graph in payload["capture"]["graphs"]
        for node in graph["nodes"] if node["op"] == "call_function"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq128-inventory", type=Path, required=True)
    parser.add_argument("--seq256-inventory", type=Path, required=True)
    parser.add_argument("--seq128-aot", type=Path, required=True)
    parser.add_argument("--seq256-aot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    eager_payloads = [load(OLD_STRONG), load(args.seq128_inventory), load(args.seq256_inventory)]
    eager = [payload["trace"]["events"] for payload in eager_payloads]
    if len({len(rows) for rows in eager}) != 1:
        raise RuntimeError("strong eager invocation denominator changed across shapes")
    structural_fields = ("invocation_id", "ordinal", "phase", "overload", "module_context")
    tensor_fields = ("dtype", "layout", "requires_grad", "source_ordinal")
    argument_fields = ("name", "schema_type", "source", "value_type", "tensor_input_indices")
    eager_constant = eager_parametric = 0
    for index, triple in enumerate(zip(*eager)):
        left, middle, right = triple
        for field in structural_fields:
            if not left.get(field) == middle.get(field) == right.get(field):
                raise RuntimeError(f"eager structural field changed at {index}/{field}")
        for group in ("input_tensors", "output_tensors"):
            values = [row[group] for row in triple]
            if len({len(value) for value in values}) != 1:
                raise RuntimeError(f"eager tensor arity changed at {index}/{group}")
            for tensor_index, tensors in enumerate(zip(*values)):
                for field in tensor_fields:
                    if not tensors[0].get(field) == tensors[1].get(field) == tensors[2].get(field):
                        raise RuntimeError(f"eager tensor identity changed at {index}/{group}/{tensor_index}/{field}")
                for field in ("shape", "stride"):
                    c, p = validate_parametric(
                        tensors[0][field], tensors[1][field], tensors[2][field],
                        f"eager[{index}].{group}[{tensor_index}].{field}",
                    )
                    eager_constant += c; eager_parametric += p
        bindings = [row["argument_bindings"] for row in triple]
        if len({len(value) for value in bindings}) != 1:
            raise RuntimeError(f"argument arity changed at eager[{index}]")
        for binding_index, values in enumerate(zip(*bindings)):
            for field in argument_fields:
                if not values[0].get(field) == values[1].get(field) == values[2].get(field):
                    raise RuntimeError(f"argument identity changed at eager[{index}]/arg[{binding_index}]/{field}")
            c, p = validate_parametric(
                values[0].get("value"), values[1].get("value"), values[2].get("value"),
                f"eager[{index}].arg[{binding_index}].value",
            )
            eager_constant += c; eager_parametric += p

    old_nodes = old_aot_nodes()
    aot128_payload, aot256_payload = load(args.seq128_aot), load(args.seq256_aot)
    aot128, aot256 = captured_aot_nodes(aot128_payload), captured_aot_nodes(aot256_payload)
    if old_nodes.keys() != aot128.keys() or old_nodes.keys() != aot256.keys():
        raise RuntimeError("AOT call-function node denominator changed across shapes")
    aot_constant = aot_parametric = 0
    for node_id in sorted(old_nodes):
        triple = (old_nodes[node_id], aot128[node_id], aot256[node_id])
        for field in ("target", "input_nodes", "original_aten"):
            if not triple[0].get(field) == triple[1].get(field) == triple[2].get(field):
                raise RuntimeError(f"AOT topology changed at {node_id}/{field}")
        for field in ("arguments", "tensor_meta"):
            c, p = validate_parametric(
                triple[0].get(field), triple[1].get(field), triple[2].get(field),
                f"aot[{node_id}].{field}",
            )
            aot_constant += c; aot_parametric += p

    payload = {
        "schema": "kernel-analyzer-qwen-three-shape-program-isomorphism-v1",
        "status": "COMPLETE_EXACT_SHAPE_PARAMETRIC_PROGRAM_ISOMORPHISM",
        "bindings": {
            "seq64_strong_inventory_sha256": eager_payloads[0]["result_sha256"],
            "seq128_strong_inventory_sha256": eager_payloads[1]["result_sha256"],
            "seq256_strong_inventory_sha256": eager_payloads[2]["result_sha256"],
            "seq64_aot_dossier_sha256": hashlib.sha256(DOSSIER.read_bytes()).hexdigest(),
            "seq128_aot_sha256": aot128_payload["result_sha256"],
            "seq256_aot_sha256": aot256_payload["result_sha256"],
        },
        "denominator": {
            "strong_eager_invocations_per_shape": len(eager[0]),
            "aot_call_function_nodes_per_shape": len(old_nodes),
            "eager_constant_fields": eager_constant,
            "eager_shape_parametric_fields": eager_parametric,
            "aot_constant_fields": aot_constant,
            "aot_shape_parametric_fields": aot_parametric,
        },
        "gates": {
            "complete_eager_dataflow_isomorphism": True,
            "complete_aot_target_and_edge_isomorphism": True,
            "all_nonconstant_values_are_device_or_sequence_parametric": True,
            "operator_name_shape_similarity_pairing_used": False,
            "candidate_values_used": False,
        },
        "claim_boundary": (
            "The three eager traces and AOT programs are the same operation/dataflow program "
            "evaluated at sequence lengths 64, 128 and 256. This transfers structural F+B "
            "identity only; each shape still requires its own execution and numerical witness."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **payload["denominator"], "result_sha256": payload["result_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
