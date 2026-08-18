#!/usr/bin/env python3
"""Census exact weak/strong F+B program patterns without granting math proof."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from build_architecture_invocation_ledger import align_origin_witness, argument_signature


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def write(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wb", compresslevel=6) as handle:
            handle.write(encoded)
    else:
        path.write_bytes(encoded)


def shape_dtype(tensor: dict[str, Any]) -> dict[str, Any]:
    return {
        "shape": tensor["shape"], "dtype": tensor["dtype"],
        "stride": tensor["stride"], "requires_grad": tensor["requires_grad"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--weak", type=Path, required=True)
    parser.add_argument("--strong", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    weak = load(args.weak)
    strong = load(args.strong)
    events = weak["trace"]["events"]
    aligned, extras = align_origin_witness(events, strong["trace"]["events"])
    by_backward_sequence: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        witness = aligned[event["invocation_id"]]
        sequence = witness.get("backward_autograd_sequence_nr")
        if event["phase"] == "BACKWARD" and sequence is not None:
            by_backward_sequence[int(sequence)].append(event)

    patterns: dict[str, dict[str, Any]] = {}
    classifications = Counter()
    forward_count = 0
    for event in events:
        if event["phase"] != "FORWARD":
            continue
        forward_count += 1
        witness = aligned[event["invocation_id"]]
        sequence = witness.get("forward_autograd_sequence_nr")
        backward = by_backward_sequence.get(int(sequence), []) if sequence is not None else []
        if backward:
            classification = "ACTUAL_BACKWARD_PROGRAM"
        elif sequence is not None:
            classification = "EMPTY_ELIDED_OR_UNREACHED_VJP"
        elif witness.get("sequence_binding_status") == \
                "EXACT_NESTED_FORWARD_NO_INDEPENDENT_AUTOGRAD_BOUNDARY":
            classification = "NESTED_DISPATCH_ELIDED_VJP"
        elif not any(value["requires_grad"] for value in event["output_tensors"]):
            classification = "STOP_GRAD_OR_NONDifferentiable_OUTPUT"
        else:
            classification = "UNRESOLVED"
        classifications[classification] += 1
        non_tensor_arguments = [
            argument_signature(value) for value in event["argument_bindings"]
            if not value["tensor_input_indices"]
        ]
        program = {
            "forward_overload": event["overload"],
            "autograd_node": witness.get("autograd_node"),
            "classification": classification,
            "backward_overloads": [value["overload"] for value in backward],
            "non_tensor_arguments": non_tensor_arguments,
            "input_arity": len(event["input_tensors"]),
            "output_arity": len(event["output_tensors"]),
        }
        pattern_id = f"fb-pattern::{digest(program)}"
        row = patterns.setdefault(pattern_id, {
            "pattern_id": pattern_id, "program": program, "count": 0,
            "instance_ids": [], "instance_sample": [], "shape_dtype_examples": [],
        })
        row["count"] += 1
        row["instance_ids"].append(event["invocation_id"])
        if len(row["instance_sample"]) < 8:
            row["instance_sample"].append({
                "base_invocation_id": event["invocation_id"],
                "origin_witness_invocation_id": witness["invocation_id"],
                "sequence_nr": sequence,
                "backward_base_invocation_ids": [value["invocation_id"] for value in backward],
                "backward_origin_witness_invocation_ids": [
                    aligned[value["invocation_id"]]["invocation_id"] for value in backward
                ],
            })
        example = {
            "inputs": [shape_dtype(value) for value in event["input_tensors"]],
            "outputs": [shape_dtype(value) for value in event["output_tensors"]],
        }
        if example not in row["shape_dtype_examples"] and len(row["shape_dtype_examples"]) < 8:
            row["shape_dtype_examples"].append(example)

    for row in patterns.values():
        row["instance_ids_sha256"] = digest(row.pop("instance_ids"))
    unresolved = classifications["UNRESOLVED"]
    payload = {
        "schema": "kernel-analyzer-fb-proof-pattern-census-v1",
        "status": "COMPLETE_PATTERN_CENSUS" if not unresolved else "PARTIAL_FAIL_CLOSED",
        "architecture": args.architecture,
        "sequence_length": weak["input"]["sequence_length"],
        "bindings": {
            "weak_result_sha256": weak["result_sha256"],
            "strong_result_sha256": strong["result_sha256"],
            "base_to_origin_witness_alignment_sha256": digest(sorted(
                (base_id, witness["invocation_id"]) for base_id, witness in aligned.items()
            )),
        },
        "counts": {
            "weak_events": len(events), "strong_events": len(strong["trace"]["events"]),
            "observer_detach_extras": len(extras), "forward_invocations": forward_count,
            "unique_exact_program_patterns": len(patterns),
            "classifications": dict(sorted(classifications.items())),
        },
        "patterns": [patterns[key] for key in sorted(patterns)],
        "claim_boundary": (
            "This is an exact event-namespace and program-pattern census. It deliberately does "
            "not assert that any backward program implements the registered analytic VJP."
        ),
    }
    payload["result_sha256"] = digest(payload)
    write(args.output, payload)
    print(json.dumps({"output": str(args.output), "status": payload["status"],
                      "counts": payload["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
