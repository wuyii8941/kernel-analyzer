#!/usr/bin/env python3
"""Freeze the candidate-independent precision screen protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    payload = {
        "schema": "kernel-analyzer-generated-fp32-protocol-v1",
        "population": {"states_per_shape": 32, "same_state_repeats": 2},
        "selection": {
            "regions": "EVERY_ACTUAL_GENERATED_COMPUTE_INVOCATION",
            "coordinates": "EVENLY_SPACED_FLAT_POSITIONS_FIXED_BEFORE_READING_VALUES",
            "candidate_values_used_for_selection": False,
        },
        "references": {
            "TRITON": "IDENTICAL_GENERATED_PROGRAM_GRID_SCALARS_ALIAS_AND_VIEW_TOPOLOGY_WITH_FP32_FLOATING_STORAGES",
            "NONTRITON": "SAME_DECLARED_EXTERNAL_OR_DIRECT_OPERATION_WITH_FP32_FLOATING_STORAGES",
        },
        "metrics": {
            "full_endpoint_scan": True,
            "streaming_chunk_default_elements": 1048576,
            "directional_sketch_default_elements": 64,
            "accumulation_dtype": "float64",
            "nonfinite": "COUNT_NAN_POSINF_NEGINF_AND_MISMATCH_SEPARATELY",
            "same_nonfinite_finite_metric_sentinel": 0.0,
        },
        "runtime_gates": {
            "observer_must_not_change_loss_or_any_parameter_gradient": True,
            "invocation_identity_must_be_repeat_and_state_stable": True,
            "static_generated_but_unexecuted_calls_remain_with_explicit_disposition": True,
            "cross_dtype_storage_alias_without_exact_fp32_mapping": "FAIL_CLOSED",
        },
        "claim_boundary": (
            "This protocol detects precision-associated local directional/nonfinite risk. "
            "It does not alone establish eager semantic equivalence or a complete F+B carrier mechanism."
        ),
    }
    payload["protocol_sha256"] = digest(payload)
    output = ROOT / "results/coverage/generated_fp32_protocol.json"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output.relative_to(ROOT)), "protocol_sha256": payload["protocol_sha256"]}))


if __name__ == "__main__":
    main()
