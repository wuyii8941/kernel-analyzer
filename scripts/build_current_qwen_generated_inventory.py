#!/usr/bin/env python3
"""Build the executed generated-region and exact pointer-dataflow denominator."""

from __future__ import annotations

import gzip
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))

from forkcert.generated_compute_dataflow_audit import (  # noqa: E402
    build_generated_compute_dataflow_audit,
    validate_generated_compute_dataflow_audit,
)
from forkcert.generated_runtime_call_completeness_audit import (  # noqa: E402
    build_generated_runtime_call_completeness_audit,
    validate_generated_runtime_call_completeness_audit,
)
import forkcert.generated_runtime_call_completeness_audit as runtime_audit_module  # noqa: E402
from forkcert.inductor_direct_runtime_call_inventory import (  # noqa: E402
    build_inductor_direct_runtime_call_inventory,
    validate_inductor_direct_runtime_call_inventory,
)
from forkcert.inductor_generated_region_inventory import (  # noqa: E402
    build_inductor_generated_region_inventory,
)


TRACE = Path("/data1/tzh/cache/kernel_analyzer/qwen_proof_inductor_trace")
CAPTURE = ROOT / "results/coverage/qwen_inductor_proof_ids.json.gz"
OUTPUT = ROOT / "results/coverage/qwen_generated_inventory.json.gz"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=TRACE)
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--allow-partial-dataflow",
        action="store_true",
        help="Retain a complete runtime-call census when a graph-break pointer producer is unresolved.",
    )
    args = parser.parse_args()
    # PyTorch nightly added this value-preserving input guard after the frozen
    # audit package was archived.  It is a runtime helper, not a compute site.
    runtime_audit_module._HELPER_ROLES.setdefault(
        "copy_if_misaligned", "INPUT_ALIGNMENT_GUARD"
    )
    runtime_audit_module._HELPER_ROLES.setdefault(
        "assert_alignment", "OUTPUT_ALIGNMENT_POSTCONDITION"
    )
    opener = gzip.open if args.capture.suffix == ".gz" else open
    with opener(args.capture, "rt", encoding="utf-8") as handle:
        capture = json.load(handle)
    inventory = build_inductor_generated_region_inventory(trace_dir=args.trace_dir)
    try:
        candidate_artifact = str(args.capture.resolve().relative_to(ROOT))
    except ValueError:
        candidate_artifact = str(args.capture.resolve())
    wrapped = {
        "schema_version": "kernel-analyzer-executed-generated-region-v1",
        "architecture": capture.get("architecture", "qwen"),
        "state": {
            "sequence_id": capture["input"].get(
                "state_id", f"state-{capture['input']['state']}"
            ),
            "length": capture["input"]["sequence_length"],
            "record_sha256": capture["input"]["token_ids_sha256"],
            "split": "identity_witness",
        },
        "candidate_artifact": candidate_artifact,
        "candidate_status": capture["status"],
        "inventory": inventory,
    }
    direct = build_inductor_direct_runtime_call_inventory(
        trace_dir=args.trace_dir, base_generated_inventory=wrapped
    )
    validate_inductor_direct_runtime_call_inventory(direct)
    runtime = build_generated_runtime_call_completeness_audit(
        trace_dir=args.trace_dir,
        base_generated_inventory=wrapped,
        direct_runtime_inventory=direct,
    )
    validate_generated_runtime_call_completeness_audit(runtime)
    dataflow_error = None
    try:
        dataflow = build_generated_compute_dataflow_audit(
            trace_dir=args.trace_dir,
            base_generated_inventory=wrapped,
            direct_runtime_inventory=direct,
            runtime_call_audit=runtime,
        )
        validate_generated_compute_dataflow_audit(dataflow)
    except ValueError as error:
        if not args.allow_partial_dataflow:
            raise
        dataflow_error = str(error)
        dataflow = {
            "status": "PARTIAL_FAIL_CLOSED_UNRESOLVED_GRAPH_BREAK_POINTER",
            "denominator": {
                "expected_compute_invocations": runtime["denominator"]["compute_invocations"],
                "resolved_compute_invocations": 0,
                "unresolved_compute_invocations": runtime["denominator"]["compute_invocations"],
            },
            "rows": [],
            "error": dataflow_error,
        }
    payload = {
        "schema": "kernel-analyzer-executed-generated-inventory-v1",
        "status": (
            "COMPLETE_GENERATED_SCHEDULE_AND_POINTER_DATAFLOW"
            if dataflow_error is None
            else "COMPLETE_GENERATED_SCHEDULE_PARTIAL_POINTER_DATAFLOW"
        ),
        "architecture": capture.get("architecture", "qwen"),
        "generated_regions": wrapped,
        "direct_runtime_calls": direct,
        "runtime_call_audit": runtime,
        "compute_dataflow": dataflow,
        "pointer_dataflow_error": dataflow_error,
        "claim_boundary": (
            "Exact current generated callsites and input/output pointer dataflow; "
            "mathematical reference dispatch and heldout values are separate gates."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    summary_path = args.output.with_name(
        args.output.name.removesuffix(".json.gz") + ".summary.json"
    )
    summary = {
        "schema": "kernel-analyzer-executed-generated-inventory-summary-v1",
        "status": payload["status"], "architecture": payload["architecture"],
        "result_sha256": payload["result_sha256"],
        "denominator": runtime["denominator"],
        "source_inventory": str(args.output.resolve()),
        "source_inventory_bytes": args.output.stat().st_size,
    }
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    try:
        output_label = str(args.output.resolve().relative_to(ROOT))
    except ValueError:
        output_label = str(args.output.resolve())
    print(json.dumps({
        "output": output_label,
        "status": payload["status"],
        "regions": inventory["denominator"],
        "direct": direct["denominator"],
        "runtime_calls": runtime["denominator"],
        "dataflow": dataflow["denominator"],
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
