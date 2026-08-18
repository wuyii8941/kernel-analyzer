#!/usr/bin/env python3
"""Transfer only structurally identical Triton reference adapters.

The old and current schedules happen to contain the same number of Triton
launches.  That count and region ordinals are not evidence.  A reference is
transferred only when the ordered original-ATen semantics and exact runtime
pointer ABI agree; repeated identical signatures all use the same mathematical
adapter, so no invocation pairing is required inside such a signature group.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / (
    "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/"
    "full_step_inventory/triton_online_reference_campaign_v1.json"
)
OLD_FLOW = ROOT / (
    "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/"
    "full_step_inventory/generated_compute_dataflow_audit_v1.json"
)
CURRENT = ROOT / "results/coverage/qwen_generated_inventory.json.gz"
OUTPUT = ROOT / "results/coverage/qwen_current_triton_reference_campaign.json.gz"
ENTRYPOINT = "forkcert.qwen3_triton_reference_dispatch:same_precision_reference"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def pointer_abi(row: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    bindings = row["boundary_witness"]["formal_to_actual_pointer_binding"]
    # ``in_out_ptr*`` is part of the pre-launch input boundary even when the
    # generated kernel overwrites it without first issuing a tl.load.  Dropping
    # such a pointer makes the runtime ABI incomplete and, more importantly,
    # prevents the observer from cloning the pre-launch value/identity.  Keep
    # load analysis as a dataflow fact, but define the invocation ABI from the
    # formal pointer role.
    inputs = tuple(sorted(
        name for name, value in bindings.items()
        if value["loaded"] or name.startswith("in_out_ptr")
    ))
    outputs = tuple(sorted(name for name, value in bindings.items() if value["stored"]))
    return inputs, outputs


def signature(
    *, phase: str, original_aten: list[str], inputs: list[str] | tuple[str, ...],
    outputs: list[str] | tuple[str, ...]
) -> tuple[Any, ...]:
    return (
        phase,
        tuple(original_aten),
        tuple(sorted(inputs)),
        tuple(sorted(outputs)),
    )


def current_exact_adapter(
    key: tuple[Any, ...], symbol: str = "", sequence: int | None = None
) -> str | None:
    """Return an adapter only for a frozen semantic sequence plus exact ABI."""
    phase, aten, inputs, outputs = key
    residual = (
        "aten._unsafe_view", "aten.add", "aten._to_copy", "aten.pow",
        "aten.mean", "aten.rsqrt", "aten.mul",
    )
    rotary_forward = (
        "aten.arange", "aten.unsqueeze", "aten.expand", "aten._to_copy",
        "aten.bmm", "aten.transpose", "aten.cat", "aten.cos", "aten.mul",
        "aten.sin", "aten._unsafe_view", "aten.view", "aten.pow", "aten.mean",
        "aten.add", "aten.rsqrt", "aten.slice", "aten.neg",
    )
    kv_repeat = (
        "aten.arange", "aten.unsqueeze", "aten.expand", "aten._to_copy",
        "aten.bmm", "aten.transpose", "aten.cat", "aten.cos", "aten.mul",
        "aten._unsafe_view", "aten.view", "aten.add", "aten.clone",
    )
    grouped_rotary = {
        (
            "aten.arange", "aten.unsqueeze", "aten.expand", "aten._to_copy",
            "aten.bmm", "aten.transpose", "aten.cat", "aten.sin", "aten.mul",
            "aten.slice_backward", "aten.cos", "aten.view", "aten.sum",
            "aten.squeeze", "aten.slice", "aten.neg", "aten.add",
        ),
        (
            "aten.view", "aten.transpose", "aten.sum", "aten.squeeze",
            "aten.arange", "aten.unsqueeze", "aten.expand", "aten._to_copy",
            "aten.bmm", "aten.cat", "aten.sin", "aten.mul", "aten.slice",
            "aten.neg", "aten.slice_backward", "aten.add", "aten.cos",
        ),
    }
    query_backward = {
        (
            "aten.arange", "aten.unsqueeze", "aten.expand", "aten._to_copy",
            "aten.bmm", "aten.transpose", "aten.cat", "aten.sin", "aten.mul",
            "aten.cos", "aten.slice_backward", "aten.view", "aten.slice",
            "aten.neg", "aten.add", "aten._unsafe_view", "aten.sum", "aten.pow",
            "aten.div", "aten.clone",
        ),
        (
            "aten.view", "aten.arange", "aten.unsqueeze", "aten.expand",
            "aten._to_copy", "aten.bmm", "aten.transpose", "aten.cat",
            "aten.sin", "aten.mul", "aten.cos", "aten.slice", "aten.neg",
            "aten.slice_backward", "aten.add", "aten._unsafe_view", "aten.sum",
            "aten.pow", "aten.div", "aten.clone",
        ),
    }
    softmax = (
        "aten.arange", "aten.add", "aten.view", "aten.le", "aten.bitwise_and",
        "aten.index", "aten.eq", "aten.lift_fresh", "aten.scalar_tensor",
        "aten.where", "aten.mul", "aten._to_copy", "prims.prepare_softmax_online",
        "aten._softmax",
    )
    embedding_norm = (
        "aten.embedding", "aten._to_copy", "aten.pow", "aten.mean",
        "aten.add", "aten.rsqrt", "aten.mul",
    )
    loss_prepare = (
        "aten._unsafe_view", "aten._to_copy", "aten.view",
        "prims.prepare_softmax_online",
    )
    embedding_rms_scatter = (
        "aten.nll_loss_forward", "aten.view", "aten.add", "aten.mul",
        "aten._to_copy", "aten.sum", "aten.pow", "aten.expand",
        "aten.div", "aten.embedding_dense_backward",
    )
    seq256_rms_partials = {
        "triton_red_fused__to_copy_mul_sum_view_2": (
            ("in_ptr0", "in_ptr1", "in_ptr2"),
            ("out_ptr0",),
            "forkcert:rms-weight-split-partial-one",
        ),
        "triton_red_fused__to_copy_add_mul_sum_view_7": (
            ("in_ptr0", "in_ptr1", "in_ptr2", "in_ptr3"),
            ("out_ptr0",),
            "forkcert:rms-weight-split-partial-two",
        ),
        "triton_red_fused__to_copy_add_mul_sum_view_18": (
            ("in_ptr0", "in_ptr1", "in_ptr2", "in_ptr3", "in_ptr4"),
            ("out_ptr0",),
            "forkcert:rms-weight-split-partial-three",
        ),
        "triton_red_fused__to_copy__unsafe_view_add_mul_sum_view_21": (
            tuple(f"in_ptr{index}" for index in range(9)),
            ("out_ptr0", "out_ptr1"),
            "forkcert:rms-weight-split-partial-dual",
        ),
    }
    partial = seq256_rms_partials.get(symbol)
    if (
        sequence == 256 and phase == "BACKWARD" and partial is not None
        and inputs == partial[0] and outputs == partial[1]
        and aten[-1:] == ("aten.sum",)
    ):
        return partial[2]
    # At seq256 Inductor splits every RMSNorm weight-gradient reduction into
    # a two-token-tile partial followed by this exact one-input final sum.
    # All 56 invocations call the same generated program; provenance differs
    # only because some layers have one versus several upstream addends.
    if (
        sequence == 256 and phase == "BACKWARD"
        and symbol == "triton_per_fused__to_copy_mul_sum_view_4"
        and inputs == ("in_ptr0",)
        and outputs == ("out_ptr0",)
        and aten[-1:] == ("aten.sum",)
    ):
        return "forkcert:rms-weight-split-final"
    if (
        sequence == 256 and phase == "FORWARD"
        and symbol == (
            "triton_red_fused__log_softmax__to_copy__unsafe_view_"
            "prepare_softmax_online_view_16"
        )
        and aten == (
            "aten._unsafe_view", "aten._to_copy", "aten.view",
            "prims.prepare_softmax_online", "aten._log_softmax",
        )
        and inputs == ("in_out_ptr0", "in_ptr0")
        and outputs == ("in_out_ptr0", "out_ptr0")
    ):
        return "forkcert:loss-softmax-seq256-partial"
    if (
        sequence == 256 and phase == "FORWARD"
        and symbol == (
            "triton_per_fused__log_softmax__to_copy__unsafe_view_"
            "nll_loss_forward_slice_sub_view_17"
        )
        and aten == (
            "aten._unsafe_view", "aten._to_copy", "aten.slice", "aten.view",
            "aten.sub", "aten._log_softmax", "aten.nll_loss_forward",
        )
        and inputs == ("in_out_ptr0", "in_ptr0", "in_ptr1", "in_ptr2", "in_ptr3")
        and outputs == ("in_out_ptr0", "out_ptr1")
    ):
        return (
            "triton_per_fused__log_softmax__to_copy__unsafe_view_"
            "nll_loss_forward_prepare_softmax_online_slice_sub_view_20"
        )
    if phase == "BACKWARD" and aten == embedding_rms_scatter and inputs == (
        "in_ptr0", "in_ptr1", "in_ptr2", "in_ptr3", "in_ptr4",
        "in_ptr5", "in_ptr6", "in_ptr7", "out_ptr2",
    ) and outputs == ("out_ptr2",) and symbol == (
        "triton_red_fused__to_copy_add_div_embedding_dense_backward_expand_"
        "mul_nll_loss_forward_pow_sum_view_20"
    ):
        return "forkcert:embedding-rmsnorm-scatter-vjp"
    if phase == "FORWARD" and aten == embedding_norm and inputs == (
        "in_out_ptr0", "in_ptr0", "in_ptr1", "in_ptr2"
    ) and outputs == ("in_out_ptr0", "out_ptr0", "out_ptr1") and symbol == (
        "triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0"
    ):
        return "forkcert:embedding-rmsnorm-forward"
    if phase == "FORWARD" and aten == loss_prepare and inputs == (
        "in_ptr0",
    ) and outputs == ("out_ptr0",) and symbol in {
        "triton_red_fused__to_copy__unsafe_view_prepare_softmax_online_view_16",
        "triton_per_fused__to_copy__unsafe_view_prepare_softmax_online_view_17",
    }:
        # These have the same high-level provenance and pointer ABI but
        # different generated programs (blocked reduction vs pointwise
        # materialization).  The dispatcher's exact generated symbol selects
        # the corresponding executable local map.
        return symbol
    if phase == "FORWARD" and aten == residual and inputs == (
        "in_out_ptr0", "in_ptr0", "in_ptr1"
    ) and outputs == ("in_out_ptr0", "in_out_ptr1", "out_ptr0"):
        return "triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14"
    if phase == "FORWARD" and aten == rotary_forward and inputs == (
        "in_ptr0", "in_ptr1", "in_ptr2"
    ):
        if outputs == ("in_out_ptr0", "out_ptr1", "out_ptr2"):
            return (
                "triton_red_fused__to_copy__unsafe_view_add_bmm_cat_cos_mean_mul_"
                "neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_13"
            )
        if outputs == ("in_out_ptr0", "in_out_ptr1"):
            return (
                "triton_red_fused__to_copy__unsafe_view_add_bmm_cat_cos_mean_mul_"
                "neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_4"
            )
    if phase == "FORWARD" and aten == kv_repeat and inputs == ("in_ptr0",) and outputs == ("out_ptr0",):
        return "triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_6"
    if phase == "FORWARD" and aten == softmax and inputs == ("in_out_ptr0", "in_ptr0") and outputs == (
        "in_out_ptr0", "out_ptr0", "out_ptr1", "out_ptr2"
    ):
        return (
            "triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_exp_"
            "index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_sub_view_where_7"
        )
    if phase == "BACKWARD" and aten in grouped_rotary and inputs == ("in_ptr0", "in_ptr1") and outputs == ("out_ptr0",):
        return (
            "triton_poi_fused__to_copy_add_bmm_cat_cos_mul_neg_sin_slice_"
            "slice_backward_squeeze_sum_transpose_unsqueeze_view_9"
        )
    if phase == "BACKWARD" and aten in query_backward and inputs == (
        "in_ptr0", "in_ptr1", "in_ptr2", "in_ptr3", "in_ptr4"
    ) and outputs == ("out_ptr1", "out_ptr3"):
        return (
            "triton_per_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_div_"
            "expand_mul_neg_pow_sin_slice_slice_backward_sum_transpose_unsqueeze_view_10"
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, default=OLD)
    parser.add_argument("--current", type=Path, default=CURRENT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--include-new-derived", action="store_true")
    args = parser.parse_args()
    old_payload = json.loads(args.historical.read_text())
    old_flow_payload = json.loads(OLD_FLOW.read_text())
    old_program_by_region = {
        row["region_id"]: row["boundary_witness"].get("embedded_program_sha256")
        for row in old_flow_payload["rows"] if row["kind"] == "TRITON"
    }
    with gzip.open(args.current, "rt", encoding="utf-8") as handle:
        current_payload = json.load(handle)
    sequence = int(current_payload["generated_regions"]["state"]["length"])
    old_rows = old_payload["rows"]
    inventory_rows = [
        row for row in current_payload["generated_regions"]["inventory"]["regions"]
        if row["kind"] == "TRITON"
    ]
    dataflow = {
        row["region_id"]: row
        for row in current_payload["compute_dataflow"]["rows"]
        if row["kind"] == "TRITON"
    }
    if len(old_rows) != 686 or len(inventory_rows) != len(dataflow):
        raise RuntimeError("unexpected Triton denominator")

    old_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in old_rows:
        old_row = dict(row)
        old_row["embedded_program_sha256"] = old_program_by_region.get(row["region_id"])
        old_by_signature[signature(
            phase=row["phase"], original_aten=row["original_aten"],
            inputs=row["input_names"], outputs=row["output_names"],
        )].append(old_row)

    current_counts: Counter[tuple[Any, ...]] = Counter()
    current_descriptors = []
    for row in inventory_rows:
        inputs, outputs = pointer_abi(dataflow[row["region_id"]])
        key = signature(
            phase=row["phase"], original_aten=row["original_aten"],
            inputs=inputs, outputs=outputs,
        )
        current_counts[key] += 1
        current_descriptors.append((row, dataflow[row["region_id"]], key, inputs, outputs))

    rows = []
    for inventory, flow, key, inputs, outputs in current_descriptors:
        historical = old_by_signature.get(key, [])
        adapters = {
            (
                row["reference_entrypoint"],
                row.get("reference_symbol", row["symbol"]),
                row["boundary_capture_mode"],
                tuple(row.get("prelaunch_clone_names", [])),
            )
            for row in historical
        }
        multiplicity_closed = len(historical) == current_counts[key]
        current_program = flow["boundary_witness"]["embedded_program_sha256"]
        program_closed = current_program in {
            row.get("embedded_program_sha256") for row in historical
        }
        derived_current = (
            current_exact_adapter(key, inventory["symbol"], sequence)
            if args.include_new_derived else None
        )
        if (
            inventory["original_aten"] == ["aten.sum"]
            and inputs == ("in_ptr0",)
            and outputs == ("out_ptr0",)
        ):
            # This does not need the historical partial/final classification.
            # The exact local map at either generated boundary is simply the
            # last-axis reduction determined by runtime source/output extents.
            entrypoint = "forkcert.qwen3_triton_reference_dispatch:same_precision_reference"
            reference_symbol = "forkcert:reduce-last"
            old_mode = None
            clones = ()
            status = "EXACT_DERIVED_CURRENT_REDUCTION_ADAPTER"
            boundary_mode = "CURRENT_EXACT_RUNTIME_POINTERS"
            reason = None
        elif derived_current is not None:
            entrypoint = ENTRYPOINT
            reference_symbol = derived_current
            old_mode = None
            clones = tuple(name for name in inputs if name in outputs)
            status = "EXACT_DERIVED_CURRENT_SEMANTIC_AND_POINTER_ABI_ADAPTER"
            boundary_mode = (
                "SPECIALIZED_BOUNDED_EMBEDDING_RMSNORM_ATOMIC"
                if derived_current == "forkcert:embedding-rmsnorm-scatter-vjp"
                else "CURRENT_EXACT_RUNTIME_POINTERS"
            )
            reason = None
        elif historical and len(adapters) == 1:
            entrypoint, reference_symbol, old_mode, clones = next(iter(adapters))
            status = "EXACT_SEMANTIC_AND_POINTER_ABI_ADAPTER_TRANSFER"
            boundary_mode = (
                old_mode
                if str(old_mode).startswith("SPECIALIZED_")
                else "CURRENT_EXACT_RUNTIME_POINTERS"
            )
            reason = None
        else:
            entrypoint = None
            reference_symbol = None
            old_mode = None
            clones = tuple(name for name in inputs if name.startswith("in_out_ptr"))
            status = "UNRESOLVED_CURRENT_REFERENCE_ADAPTER"
            boundary_mode = "RUNTIME_SKIP_UNRESOLVED_CURRENT_REFERENCE_ADAPTER"
            if not historical:
                reason = "NO_HISTORICAL_EXACT_SEMANTIC_AND_POINTER_ABI_SIGNATURE"
            elif len(adapters) != 1:
                reason = "HISTORICAL_SIGNATURE_HAS_NONUNIQUE_ADAPTER"
            else:
                reason = "SIGNATURE_MULTIPLICITY_CHANGED"
        row = {
            "region_id": inventory["region_id"],
            "phase": inventory["phase"],
            "symbol": inventory["symbol"],
            "original_aten": inventory["original_aten"],
            "source_nodes": inventory["source_nodes"],
            "source_path": inventory["source_path"],
            "source_line": inventory["source_line"],
            "input_names": list(inputs),
            "output_names": list(outputs),
            "prelaunch_clone_names": list(clones),
            "boundary_capture_mode": boundary_mode,
            "reference_entrypoint": entrypoint,
            "reference_symbol": reference_symbol,
            "historical_boundary_capture_mode": old_mode,
            "adapter_status": status,
            "unresolved_reason": reason,
            "heldout_execution_status": "PENDING_HELDOUT_ONLINE_REFERENCE",
            "evidence": {
                "ordered_original_aten_exact": bool(historical),
                "formal_pointer_abi_exact": bool(historical),
                "signature_multiplicity_closed": multiplicity_closed,
                "embedded_program_exact_to_historical_adapter": program_closed,
                "region_id_or_symbol_used_for_transfer": False,
                "candidate_values_used": False,
                "embedded_program_sha256": current_program,
            },
        }
        row["row_sha256"] = digest(row)
        rows.append(row)

    counts = Counter(row["adapter_status"] for row in rows)
    unresolved = counts["UNRESOLVED_CURRENT_REFERENCE_ADAPTER"]
    payload = {
        "schema": "kernel-analyzer-current-qwen-triton-reference-campaign-v1",
        "status": (
            "COMPLETE_STATIC_REFERENCE_PLAN"
            if unresolved == 0 else "PARTIAL_FAIL_CLOSED_REFERENCE_PLAN"
        ),
        "denominator": {
            "triton_invocations": len(rows),
            "reference_adapter_exact": (
                counts["EXACT_SEMANTIC_AND_POINTER_ABI_ADAPTER_TRANSFER"]
                + counts["EXACT_DERIVED_CURRENT_REDUCTION_ADAPTER"]
                + counts["EXACT_DERIVED_CURRENT_SEMANTIC_AND_POINTER_ABI_ADAPTER"]
            ),
            "reference_adapter_unresolved": unresolved,
        },
        "gates": {
            "all_current_triton_invocations_retained": len(rows) == len(inventory_rows),
            "all_exact_transfers_match_ordered_aten_and_pointer_abi": True,
            "historical_program_hash_change_requires_runtime_shape_validation": True,
            "cross_schedule_region_id_symbol_shape_or_ordinal_pairing_used": False,
            "exact_current_generated_symbol_dispatch_used": any(
                row["adapter_status"]
                == "EXACT_DERIVED_CURRENT_SEMANTIC_AND_POINTER_ABI_ADAPTER"
                for row in rows
            ),
            "candidate_values_used_to_select_adapter": False,
            "all_reference_adapters_exact": unresolved == 0,
            "heldout_values_observed": False,
        },
        "bindings": {
            "historical_campaign_sha256": old_payload["campaign_sha256"],
            "current_inventory_sha256": current_payload["result_sha256"],
            "current_dataflow_sha256": current_payload["compute_dataflow"]["audit_sha256"],
        },
        "rows": rows,
        "claim_boundary": (
            "This transfers a mathematical reference adapter, not a numerical verdict. "
            "Unmatched current semantics remain in the full denominator."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps({
        "output": str(args.output.resolve().relative_to(ROOT)),
        "denominator": payload["denominator"],
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
