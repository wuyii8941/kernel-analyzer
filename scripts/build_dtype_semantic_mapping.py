#!/usr/bin/env python3
"""Build a conservative dtype-specific symbol -> F+B reference mapping.

The mapping is name/topology based only.  It never reads candidate tensor
values and it never turns an unresolved topology into a numerical verdict.
Only a unique pointer-topology match to the frozen reference catalog is marked
``MAPPED``; everything else remains explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_SRC = ROOT / "archive" / "round1_code" / "src"
if str(OLD_SRC) not in sys.path:
    sys.path.insert(0, str(OLD_SRC))


def unique_catalog(path: Path) -> dict[tuple[str, tuple[str, ...], tuple[str, ...]], set[str]]:
    rows = json.loads(path.read_text())["rows"]
    catalog: dict[tuple[str, tuple[str, ...], tuple[str, ...]], set[str]] = defaultdict(set)
    for row in rows:
        key = (
            str(row["reference_entrypoint"]),
            tuple(str(x) for x in row["input_names"]),
            tuple(str(x) for x in row["output_names"]),
        )
        catalog[key].add(str(row.get("reference_symbol", row["symbol"])))
    return catalog


def io_names(pointer_names: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    inputs = tuple(name for name in pointer_names if name.startswith("in_"))
    outputs = tuple(
        name
        for name in pointer_names
        if name.startswith("out_") or name.startswith("in_out_")
    )
    return inputs, outputs


def classify(symbol: str, pointers: list[str]) -> str | None:
    """Return a reference entrypoint suffix only for unambiguous families."""
    if "softmax_backward_data" in symbol:
        return "replay_softmax_backward_region"
    if "__softmax_add_arange" in symbol:
        return "replay_causal_softmax_region"
    if "unsafe_view_clone_expand" in symbol:
        return "replay_value_head_expand"
    if "clone_squeeze_sum" in symbol:
        return "replay_value_head_reduce"
    if "clone_transpose" in symbol:
        return "replay_attention_output_transpose"
    if "unsafe_view_mul_silu_silu_backward" in symbol:
        return "replay_swiglu_backward"
    if "unsafe_view_mul_silu" in symbol:
        return "replay_swiglu_forward"
    if "unsafe_view_add_bmm_cat_cos_mean" in symbol:
        if "in_out_ptr1" in pointers:
            return "replay_key_rotary_norm_forward"
        if "out_ptr2" in pointers:
            return "replay_query_rotary_norm_forward"
        return None
    if "unsafe_view_add_bmm_cat_clone_cos_expand" in symbol:
        return "replay_kv_head_repeat"
    if "add_bmm_cat_cos_mul_neg_sin" in symbol and "backward" in symbol:
        if len([x for x in pointers if x.startswith("in_")]) == 2:
            return "replay_grouped_rotary_backward"
        return None
    if "unsafe_view_add_bmm_cat_clone_cos_div_expand_mul_neg_pow_sin" in symbol:
        return "replay_query_rotary_norm_backward"
    if "red_fused__log_softmax__log_softmax_backward" in symbol:
        return "replay_remaining_singleton"
    if "per_fused__log_softmax_lift_fresh_nll_loss_forward" in symbol:
        return "replay_remaining_singleton"
    if "red_fused_add_div_expand_mul_pow_sum" in symbol:
        return "replay_rmsnorm_input_gradient"
    if "per_fused_add_mul_sum" in symbol:
        return "replay_rmsnorm_weight_gradient"
    if "red_fused__unsafe_view_add_mean_mul_pow_rsqrt" in symbol:
        return "replay_fused_residual_rmsnorm"
    return None


# Generated FP32/TF32 names can differ from the BF16 campaign by a
# ``__to_copy`` materialization token or by a schedule-local ordinal.  These
# aliases are still candidate-blind: they are keyed only by the generated
# symbol and its frozen pointer topology, and each target is an existing exact
# F+B reference program (or an explicit internal schedule sentinel).
DIRECT_REFERENCE_ALIASES = {
    "triton_per_fused__softmax_add_arange_bitwise_and_eq_exp_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_sub_view_where_6":
        "forkcert:softmax-forward-inout",
    "triton_per_fused__softmax_backward_data_mul_view_16":
        "forkcert:softmax-backward-two-input",
    "triton_per_fused__softmax_backward_data_mul_view_6":
        "forkcert:softmax-backward-two-input-inout-upstream",
    "triton_per_fused__softmax_backward_data_mul_view_17":
        "forkcert:softmax-backward-two-input",
    "triton_per_fused__softmax_backward_data_mul_view_7":
        "forkcert:softmax-backward-two-input-inout-upstream",
    "triton_per_fused__log_softmax_prepare_softmax_online_17":
        "triton_per_fused__log_softmax__to_copy__unsafe_view_prepare_softmax_online_view_19",
    "triton_per_fused_arange_cat_cumsum_ne_slice_sub_unsqueeze_1":
        "triton_per_fused_arange_cat_cumsum_ne_slice_sub_unsqueeze_1",
    "triton_poi_fused__to_copy_arange_unsqueeze_2":
        "triton_poi_fused__to_copy_arange_unsqueeze_2",
    # The generated seq64/128/256 clone-expand schedule repeats each
    # contiguous KV-head block twice.  It is *not* the transpose/unsqueeze
    # schedule used by ``...clone_expand_transpose...``; that tempting alias
    # was removed after the seq64 source-level check exposed a large false
    # residual.
    "triton_poi_fused_clone_expand_unsqueeze_5":
        "forkcert:kv-head-repeat",
    "triton_per_fused__unsafe_view_add_bmm_cat_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10":
        "forkcert:reduce-last",
    "triton_per_fused__unsafe_view_add_bmm_cat_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_11":
        "forkcert:reduce-last",
    "triton_per_fused__unsafe_view_add_bmm_cat_cos_mul_neg_sin_slice_slice_backward_sum_transpose_unsqueeze_view_18":
        "forkcert:reduce-last",
    "triton_per_fused__unsafe_view_add_bmm_cat_cos_mul_neg_sin_slice_slice_backward_sum_transpose_unsqueeze_view_19":
        "forkcert:reduce-last",
    "triton_per_fused__unsafe_view_add_mul_sum_view_23":
        "forkcert:rms-weight-split-full",
    "triton_per_fused_mul_sum_view_1":
        "forkcert:rms-weight-single",
    "triton_per_fused_mul_sum_view_2":
        "forkcert:rms-weight-final",
    "triton_poi_fused_embedding_dense_backward_25":
        "forkcert:embedding-zero-fill",
    "triton_poi_fused_embedding_dense_backward_26":
        "forkcert:embedding-zero-fill",
    "triton_red_fused__unsafe_view_add_div_expand_mul_pow_sum_view_21":
        "forkcert:rmsnorm-input-gradient-variant",
    "triton_red_fused__unsafe_view_add_div_expand_mul_pow_sum_view_22":
        "forkcert:rmsnorm-input-gradient-variant",
    "triton_red_fused_add_div_embedding_dense_backward_expand_lift_fresh_mul_pow_sum_view_24":
        "forkcert:embedding-rmsnorm-input-gradient",
    "triton_red_fused_add_div_embedding_dense_backward_expand_lift_fresh_mul_pow_sum_view_25":
        "forkcert:embedding-rmsnorm-input-gradient",
    "triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0":
        "forkcert:embedding-rmsnorm-forward",
    "triton_poi_fused_constant_pad_nd_13":
        "triton_poi_fused_constant_pad_nd_15",
    "triton_red_fused_prepare_softmax_online_14":
        "triton_red_fused__to_copy__unsafe_view_prepare_softmax_online_view_16",
    # Seq256 strict-FP32 loss reduction changes the generated name and fuses
    # the max/log-denominator stores into one in-place boundary.  Its source
    # addresses are exactly the dedicated two-output seq256 partial program.
    "triton_red_fused__log_softmax_prepare_softmax_online_14":
        "forkcert:loss-softmax-seq256-partial",
    "triton_red_fused_prepare_softmax_online_16":
        "triton_red_fused__to_copy__unsafe_view_prepare_softmax_online_view_18",
    "triton_per_fused_prepare_softmax_online_15":
        "triton_per_fused__to_copy__unsafe_view_prepare_softmax_online_view_17",
    "triton_red_fused_add_mul_sum_view_4":
        "triton_per_fused__to_copy_add_mul_sum_view_6",
    "triton_red_fused_add_mul_sum_view_5":
        "forkcert:rms-weight-split-partial-two",
    "triton_red_fused_add_mul_sum_view_19":
        "triton_per_fused__to_copy_add_mul_sum_view_17",
    "triton_red_fused_add_mul_sum_view_20":
        "forkcert:rms-weight-split-partial-three",
    "triton_red_fused__unsafe_view_add_mul_sum_view_24":
        "forkcert:rms-weight-split-full",
    "triton_red_fused__unsafe_view_add_mul_sum_view_23":
        "forkcert:rms-weight-split-partial-dual",
    "triton_red_fused__unsafe_view_add_mean_mul_pow_rsqrt_9":
        "triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_10",
    "triton_red_fused__unsafe_view_add_mean_mul_pow_rsqrt_11":
        "triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12",
    "triton_red_fused_mul_sum_view_1":
        "forkcert:rms-weight-single",
    "triton_per_fused__unsafe_view_add_bmm_cat_cos_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_3":
        "forkcert:query-rotary-forward-variant",
    "triton_per_fused__unsafe_view_add_bmm_cat_cos_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_4":
        "forkcert:key-rotary-forward-variant",
    "triton_per_fused__unsafe_view_add_bmm_cat_clone_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_sum_transpose_unsqueeze_view_13":
        "forkcert:query-rotary-vjp",
    "triton_per_fused__unsafe_view_add_bmm_cat_clone_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_sum_transpose_unsqueeze_view_20":
        "forkcert:query-rotary-vjp",
    "triton_per_fused__unsafe_view_add_bmm_cat_clone_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_sum_transpose_unsqueeze_view_14":
        "forkcert:query-rotary-vjp",
    "triton_per_fused__unsafe_view_add_bmm_cat_clone_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_sum_transpose_unsqueeze_view_21":
        "forkcert:query-rotary-vjp",
    "triton_poi_fused__unsafe_view_add_bmm_cat_clone_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_12":
        "forkcert:rmsnorm-vjp-pointwise",
    "triton_poi_fused__unsafe_view_add_bmm_cat_clone_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_22":
        "forkcert:rmsnorm-vjp-pointwise",
    "triton_poi_fused__unsafe_view_add_bmm_cat_clone_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_13":
        "forkcert:rmsnorm-vjp-pointwise",
    "triton_poi_fused__unsafe_view_add_bmm_cat_clone_cos_div_expand_mul_neg_pow_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_23":
        "forkcert:rmsnorm-vjp-pointwise",
    "triton_red_fused__unsafe_view_add_bmm_cat_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9":
        "forkcert:rotary-reduction",
    "triton_red_fused__unsafe_view_add_bmm_cat_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10":
        "forkcert:rotary-reduction",
    "triton_red_fused__unsafe_view_add_bmm_cat_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_11":
        "forkcert:rotary-reduction",
    "triton_red_fused__unsafe_view_add_bmm_cat_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_12":
        "forkcert:rotary-reduction",
    "triton_red_fused__unsafe_view_add_bmm_cat_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_13":
        "forkcert:rotary-reduction",
    "triton_red_fused__unsafe_view_add_bmm_cat_cos_mul_neg_sin_slice_slice_backward_sum_transpose_unsqueeze_view_17":
        "forkcert:rotary-weight-split-seven",
    "triton_red_fused__unsafe_view_add_bmm_cat_cos_mul_neg_sin_slice_slice_backward_sum_transpose_unsqueeze_view_18":
        "forkcert:rotary-weight-split-seven",
    "triton_red_fused_add_div_expand_mul_pow_sum_view_15":
        "forkcert:rmsnorm-input-gradient-inout",
    "triton_red_fused_add_div_expand_mul_pow_sum_view_16":
        "forkcert:rmsnorm-input-gradient-inout",
    "triton_red_fused_add_div_expand_mul_pow_sum_view_2":
        "forkcert:rmsnorm-input-gradient-simple-inout",
    "triton_red_fused_add_div_expand_mul_pow_sum_view_3":
        "forkcert:rmsnorm-input-gradient-simple-inout",
    "triton_red_fused__log_softmax__log_softmax_backward_data_arange_eq_expand_lift_fresh_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_where_0":
        "triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_where_0",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/triton_online_reference_campaign_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = unique_catalog(args.campaign)
    topology = json.loads(args.topology.read_text())
    rows = []
    for symbol, signature in sorted(topology["symbol_signatures"].items()):
        pointers = [str(x) for x in signature["pointer_names"]]
        inputs, outputs = io_names(pointers)
        entry_suffix = classify(symbol, pointers)
        candidates = set()
        direct_reference_symbol = DIRECT_REFERENCE_ALIASES.get(symbol)
        if direct_reference_symbol is not None:
            candidates.add(direct_reference_symbol)
        # An explicit source-derived alias is authoritative.  Do not union it
        # with a looser catalog classifier: that turns a known topology into a
        # spurious AMBIGUOUS row when another family happens to share the same
        # pointer count.
        if direct_reference_symbol is None and entry_suffix is not None:
            for (entrypoint, catalog_inputs, catalog_outputs), symbols in catalog.items():
                if entrypoint.rsplit(":", 1)[-1] != entry_suffix:
                    continue
                if catalog_inputs == inputs and catalog_outputs == outputs:
                    candidates.update(symbols)
        if len(candidates) == 1:
            status = "MAPPED"
            reference_symbol = sorted(candidates)[0]
            reason = "unique reference-entrypoint and exact pointer topology"
        elif len(candidates) > 1:
            status = "AMBIGUOUS"
            reference_symbol = None
            reason = "multiple reference symbols share the inferred topology"
        elif entry_suffix is None:
            status = "UNRESOLVED_FAMILY"
            reference_symbol = None
            reason = "generated family has no conservative classifier"
        else:
            status = "UNRESOLVED_POINTER_TOPOLOGY"
            reference_symbol = None
            reason = "family is recognized but pointer topology is absent from the BF16 catalog"
        rows.append({
            "symbol": symbol,
            "invocations": int(signature["invocations"]),
            "pointer_names": pointers,
            "input_names": list(inputs),
            "output_names": list(outputs),
            "inferred_entrypoint": entry_suffix,
            "reference_symbol": reference_symbol,
            "status": status,
            "reason": reason,
        })
    output = {
        "schema": "kernel-analyzer-dtype-semantic-mapping-v1",
        "subject": topology["subject"],
        "dtype": topology["dtype"],
        "tf32": bool(topology["tf32"]),
        "seq_len": topology["seq_len"],
        "checkpoint_step": topology["checkpoint_step"],
        "topology_source": str(args.topology),
        "topology_sha256": hashlib.sha256(args.topology.read_bytes()).hexdigest(),
        "reference_catalog": str(args.campaign),
        "reference_catalog_sha256": hashlib.sha256(args.campaign.read_bytes()).hexdigest(),
        "candidate_values_used_to_select_or_classify": False,
        "denominator": {
            "runtime_symbols": len(rows),
            "runtime_invocations": sum(row["invocations"] for row in rows),
            "mapped_symbols": sum(row["status"] == "MAPPED" for row in rows),
            "mapped_invocations": sum(row["invocations"] for row in rows if row["status"] == "MAPPED"),
            "unresolved_symbols": sum(row["status"] != "MAPPED" for row in rows),
            "unresolved_invocations": sum(row["invocations"] for row in rows if row["status"] != "MAPPED"),
        },
        "rows": rows,
        "mapping_scope": "candidate-blind endpoint reference dispatch; MAPPED does not imply a closed F+B semantic reduction",
        "boundary": "This is a candidate-blind name/topology mapping census. MAPPED rows are eligible for a dtype-matched dynamic campaign; unresolved rows remain in the denominator and have no correctness verdict.",
    }
    output["result_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **output["denominator"]}, sort_keys=True))


if __name__ == "__main__":
    main()
