#!/usr/bin/env python3
"""Decompose the confirmed q_proj tile into exact G and H operand effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD_SRC = REPO / "archive" / "round1_code" / "src"
if str(OLD_SRC) not in sys.path:
    sys.path.insert(0, str(OLD_SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.long_horizon_trigger import atomic_json, build_model, load_eval_states, load_milestone, under_root
from scripts import evolving_triton_observation as observation


PARAMETER = "model.layers.23.self_attn.q_proj.weight"
ROWS = slice(1152, 1280)
COLUMNS = slice(1664, 1792)
SOURCE_MARKER = "linear_161, view_69, hidden_states_325, mul_641"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("results/final/long_horizon_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--direction", type=Path, default=Path("results/final/l23_qproj_tile_direction.pt"))
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--state-index", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("results/final/l23_qproj_operand_decomposition.json"))
    parser.add_argument("--query-campaign", type=Path, default=Path("results/final/seq1024_query_campaign.json"))
    parser.add_argument("--full-campaign", type=Path, default=Path(
        "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/"
        "triton_online_reference_campaign_v1.json"
    ))
    parser.add_argument("--screen-local-groups", action="store_true")
    parser.add_argument(
        "--local-group", action="append", default=[],
        choices=("query_forward_backward", "key_forward", "softmax_backward"),
        help="screen selected local group(s); query_backward is always retained",
    )
    parser.add_argument(
        "--key-reference-variant",
        choices=("eager_materialized", "key_actual_scale_materialized"),
        default="eager_materialized",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def projection(value, direction, torch) -> float:
    return float(torch.dot(value.reshape(-1).float(), direction.reshape(-1).float()) / direction.float().norm())


def main() -> None:
    args = parse_args()
    bank_path = under_root(args.bank, "bank")
    model_path = under_root(args.model, "model")
    direction_path = under_root(args.direction, "direction")
    output_path = under_root(args.output, "output")
    campaign_path = under_root(args.query_campaign, "query campaign")
    full_campaign_path = under_root(args.full_campaign, "full campaign")
    state_indices = args.state_index or [8, 9]
    if min(state_indices) < 8 or len(set(state_indices)) != len(state_indices):
        raise ValueError("operand decomposition requires unique held-out state indices >= 8")

    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/kernel_analyzer/tile_causal_compile")

    import torch
    from torch._dynamo.backends.registry import lookup_backend
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.select_algorithm import extern_kernels
    from transformers import AutoTokenizer
    from forkcert.generated_triton_reference_observer import GeneratedTritonReferenceObserver

    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    bank = json.loads(bank_path.read_text())
    milestone = next(row for row in bank["milestones"] if int(row["step"]) == args.step)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    all_states, evaluation = load_eval_states(tokenizer, 1024, max(state_indices) + 1, device)
    model = build_model(model_path, device)
    load_milestone(model, milestone, model_path)

    direction_payload = torch.load(direction_path, map_location="cpu", weights_only=False)
    if direction_payload.get("schema") != "kernel-analyzer-frozen-tile-direction-v1":
        raise ValueError("direction schema mismatch")
    direction = direction_payload["direction"].float().to(device)

    class LossStep(torch.nn.Module):
        def __init__(self, subject):
            super().__init__()
            self.subject = subject

        def forward(self, input_ids, labels):
            return self.subject(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]

    module_start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend=lookup_backend("inductor"), fullgraph=True, dynamic=False)
    model.zero_grad(set_to_none=True)
    warm = candidate(*all_states[state_indices[0]])
    warm.backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[module_start:])
    source_matches = []
    for module in modules:
        path = Path(module.__file__)
        source = path.read_text()
        if SOURCE_MARKER in source and "mm_267" in source:
            source_matches.append((path, source))
    if len(source_matches) != 1:
        raise RuntimeError(f"expected one exact backward source match, got {len(source_matches)}")
    source_path, source = source_matches[0]
    marker_position = source.index(SOURCE_MARKER)
    call_start = source.rfind("def call(", 0, marker_position)
    target_ordinal = source[call_start:marker_position].count("extern_kernels.mm(")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    warmed_symbols = observation.discover_all_triton_symbols(modules)
    original_campaign = json.loads(campaign_path.read_text())["rows"]
    target_row = next(row for row in original_campaign if row["region_id"] == "backward:157")
    target_symbol = target_row["symbol"]
    query_symbol_rows = [row for row in original_campaign if row["symbol"] == target_symbol]
    campaign_rows, unmatched = observation.remap_campaign_to_warmed_symbols(
        query_symbol_rows, warmed_symbols, allow_extra_same_stem=True
    )
    if "backward:157" not in {row["region_id"] for row in campaign_rows}:
        raise RuntimeError("layer-23 query VJP region is absent after exact remapping")
    intervention_campaigns = {"query_backward": campaign_rows}
    intervention_region_ids = {"query_backward": ["backward:157"]}
    requested_groups = (
        ["query_forward_backward", "key_forward", "softmax_backward"]
        if args.screen_local_groups else list(args.local_group)
    )
    if requested_groups:
        full_rows = json.loads(full_campaign_path.read_text())["rows"]
        group_ids = {
            "query_forward_backward": ["forward:1350", "backward:157"],
            "key_forward": ["forward:1352"],
            "softmax_backward": ["backward:151"],
        }
        by_id = {row["region_id"]: row for row in full_rows}
        for group, region_ids in group_ids.items():
            if group not in requested_groups:
                continue
            remap_region_ids = (
                [region_id for region_id in region_ids if region_id != "backward:157"]
                if group == "query_forward_backward" else region_ids
            )
            symbols = {by_id[region_id]["symbol"] for region_id in remap_region_ids}
            selected = [row for row in full_rows if row["symbol"] in symbols]
            mapped, group_unmatched = observation.remap_campaign_to_warmed_symbols(
                selected, warmed_symbols, allow_extra_same_stem=True
            )
            if group == "query_forward_backward":
                mapped = mapped + campaign_rows
            if not set(region_ids) <= {row["region_id"] for row in mapped}:
                raise RuntimeError(f"{group} regions are absent after exact remapping")
            intervention_campaigns[group] = mapped
            intervention_region_ids[group] = region_ids
            unmatched.extend(group_unmatched)

    rows = []
    q_proj = model.model.layers[23].self_attn.q_proj
    for state_index in state_indices:
        inputs = all_states[state_index]
        eager_capture = {}

        def forward_hook(_module, values, output):
            eager_capture["H"] = values[0].detach()[0].clone()

            def gradient_hook(gradient):
                eager_capture["G"] = gradient.detach()[0].clone()
                return gradient

            output.register_hook(gradient_hook)

        handle = q_proj.register_forward_hook(forward_hook)
        model.zero_grad(set_to_none=True)
        eager_loss = model(input_ids=inputs[0], labels=inputs[1], use_cache=False, return_dict=False)[0]
        eager_loss.backward()
        handle.remove()
        eager_tile = dict(model.named_parameters())[PARAMETER].grad.detach()[ROWS, COLUMNS].clone()
        if set(eager_capture) != {"G", "H"}:
            raise RuntimeError("eager q_proj operands were not captured")

        model.zero_grad(set_to_none=True)
        candidate_loss = candidate(*inputs)
        capture = {}
        counter = {"mm": 0}
        original_mm = extern_kernels.mm

        def wrapped_mm(*values, **kwargs):
            ordinal = counter["mm"]
            counter["mm"] += 1
            result = original_mm(*values, **kwargs)
            if ordinal == target_ordinal:
                left, right = values[:2]
                out = kwargs.get("out")
                if tuple(left.shape) != (2048, 1024) or tuple(right.shape) != (1024, 2048):
                    raise RuntimeError("exact mm_267 ordinal has unexpected operands")
                capture["Gt"] = left.detach().clone()
                capture["H"] = right.detach().clone()
                capture["dW"] = out[ROWS, COLUMNS].detach().clone()
            return result

        extern_kernels.mm = wrapped_mm
        try:
            candidate_loss.backward()
            torch.cuda.synchronize(device)
        finally:
            extern_kernels.mm = original_mm
        candidate_tile = dict(model.named_parameters())[PARAMETER].grad.detach()[ROWS, COLUMNS].clone()
        if set(capture) != {"Gt", "H", "dW"}:
            raise RuntimeError("exact candidate mm_267 operands were not captured")
        output_matches_parameter = bool(torch.equal(capture["dW"], candidate_tile))

        gr_full, hr_full = eager_capture["G"], eager_capture["H"]
        gc_full = capture["Gt"].transpose(0, 1)
        hc_full = capture["H"]
        gr, hr = gr_full[:, ROWS].float(), hr_full[:, COLUMNS].float()
        gc, hc = gc_full[:, ROWS].float(), hc_full[:, COLUMNS].float()
        g_effect = (gc - gr).transpose(0, 1) @ hr
        h_effect = gr.transpose(0, 1) @ (hc - hr)
        interaction = (gc - gr).transpose(0, 1) @ (hc - hr)
        fp32_total = gc.transpose(0, 1) @ hc - gr.transpose(0, 1) @ hr
        closure = g_effect + h_effect + interaction
        actual_delta = candidate_tile.float() - eager_tile.float()

        def exact_mm_tile(left, right):
            out = torch.empty((2048, 2048), dtype=torch.bfloat16, device=device)
            original_mm(left, right, out=out)
            return out[ROWS, COLUMNS].clone()

        finite_cc = exact_mm_tile(capture["Gt"], hc_full)
        finite_rr = exact_mm_tile(gr_full.transpose(0, 1), hr_full)
        finite_cr = exact_mm_tile(capture["Gt"], hr_full)
        finite_rc = exact_mm_tile(gr_full.transpose(0, 1), hc_full)
        finite_total = finite_cc.float() - finite_rr.float()
        g_at_reference_h = finite_cr.float() - finite_rr.float()
        g_at_candidate_h = finite_cc.float() - finite_rc.float()
        h_at_reference_g = finite_rc.float() - finite_rr.float()
        h_at_candidate_g = finite_cc.float() - finite_cr.float()
        g_shapley = 0.5 * (g_at_reference_h + g_at_candidate_h)
        h_shapley = 0.5 * (h_at_reference_g + h_at_candidate_g)

        baseline_projection = projection(actual_delta, direction, torch)
        intervention_results = {}
        for group, group_campaign in intervention_campaigns.items():
            region_ids = intervention_region_ids[group]
            endpoint_filter = (
                {"backward:157": ["out_ptr3"]}
                if group == "query_backward" else None
            )
            observer = GeneratedTritonReferenceObserver(
                modules=modules,
                campaign_rows=group_campaign,
                sequence=1024,
                intervene_region_ids=region_ids,
                intervene_endpoints_by_region=endpoint_filter,
                intervention_reference_variant=(
                    args.key_reference_variant
                    if group == "key_forward" else "eager_materialized"
                ),
            )
            model.zero_grad(set_to_none=True)
            with observer:
                intervened_loss = candidate(*inputs)
                intervened_loss.backward()
            torch.cuda.synchronize(device)
            intervened_tile = dict(model.named_parameters())[PARAMETER].grad.detach()[ROWS, COLUMNS].clone()
            residual = intervened_tile.float() - eager_tile.float()
            residual_projection = projection(residual, direction, torch)
            observer_summary = observer.summary()
            intervention_results[group] = {
                "region_ids": region_ids,
                "loss": float(intervened_loss.detach().float().cpu()),
                "residual_projection": residual_projection,
                "removal_projection": baseline_projection - residual_projection,
                "directional_removal_fraction": (
                    1.0 - residual_projection / baseline_projection
                    if baseline_projection != 0.0 else None
                ),
                "all_regions_observed": set(observer_summary["intervention"]["observed_region_ids"]) == set(region_ids),
            }
        query_backward_result = intervention_results["query_backward"]
        rows.append({
            "state_index": state_index,
            "offset": evaluation["offsets"][state_index],
            "token_sha256": evaluation["token_sha256"][state_index],
            "eager_loss": float(eager_loss.detach().float().cpu()),
            "candidate_loss": float(candidate_loss.detach().float().cpu()),
            "exact_mm_output_matches_parameter_gradient_tile": output_matches_parameter,
            "candidate_minus_eager_projection": projection(actual_delta, direction, torch),
            "fp32_operand_total_projection": projection(fp32_total, direction, torch),
            "g_effect_projection": projection(g_effect, direction, torch),
            "h_effect_projection": projection(h_effect, direction, torch),
            "interaction_projection": projection(interaction, direction, torch),
            "finite_mm_cc_matches_actual_candidate_tile": bool(torch.equal(finite_cc, candidate_tile)),
            "finite_mm_rr_matches_actual_eager_tile": bool(torch.equal(finite_rr, eager_tile)),
            "finite_mm_total_projection": projection(finite_total, direction, torch),
            "finite_mm_total_matches_actual_delta": bool(torch.equal(finite_total, actual_delta)),
            "finite_mm_g_shapley_projection": projection(g_shapley, direction, torch),
            "finite_mm_h_shapley_projection": projection(h_shapley, direction, torch),
            "finite_mm_shapley_closure_max_abs": float((g_shapley + h_shapley - finite_total).abs().max()),
            "finite_mm_g_at_reference_h_projection": projection(g_at_reference_h, direction, torch),
            "finite_mm_g_at_candidate_h_projection": projection(g_at_candidate_h, direction, torch),
            "finite_mm_h_at_reference_g_projection": projection(h_at_reference_g, direction, torch),
            "finite_mm_h_at_candidate_g_projection": projection(h_at_candidate_g, direction, torch),
            "local_vjp_intervened_loss": query_backward_result["loss"],
            "local_vjp_residual_projection": query_backward_result["residual_projection"],
            "local_vjp_removal_projection": query_backward_result["removal_projection"],
            "local_vjp_directional_removal_fraction": query_backward_result["directional_removal_fraction"],
            "local_vjp_region_observed": query_backward_result["all_regions_observed"],
            "local_group_interventions": intervention_results,
            "algebraic_closure_max_abs": float((closure - fp32_total).abs().max()),
            "g_delta_l2": float((gc - gr).norm()),
            "h_delta_l2": float((hc - hr).norm()),
            "actual_tile_delta_l2": float(actual_delta.norm()),
            "fp32_operand_total_l2": float(fp32_total.norm()),
        })

    result = {
        "schema": "kernel-analyzer-l23-qproj-operand-decomposition-v1",
        "status": "COMPLETE",
        "parameter": PARAMETER,
        "tile": {"rows": [ROWS.start, ROWS.stop], "columns": [COLUMNS.start, COLUMNS.stop]},
        "checkpoint_step": args.step,
        "state_indices": state_indices,
        "binding": {
            "forward": "mm_161: Y=H W^T",
            "actual_weight_backward": "mm_267: dW=G^T H",
            "source_marker": SOURCE_MARKER,
            "backward_mm_zero_based_ordinal": target_ordinal,
            "generated_source_sha256": source_sha256,
            "candidate_values_used_to_select_boundary": False,
        },
        "local_vjp_intervention": {
            "region_id": "backward:157",
            "endpoint": "out_ptr3",
            "reference": "same-input BF16 eager materialized VJP",
            "campaign": str(campaign_path),
            "unmatched_warmed_symbols": unmatched,
            "key_reference_variant": args.key_reference_variant,
        },
        "equation": "delta_dW=(delta_G)^T H_ref + G_ref^T(delta_H) + (delta_G)^T(delta_H)",
        "rows": rows,
        "tensor_values_saved": False,
    }
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(output_path, result)
    print(json.dumps({"output": str(output_path), "rows": len(rows), "target_ordinal": target_ordinal}, sort_keys=True))


if __name__ == "__main__":
    main()
