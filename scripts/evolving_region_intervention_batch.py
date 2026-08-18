#!/usr/bin/env python3
"""Run isolated region interventions in one compiled natural checkpoint.

Each arm replaces exactly one candidate-blind mapped generated region and then
lets the real downstream forward/backward continue.  The compiler/model are
warmed once per process; arms are still executed separately so their carrier
deltas remain causally attributable to one region.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path("/data1/tzh").resolve()
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import evolving_triton_observation as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-manifest", type=Path, default=Path("results/final/natural_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--dtype-mapping", type=Path, required=False)
    parser.add_argument("--campaign", type=Path, default=None,
                        help="optional original shape-specific campaign; preserves exact region IDs")
    parser.add_argument("--arms", type=Path, required=True, help="JSON list of {region_id} arm records")
    parser.add_argument(
        "--joint-arms", action="store_true",
        help="replace every declared arm simultaneously as one grouped causal intervention",
    )
    parser.add_argument("--seq-len", type=int, choices=(64, 128, 256, 1024), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument(
        "--seq1024-state-index", type=int, default=0,
        help="contiguous validation-state index; permits held-out seq1024 states",
    )
    parser.add_argument(
        "--frozen-tile-direction", type=Path, default=None,
        help="optional exact tile direction produced by build_tile_carrier_direction.py",
    )
    parser.add_argument("--carrier-parameter", action="append", default=[])
    parser.add_argument("--carrier-sketch-input-dir", type=Path, default=None)
    parser.add_argument("--carrier-sketch-output-dir", type=Path, default=None)
    parser.add_argument(
        "--carrier-vector-output-dir", type=Path, default=None,
        help="persist complete FP32 carrier deltas for T3; one file per arm",
    )
    parser.add_argument(
        "--state-id", default=None,
        help="frozen state identifier required with --carrier-vector-output-dir",
    )
    parser.add_argument("--carrier-sketch-sample-size", type=int, default=131072)
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--metric-chunk-elements", type=int, default=131072)
    parser.add_argument(
        "--intervention-candidate-scale", type=float, default=0.0,
        help="0 copies the reference; 1 is a candidate-restoration sham control",
    )
    parser.add_argument(
        "--intervention-reference-role",
        choices=("schedule", "semantic"),
        default="schedule",
        help="schedule uses the generated-kernel reference; semantic uses the closed eager reduction",
    )
    parser.add_argument(
        "--intervention-endpoint", action="append", default=[],
        help="optional output pointer(s) to replace within every selected region",
    )
    parser.add_argument(
        "--intervention-reference-variant",
        choices=("eager_materialized", "fp32_rotary_rounded", "fp32_weight_multiply", "multiply_r3", "fp32_rotary_and_weight", "softmax_delayed_rounding", "key_fp32_fused", "key_actual_scale_fp32_fused", "key_actual_scale_materialized"),
        default="eager_materialized",
        help="controlled arithmetic variant for query-backward out_ptr3",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def under_root(path: Path, label: str) -> Path:
    return base.under_root(path, label)


def arm_file_name(region_id: str) -> str:
    return hashlib.sha256(region_id.encode()).hexdigest()[:20] + ".pt"


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "candidate_finite", "reference_finite", "exact", "full_value_scan",
        "rms", "max_abs", "signed_mean", "nonzero_elements", "nonzero_fraction",
    }
    return {key: value for key, value in metrics.items() if key in allowed}


def load_campaign_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.campaign is not None:
        opener = gzip.open if args.campaign.suffix == ".gz" else open
        with opener(args.campaign, "rt") as handle:
            campaign = json.load(handle)
        rows = campaign.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("campaign has no rows")
        return [dict(row) for row in rows]
    mapping = json.loads(args.dtype_mapping.read_text()) if args.dtype_mapping is not None else None
    if mapping is not None and (
        str(mapping["dtype"]) != args.dtype
        or bool(mapping["tf32"]) != bool(args.tf32)
        or int(mapping["seq_len"]) != args.seq_len
    ):
        raise ValueError("dtype mapping configuration does not match worker configuration")
    rows: list[dict[str, Any]] = []
    for mapping_row in mapping["rows"]:
        if mapping_row["status"] != "MAPPED":
            continue
        symbol = str(mapping_row["symbol"])
        reference_symbol = str(mapping_row["reference_symbol"])
        phase = (
            "BACKWARD"
            if "backward" in symbol or "mul_sum" in symbol or "add_div" in symbol
            else "FORWARD"
        )
        inputs = list(mapping_row["input_names"])
        outputs = list(mapping_row["output_names"])
        for invocation_index in range(int(mapping_row["invocations"])):
            rows.append({
                "boundary_capture_mode": "DTYPE_MAPPED_EXACT_RUNTIME_POINTERS",
                "heldout_execution_status": "PENDING_GPU_HELDOUT_ONLINE_REFERENCE",
                "input_names": inputs,
                "output_names": outputs,
                "prelaunch_clone_names": [name for name in inputs if name.startswith("in_out_")],
                "phase": phase,
                "reference_entrypoint": "forkcert.qwen3_triton_reference_dispatch:same_precision_reference",
                "reference_symbol": reference_symbol,
                "canonical_reference_symbol": reference_symbol,
                "region_id": f"dtype:{args.dtype}:{'tf32' if args.tf32 else 'strict'}:seq{args.seq_len}:{symbol}:{invocation_index}",
                "single_state_replay_schema": "forkcert.dtype-mapped-runtime-record.v1",
                "source_nodes": [],
                "symbol": symbol,
            })
    return rows


def load_pilot_sketch(path: Path, parameter_names: list[str], torch: Any) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema") != "kernel-analyzer-carrier-difference-sketch-v1":
        raise ValueError(f"carrier sketch schema mismatch: {path}")
    if value.get("parameter_names") != parameter_names:
        raise ValueError(f"carrier sketch parameter mismatch: {path}")
    return value


def exact_causal_contrast(
    eager: dict[str, Any], candidate: dict[str, Any], intervened: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name in sorted(set(eager) | set(candidate) | set(intervened)):
        reference, baseline, observed = eager.get(name), candidate.get(name), intervened.get(name)
        if reference is None or baseline is None or observed is None:
            rows[name] = {"status": "MISSING"}
            continue
        baseline_delta = baseline - reference
        residual = observed - reference
        baseline_l2 = float(baseline_delta.norm())
        residual_l2 = float(residual.norm())
        direction = baseline_delta / (baseline_delta.norm() + 1e-30)
        residual_projection = float((residual * direction).sum())
        intervention_projection = float(((observed - baseline) * direction).sum())
        rows[name] = {
            "status": "OK",
            "exact_vector": True,
            "numel": int(baseline_delta.numel()),
            "candidate_minus_eager_l2": baseline_l2,
            "intervened_minus_eager_l2": residual_l2,
            "intervened_minus_eager_projection": residual_projection,
            "intervened_minus_candidate_projection": intervention_projection,
            "directional_residual_ratio": residual_projection / (baseline_l2 + 1e-30),
            "directional_removal_fraction": 1.0 - residual_projection / (baseline_l2 + 1e-30),
        }
    return rows


def main() -> None:
    args = parse_args()
    args.bank_manifest = under_root(args.bank_manifest, "bank manifest")
    args.model = under_root(args.model, "model")
    if args.dtype_mapping is not None:
        args.dtype_mapping = under_root(args.dtype_mapping, "dtype mapping")
    if args.campaign is not None:
        args.campaign = under_root(args.campaign, "campaign")
    if args.dtype_mapping is None and args.campaign is None:
        raise ValueError("one of --dtype-mapping or --campaign is required")
    args.arms = under_root(args.arms, "arms")
    args.output = under_root(args.output, "output")
    if args.carrier_sketch_input_dir is not None:
        args.carrier_sketch_input_dir = under_root(args.carrier_sketch_input_dir, "carrier sketch input directory")
    if args.carrier_sketch_output_dir is not None:
        args.carrier_sketch_output_dir = under_root(args.carrier_sketch_output_dir, "carrier sketch output directory")
    if args.carrier_vector_output_dir is not None:
        args.carrier_vector_output_dir = under_root(args.carrier_vector_output_dir, "carrier vector output directory")
        if not args.state_id:
            raise ValueError("--state-id is required when saving complete carrier vectors")
    if args.frozen_tile_direction is not None:
        args.frozen_tile_direction = under_root(args.frozen_tile_direction, "frozen tile direction")
    if args.tf32 and args.dtype != "fp32":
        raise ValueError("--tf32 requires --dtype fp32")
    if args.carrier_sketch_sample_size < 1:
        raise ValueError("carrier sketch sample size must be positive")

    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from torch._dynamo.backends.registry import lookup_backend
    from torch._inductor.codecache import PyCodeCache

    from scripts.checkpoint_inductor import build_model as build_checkpoint_model, load_natural_validation
    from scripts.long_horizon_trigger import build_model as build_long_model, load_milestone
    from forkcert.generated_triton_reference_observer import GeneratedTritonReferenceObserver

    arms_payload = json.loads(args.arms.read_text())
    if isinstance(arms_payload, dict) and arms_payload.get("schema") == "kernel-analyzer-t2-region-queue-v1":
        arms = list(arms_payload.get("selected_arms", ()))
    else:
        arms = arms_payload
    if not isinstance(arms, list) or not arms:
        raise ValueError("arms must be a nonempty JSON list or T2 region queue")
    arm_ids = [str(row["region_id"]) for row in arms]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("arm region IDs must be unique")
    arm_rows_by_id = {str(row["region_id"]): row for row in arms}
    carrier_parameter_names = list(args.carrier_parameter) or ["model.embed_tokens.weight"]
    bank = json.loads(args.bank_manifest.read_text())
    bank_rows = bank.get("milestones", bank.get("checkpoints", []))
    checkpoint_row = next((row for row in bank_rows if int(row["step"]) == args.step), None)
    if checkpoint_row is None:
        raise RuntimeError(f"checkpoint step {args.step} is absent from bank")
    checkpoint = under_root(Path(checkpoint_row["path"]), "checkpoint")

    tokenizer_mod = __import__("transformers", fromlist=["AutoTokenizer"])
    tokenizer = tokenizer_mod.AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    eval_args = SimpleNamespace(
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        eval_split=args.eval_split,
        eval_offset=args.eval_offset,
        seq_len=args.seq_len,
        device=torch.device(args.device),
    )
    if args.seq_len == 1024:
        if args.eval_offset != 0:
            raise ValueError("seq1024 uses --seq1024-state-index, not --eval-offset")
        if args.seq1024_state_index < 0:
            raise ValueError("seq1024 state index must be nonnegative")
        from scripts.long_horizon_trigger import load_eval_states
        seq1024_states, all_eval_protocol = load_eval_states(
            tokenizer, args.seq_len, args.seq1024_state_index + 1, torch.device(args.device)
        )
        inputs = seq1024_states[args.seq1024_state_index]
        eval_protocol = {
            **all_eval_protocol,
            "states": 1,
            "selected_state_index": args.seq1024_state_index,
            "offsets": [all_eval_protocol["offsets"][args.seq1024_state_index]],
            "token_sha256": [all_eval_protocol["token_sha256"][args.seq1024_state_index]],
        }
    else:
        inputs, eval_protocol = load_natural_validation(tokenizer, eval_args)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.benchmark = False

    # The intervention must use the exact same executable reference program as
    # the long-horizon trigger.  The older from_config/assign=True checkpoint
    # constructor produced a different eager loss despite identical weights.
    # Reusing the trigger constructor prevents that program mismatch from being
    # mistaken for a causal kernel effect.
    if bank.get("schema") == "kernel-analyzer-long-horizon-bank-v1":
        if args.dtype != "bf16":
            raise ValueError("the canonical long-horizon bank is BF16")
        model = build_long_model(args.model, torch.device(args.device))
        load_milestone(model, checkpoint_row, args.model)
        model_construction = "scripts.long_horizon_trigger.build_model+load_milestone"
    else:
        if checkpoint.exists():
            model = build_checkpoint_model(
                args.model, checkpoint, "eager", dtype, torch.device(args.device)
            )
            model_construction = "scripts.checkpoint_matrix.build_model"
        elif int(args.step) == 0:
            # Step zero is exactly the pinned pretrained checkpoint.  Compact
            # cleanup may remove its redundant 4 GB single-file copy while
            # retaining the original sharded model.  Reconstruct only this
            # initial state and verify the manifest's tensor digest before any
            # candidate execution; later trained states remain fail-closed.
            from transformers import AutoModelForCausalLM
            from scripts.natural_bank import parameter_digest

            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=dtype,
                attn_implementation="eager",
                local_files_only=True,
            ).to(torch.device(args.device))
            observed_digest = parameter_digest(model)
            expected_digest = checkpoint_row.get("parameter_sha256")
            if not expected_digest or observed_digest != expected_digest:
                raise RuntimeError(
                    "reconstructed step-0 parameter digest does not match frozen bank"
                )
            model_construction = "PINNED_SHARDED_PRETRAINED_STEP0_DIGEST_VERIFIED"
        else:
            raise FileNotFoundError(
                f"trained checkpoint was cleaned and cannot be reconstructed: {checkpoint}"
            )
    model.config.use_cache = False

    # Capture the eager endpoint before compilation.  This makes every arm's
    # removal metric an exact (intervened-eager) projection onto the actual
    # full-step (candidate-eager) direction, not merely an arm delta.
    input_ids, labels = inputs
    model.zero_grad(set_to_none=True)
    eager_loss = model(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]
    eager_loss.backward()
    eager_gradients = base.capture_named_gradients(model, carrier_parameter_names)

    class LossStep(torch.nn.Module):
        def __init__(self, subject: torch.nn.Module) -> None:
            super().__init__()
            self.subject = subject

        def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return self.subject(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]

    audit = {"backend_compiles": 0, "runtime_invocations": 0, "graph_hashes": []}
    inductor = lookup_backend("inductor")

    def backend(graph_module: Any, example_inputs: list[Any]) -> Any:
        audit["backend_compiles"] += 1
        audit["graph_hashes"].append(hashlib.sha256(graph_module.code.encode()).hexdigest())
        compiled = inductor(graph_module, example_inputs)

        def counted(*values: Any) -> Any:
            audit["runtime_invocations"] += 1
            return compiled(*values)

        return counted

    module_start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend=backend, fullgraph=True, dynamic=False)
    model.zero_grad(set_to_none=True)
    warm_loss = candidate(input_ids, labels)
    warm_loss.backward()
    torch.cuda.synchronize(torch.device(args.device))
    warmed_symbols = base.discover_all_triton_symbols(list(PyCodeCache.modules[module_start:]))
    original_campaign_rows = load_campaign_rows(args)
    # An original shape-specific campaign contains every generated region.  A
    # causal arm run, however, intentionally warms only the selected arm set.
    # Restrict the campaign before symbol remapping so unrelated regions cannot
    # make the exact warmed-symbol census fail (and so the denominator remains
    # the declared arm set).
    if args.campaign is not None:
        selected_arm_ids = set(arm_ids)
        selected_rows = [
            row for row in original_campaign_rows
            if str(row.get("region_id")) in selected_arm_ids
        ]
        missing_campaign_rows = sorted(selected_arm_ids - {
            str(row.get("region_id")) for row in selected_rows
        })
        if missing_campaign_rows:
            raise RuntimeError(
                f"intervention arms are absent from supplied campaign: {missing_campaign_rows}"
            )
        # A generated symbol can be invoked many times.  Keep every campaign
        # row for the selected arm's symbol(s), otherwise the observer would
        # correctly reject the second ordinary invocation as outside campaign.
        selected_symbols = {str(row["symbol"]) for row in selected_rows}
        selected_reference_symbols = {
            str(row.get("reference_symbol", row["symbol"]))
            for row in selected_rows
        }
        # A closed semantic final reduction needs the corresponding partial
        # invocation to seed its same-input eager reference.  Preserve the
        # complete partial/final pair even when only the final is intervened.
        closure_pairs = (
            {"triton_red_fused_sum_14", "triton_per_fused_sum_18"},
            {
                "triton_red_fused__to_copy__unsafe_view_mul_sum_transpose_view_11",
                "triton_per_fused__to_copy__unsafe_view_mul_sum_transpose_view_15",
            },
        )
        closure_references: set[str] = set()
        for pair in closure_pairs:
            if selected_reference_symbols & pair:
                closure_references.update(pair)
        original_campaign_rows = [
            row for row in original_campaign_rows
            if (
                str(row.get("symbol")) in selected_symbols
                or str(row.get("reference_symbol", row["symbol"]))
                in closure_references
            )
        ]
    campaign_rows, unmatched = base.remap_campaign_to_warmed_symbols(
        original_campaign_rows,
        warmed_symbols,
        allow_extra_same_stem=True,
    )
    expected_regions = {str(row["region_id"]) for row in campaign_rows}
    missing_arms = sorted(set(arm_ids) - expected_regions)
    if missing_arms:
        raise RuntimeError(f"intervention arms are not present in exact mapping: {missing_arms}")

    baseline_gradients = base.capture_named_gradients(model, carrier_parameter_names)
    frozen_tile = None
    if args.frozen_tile_direction is not None:
        frozen_tile = torch.load(args.frozen_tile_direction, map_location="cpu", weights_only=False)
        if frozen_tile.get("schema") != "kernel-analyzer-frozen-tile-direction-v1":
            raise ValueError("frozen tile direction schema mismatch")
        if frozen_tile["parameter"] not in carrier_parameter_names:
            raise ValueError("frozen tile parameter must be included via --carrier-parameter")

    def frozen_tile_contrast(observed_gradients: dict[str, Any]) -> dict[str, Any] | None:
        if frozen_tile is None:
            return None
        name = frozen_tile["parameter"]
        rs, re = int(frozen_tile["row_start"]), int(frozen_tile["row_stop"])
        cs, ce = int(frozen_tile["column_start"]), int(frozen_tile["column_stop"])
        direction = frozen_tile["direction"].float().reshape(-1)
        eager_tile = eager_gradients[name][rs:re, cs:ce].float().reshape(-1)
        baseline_tile = baseline_gradients[name][rs:re, cs:ce].float().reshape(-1)
        observed_tile = observed_gradients[name][rs:re, cs:ce].float().reshape(-1)
        baseline_delta = baseline_tile - eager_tile
        residual = observed_tile - eager_tile
        arm_delta = observed_tile - baseline_tile
        norm = float(direction.norm())
        baseline_projection = float(torch.dot(baseline_delta, direction) / (norm + 1e-30))
        residual_projection = float(torch.dot(residual, direction) / (norm + 1e-30))
        arm_projection = float(torch.dot(arm_delta, direction) / (norm + 1e-30))
        return {
            "parameter": name,
            "row_start": rs, "row_stop": re, "column_start": cs, "column_stop": ce,
            "coordinate_count": int(direction.numel()),
            "direction_l2": norm,
            "candidate_minus_eager_projection": baseline_projection,
            "intervened_minus_eager_projection": residual_projection,
            "intervened_minus_candidate_projection": arm_projection,
            "directional_residual_ratio": residual_projection / (baseline_projection + 1e-30),
            "directional_removal_fraction": 1.0 - residual_projection / (baseline_projection + 1e-30),
            "candidate_minus_eager_tile_l2": float(baseline_delta.norm()),
            "intervened_minus_eager_tile_l2": float(residual.norm()),
            "intervened_equals_candidate_tile": bool(torch.equal(observed_tile, baseline_tile)),
        }
    pilot_sketches: dict[str, dict[str, Any]] = {}
    if args.carrier_sketch_input_dir is not None:
        for region_id in arm_ids:
            sketch_path = args.carrier_sketch_input_dir / arm_file_name(region_id)
            if not sketch_path.exists():
                raise FileNotFoundError(f"missing pilot sketch for {region_id}: {sketch_path}")
            pilot_sketches[region_id] = load_pilot_sketch(sketch_path, carrier_parameter_names, torch)

    arm_results: list[dict[str, Any]] = []
    intervention_groups = (
        [("joint:" + hashlib.sha256("\n".join(arm_ids).encode()).hexdigest()[:16], arm_ids)]
        if args.joint_arms
        else [(region_id, [region_id]) for region_id in arm_ids]
    )
    for arm_index, (region_id, intervention_ids) in enumerate(intervention_groups):
        repeats: list[dict[str, Any]] = []
        first_sketch_payload = None
        first_complete_deltas = None
        for repeat_id in range(2):
            endpoint_map = {}
            for selected_region in intervention_ids:
                declared = arm_rows_by_id[selected_region].get("endpoints")
                endpoints = list(declared) if declared is not None else list(args.intervention_endpoint)
                if endpoints:
                    endpoint_map[selected_region] = endpoints
            observer = GeneratedTritonReferenceObserver(
                modules=PyCodeCache.modules[module_start:],
                campaign_rows=campaign_rows,
                sequence=args.seq_len,
                sample_size=args.sample_size,
                metric_chunk_elements=args.metric_chunk_elements,
                intervene_region_ids=intervention_ids,
                intervene_endpoints_by_region=endpoint_map or None,
                intervention_candidate_scale=args.intervention_candidate_scale,
                intervention_reference_role=args.intervention_reference_role,
                intervention_reference_variant=args.intervention_reference_variant,
            )
            model.zero_grad(set_to_none=True)
            started = __import__("time").perf_counter()
            with observer:
                loss = candidate(input_ids, labels)
                loss.backward()
            torch.cuda.synchronize(torch.device(args.device))
            elapsed = __import__("time").perf_counter() - started
            observed_gradients = base.capture_named_gradients(model, carrier_parameter_names)
            complete_deltas = {
                name: (observed_gradients[name] - baseline_gradients[name]).contiguous()
                for name in carrier_parameter_names
                if observed_gradients.get(name) is not None and baseline_gradients.get(name) is not None
            }
            if args.carrier_vector_output_dir is not None:
                if repeat_id == 0:
                    first_complete_deltas = complete_deltas
                elif first_complete_deltas is None or any(
                    name not in complete_deltas
                    or not torch.equal(first_complete_deltas[name], complete_deltas[name])
                    for name in set(first_complete_deltas) | set(complete_deltas)
                ):
                    raise RuntimeError(f"complete carrier delta repeat mismatch for {region_id}")
            stats = base.gradient_difference_stats(baseline_gradients, observed_gradients)
            causal = exact_causal_contrast(eager_gradients, baseline_gradients, observed_gradients)
            tile_causal = frozen_tile_contrast(observed_gradients)
            sketches = base.sampled_gradient_difference(
                baseline_gradients,
                observed_gradients,
                parameter_names=carrier_parameter_names,
                sample_size=args.carrier_sketch_sample_size,
            )
            sketch_summary: dict[str, dict[str, Any]] = {}
            for name, item in sketches.items():
                if item.get("status") != "OK":
                    sketch_summary[name] = item
                    continue
                values = item["values"]
                row = {
                    "status": "OK",
                    "numel": item["numel"],
                    "sample_size": item["sample_size"],
                    "sketch_l2": float(values.norm()),
                }
                pilot = pilot_sketches.get(region_id)
                if pilot is not None:
                    p = pilot["sketches"][name]
                    if not torch.equal(item["indices"], p["indices"]):
                        raise ValueError(f"carrier sketch coordinates differ for {region_id}/{name}")
                    pv = p["values"].to(values.dtype)
                    dot = float((values * pv).sum())
                    denom = float(values.norm() * pv.norm())
                    row.update({
                        "pilot_step": pilot["pilot_step"],
                        "pilot_dot": dot,
                        "pilot_cosine": dot / (denom + 1e-30),
                    })
                sketch_summary[name] = row
                if repeat_id == 0:
                    first_sketch_payload = {
                        "schema": "kernel-analyzer-carrier-difference-sketch-v1",
                        "pilot_step": args.step,
                        "parameter_names": carrier_parameter_names,
                        "sample_size": args.carrier_sketch_sample_size,
                        "sketches": {
                            name: {
                                "numel": value["numel"],
                                "sample_size": value["sample_size"],
                                "indices": value["indices"],
                                "values": value["values"],
                            }
                            for name, value in sketches.items()
                            if value.get("status") == "OK"
                        },
                    }
            records = [
                row for row in observer.records
                if row.region_id in set(intervention_ids)
            ]
            if {row.region_id for row in records} != set(intervention_ids):
                raise RuntimeError(f"intervention records missing for {region_id}")
            primary_record = records[0]
            observer_summary = observer.summary()
            repeats.append({
                "repeat_id": repeat_id,
                "loss": float(loss.detach().float().cpu()),
                "elapsed_seconds": elapsed,
                "carrier": stats,
                "exact_causal_contrast": causal,
                "frozen_tile_causal_contrast": tile_causal,
                "carrier_sketch": sketch_summary,
                "record": {
                    "region_id": primary_record.region_id,
                    "phase": primary_record.phase,
                    "symbol": primary_record.symbol,
                    "reference_symbol": primary_record.reference_symbol,
                    "intervened_endpoints": list(primary_record.intervened_endpoints),
                    "endpoint_metrics": {
                        name: compact_metrics(metric)
                        for name, metric in primary_record.endpoint_metrics.items()
                    },
                },
                "intervened_region_count": len(records),
                "gates": {
                    "region_observed": set(observer_summary["intervention"]["observed_region_ids"]) == set(intervention_ids),
                    "all_mapped_regions_observed": observer_summary["denominator"]["observed_ordinary_triton_regions"] == len(expected_regions),
                },
            })
            del observed_gradients, sketches, observer, loss
            gc.collect()
            torch.cuda.empty_cache()
        if first_sketch_payload is not None and args.carrier_sketch_output_dir is not None:
            args.carrier_sketch_output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(first_sketch_payload, args.carrier_sketch_output_dir / arm_file_name(region_id))
        if first_complete_deltas is not None and args.carrier_vector_output_dir is not None:
            args.carrier_vector_output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "schema": "kernel-analyzer-complete-carrier-delta-v1",
                "state_id": args.state_id,
                "eval_offset": args.eval_offset,
                "checkpoint_step": args.step,
                "region_id": region_id,
                "region_ids": intervention_ids,
                "parameter_names": carrier_parameter_names,
                "complete_coordinates": True,
                "repeat_exact": True,
                "deltas": first_complete_deltas,
            }, args.carrier_vector_output_dir / arm_file_name(region_id))
        arm_results.append({
            "arm_index": arm_index,
            "region_id": region_id,
            "region_ids": intervention_ids,
            "repeats": repeats,
            "pilot_sketch_file": arm_file_name(region_id) if first_sketch_payload is not None else None,
            "input_sketch_file": arm_file_name(region_id) if region_id in pilot_sketches else None,
        })

    output = {
        "schema": "kernel-analyzer-evolving-region-intervention-batch-v1",
        "subject": "Qwen3-1.7B exact-mapped generated region interventions",
        "checkpoint_step": args.step,
        "checkpoint_file_sha256": checkpoint_row.get("sha256"),
        "model_construction": model_construction,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "tf32": bool(args.tf32),
        "dtype_mapping": str(args.dtype_mapping) if args.dtype_mapping is not None else None,
        "arms_file": str(args.arms),
        "arm_count": len(arm_results),
        "declared_region_count": len(arm_ids),
        "joint_arms": bool(args.joint_arms),
        "carrier_parameter_names": carrier_parameter_names,
        "candidate_blind": True,
        "reference_replacement": "copy same-input reference outputs at one exact generated boundary per arm before real downstream backward",
        "intervention_candidate_scale": args.intervention_candidate_scale,
        "intervention_reference_role": args.intervention_reference_role,
        "intervention_endpoints": list(args.intervention_endpoint),
        "intervention_endpoints_by_region": {
            region: list(row["endpoints"])
            for region, row in arm_rows_by_id.items()
            if "endpoints" in row
        },
        "intervention_reference_variant": args.intervention_reference_variant,
        "eager_loss": float(eager_loss.detach().float().cpu()),
        "baseline_candidate_loss": float(warm_loss.detach().float().cpu()),
        "evaluation": eval_protocol,
        "frozen_tile_direction": str(args.frozen_tile_direction) if args.frozen_tile_direction is not None else None,
        "warmed_symbol_count": len(warmed_symbols),
        "unmatched_warmed_symbols": unmatched,
        "compile_audit": audit,
        "arms": arm_results,
        "gates": {
            "every_arm_has_two_repeats": all(len(row["repeats"]) == 2 for row in arm_results),
            "every_arm_observed_at_exact_boundary": all(
                all(repeat["gates"]["region_observed"] for repeat in row["repeats"])
                for row in arm_results
            ),
            "all_mapped_regions_census_complete_per_repeat": all(
                all(repeat["gates"]["all_mapped_regions_observed"] for repeat in row["repeats"])
                for row in arm_results
            ),
            "tensor_values_retained": False,
            "natural_bias_case_added": False,
            "property_generalization_allowed": False,
        },
        "boundary": "Batch causal screening only. Each arm is isolated, but a finite arm set and coordinate carrier sketches do not certify coherent bias or exhaust unresolved topology.",
    }
    output["result_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "arm_count": len(arm_results), "step": args.step}, sort_keys=True))
    del candidate, model, eager_gradients
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
