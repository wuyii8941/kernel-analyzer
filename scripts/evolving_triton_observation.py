#!/usr/bin/env python3
"""Observe every generated Triton region on one natural checkpoint.

The mathematical unit ledger is immutable.  This worker only reuses the
existing shape-specific campaign and records online same-input reference
metrics for every runtime Triton invocation during a complete forward and
backward step.  One process handles one checkpoint so compiler modules and
GPU memory cannot leak across states; callers can run workers in parallel and
merge their compact JSON artifacts afterwards.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from types import SimpleNamespace


ROOT = Path("/data1/tzh").resolve()
REPO = Path(__file__).resolve().parents[1]
OLD_SRC = REPO / "archive" / "round1_code" / "src"
if str(OLD_SRC) not in sys.path:
    sys.path.insert(0, str(OLD_SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def under_root(path: Path, label: str) -> Path:
    value = path.expanduser().resolve()
    if ROOT not in (value, *value.parents):
        raise ValueError(f"{label} must stay under {ROOT}: {value}")
    return value


def tensor_digest(value: Any) -> str:
    import torch

    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(repr(tuple(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def gradient_digest(
    model: Any,
    parameter_names: list[str] | None = None,
) -> tuple[str, dict[str, str]]:
    combined = hashlib.sha256()
    per_parameter: dict[str, str] = {}
    named_parameters = dict(model.named_parameters())
    names = sorted(named_parameters) if parameter_names is None else sorted(parameter_names)
    missing = [name for name in names if name not in named_parameters]
    if missing:
        raise KeyError(f"unknown gradient parameter(s): {missing}")
    for name in names:
        parameter = named_parameters[name]
        gradient = parameter.grad
        value = "NONE" if gradient is None else tensor_digest(gradient)
        per_parameter[name] = value
        combined.update(name.encode())
        combined.update(value.encode())
    return combined.hexdigest(), per_parameter


def capture_named_gradients(
    model: Any,
    parameter_names: list[str],
) -> dict[str, Any]:
    """Copy only explicitly requested carrier gradients to host memory."""
    named_parameters = dict(model.named_parameters())
    missing = [name for name in parameter_names if name not in named_parameters]
    if missing:
        raise KeyError(f"unknown gradient parameter(s): {missing}")
    return {
        name: (
            None
            if named_parameters[name].grad is None
            else named_parameters[name].grad.detach().float().cpu().clone()
        )
        for name in parameter_names
    }


def gradient_difference_stats(
    baseline: dict[str, Any],
    observed: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name in sorted(set(baseline) | set(observed)):
        left, right = baseline.get(name), observed.get(name)
        if left is None or right is None:
            rows[name] = {"status": "MISSING"}
            continue
        delta = right - left
        left_norm = float(left.norm())
        delta_norm = float(delta.norm())
        rows[name] = {
            "status": "OK",
            "l2": delta_norm,
            "rms": float(delta.square().mean().sqrt()),
            "max_abs": float(delta.abs().max()),
            "mean": float(delta.mean()),
            "baseline_l2": left_norm,
            "delta_baseline_cosine": float(
                (delta * left).sum() / (delta_norm * left_norm + 1e-30)
            ),
            "nonzero": int((delta != 0).sum()),
        }
    return rows


def sampled_gradient_difference(
    baseline: dict[str, Any],
    observed: dict[str, Any],
    *,
    parameter_names: list[str],
    sample_size: int,
) -> dict[str, Any]:
    """Return a deterministic coordinate sketch of intervention deltas.

    The coordinates are fixed by tensor shape and evenly spaced before any
    values are read.  This is a screening sketch, not a full-vector carrier
    certificate; positive candidates must be confirmed with an exact vector.
    """
    import torch

    if sample_size < 1:
        raise ValueError("carrier sketch sample size must be positive")
    result: dict[str, Any] = {}
    for name in parameter_names:
        left, right = baseline.get(name), observed.get(name)
        if left is None or right is None:
            result[name] = {"status": "MISSING"}
            continue
        if left.numel() != right.numel():
            raise ValueError(f"carrier shape changed for {name}")
        count = min(sample_size, int(left.numel()))
        indices = torch.linspace(
            0,
            int(left.numel()) - 1,
            count,
            dtype=torch.long,
        )
        delta = (right.reshape(-1) - left.reshape(-1))[indices]
        result[name] = {
            "status": "OK",
            "numel": int(left.numel()),
            "sample_size": count,
            "indices": indices,
            "values": delta,
        }
    return result


def _kernel_stem(symbol: str) -> str:
    return re.sub(r"_\d+$", "", symbol)


def _kernel_ordinal(symbol: str) -> int:
    match = re.search(r"_(\d+)$", symbol)
    return int(match.group(1)) if match else -1


def discover_all_triton_symbols(modules: list[Any]) -> list[str]:
    symbols: set[str] = set()
    for module in modules:
        for symbol, value in vars(module).items():
            if symbol.startswith("triton_") and callable(getattr(value, "run", None)) and hasattr(value, "triton_meta"):
                symbols.add(symbol)
    return sorted(symbols)


def remap_campaign_to_warmed_symbols(
    campaign_rows: list[dict[str, Any]],
    warmed_symbols: list[str],
    *,
    allow_extra_same_stem: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Map harmless generated suffix renumbering without changing references.

    Inductor may insert one helper kernel when the natural validation labels
    differ from the historical discovery state.  The fused body stem remains
    the same, while its numeric suffix shifts.  Reference dispatch must keep
    the historical symbol, so the warmed symbol is stored separately in the
    row's ``symbol`` field and the original is preserved as
    ``reference_symbol``.  Unmatched warmed symbols remain explicit and are
    never silently counted as observed campaign rows.
    """
    expected_names = sorted({str(row["symbol"]) for row in campaign_rows})
    expected_by_stem: dict[str, list[str]] = {}
    warmed_by_stem: dict[str, list[str]] = {}
    for symbol in expected_names:
        expected_by_stem.setdefault(_kernel_stem(symbol), []).append(symbol)
    for symbol in warmed_symbols:
        warmed_by_stem.setdefault(_kernel_stem(symbol), []).append(symbol)
    mapping: dict[str, str] = {}
    for stem, expected in expected_by_stem.items():
        warmed = warmed_by_stem.get(stem, [])
        if allow_extra_same_stem:
            exact = sorted(set(expected) & set(warmed))
            if len(exact) == len(expected):
                mapping.update({name: name for name in exact})
                continue
            if len(expected) == 1 and len(warmed) == 1:
                mapping[expected[0]] = warmed[0]
                continue
        if len(expected) != len(warmed):
            raise RuntimeError(f"generated symbol stem mismatch for {stem}: expected {expected}, warmed {warmed}")
        for old, new in zip(sorted(expected, key=_kernel_ordinal), sorted(warmed, key=_kernel_ordinal)):
            mapping[old] = new
    remapped = []
    for raw in campaign_rows:
        old = str(raw["symbol"])
        row = dict(raw)
        row["symbol"] = mapping[old]
        # Runtime suffix remapping must never replace the campaign's frozen
        # reference-dispatch identity.  Some current symbols alternate two
        # different semantic regions, so dispatching by the runtime symbol
        # would call an unsupported or, worse, incorrect reference.
        reference_symbol = raw.get("canonical_reference_symbol")
        if reference_symbol is None:
            reference_symbol = raw.get("reference_symbol")
        row["reference_symbol"] = str(
            old if reference_symbol is None else reference_symbol
        )
        remapped.append(row)
    unmatched = sorted(set(warmed_symbols) - set(mapping.values()))
    return remapped, unmatched


def _canonical_reference_symbol(
    row: dict[str, Any],
    canonical_rows: list[dict[str, Any]],
) -> str:
    """Resolve a shape-specific generated row to the frozen reference symbol.

    Generated suffixes and even region ordinals change at seq256.  Dispatching
    by those names would either call the wrong VJP or fail closed for an
    otherwise valid row.  The campaign's semantic entrypoint and exact
    pointer topology are stable, so use those as the primary key.  This is
    only a reference-dispatch mapping; it never uses candidate tensor values.
    """
    entry = str(row.get("reference_entrypoint", ""))
    phase = str(row.get("phase", ""))
    inputs = [str(value) for value in row.get("input_names", [])]
    outputs = [str(value) for value in row.get("output_names", [])]

    # Internal seq256 schedule boundaries have no generated-name equivalent.
    if entry.endswith("replay_rmsnorm_weight_gradient_partials"):
        count = len(inputs)
        return {
            3: "forkcert:rms-weight-split-partial-one",
            4: "forkcert:rms-weight-split-partial-two",
            5: "forkcert:rms-weight-split-partial-three",
            9: "forkcert:rms-weight-split-partial-dual",
        }[count]
    if entry.endswith("replay_rmsnorm_weight_gradient_final"):
        return "forkcert:rms-weight-split-final"
    if entry.endswith("qwen3_triton_reference_dispatch:same_precision_reference"):
        return "forkcert:loss-softmax-seq256-partial"
    if str(row.get("boundary_capture_mode", "")).startswith("SPECIALIZED_INDEXED_EMBEDDING_ROWS"):
        return "triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0"

    # The remaining singleton catalog contains several semantic families with
    # the same phase/entrypoint.  Pointer topology distinguishes them exactly.
    if entry.endswith("replay_remaining_singleton"):
        if phase == "BACKWARD":
            by_shape = {
                ("in_out_ptr0", 5, 1): (
                    "triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_where_0"
                ),
                ("out_ptr0", 3, 1): "triton_per_fused__to_copy_mul_sum_view_2",
                ("in_out_ptr0", 3, 1): "triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_3",
                ("in_out_ptr0", 6, 1): "triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view_19",
                ("out_ptr0", 9, 2): "triton_per_fused__to_copy__unsafe_view_add_mul_sum_view_20",
                ("out_ptr1", 8, 1): "triton_red_fused__to_copy_add_div_embedding_dense_backward_expand_mul_nll_loss_forward_pow_sum_view_21",
            }
            key = ("in_out_ptr0" if "in_out_ptr0" in inputs else ("out_ptr1" if "out_ptr1" in outputs else "out_ptr0"), len(inputs) - (1 if "in_out_ptr0" in inputs else 0), len(outputs))
            # NLL has five auxiliary in_ptrs, while the two RMS input-gradient
            # variants have three or six auxiliary pointers.
            if "in_out_ptr0" in inputs and len(inputs) == 6 and len(outputs) == 1:
                return by_shape[("in_out_ptr0", 5, 1)]
            if "in_out_ptr0" in inputs and len(inputs) == 4:
                return by_shape[("in_out_ptr0", 3, 1)]
            if "in_out_ptr0" in inputs and len(inputs) == 7:
                return by_shape[("in_out_ptr0", 6, 1)]
            if len(outputs) == 2:
                return by_shape[("out_ptr0", 9, 2)]
            if "out_ptr1" in outputs:
                return by_shape[("out_ptr1", 8, 1)]
            if len(inputs) == 3:
                return by_shape[("out_ptr0", 3, 1)]
            raise KeyError(f"unclassified remaining backward topology: {row.get('region_id')} {inputs} {outputs}")
        # Forward singleton families are similarly separated by pointer/output
        # topology and by the input-free position-mask symbols.
        source_nodes = {str(value) for value in row.get("source_nodes", [])}
        if "loss" in source_nodes and len(inputs) == 2 and len(outputs) == 1:
            return "triton_per_fused__log_softmax__to_copy__unsafe_view_prepare_softmax_online_view_19"
        if "loss" in source_nodes and len(inputs) == 5 and len(outputs) == 2:
            return "triton_per_fused__log_softmax__to_copy__unsafe_view_nll_loss_forward_prepare_softmax_online_slice_sub_view_20"
        if not inputs and outputs == ["out_ptr0"]:
            return (
                "triton_poi_fused__to_copy_arange_unsqueeze_2"
                if "position_ids_expanded" in source_nodes
                else "triton_per_fused_arange_cat_cumsum_ne_slice_sub_unsqueeze_1"
            )
        if len(inputs) == 2 and "in_out_ptr0" in inputs:
            return "triton_per_fused__to_copy__unsafe_view_add_mean_pow_rsqrt_view_3"
        if len(inputs) == 4 and len(outputs) == 2 and "in_out_ptr0" not in inputs:
            return "triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_cos_mul_neg_sin_slice_transpose_unsqueeze_view_5"
        if len(inputs) == 4 and len(outputs) == 2 and "in_out_ptr0" in inputs:
            return "triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_10"
        if len(inputs) == 5 and len(outputs) == 3:
            return "triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12"
        if len(inputs) == 5 and len(outputs) == 2:
            # Terminal NLL has the logits alias plus four auxiliary pointers.
            return "triton_per_fused__log_softmax__to_copy__unsafe_view_nll_loss_forward_prepare_softmax_online_slice_sub_view_20"
        # Loss partial rows are handled above by the dedicated dispatcher.
        if len(inputs) == 1 and outputs == ["out_ptr0"]:
            if "labels" in {str(value) for value in row.get("source_nodes", [])}:
                return "triton_poi_fused_constant_pad_nd_15"
            return "triton_red_fused__to_copy__unsafe_view_prepare_softmax_online_view_16"
        if len(inputs) == 2 and outputs == ["out_ptr0"]:
            return "triton_red_fused__to_copy__unsafe_view_prepare_softmax_online_view_18"
        raise KeyError(f"unclassified remaining forward topology: {row.get('region_id')} {inputs} {outputs}")

    # For all other semantic entrypoints the frozen campaign has one canonical
    # generated symbol, except the two RMSNorm gradient variants.  Pointer
    # topology selects their exact mathematical program.
    if entry.endswith("replay_rmsnorm_input_gradient"):
        return (
            "triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_16"
            if len(inputs) == 7
            else "triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5"
        )
    if entry.endswith("replay_rmsnorm_weight_gradient"):
        return (
            "triton_per_fused__to_copy_add_mul_sum_view_17"
            if len(inputs) == 5
            else "triton_per_fused__to_copy_add_mul_sum_view_6"
        )
    candidates = sorted({str(item["symbol"]) for item in canonical_rows if str(item.get("phase")) == phase and str(item.get("reference_entrypoint")) == entry})
    if len(candidates) == 1:
        return candidates[0]
    # Exact region correspondence remains a safe fallback for seq64/128.
    for item in canonical_rows:
        if str(item.get("region_id")) == str(row.get("region_id")):
            return str(item["symbol"])
    raise KeyError(f"no canonical reference symbol for {row.get('region_id')} {entry}")


def _dtype_mapping_phase(
    symbol: str,
    reference_symbol: str,
    canonical_phases: dict[str, set[str]],
) -> str:
    """Resolve the F/B side without inspecting candidate tensor values.

    The original static dtype maps did not carry a phase field.  Prefer the
    frozen reference campaign's phase for ordinary symbols; source-derived
    ``forkcert:`` aliases are assigned from their semantic name.  The final
    substring fallback is deliberately explicit and candidate-blind.
    """
    phases = canonical_phases.get(reference_symbol, set())
    if len(phases) == 1:
        return next(iter(phases))
    text = f"{symbol} {reference_symbol}".lower()
    backward_tokens = (
        "backward", "vjp", "rms-weight", "rmsnorm-input-gradient",
        "rmsnorm-vjp", "rotary-reduction", "rotary-weight", "reduce-last",
        "embedding-zero-fill", "softmax-backward",
    )
    if any(token in text for token in backward_tokens):
        return "BACKWARD"
    forward_tokens = (
        "forward", "softmax-forward", "query-rotary-forward",
        "key-rotary-forward", "kv-head-repeat", "embedding-rmsnorm-forward",
        "loss-softmax-seq256-partial",
    )
    if any(token in text for token in forward_tokens):
        return "FORWARD"
    if any(token in text for token in ("mul_sum", "add_div")):
        return "BACKWARD"
    if any(token in text for token in (
        "prepare_softmax_online", "arange_cat", "constant_pad_nd",
        "clone_expand", "embedding_mean", "add_bmm_cat_cos_mean",
    )):
        return "FORWARD"
    raise ValueError(
        f"cannot resolve candidate-blind F/B phase for dtype mapping: "
        f"{symbol} -> {reference_symbol}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-manifest", type=Path, default=Path("results/final/natural_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--campaign", type=Path, default=None)
    parser.add_argument(
        "--dtype-mapping",
        type=Path,
        default=None,
        help="conservative dtype-specific symbol/topology mapping; mapped rows are observed and unresolved rows stay out of the semantic layer",
    )
    parser.add_argument("--seq-len", type=int, choices=(64, 128, 256), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--metric-chunk-elements", type=int, default=1_048_576)
    parser.add_argument(
        "--intervene-region",
        action="append",
        default=[],
        help=(
            "replace the selected ordinary generated region's output with its "
            "candidate-blind reference before downstream execution; repeat "
            "for multiple regions"
        ),
    )
    parser.add_argument(
        "--carrier-parameter",
        action="append",
        default=[],
        help=(
            "parameter whose gradient is retained for intervention attribution; "
            "repeat for multiple carriers (default: tied embedding when intervening)"
        ),
    )
    parser.add_argument(
        "--carrier-sketch-input",
        type=Path,
        default=None,
        help="torch.save carrier-difference coordinate sketch from the pilot checkpoint",
    )
    parser.add_argument(
        "--carrier-sketch-output",
        type=Path,
        default=None,
        help="write the first-repeat intervention-difference coordinate sketch",
    )
    parser.add_argument("--carrier-sketch-sample-size", type=int, default=131072)
    parser.add_argument("--implementation-atlas", type=Path, default=Path("results/final/implementation_atlas.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.bank_manifest = under_root(args.bank_manifest, "bank manifest")
    args.model = under_root(args.model, "model")
    args.output = under_root(args.output, "output")
    args.implementation_atlas = under_root(args.implementation_atlas, "implementation atlas")
    if args.dtype_mapping is not None:
        args.dtype_mapping = under_root(args.dtype_mapping, "dtype semantic mapping")
    if args.carrier_sketch_input is not None:
        args.carrier_sketch_input = under_root(args.carrier_sketch_input, "carrier sketch input")
    if args.carrier_sketch_output is not None:
        args.carrier_sketch_output = under_root(args.carrier_sketch_output, "carrier sketch output")
    if args.campaign is None:
        if args.seq_len == 64:
            args.campaign = REPO / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/triton_online_reference_campaign_v1.json"
        else:
            args.campaign = REPO / f"results/training_semantic_oracle/qwen3_1p7b/configuration_specific_inventories/bf16_inductor/seq{args.seq_len}/triton_online_reference_campaign.json"
    args.campaign = under_root(args.campaign, "reference campaign")
    if args.tf32 and args.dtype != "fp32":
        raise ValueError("--tf32 requires --dtype fp32")
    if args.sample_size < 1 or args.metric_chunk_elements < 1:
        raise ValueError("observation bounds must be positive")
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

    from scripts.checkpoint_inductor import build_model, load_natural_validation, under_root as checkpoint_under_root
    from forkcert.generated_triton_reference_observer import GeneratedTritonReferenceObserver
    from forkcert.generated_embedding_reference_observer import GeneratedEmbeddingReferenceObserver

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.benchmark = False

    bank = json.loads(args.bank_manifest.read_text())
    checkpoint_row = next((row for row in bank["checkpoints"] if int(row["step"]) == args.step), None)
    if checkpoint_row is None:
        raise RuntimeError(f"checkpoint step {args.step} is absent from bank")
    checkpoint = checkpoint_under_root(Path(checkpoint_row["path"]), "checkpoint")
    campaign = json.loads(args.campaign.read_text())
    canonical_campaign_path = REPO / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/triton_online_reference_campaign_v1.json"
    canonical_rows = json.loads(canonical_campaign_path.read_text())["rows"]
    canonical_phases: dict[str, set[str]] = {}
    for canonical_row in canonical_rows:
        canonical_symbol = str(canonical_row.get("reference_symbol", canonical_row["symbol"]))
        canonical_phases.setdefault(canonical_symbol, set()).add(str(canonical_row["phase"]))
    dtype_mapping = None
    if args.dtype_mapping is not None:
        dtype_mapping = json.loads(args.dtype_mapping.read_text())
        if str(dtype_mapping["dtype"]) != args.dtype or bool(dtype_mapping["tf32"]) != bool(args.tf32) or int(dtype_mapping["seq_len"]) != args.seq_len:
            raise ValueError("dtype mapping configuration does not match observation configuration")
        original_campaign_rows = []
        for mapping_row in dtype_mapping["rows"]:
            if mapping_row["status"] != "MAPPED":
                continue
            symbol = str(mapping_row["symbol"])
            reference_symbol = str(mapping_row["reference_symbol"])
            input_names = list(mapping_row["input_names"])
            output_names = list(mapping_row["output_names"])
            phase = _dtype_mapping_phase(symbol, reference_symbol, canonical_phases)
            for invocation_index in range(int(mapping_row["invocations"])):
                original_campaign_rows.append({
                    "boundary_capture_mode": "DTYPE_MAPPED_EXACT_RUNTIME_POINTERS",
                    "heldout_execution_status": "PENDING_GPU_HELDOUT_ONLINE_REFERENCE",
                    "input_names": input_names,
                    "output_names": output_names,
                    "prelaunch_clone_names": [name for name in input_names if name.startswith("in_out_")],
                    "phase": phase,
                    "reference_entrypoint": "forkcert.qwen3_triton_reference_dispatch:same_precision_reference",
                    "reference_symbol": reference_symbol,
                    "canonical_reference_symbol": reference_symbol,
                    "region_id": f"dtype:{args.dtype}:{'tf32' if args.tf32 else 'strict'}:seq{args.seq_len}:{symbol}:{invocation_index}",
                    "single_state_replay_schema": "forkcert.dtype-mapped-runtime-record.v1",
                    "source_nodes": [],
                    "symbol": symbol,
                })
    else:
        original_campaign_rows = list(campaign["rows"])
        for row in original_campaign_rows:
            row["canonical_reference_symbol"] = _canonical_reference_symbol(row, canonical_rows)
    expected_original_symbols = sorted({str(row["symbol"]) for row in original_campaign_rows})

    tokenizer_mod = __import__("transformers", fromlist=["AutoTokenizer"])
    tokenizer = tokenizer_mod.AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    eval_args = SimpleNamespace(
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        eval_split=args.eval_split,
        eval_offset=args.eval_offset,
        seq_len=args.seq_len,
        device=device,
    )
    inputs, eval_protocol = load_natural_validation(tokenizer, eval_args)

    model = build_model(args.model, checkpoint, "eager", dtype, device)
    model.config.use_cache = False

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
    input_ids, labels = inputs
    model.zero_grad(set_to_none=True)
    warm_loss = candidate(input_ids, labels)
    warm_loss.backward()
    torch.cuda.synchronize(device)
    warmed_symbols = discover_all_triton_symbols(list(PyCodeCache.modules[module_start:]))
    campaign_rows, unmatched_warmed_symbols = remap_campaign_to_warmed_symbols(
        original_campaign_rows,
        warmed_symbols,
        allow_extra_same_stem=dtype_mapping is not None,
    )
    expected_regions = {str(row["region_id"]) for row in campaign_rows if not str(row.get("boundary_capture_mode", "")).startswith("SPECIALIZED_")}
    if not expected_regions:
        raise RuntimeError("reference campaign has no ordinary Triton regions")
    atlas = json.loads(args.implementation_atlas.read_text())
    changed_region_ids = (
        {str(row["region_id"]) for row in campaign_rows}
        if dtype_mapping is not None
        else {
            str(row["region"])
            for row in atlas["rows"]
            if bool(row.get("implementation_changed"))
            and str(row.get("mechanism")) in {
                "EXPLICIT_FP32_REDUCTION_SCHEDULE_DIFFERENCE",
                "MATERIALIZATION_OR_ROUNDING_SCHEDULE_INTERVENTION",
                "SAME_PRECISION_GENERATED_SCHEDULE_DIFFERENCE",
            }
        }
    )
    baseline_loss_digest = tensor_digest(warm_loss)
    intervention_mode = bool(args.intervene_region)
    carrier_parameter_names = list(args.carrier_parameter)
    if intervention_mode and not carrier_parameter_names:
        carrier_parameter_names = ["model.embed_tokens.weight"]
    digest_parameter_names = carrier_parameter_names if intervention_mode else None
    baseline_gradient_digest, baseline_parameter_digests = gradient_digest(
        model,
        digest_parameter_names,
    )
    baseline_carrier_gradients = (
        capture_named_gradients(model, carrier_parameter_names)
        if intervention_mode
        else {}
    )
    pilot_carrier_sketch = None
    if args.carrier_sketch_input is not None:
        pilot_carrier_sketch = torch.load(
            args.carrier_sketch_input,
            map_location="cpu",
            weights_only=False,
        )
        if pilot_carrier_sketch.get("schema") != "kernel-analyzer-carrier-difference-sketch-v1":
            raise ValueError("carrier sketch schema mismatch")
        if pilot_carrier_sketch.get("parameter_names") != carrier_parameter_names:
            raise ValueError("carrier sketch parameter list mismatch")
    module_count = len(PyCodeCache.modules)

    repeats = []
    carrier_sketch_payload = None
    for repeat_id in range(2):
        observer = GeneratedTritonReferenceObserver(
            modules=PyCodeCache.modules[module_start:],
            campaign_rows=campaign_rows,
            sequence=args.seq_len,
            sample_size=args.sample_size,
            metric_chunk_elements=args.metric_chunk_elements,
            intervene_region_ids=args.intervene_region,
            allow_unclosed_closures=args.dtype_mapping is not None,
        )
        embedding_observer = GeneratedEmbeddingReferenceObserver(
            modules=PyCodeCache.modules[module_start:],
            chunk_elements=args.metric_chunk_elements,
            sample_size=args.sample_size,
        )
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with observer, embedding_observer:
            loss = candidate(input_ids, labels)
            loss.backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        loss_digest = tensor_digest(loss)
        gradient_digest_value, parameter_digests = gradient_digest(
            model,
            digest_parameter_names,
        )
        intervention_carrier_stats = None
        intervention_carrier_sketch = None
        if intervention_mode:
            observed_carrier_gradients = capture_named_gradients(
                model, carrier_parameter_names
            )
            intervention_carrier_stats = gradient_difference_stats(
                baseline_carrier_gradients,
                observed_carrier_gradients,
            )
            sketches = sampled_gradient_difference(
                baseline_carrier_gradients,
                observed_carrier_gradients,
                parameter_names=carrier_parameter_names,
                sample_size=args.carrier_sketch_sample_size,
            )
            intervention_carrier_sketch = {}
            for name, item in sketches.items():
                if item.get("status") != "OK":
                    intervention_carrier_sketch[name] = item
                    continue
                values = item["values"]
                row = {
                    "status": "OK",
                    "numel": item["numel"],
                    "sample_size": item["sample_size"],
                    "sketch_l2": float(values.norm()),
                }
                if pilot_carrier_sketch is not None:
                    pilot = pilot_carrier_sketch["sketches"][name]
                    pilot_values = pilot["values"].to(values.dtype)
                    dot = float((values * pilot_values).sum())
                    denom = float(values.norm() * pilot_values.norm())
                    row["pilot_dot"] = dot
                    row["pilot_cosine"] = dot / (denom + 1e-30)
                    row["pilot_step"] = pilot_carrier_sketch["pilot_step"]
                intervention_carrier_sketch[name] = row
            if repeat_id == 0 and args.carrier_sketch_output is not None:
                carrier_sketch_payload = {
                    "schema": "kernel-analyzer-carrier-difference-sketch-v1",
                    "pilot_step": args.step,
                    "parameter_names": carrier_parameter_names,
                    "sample_size": args.carrier_sketch_sample_size,
                    "sketches": {
                        name: {
                            "numel": item["numel"],
                            "sample_size": item["sample_size"],
                            "indices": item["indices"],
                            "values": item["values"],
                        }
                        for name, item in sketches.items()
                        if item.get("status") == "OK"
                    },
                }
        summary = observer.summary()
        summary_for_output = dict(summary)
        summary_for_output.pop("records", None)
        stability = (
            loss_digest == baseline_loss_digest
            and gradient_digest_value == baseline_gradient_digest
            and parameter_digests == baseline_parameter_digests
            and len(PyCodeCache.modules) == module_count
        )
        intervention_observed = sorted(
            str(value)
            for value in observer.summary().get("intervention", {}).get(
                "observed_region_ids", []
            )
        )
        def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
            value = dict(metrics)
            sketch = value.get("directional_error_sketch")
            if isinstance(sketch, dict):
                value["directional_error_sketch"] = {
                    key: item
                    for key, item in sketch.items()
                    if key not in {"candidate_values", "reference_values", "signed_delta_values", "flat_coordinate_indices"}
                }
            return value

        records = []
        for raw_record in observer.records:
            if str(raw_record.region_id) not in changed_region_ids:
                continue
            record = raw_record.as_dict()
            record["endpoint_metrics"] = {
                name: compact_metrics(value)
                for name, value in record["endpoint_metrics"].items()
            }
            if record.get("semantic_closure") is not None:
                record["semantic_closure"]["endpoint_metrics"] = {
                    name: compact_metrics(value)
                    for name, value in record["semantic_closure"]["endpoint_metrics"].items()
                }
            records.append(record)
        specialized_rows = {
            (str(row["symbol"]), int(row.get("invocation_index", 0))): row
            for row in campaign_rows
            if str(row.get("boundary_capture_mode", "")).startswith("SPECIALIZED_")
        }
        for raw_record in embedding_observer.forward_online_records:
            key = (str(raw_record["symbol"]), int(raw_record["invocation_index"]))
            row = specialized_rows.get(key)
            if row is None or str(row["region_id"]) not in changed_region_ids:
                continue
            records.append({
                "region_id": str(row["region_id"]),
                "phase": str(row["phase"]),
                "symbol": key[0],
                "reference_symbol": key[0],
                "invocation_index": key[1],
                "kernel_hash": raw_record.get("kernel_hash"),
                "reference_entrypoint": str(row["reference_entrypoint"]),
                "endpoint_metrics": {
                    name: compact_metrics(value)
                    for name, value in raw_record["endpoint_metrics"].items()
                },
                "semantic_closure": None,
            })
        for raw_record in embedding_observer.online_records:
            key = (str(raw_record["symbol"]), int(raw_record["invocation_index"]))
            row = specialized_rows.get(key)
            if row is None or str(row["region_id"]) not in changed_region_ids:
                continue
            metric = compact_metrics(raw_record["metrics"])
            if raw_record.get("semantic_closure_metrics") is not None:
                metric = compact_metrics(raw_record["semantic_closure_metrics"])
            records.append({
                "region_id": str(row["region_id"]),
                "phase": str(row["phase"]),
                "symbol": key[0],
                "reference_symbol": key[0],
                "invocation_index": key[1],
                "kernel_hash": raw_record.get("kernel_hash"),
                "reference_entrypoint": str(row["reference_entrypoint"]),
                "endpoint_metrics": {"out_ptr0": metric},
                "semantic_closure": None,
            })
        repeats.append({
            "repeat_id": repeat_id,
            "loss": float(loss.detach().float().cpu()),
            "loss_digest": loss_digest,
            "gradient_digest": gradient_digest_value,
            "observation_stable_against_unobserved_full_step": stability,
            "intervention_observed_region_ids": intervention_observed,
            "intervention_carrier_parameter_stats": intervention_carrier_stats,
            "intervention_carrier_sketch": intervention_carrier_sketch,
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "triton_summary": summary_for_output,
            "embedding_summary": {
                "discovered_symbols": embedding_observer.discovered_symbols,
                "forward_online_record_count": len(embedding_observer.forward_online_records),
                "online_record_count": len(embedding_observer.online_records),
            },
            "records": records,
        })

    output = {
        "schema": "kernel-analyzer-evolving-triton-observation-v1",
        "subject": "Qwen3-1.7B generated Triton regions on natural checkpoint",
        "checkpoint_step": args.step,
        "checkpoint_parameter_sha256": checkpoint_row["parameter_sha256"],
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "tf32": bool(args.tf32),
        "campaign": str(args.campaign),
        "campaign_sha256": hashlib.sha256(args.campaign.read_bytes()).hexdigest(),
        "dtype_mapping": str(args.dtype_mapping) if args.dtype_mapping is not None else None,
        "dtype_mapping_sha256": hashlib.sha256(args.dtype_mapping.read_bytes()).hexdigest() if args.dtype_mapping is not None else None,
        "intervention_region_ids": sorted(str(value) for value in args.intervene_region),
        "carrier_parameter_names": sorted(carrier_parameter_names),
        "carrier_sketch_input": (
            args.carrier_sketch_input.name
            if args.carrier_sketch_input is not None
            else None
        ),
        "carrier_sketch_output": (
            args.carrier_sketch_output.name
            if args.carrier_sketch_output is not None
            else None
        ),
        "bank_manifest": "results/final/natural_bank.json",
        "bank_protocol_sha256": bank["protocol_sha256"],
        "evaluation": eval_protocol,
        "expected_ordinary_triton_regions": len(expected_regions),
        "changed_region_ids_expected": len(changed_region_ids),
        "expected_campaign_symbol_count": len(expected_original_symbols),
        "warmed_symbol_count": len(warmed_symbols),
        "unmatched_warmed_symbols": unmatched_warmed_symbols,
            "repeats": repeats,
        "compile_audit": audit,
            "gates": {
            "all_expected_ordinary_regions_observed_twice": all(
                row["triton_summary"]["denominator"]["observed_ordinary_triton_regions"] == len(expected_regions)
                for row in repeats
            ),
            "all_changed_region_ids_retained_twice": all(
                len(row["records"]) == len(changed_region_ids)
                and {str(x["region_id"]) for x in row["records"]} == changed_region_ids
                for row in repeats
            ),
            "all_observation_repeats_stable": (
                all(row["observation_stable_against_unobserved_full_step"] for row in repeats)
                if not intervention_mode
                else False
            ),
            "intervention_runs_are_explicitly_nonbaseline": intervention_mode,
            "all_requested_intervention_regions_observed": all(
                set(row["intervention_observed_region_ids"]) == set(args.intervene_region)
                for row in repeats
            ),
            "candidate_values_used_to_select_regions": False,
            "intervention_region_ids_are_explicit": sorted(str(value) for value in args.intervene_region),
        },
        "boundary": (
            "Online same-input reference metrics for every ordinary Triton invocation; external exact-replay calls and internal schedule boundaries remain separately classified. "
            "When dtype_mapping is supplied, only candidate-blind unique symbol/topology mappings enter this layer; all unresolved generated symbols remain outside semantic attribution. An intervention run changes downstream execution and therefore is not expected to match the unobserved full-step digest."
        ),
    }
    output["result_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if carrier_sketch_payload is not None:
        args.carrier_sketch_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(carrier_sketch_payload, args.carrier_sketch_output)
    print(json.dumps({"output": str(args.output), "step": args.step, "seq_len": args.seq_len, "regions": len(expected_regions), "stable": output["gates"]["all_observation_repeats_stable"]}, sort_keys=True))
    del candidate, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
