#!/usr/bin/env python
"""Fail-closed matched-state inference Oracle for real Transformer/CNN subjects."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "forkcert.real-model-oracle.v2"


@dataclass
class CompileAudit:
    backend_compiles: int = 0
    runtime_invocations: int = 0
    graph_code_sha256: list[str] = field(default_factory=list)
    graph_node_counts: list[int] = field(default_factory=list)


@dataclass
class State:
    state_id: str
    inputs: tuple[Any, ...]
    observation_positions: list[int] | None
    targets: list[int] | None
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", choices=["bert_sst2", "resnet18", "qwen_causal"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest(path: Path) -> dict[str, Any]:
    names = ["config.json", "model.safetensors", "pytorch_model.bin", "tokenizer.json", "vocab.txt"]
    files = []
    for name in names:
        item = path / name
        if item.is_file():
            files.append({"name": name, "size": item.stat().st_size, "sha256": sha256_file(item)})
    return {"path": str(path.resolve()), "files": files}


def make_tracking_backend(torch: Any, audit: CompileAudit) -> Callable[..., Any]:
    def backend(graph_module: Any, example_inputs: list[Any]) -> Callable[..., Any]:
        audit.backend_compiles += 1
        audit.graph_code_sha256.append(sha256_text(graph_module.code))
        audit.graph_node_counts.append(sum(1 for _ in graph_module.graph.nodes))
        compiled = torch._inductor.compile(graph_module, example_inputs)

        def counted(*args: Any) -> Any:
            audit.runtime_invocations += 1
            return compiled(*args)

        return counted

    return backend


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_subject(torch: Any, args: argparse.Namespace) -> tuple[Any, Callable[..., Any], list[State], dict[str, Any]]:
    from datasets import load_from_disk
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageClassification,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    model_path = Path(args.model_path)
    if args.subject == "bert_sst2":
        if args.sequence_length is None:
            args.sequence_length = 64
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, dtype=torch.float16)
        model.eval().to("cuda")
        dataset = load_from_disk(args.data_path)
        if args.start != 0 or args.count > len(dataset):
            raise ValueError(f"saved dataset already represents a frozen partition; require start=0 and count<={len(dataset)}")
        rows = dataset.select(range(args.count))
        states = []
        for ordinal, row in enumerate(rows):
            encoded = tokenizer(
                row["sentence"],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=args.sequence_length,
            )
            states.append(
                State(
                    state_id=f"sst2-{int(row['idx']):06d}",
                    inputs=(encoded["input_ids"].to("cuda"), encoded["attention_mask"].to("cuda")),
                    observation_positions=None,
                    targets=[int(row["label"])],
                    metadata={"ordinal": ordinal, "dataset_idx": int(row["idx"]), "label": int(row["label"])},
                )
            )

        def forward(input_ids: Any, attention_mask: Any) -> Any:
            return model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]

        return model, forward, states, {"label_space": "sst2", "label_compatible": True, "output_classes": 2}

    if args.subject == "resnet18":
        import numpy as np
        from PIL import Image

        processor_config = json.loads((model_path / "preprocessor_config.json").read_text())
        crop_size = int(processor_config["size"])
        crop_pct = float(processor_config.get("crop_pct", 1.0))
        resize_size = int(round(crop_size / crop_pct))
        image_mean = np.asarray(processor_config["image_mean"], dtype=np.float32)[:, None, None]
        image_std = np.asarray(processor_config["image_std"], dtype=np.float32)[:, None, None]
        model = AutoModelForImageClassification.from_pretrained(model_path, dtype=torch.float16)
        model.eval().to("cuda")
        dataset = load_from_disk(args.data_path)
        if args.start != 0 or args.count > len(dataset):
            raise ValueError(f"saved dataset already represents a frozen partition; require start=0 and count<={len(dataset)}")
        rows = dataset.select(range(args.count))
        states = []
        for ordinal, row in enumerate(rows):
            image = row["img"].convert("RGB")
            width, height = image.size
            scale = resize_size / min(width, height)
            resized = image.resize(
                (round(width * scale), round(height * scale)),
                resample=Image.Resampling.BICUBIC,
            )
            left = (resized.width - crop_size) // 2
            top = (resized.height - crop_size) // 2
            cropped = resized.crop((left, top, left + crop_size, top + crop_size))
            values = np.asarray(cropped, dtype=np.float32).transpose(2, 0, 1) / 255.0
            values = (values - image_mean) / image_std
            pixel_values = torch.from_numpy(values.copy()).unsqueeze(0).to(device="cuda", dtype=torch.float16)
            states.append(
                State(
                    state_id=f"cifar10-{ordinal:06d}",
                    inputs=(pixel_values,),
                    observation_positions=None,
                    targets=None,
                    metadata={"ordinal": ordinal, "cifar10_label": int(row["label"]), "label_compatible": False},
                )
            )

        def forward(pixel_values: Any) -> Any:
            return model(pixel_values=pixel_values, return_dict=False)[0]

        return model, forward, states, {
            "label_space": "imagenet-1k",
            "label_compatible": False,
            "output_classes": 1000,
            "preprocessing": {
                "resize_shorter_side": resize_size,
                "center_crop": crop_size,
                "resample": "PIL bicubic",
                "image_mean": processor_config["image_mean"],
                "image_std": processor_config["image_std"],
                "source_config": str((model_path / "preprocessor_config.json").resolve()),
            },
        }

    if args.sequence_length is None:
        args.sequence_length = 166
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.eval().to("cuda")
    rows = read_jsonl(Path(args.data_path))
    stop = args.start + args.count
    if stop > len(rows):
        raise ValueError(f"requested [{args.start},{stop}) from {len(rows)} Qwen samples")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Qwen tokenizer lacks both pad and eos token")
    states = []
    for source_index, row in enumerate(rows[args.start:stop], start=args.start):
        prompt_ids = [int(value) for value in row["prompt_ids"]]
        response_ids = [int(value) for value in row["response_ids"]]
        full = (prompt_ids + response_ids)[: args.sequence_length]
        attention = [1] * len(full)
        if len(full) < args.sequence_length:
            padding = args.sequence_length - len(full)
            full += [int(pad_id)] * padding
            attention += [0] * padding
        response_count = max(0, min(len(response_ids), args.sequence_length - len(prompt_ids)))
        positions = [len(prompt_ids) - 1 + offset for offset in range(response_count)]
        targets = response_ids[:response_count]
        if not positions or positions[-1] >= args.sequence_length:
            raise ValueError(f"invalid response positions for {row['case_id']}")
        states.append(
            State(
                state_id=str(row["case_id"]),
                inputs=(
                    torch.tensor([full], device="cuda", dtype=torch.long),
                    torch.tensor([attention], device="cuda", dtype=torch.long),
                ),
                observation_positions=positions,
                targets=targets,
                metadata={
                    "source_index": source_index,
                    "rollout_batch": row.get("metadata", {}).get("rollout_batch"),
                    "prompt_tokens": len(prompt_ids),
                    "response_tokens": response_count,
                },
            )
        )

    def forward(input_ids: Any, attention_mask: Any) -> Any:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )[0]

    return model, forward, states, {"label_space": "qwen-vocabulary", "label_compatible": True, "output_classes": model.config.vocab_size}


def attention_context(torch: Any, subject: str) -> Any:
    if subject not in {"bert_sst2", "qwen_causal"}:
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel(SDPBackend.MATH)


def execute(
    torch: Any,
    fn: Callable[..., Any],
    state: State,
    path: str,
    subject: str,
    audit: CompileAudit,
) -> dict[str, Any]:
    before = audit.runtime_invocations
    started = time.perf_counter_ns()
    with torch.no_grad(), attention_context(torch, subject):
        logits = fn(*state.inputs)
    torch.cuda.synchronize()
    elapsed_ns = time.perf_counter_ns() - started
    runtime_delta = audit.runtime_invocations - before
    if path == "compiled" and runtime_delta <= 0:
        raise RuntimeError(f"fail-closed: {state.state_id} candidate call did not reach compiled callable")
    if path == "eager" and runtime_delta != 0:
        raise RuntimeError("eager call changed compiled invocation counter")
    finite = bool(torch.isfinite(logits).all().item())
    if not finite:
        raise FloatingPointError(f"non-finite output for {state.state_id} {path}")
    return {
        "path": path,
        "logits": logits.detach(),
        "elapsed_ns": elapsed_ns,
        "compiled_runtime_invocations": runtime_delta,
        "shape": list(logits.shape),
        "dtype": str(logits.dtype),
    }


def selected_logits(torch: Any, output: dict[str, Any], state: State, subject: str) -> Any:
    logits = output["logits"]
    if subject == "qwen_causal":
        positions = torch.tensor(state.observation_positions, device=logits.device, dtype=torch.long)
        return logits[0].index_select(0, positions)
    return logits


def compare_outputs(
    torch: Any,
    eager: dict[str, Any],
    compiled: dict[str, Any],
    state: State,
    subject: str,
    repeat: int,
    execution_order: list[str],
) -> list[dict[str, Any]]:
    ref = selected_logits(torch, eager, state, subject).float()
    alt = selected_logits(torch, compiled, state, subject).float()
    if ref.shape != alt.shape or ref.ndim != 2:
        raise ValueError(f"selected logit shape mismatch: {ref.shape} vs {alt.shape}")
    observations, classes = ref.shape
    top_count = min(6, classes)
    ref_top_values, ref_top_indices = torch.topk(ref, k=top_count, dim=-1)
    alt_top_values, alt_top_indices = torch.topk(alt, k=top_count, dim=-1)
    delta = alt - ref
    # Split each vocabulary/class-coordinate delta into an exactly
    # decision-invariant common translation and the orthogonal centered
    # residual.  Use float64 for the diagnostic so the energy-closure audit is
    # not itself dominated by the FP32 analysis reduction.
    delta64 = delta.double()
    signed_mean64 = delta64.mean(dim=-1)
    centered64 = delta64 - signed_mean64.unsqueeze(-1)
    raw_energy64 = delta64.square().sum(dim=-1)
    common_energy64 = signed_mean64.square() * classes
    centered_energy64 = centered64.square().sum(dim=-1)
    nonzero_energy = raw_energy64 > 0
    common_energy_share = torch.where(nonzero_energy, common_energy64 / raw_energy64, 0.0)
    centered_energy_share = torch.where(nonzero_energy, centered_energy64 / raw_energy64, 0.0)
    energy_closure_relative_error = torch.where(
        nonzero_energy,
        (raw_energy64 - common_energy64 - centered_energy64).abs() / raw_energy64,
        0.0,
    )
    if bool((energy_closure_relative_error > 1e-10).any().item()):
        raise ValueError(
            "common-mode/centered logit energy decomposition failed: "
            f"max relative closure error={float(energy_closure_relative_error.max().item())}"
        )
    signed_mean = delta.mean(dim=-1)
    mean_abs = delta.abs().mean(dim=-1)
    max_abs = delta.abs().max(dim=-1).values
    centered_mean_abs = centered64.abs().mean(dim=-1)
    centered_max_abs = centered64.abs().max(dim=-1).values
    targets = None
    ref_target_logp = None
    alt_target_logp = None
    if state.targets is not None:
        if len(state.targets) != observations:
            raise ValueError(f"target count {len(state.targets)} != observations {observations}")
        targets = torch.tensor(state.targets, device=ref.device, dtype=torch.long)
        indices = targets.unsqueeze(-1)
        ref_target_logp = ref.gather(-1, indices).squeeze(-1) - torch.logsumexp(ref, dim=-1)
        alt_target_logp = alt.gather(-1, indices).squeeze(-1) - torch.logsumexp(alt, dim=-1)
    rows = []
    for index in range(observations):
        ref_order = [int(value) for value in ref_top_indices[index].tolist()]
        alt_order = [int(value) for value in alt_top_indices[index].tolist()]
        top1_margin_ref = float((ref_top_values[index, 0] - ref_top_values[index, 1]).item())
        top1_margin_alt = float((alt_top_values[index, 0] - alt_top_values[index, 1]).item())
        epsilon = float(max_abs[index].item())
        row = {
            "state_id": state.state_id,
            "observation_index": index,
            "repeat": repeat,
            "execution_order": execution_order,
            "subject": subject,
            "logit_mean_signed_delta": float(signed_mean[index].item()),
            "logit_common_mode_shift": float(signed_mean64[index].item()),
            "logit_mean_abs_delta": float(mean_abs[index].item()),
            "logit_max_abs_delta": epsilon,
            "centered_logit_mean_abs_delta": float(centered_mean_abs[index].item()),
            "centered_logit_max_abs_delta": float(centered_max_abs[index].item()),
            "logit_common_mode_energy_share": float(common_energy_share[index].item()),
            "centered_logit_energy_share": float(centered_energy_share[index].item()),
            "logit_energy_decomposition_relative_error": float(
                energy_closure_relative_error[index].item()
            ),
            "reference_argmax": ref_order[0],
            "candidate_argmax": alt_order[0],
            "argmax_disagreement": ref_order[0] != alt_order[0],
            "reference_top1_margin": top1_margin_ref,
            "candidate_top1_margin": top1_margin_alt,
            "top1_margin_signed_delta": top1_margin_alt - top1_margin_ref,
            "reference_top1_exact_tie": top1_margin_ref == 0.0,
            "argmax_stability_ratio": (2.0 * epsilon / top1_margin_ref) if top1_margin_ref > 0 else None,
            "top_order_reference": ref_order,
            "top_order_candidate": alt_order,
            "top_order_disagreement": ref_order != alt_order,
            "state_metadata": state.metadata,
        }
        if top_count >= 6:
            ref_top5 = ref_order[:5]
            alt_top5 = alt_order[:5]
            top5_margin_ref = float((ref_top_values[index, 4] - ref_top_values[index, 5]).item())
            top5_margin_alt = float((alt_top_values[index, 4] - alt_top_values[index, 5]).item())
            row.update(
                {
                    "reference_top5_margin": top5_margin_ref,
                    "candidate_top5_margin": top5_margin_alt,
                    "top5_margin_signed_delta": top5_margin_alt - top5_margin_ref,
                    "top5_set_disagreement": set(ref_top5) != set(alt_top5),
                    "reference_top5_exact_tie": top5_margin_ref == 0.0,
                    "top5_stability_ratio": (2.0 * epsilon / top5_margin_ref) if top5_margin_ref > 0 else None,
                }
            )
        else:
            row.update(
                {
                    "reference_top5_margin": None,
                    "candidate_top5_margin": None,
                    "top5_margin_signed_delta": None,
                    "top5_set_disagreement": None,
                    "reference_top5_exact_tie": None,
                    "top5_stability_ratio": None,
                }
            )
        if targets is not None and ref_target_logp is not None and alt_target_logp is not None:
            target = int(targets[index].item())
            reference_logp = float(ref_target_logp[index].item())
            candidate_logp = float(alt_target_logp[index].item())
            row.update(
                {
                    "target_id": target,
                    "reference_target_logp": reference_logp,
                    "candidate_target_logp": candidate_logp,
                    "target_logp_signed_delta": candidate_logp - reference_logp,
                    "reference_correct": ref_order[0] == target,
                    "candidate_correct": alt_order[0] == target,
                    "correctness_event_disagreement": (ref_order[0] == target) != (alt_order[0] == target),
                }
            )
        else:
            row.update(
                {
                    "target_id": None,
                    "reference_target_logp": None,
                    "candidate_target_logp": None,
                    "target_logp_signed_delta": None,
                    "reference_correct": None,
                    "candidate_correct": None,
                    "correctness_event_disagreement": None,
                }
            )
        if subject == "bert_sst2":
            row.update(
                {
                    "reference_positive": ref_order[0] == 1,
                    "candidate_positive": alt_order[0] == 1,
                    "positive_up": ref_order[0] == 0 and alt_order[0] == 1,
                    "positive_down": ref_order[0] == 1 and alt_order[0] == 0,
                }
            )
        if subject == "qwen_causal":
            row["token_prediction_position"] = int(state.observation_positions[index])
        rows.append(row)
    return rows


def exact_mcnemar(up: int, down: int) -> float:
    discordant = up + down
    if discordant == 0:
        return 1.0
    tail = min(up, down)
    return min(1.0, 2.0 * sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def quantile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def state_bootstrap_ci(rows: list[dict[str, Any]], field: str, draws: int, seed: int) -> list[float | None]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None and math.isfinite(float(value)):
            grouped[row["state_id"]].append(float(value))
    state_values = {key: mean(values) for key, values in grouped.items()}
    keys = sorted(state_values)
    if not keys:
        return [None, None]
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        estimates.append(mean([state_values[rng.choice(keys)] for _ in keys]))
    return [quantile(estimates, 0.025), quantile(estimates, 0.975)]


def repeated_variance(rows: list[dict[str, Any]], field: str) -> float:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None and math.isfinite(float(value)):
            grouped[(row["state_id"], int(row["observation_index"]))].append(float(value))
    values = [statistics.variance(items) if len(items) > 1 else 0.0 for items in grouped.values()]
    return mean(values) if values else float("nan")


def state_heterogeneity(rows: list[dict[str, Any]], field: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None and math.isfinite(float(value)):
            grouped[row["state_id"]].append(float(value))
    values = [mean(items) for items in grouped.values()]
    return statistics.variance(values) if len(values) > 1 else 0.0


def build_summary(
    rows: list[dict[str, Any]],
    state_records: list[dict[str, Any]],
    audit: CompileAudit,
    args: argparse.Namespace,
    warmup_invocations: int,
) -> dict[str, Any]:
    primary = [row for row in rows if row["repeat"] == 0]
    argmax_disagreements = [row for row in primary if row["argmax_disagreement"]]
    argmax_risk = [
        row
        for row in primary
        if row["argmax_stability_ratio"] is None or float(row["argmax_stability_ratio"]) >= 1.0
    ]
    top5_rows = [row for row in primary if row["top5_set_disagreement"] is not None]
    top5_disagreements = [row for row in top5_rows if row["top5_set_disagreement"]]
    top5_risk = [
        row
        for row in top5_rows
        if row["top5_stability_ratio"] is None or float(row["top5_stability_ratio"]) >= 1.0
    ]
    target_rows = [row for row in primary if row["target_logp_signed_delta"] is not None]
    state_count = len({row["state_id"] for row in primary})
    self_nonzero = [row for row in state_records if row["repeat"] > 0 and not row["self_pair_exact"]]
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "subject": args.subject,
        "validity": {
            "candidate_calls_valid": all(row["candidate_execution_valid"] for row in state_records),
            "backend_compiles": audit.backend_compiles,
            "measurement_runtime_invocations": audit.runtime_invocations - warmup_invocations,
            "warmup_runtime_invocations": warmup_invocations,
            "graph_code_sha256": audit.graph_code_sha256,
            "graph_node_counts": audit.graph_node_counts,
            "self_pair_nonzero_count": len(self_nonzero),
            "self_pair_nonzero_examples": self_nonzero[:10],
        },
        "sampling": {
            "states": state_count,
            "observations": len(primary),
            "repeats": args.repeats,
            "bootstrap_unit": "state/case; token positions remain nested for qwen_causal",
        },
        "numerical": {
            "mean_signed_logit_delta": mean([row["logit_mean_signed_delta"] for row in primary]),
            "mean_signed_logit_delta_state_bootstrap_95ci": state_bootstrap_ci(
                primary, "logit_mean_signed_delta", args.bootstrap, args.seed + 1
            ),
            "mean_abs_logit_delta": mean([row["logit_mean_abs_delta"] for row in primary]),
            "max_abs_logit_delta": max((row["logit_max_abs_delta"] for row in primary), default=0.0),
            "mean_abs_common_mode_shift": mean(
                [abs(row["logit_common_mode_shift"]) for row in primary]
            ),
            "mean_centered_abs_logit_delta": mean(
                [row["centered_logit_mean_abs_delta"] for row in primary]
            ),
            "mean_centered_abs_logit_delta_state_bootstrap_95ci": state_bootstrap_ci(
                primary, "centered_logit_mean_abs_delta", args.bootstrap, args.seed + 3
            ),
            "max_centered_abs_logit_delta": max(
                (row["centered_logit_max_abs_delta"] for row in primary), default=0.0
            ),
            "mean_common_mode_energy_share": mean(
                [row["logit_common_mode_energy_share"] for row in primary]
            ),
            "mean_centered_logit_energy_share": mean(
                [row["centered_logit_energy_share"] for row in primary]
            ),
            "max_logit_energy_decomposition_relative_error": max(
                (row["logit_energy_decomposition_relative_error"] for row in primary),
                default=0.0,
            ),
            "mean_signed_top1_margin_delta": mean([row["top1_margin_signed_delta"] for row in primary]),
            "mean_abs_top1_margin_delta": mean([abs(row["top1_margin_signed_delta"]) for row in primary]),
            "top1_margin_state_heterogeneity_variance": state_heterogeneity(primary, "top1_margin_signed_delta"),
            "top1_margin_same_state_repeat_variance": repeated_variance(rows, "top1_margin_signed_delta"),
            "sampling_uncertainty_note": "bootstrap confidence intervals are not runtime variance",
        },
        "semantic": {
            "argmax_disagreements": len(argmax_disagreements),
            "argmax_disagreement_rate": len(argmax_disagreements) / len(primary) if primary else float("nan"),
            "argmax_boundary_risk_count": len(argmax_risk),
            "argmax_disagreement_within_risk_rate": (
                len(argmax_disagreements) / len(argmax_risk) if argmax_risk else 0.0
            ),
            "argmax_stability_condition_violations": sum(
                row["argmax_disagreement"]
                and row["argmax_stability_ratio"] is not None
                and float(row["argmax_stability_ratio"]) < 1.0
                for row in primary
            ),
            "top_order_disagreements": sum(row["top_order_disagreement"] for row in primary),
            "top5_set_disagreements": len(top5_disagreements),
            "top5_set_disagreement_rate": len(top5_disagreements) / len(top5_rows) if top5_rows else None,
            "top5_boundary_risk_count": len(top5_risk),
            "top5_stability_condition_violations": sum(
                bool(row["top5_set_disagreement"])
                and row["top5_stability_ratio"] is not None
                and float(row["top5_stability_ratio"]) < 1.0
                for row in top5_rows
            ),
            "reference_top1_margin_quantiles": {
                str(p): quantile([row["reference_top1_margin"] for row in primary], p)
                for p in [0.0, 0.01, 0.05, 0.5, 0.95, 1.0]
            },
        },
        "outcome": {
            "target_observations": len(target_rows),
            "mean_signed_target_logp_delta": (
                mean([row["target_logp_signed_delta"] for row in target_rows]) if target_rows else None
            ),
            "mean_signed_target_logp_delta_state_bootstrap_95ci": (
                state_bootstrap_ci(target_rows, "target_logp_signed_delta", args.bootstrap, args.seed + 2)
                if target_rows
                else [None, None]
            ),
            "mean_abs_target_logp_delta": (
                mean([abs(row["target_logp_signed_delta"]) for row in target_rows]) if target_rows else None
            ),
            "correctness_event_disagreements": sum(
                bool(row["correctness_event_disagreement"]) for row in target_rows
            ),
        },
        "claim_scope": {
            "supported": "implementation-relative deterministic discrepancy on the frozen subject/state distribution",
            "not_supported": [
                "mathematical/compiler correctness failure",
                "operational non-equivalence without a tolerance",
                "long-run training impact",
                "generalization outside the named subject and state population",
            ],
        },
    }
    if args.subject == "bert_sst2":
        up = sum(bool(row["positive_up"]) for row in primary)
        down = sum(bool(row["positive_down"]) for row in primary)
        summary["semantic"].update(
            {
                "positive_up": up,
                "positive_down": down,
                "directional_positive_shift": (up - down) / len(primary),
                "directional_exact_mcnemar_pvalue": exact_mcnemar(up, down),
                "reference_positive_rate": mean([float(row["reference_positive"]) for row in primary]),
                "candidate_positive_rate": mean([float(row["candidate_positive"]) for row in primary]),
            }
        )
        summary["outcome"].update(
            {
                "reference_accuracy": mean([float(row["reference_correct"]) for row in target_rows]),
                "candidate_accuracy": mean([float(row["candidate_correct"]) for row in target_rows]),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("at least two repeats are required")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; fail instead of falling back to CPU")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.recompile_limit = 2

    model, eager_fn, states, subject_metadata = load_subject(torch, args)
    if len(states) != args.count:
        raise ValueError(f"loaded {len(states)} states, expected {args.count}")
    audit = CompileAudit()
    compiled_fn = torch.compile(
        eager_fn,
        backend=make_tracking_backend(torch, audit),
        fullgraph=True,
        dynamic=False,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "started_unix": time.time(),
        "arguments": vars(args),
        "subject_metadata": subject_metadata,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "model_artifact": artifact_manifest(Path(args.model_path)),
        "data_artifact": {
            "path": str(Path(args.data_path).resolve()),
            "file_sha256": sha256_file(Path(args.data_path)) if Path(args.data_path).is_file() else None,
        },
        "state_ids": [state.state_id for state in states],
        "comparison_contract": {
            "reference": "PyTorch eager CUDA",
            "candidate": "tracked Inductor, fullgraph=True",
            "execution_batch_size": 1,
            "randomness": "deterministic algorithms, eval mode, no sampling RNG",
            "execution_order": "warm candidate/eager, then alternating E-C/C-E/E-C by repeat",
            "acceptance": "descriptive difference study; no operational equivalence bound",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")

    # Compile and warm both paths before measurement. Warm-up is recorded but
    # excluded from denominators.
    warm_compiled = execute(torch, compiled_fn, states[0], "compiled", args.subject, audit)
    warm_eager = execute(torch, eager_fn, states[0], "eager", args.subject, audit)
    del warm_compiled, warm_eager
    warmup_invocations = audit.runtime_invocations

    rows: list[dict[str, Any]] = []
    state_records: list[dict[str, Any]] = []
    with (out_dir / "observations.jsonl").open("w", encoding="utf-8") as observation_handle, (
        out_dir / "states.jsonl"
    ).open("w", encoding="utf-8") as state_handle:
        for state_index, state in enumerate(states):
            baselines: dict[str, Any] = {}
            for repeat in range(args.repeats):
                order = ["eager", "compiled"] if repeat % 2 == 0 else ["compiled", "eager"]
                outputs = {}
                for path in order:
                    fn = eager_fn if path == "eager" else compiled_fn
                    outputs[path] = execute(torch, fn, state, path, args.subject, audit)
                if repeat == 0:
                    baselines = {path: outputs[path]["logits"] for path in ["eager", "compiled"]}
                    self_exact = True
                    self_details = {"eager": {"exact": True}, "compiled": {"exact": True}}
                else:
                    self_details = {}
                    for path in ["eager", "compiled"]:
                        exact = bool(torch.equal(baselines[path], outputs[path]["logits"]))
                        detail = {"exact": exact}
                        if not exact:
                            difference = outputs[path]["logits"].float() - baselines[path].float()
                            detail.update(
                                {
                                    "mean_abs_delta": float(difference.abs().mean().item()),
                                    "max_abs_delta": float(difference.abs().max().item()),
                                }
                            )
                        self_details[path] = detail
                    self_exact = all(item["exact"] for item in self_details.values())
                observation_rows = compare_outputs(
                    torch,
                    outputs["eager"],
                    outputs["compiled"],
                    state,
                    args.subject,
                    repeat,
                    order,
                )
                for row in observation_rows:
                    observation_handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                observation_handle.flush()
                rows.extend(observation_rows)
                state_record = {
                    "state_id": state.state_id,
                    "state_index": state_index,
                    "repeat": repeat,
                    "execution_order": order,
                    "candidate_execution_valid": outputs["compiled"]["compiled_runtime_invocations"] > 0,
                    "compiled_runtime_invocations": outputs["compiled"]["compiled_runtime_invocations"],
                    "elapsed_ns": {path: outputs[path]["elapsed_ns"] for path in ["eager", "compiled"]},
                    "output_shape": outputs["eager"]["shape"],
                    "output_dtype": outputs["eager"]["dtype"],
                    "self_pair_exact": self_exact,
                    "self_pair_details": self_details,
                    "observations": len(observation_rows),
                }
                state_handle.write(json.dumps(state_record, sort_keys=True, allow_nan=False) + "\n")
                state_handle.flush()
                state_records.append(state_record)
                if repeat > 0:
                    del outputs
            del baselines
            if state_index % 8 == 0:
                torch.cuda.empty_cache()

    summary = build_summary(rows, state_records, audit, args, warmup_invocations)
    if not summary["validity"]["candidate_calls_valid"]:
        raise RuntimeError("fail-closed: candidate execution identity failed")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    manifest["completed_unix"] = time.time()
    manifest["compile_audit"] = {
        "backend_compiles": audit.backend_compiles,
        "runtime_invocations": audit.runtime_invocations,
        "warmup_invocations": warmup_invocations,
        "graph_code_sha256": audit.graph_code_sha256,
        "graph_node_counts": audit.graph_node_counts,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    del model


if __name__ == "__main__":
    main()
