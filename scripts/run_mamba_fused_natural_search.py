#!/usr/bin/env python3
"""Search official Mamba fused-scan effects on fresh natural token windows.

This deliberately avoids provenance digests.  The calibration direction is
computed only from the first two fixed windows; all later windows are freshly
drawn from the local validation corpus.  The result is a search/confirmation
artifact, not a population guarantee.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, MambaForCausalLM
from transformers.models.mamba import modeling_mamba


def windows(tokenizer: Any, arrow: Path, seq_len: int, start: int, count: int) -> list[torch.Tensor]:
    from datasets import Dataset
    dataset = Dataset.from_file(str(arrow))
    text = "\n".join(row["text"] for row in dataset if row["text"].strip())
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    stride = seq_len * 17
    begin = start * stride
    end = begin + count * stride + seq_len
    if tokens.numel() < end:
        raise RuntimeError(f"validation corpus has {tokens.numel()} tokens, need {end}")
    return [tokens[begin + i * stride : begin + i * stride + seq_len].clone() for i in range(count)]


def set_fused(enabled: bool) -> None:
    if enabled:
        return
    # The official fused functions are restored by the caller before every
    # candidate run; sequential mode is selected by clearing these globals.
    modeling_mamba.selective_scan_fn = None
    modeling_mamba.mamba_inner_fn = None
    modeling_mamba.selective_state_update = None


def set_mambapy(model: MambaForCausalLM, enabled: bool) -> None:
    for layer in model.backbone.layers:
        layer.mixer.use_mambapy = enabled


def run(model: MambaForCausalLM, selected: dict[str, torch.nn.Parameter], ids: torch.Tensor, fused: bool):
    # The newer Transformers Mamba implementation enters the optional fused
    # path only while ``training`` is true.  Its sequential reference path is
    # the eval-mode branch, which is deterministic for this model (there is no
    # dropout).  Switch only this dispatch flag; gradients remain enabled.
    was_training = model.training
    model.train(fused)
    if CANDIDATE_MODE == "mambapy":
        set_mambapy(model, fused)
    elif fused:
        for name, value in OFFICIAL.items():
            setattr(modeling_mamba, name, value)
    else:
        set_fused(False)
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    gradients = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in selected.items()
        if parameter.grad is not None
    }
    if len(gradients) != len(selected):
        missing = sorted(set(selected) - set(gradients))
        raise RuntimeError(f"missing gradients: {missing}")
    model.train(was_training)
    return float(loss.detach().cpu()), gradients


def adamw_write_delta(reference: torch.Tensor, candidate: torch.Tensor, lr: float) -> tuple[float, float]:
    """Return candidate-reference first-step AdamW write RMS and reference RMS."""
    p_ref = torch.nn.Parameter(reference.float().clone())
    p_cand = torch.nn.Parameter(reference.float().clone())
    o_ref = torch.optim.AdamW([p_ref], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    o_cand = torch.optim.AdamW([p_cand], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    p_ref.grad = reference.float().clone()
    p_cand.grad = candidate.float().clone()
    before = p_ref.detach().clone()
    o_ref.step()
    o_cand.step()
    ref_write = p_ref.detach() - before
    cand_write = p_cand.detach() - before
    delta = cand_write - ref_write
    return float(torch.linalg.vector_norm(delta) / max(torch.linalg.vector_norm(ref_write), 1e-30)), float(
        torch.linalg.vector_norm(ref_write)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/state-spaces/mamba-130m-hf"))
    parser.add_argument(
        "--validation-arrow", type=Path,
        default=Path("/data1/tzh/cache/huggingface/datasets/Salesforce___wikitext/"
                     "wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/"
                     "wikitext-validation.arrow"),
    )
    parser.add_argument("--input-bank", type=Path, default=None)
    parser.add_argument("--candidate-mode", choices=("mambapy", "official_fused"), default="mambapy")
    parser.add_argument("--parameter", default="backbone.layers.3.mixer.x_proj.weight")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--calibration-count", type=int, default=2)
    parser.add_argument("--confirmation-start", type=int, default=24)
    parser.add_argument("--confirmation-count", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    global OFFICIAL, CANDIDATE_MODE
    CANDIDATE_MODE = args.candidate_mode
    # Transformers versions expose different subsets of the optional fused
    # Mamba hooks.  Preserve the hooks that actually exist instead of treating
    # a missing compatibility alias as a scientific failure.
    OFFICIAL = {
        name: getattr(modeling_mamba, name)
        for name in ("selective_scan_fn", "mamba_inner_fn", "selective_state_update")
        if hasattr(modeling_mamba, name)
    }
    if args.candidate_mode == "official_fused" and any(value is None for value in OFFICIAL.values()):
        raise RuntimeError("official fused Mamba implementation is unavailable in this environment")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.input_bank is not None:
        bank = json.loads(args.input_bank.read_text())
        if len(bank["states"]) < args.confirmation_start + args.confirmation_count:
            raise RuntimeError("input bank too small")
        all_windows = [torch.tensor(row["token_ids"], dtype=torch.long) for row in bank["states"]]
    else:
        all_windows = windows(
            tokenizer,
            args.validation_arrow,
            args.seq_len,
            0,
            args.confirmation_start + args.confirmation_count,
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, dtype=dtype, local_files_only=True).to(device).train()
    model.config.use_cache = False
    named = dict(model.named_parameters())
    if args.parameter not in named:
        raise KeyError(args.parameter)
    selected = {args.parameter: named[args.parameter]}
    calibration_ids = list(range(args.calibration_count))
    confirmation_ids = list(range(args.confirmation_start, args.confirmation_start + args.confirmation_count))
    deltas: dict[int, torch.Tensor] = {}
    losses: dict[int, dict[str, float]] = {}
    write_rms: dict[int, float] = {}
    reference_write_norm: dict[int, float] = {}
    for state_id in calibration_ids + confirmation_ids:
        ids = all_windows[state_id].unsqueeze(0).to(device)
        ref_loss, ref = run(model, selected, ids, fused=False)
        cand_loss, cand = run(model, selected, ids, fused=True)
        delta = cand[args.parameter] - ref[args.parameter]
        aligned_ratio = float(
            torch.sum(delta.double() * ref[args.parameter].double())
            / torch.sum(ref[args.parameter].double() ** 2).clamp_min(1e-30)
        )
        deltas[state_id] = delta
        write_rms[state_id], reference_write_norm[state_id] = adamw_write_delta(
            ref[args.parameter], cand[args.parameter], args.learning_rate
        )
        losses[state_id] = {"reference": ref_loss, "candidate": cand_loss, "difference": cand_loss - ref_loss}
        losses[state_id]["aligned_gradient_ratio"] = aligned_ratio
        print(json.dumps({"state": state_id, "write_rms": write_rms[state_id], "aligned_gradient_ratio": aligned_ratio, "loss_difference": losses[state_id]["difference"]}))
        del ids, ref, cand
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw_direction = sum((deltas[state_id] for state_id in calibration_ids), torch.zeros_like(deltas[calibration_ids[0]]))
    norm = torch.linalg.vector_norm(raw_direction.double())
    if norm == 0:
        raise RuntimeError("calibration direction is zero")
    direction = raw_direction.double() / norm
    projection = {state_id: float(torch.sum(deltas[state_id].double() * direction)) for state_id in deltas}
    confirmation = [projection[state_id] for state_id in confirmation_ids]
    write_values = [write_rms[state_id] for state_id in confirmation_ids]
    loss_values = [losses[state_id]["difference"] for state_id in confirmation_ids]
    aligned_values = [losses[state_id]["aligned_gradient_ratio"] for state_id in confirmation_ids]
    output = {
        "schema": "kernel-analyzer-mamba-fused-natural-search-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "parameter": args.parameter,
        "candidate": "official fused mamba_inner_fn / selective scan",
        "reference": "explicit sequential selective-state recurrence",
        "seq_len": args.seq_len,
        "calibration_state_ids": calibration_ids,
        "confirmation_state_ids": confirmation_ids,
        "calibration_direction_is_frozen_before_confirmation": True,
        "rows": [
            {
                "state_id": state_id,
                "split": "CALIBRATION" if state_id in calibration_ids else "CONFIRMATION",
                "gradient_direction_projection": projection[state_id],
                "first_step_adamw_write_rms_over_reference": write_rms[state_id],
                "reference_write_norm": reference_write_norm[state_id],
                "loss_difference": losses[state_id]["difference"],
                "aligned_gradient_ratio": losses[state_id]["aligned_gradient_ratio"],
            }
            for state_id in calibration_ids + confirmation_ids
        ],
        "confirmation_summary": {
            "gradient_projection_positive": sum(value > 0 for value in confirmation),
            "gradient_projection_negative": sum(value < 0 for value in confirmation),
            "gradient_projection_mean": sum(confirmation) / len(confirmation),
            "gradient_projection_min": min(confirmation),
            "gradient_projection_max": max(confirmation),
            "write_rms_mean": sum(write_values) / len(write_values),
            "write_rms_min": min(write_values),
            "write_rms_max": max(write_values),
            "loss_difference_mean": sum(loss_values) / len(loss_values),
            "loss_difference_positive": sum(value > 0 for value in loss_values),
            "loss_difference_negative": sum(value < 0 for value in loss_values),
            "aligned_gradient_ratio_mean": sum(aligned_values) / len(aligned_values),
            "aligned_gradient_ratio_positive": sum(value > 0 for value in aligned_values),
            "aligned_gradient_ratio_negative": sum(value < 0 for value in aligned_values),
            "aligned_gradient_ratio_interval_normal_95": [
                (sum(aligned_values) / len(aligned_values))
                - 1.96 * (sum((value - sum(aligned_values) / len(aligned_values)) ** 2 for value in aligned_values) / max(len(aligned_values) - 1, 1)) ** 0.5 / (len(aligned_values) ** 0.5),
                (sum(aligned_values) / len(aligned_values))
                + 1.96 * (sum((value - sum(aligned_values) / len(aligned_values)) ** 2 for value in aligned_values) / max(len(aligned_values) - 1, 1)) ** 0.5 / (len(aligned_values) ** 0.5),
            ],
            "all_gradient_projections_positive": all(value > 0 for value in confirmation),
        },
        "claim_boundary": (
            "Natural fixed corpus confirmation of a frozen parameter direction and first-step write magnitude; "
            "no population guarantee, warm-state conclusion, or loss-quality claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), **output["confirmation_summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
