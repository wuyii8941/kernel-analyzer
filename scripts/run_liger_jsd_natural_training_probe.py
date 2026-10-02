#!/usr/bin/env python3
"""Probe Liger fused JSD on a real language-model distillation boundary.

The teacher and student hidden states are produced from a local GPT-2
checkpoint and real tokenizer text.  Only the student linear-loss boundary is
changed: the candidate uses Liger's chunked JSD implementation with BF16
operands, while the reference evaluates the same JSD/CE expression in FP32.
This is a one-step natural distillation boundary, not a long-run training
claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss


def _texts() -> list[str]:
    return [
        "Numerical implementations can change training updates even when the mathematical operator is unchanged.",
        "A language model learns from repeated prediction errors across many token windows.",
        "The reference and candidate use the same inputs and parameters in this controlled comparison.",
        "Reduction order and intermediate materialization can create a signed update effect.",
        "A useful numerical test separates local output differences from gradient and parameter-write differences.",
        "Training state determines how a small implementation difference enters the next optimizer update.",
        "The experiment records both total update energy and the component aligned with the reference update.",
        "A held-out text window is used to check whether the direction is stable across states.",
        "Distillation losses combine a hard language-model target with a soft teacher distribution.",
        "Chunked accumulation is a distinct numerical boundary from the pointwise loss formula.",
        "A result is promoted only when its source and signed direction are both reproducible.",
        "False positives are retained as controls rather than renamed as negative bias cases.",
        "The same model weights can provide realistic hidden states for a local training-boundary probe.",
        "FP32 evaluation is used as a declared reference for this precision intervention.",
        "The candidate is evaluated with the implementation used by the training package.",
        "The final report is conditional on this checkpoint, tokenizer, and text bank.",
    ]


def _reference_loss(
    student_input: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_input: torch.Tensor,
    teacher_weight: torch.Tensor,
    labels: torch.Tensor,
    student_bias: torch.Tensor | None,
    teacher_bias: torch.Tensor | None,
    beta: float,
    temperature: float,
) -> torch.Tensor:
    student_logits = student_input.float() @ student_weight.float().t()
    teacher_logits = teacher_input.float() @ teacher_weight.float().t()
    if student_bias is not None:
        student_logits = student_logits + student_bias.float()
    if teacher_bias is not None:
        teacher_logits = teacher_logits + teacher_bias.float()
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_logp = F.log_softmax(student_logits, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits, dim=-1)
    log_mean = torch.logsumexp(
        torch.stack([student_logp + torch.log(torch.tensor(1.0 - beta, device=student_logits.device)),
                     teacher_logp + torch.log(torch.tensor(beta, device=student_logits.device))]),
        dim=0,
    )
    student_kl = F.kl_div(log_mean, student_logp, reduction="none", log_target=True).sum(dim=-1)
    teacher_kl = F.kl_div(log_mean, teacher_logp, reduction="none", log_target=True).sum(dim=-1)
    jsd = beta * teacher_kl + (1.0 - beta) * student_kl
    mask = labels != -100
    jsd = jsd.masked_fill(~mask, 0.0)
    hard = F.cross_entropy(student_logits, labels, reduction="none", ignore_index=-100)
    hard = hard.masked_fill(~mask, 0.0)
    denom = mask.sum().clamp_min(1).to(jsd.dtype)
    return 0.5 * (jsd.sum() / denom) + 0.5 * (hard.sum() / denom)


def _relative_rms(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x.float().square().mean().sqrt() / (y.float().square().mean().sqrt() + 1e-30)).detach())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--states", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260919)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16, device_map=None
    ).to(device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = _texts()
    rows: list[dict[str, Any]] = []
    for state_id in range(args.states):
        # Keep every state tied to a distinct real-text window rather than
        # silently repeating the first bank when states > len(texts).
        text = texts[state_id % len(texts)] + f" Trial window {state_id} uses a separate context."
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len + 1)
        original_ids = encoded.input_ids.to(device)
        if original_ids.shape[1] < 4:
            continue
        ids = original_ids.clone()
        labels = torch.full_like(ids, -100)
        # A deterministic masked-token objective makes this an actual MLM
        # training boundary while keeping pairwise inputs identical.
        positions = torch.arange(1, ids.shape[1] - 1, 4, device=device)
        labels[:, positions] = original_ids[:, positions]
        ids[:, positions] = tokenizer.mask_token_id
        with torch.no_grad():
            if hasattr(model, "bert"):
                hidden = model.bert(input_ids=ids).last_hidden_state
                teacher_hidden = model.bert(input_ids=original_ids).last_hidden_state
            elif hasattr(model, "roberta"):
                hidden = model.roberta(input_ids=ids).last_hidden_state
                teacher_hidden = model.roberta(input_ids=original_ids).last_hidden_state
            else:
                raise RuntimeError("the probe currently supports BERT/RoBERTa-style masked-LM bases")
            hidden = hidden.reshape(-1, hidden.shape[-1])
            teacher_hidden = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])
        labels = labels.reshape(-1)
        hidden = hidden.to(torch.bfloat16)
        if hasattr(model, "cls") and hasattr(model.cls, "predictions"):
            decoder = model.cls.predictions.decoder
        elif hasattr(model, "lm_head"):
            decoder = model.lm_head
        else:
            raise RuntimeError("cannot locate the masked-LM decoder")
        weight = decoder.weight.detach().to(torch.bfloat16)
        bias = decoder.bias.detach().to(torch.bfloat16) if decoder.bias is not None else None
        student_input = hidden.detach().clone().requires_grad_(True)
        student_weight = weight.detach().clone().requires_grad_(True)
        teacher_input = teacher_hidden.to(torch.bfloat16).detach().clone()
        teacher_weight = weight.detach().clone()
        student_bias = bias.detach().clone().requires_grad_(True) if bias is not None else None
        teacher_bias = bias.detach().clone() if bias is not None else None
        candidate_loss_fn = LigerFusedLinearJSDLoss(
            beta=0.5, temperature=1.0, compiled=True, chunk_size=args.chunk_size
        )
        candidate_loss = candidate_loss_fn(
            student_input, student_weight, teacher_input, teacher_weight, labels, student_bias, teacher_bias
        )
        candidate_loss.backward()
        candidate_grad = student_weight.grad.detach().float().clone()
        candidate_update = -1e-3 * candidate_grad

        ref_input = hidden.detach().float().requires_grad_(True)
        ref_weight = weight.detach().float().requires_grad_(True)
        ref_bias = bias.detach().float().requires_grad_(True) if bias is not None else None
        ref_teacher_bias = bias.detach().float() if bias is not None else None
        ref_loss = _reference_loss(ref_input, ref_weight, teacher_hidden.detach().float(), weight.detach().float(), labels, ref_bias, ref_teacher_bias, 0.5, 1.0)
        ref_loss.backward()
        ref_grad = ref_weight.grad.detach().float().clone()
        ref_update = -1e-3 * ref_grad
        delta_update = candidate_update - ref_update
        rows.append(
            {
                "state_id": state_id,
                "token_count": int(labels.numel()),
                "candidate_loss": float(candidate_loss.detach()),
                "reference_loss": float(ref_loss.detach()),
                "loss_difference": float((candidate_loss.detach().float() - ref_loss.detach()).cpu()),
                "gradient_effect_rms": float(delta_update.square().mean().sqrt() / 1e-3),
                "gradient_effect_relative_rms": _relative_rms(delta_update, ref_grad),
                "write_effect_rms": float(delta_update.square().mean().sqrt()),
                "write_effect_relative_rms": _relative_rms(delta_update, ref_update),
                "aligned_write": float((delta_update * ref_update).sum() / (ref_update.square().sum() + 1e-30)),
            }
        )
    values = torch.tensor([r["aligned_write"] for r in rows], dtype=torch.float64)
    half = len(rows) // 2
    confirm = values[half:]
    mean = float(confirm.mean()) if len(confirm) else 0.0
    sd = float(confirm.std(unbiased=True)) if len(confirm) > 1 else 0.0
    radius = 2.131449545559323 * sd / max(len(confirm), 1) ** 0.5 if len(confirm) > 1 else 0.0
    out = {
        "schema": "liger-jsd-natural-training-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "device": str(device),
        "states": len(rows),
        "calibration_count": half,
        "confirmation_count": len(confirm),
        "candidate": "LigerFusedLinearJSDLoss compiled=True with BF16 operands",
        "reference": "explicit FP32 JSD plus CE on the same student/teacher hidden states and weights",
        "summary": {
            "write_effect_rms_mean": float(torch.tensor([r["write_effect_rms"] for r in rows]).mean()) if rows else 0.0,
            "aligned_confirmation_mean": mean,
            "aligned_confirmation_interval_95_approx": [mean - radius, mean + radius],
            "aligned_confirmation_negative": int((confirm < 0).sum()),
            "aligned_confirmation_positive": int((confirm > 0).sum()),
        },
        "claim_boundary": "real BERT-tiny text windows with masked student and unmasked teacher states; one-step distillation update; no long-run quality claim",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
