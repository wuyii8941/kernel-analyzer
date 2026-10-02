#!/usr/bin/env python3
"""Natural DeBERTa disentangled-relative-attention probe.

The selected DeBERTa-v3-small checkpoint is run on real text windows.  The
native relative-position path is compared with a path that keeps all model
inputs and projection weights fixed but evaluates the c2p/p2c positional score
products in FP32 before returning the native attention dtype.  This tests a
different positional-attention family from RoPE/ALiBi without treating an
isolated tensor scan as a training result.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import AutoModelForMaskedLM
from transformers.models.deberta_v2 import modeling_deberta_v2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/microsoft/deberta-v3-small")
DEFAULT_TEXT = ROOT / "README.md"
DEFAULT_TOKENIZER = Path("/data1/tzh/cache/deberta_v3_small/tokenizer_unk.json")


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean())
    if x.numel() < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True)) / (x.numel() ** 0.5)
    return [mean - half, mean + half]


def ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect).item()) / max(float(torch.linalg.vector_norm(reference).item()), 1e-30)


def first_step_write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    g = gradient.float()
    return -learning_rate * g / (g.abs() + eps)


def make_states(tokenizer: Any, source: Path, states: int, sequence_length: int) -> list[dict[str, Any]]:
    text = source.read_text(encoding="utf-8", errors="ignore")
    all_ids = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
    out: list[dict[str, Any]] = []
    stride = max(1, sequence_length - 8)
    for index, start in enumerate(range(0, max(1, all_ids.numel() - sequence_length + 1), stride)):
        ids = all_ids[start : start + sequence_length].clone()
        if ids.numel() < sequence_length:
            break
        # The checkpoint is an encoder MLM model.  Using the real text token
        # windows as labels keeps the natural forward/backward path paired;
        # this probe does not claim to reproduce the original masking policy.
        labels = ids.clone()
        out.append({"state_id": f"readme_window_{index}", "input_ids": ids.tolist(), "labels": labels.tolist()})
        if len(out) >= states:
            break
    if len(out) < states:
        raise RuntimeError(f"text source supplied only {len(out)} windows")
    return out


def fp32_disentangled_bias(
    self: torch.nn.Module,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    relative_pos: torch.Tensor | None,
    rel_embeddings: torch.Tensor,
    scale_factor: int,
) -> torch.Tensor:
    """Native DeBERTa c2p/p2c formula with score products in FP32."""
    if relative_pos is None:
        relative_pos = modeling_deberta_v2.build_relative_position(
            query_layer, key_layer, bucket_size=self.position_buckets,
            max_position=self.max_relative_positions,
        )
    if relative_pos.dim() == 2:
        relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
    elif relative_pos.dim() == 3:
        relative_pos = relative_pos.unsqueeze(1)
    elif relative_pos.dim() != 4:
        raise ValueError(f"Relative position ids must be dim 2/3/4, got {relative_pos.dim()}")
    att_span = self.pos_ebd_size
    relative_pos = relative_pos.to(device=query_layer.device, dtype=torch.long)
    rel_embeddings = rel_embeddings[: att_span * 2, :]
    emb_float = rel_embeddings.float().unsqueeze(0)
    if self.share_att_key:
        pos_query_layer = self.transpose_for_scores(
            F.linear(emb_float, self.query_proj.weight.float(), None), self.num_attention_heads
        ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
        pos_key_layer = self.transpose_for_scores(
            F.linear(emb_float, self.key_proj.weight.float(), None), self.num_attention_heads
        ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
    else:
        if "c2p" in self.pos_att_type:
            pos_key_layer = self.transpose_for_scores(
                F.linear(emb_float, self.pos_key_proj.weight.float(), None), self.num_attention_heads
            ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
        if "p2c" in self.pos_att_type:
            pos_query_layer = self.transpose_for_scores(
                F.linear(emb_float, self.pos_query_proj.weight.float(), None), self.num_attention_heads
            ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
    score: torch.Tensor | int = 0
    if "c2p" in self.pos_att_type:
        scale = modeling_deberta_v2.scaled_size_sqrt(pos_key_layer, scale_factor).float()
        c2p_att = torch.bmm(query_layer.float(), pos_key_layer.transpose(-1, -2))
        c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
        c2p_att = torch.gather(
            c2p_att, dim=-1,
            index=c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]),
        )
        score = score + c2p_att / scale.to(dtype=c2p_att.dtype)
    if "p2c" in self.pos_att_type:
        scale = modeling_deberta_v2.scaled_size_sqrt(pos_query_layer, scale_factor).float()
        r_pos = modeling_deberta_v2.build_rpos(
            query_layer, key_layer, relative_pos, self.max_relative_positions, self.position_buckets
        )
        p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
        p2c_att = torch.bmm(key_layer.float(), pos_query_layer.transpose(-1, -2))
        p2c_att = torch.gather(
            p2c_att, dim=-1,
            index=p2c_pos.squeeze(0).expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)]),
        ).transpose(-1, -2)
        score = score + p2c_att / scale.to(dtype=p2c_att.dtype)
    return score.to(dtype=query_layer.dtype)


def run(args: argparse.Namespace) -> dict[str, Any]:
    # The local checkpoint does not ship a pretrained MLM head.  Fix the
    # initialization of that declared auxiliary head so layer-to-layer runs
    # are reproducible and the probe is not silently changing its loss head.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    states = make_states(tokenizer, args.text_source, args.states, args.sequence_length)
    model = AutoModelForMaskedLM.from_pretrained(
        str(args.model), local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).train()
    # DeBERTa-v3-small exposes the encoder under ``deberta``.
    attention = model.deberta.encoder.layer[args.layer].attention.self
    target = attention.query_proj.weight
    original_bias = attention.disentangled_attention_bias
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    try:
        for state in states:
            ids = torch.tensor([state["input_ids"]], dtype=torch.long, device=device)
            labels = torch.tensor([state["labels"]], dtype=torch.long, device=device)
            model.zero_grad(set_to_none=True)
            attention.disentangled_attention_bias = original_bias
            native = model(input_ids=ids, labels=labels, return_dict=True)
            native_loss = native.loss
            native_loss.backward()
            native_grad = target.grad.detach().float().cpu().clone()
            native_write = first_step_write(native_grad, args.learning_rate, args.eps)

            model.zero_grad(set_to_none=True)
            attention.disentangled_attention_bias = types.MethodType(fp32_disentangled_bias, attention)
            reference = model(input_ids=ids, labels=labels, return_dict=True)
            reference_loss = reference.loss
            reference_loss.backward()
            reference_grad = target.grad.detach().float().cpu().clone()
            reference_write = first_step_write(reference_grad, args.learning_rate, args.eps)
            effect = native_write - reference_write
            effects.append(effect)
            rows.append({
                "state_id": state["state_id"],
                "loss_difference": float((native_loss - reference_loss).detach().cpu()),
                "gradient_effect_rms_over_reference": ratio(native_grad - reference_grad, reference_grad),
                "write_effect_rms_over_reference": ratio(effect, reference_write),
                "write_aligned": float(torch.sum(effect * reference_write).item())
                / max(float(torch.sum(reference_write * reference_write).item()), 1e-30),
            })
            del ids, labels, native, reference, native_loss, reference_loss
            torch.cuda.empty_cache()
    finally:
        attention.disentangled_attention_bias = original_bias
    split = len(rows) // 2
    calibration = torch.stack([x.double() for x in effects[:split]], dim=0).mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(calibration).item())
    projections: list[float] = []
    if direction_norm > 0:
        direction = calibration / direction_norm
        projections = [float(torch.sum(x.double() * direction).item()) for x in effects[split:]]
    aligned = [rows[i]["write_aligned"] for i in range(split, len(rows))]
    return {
        "schema": "kernel-analyzer-deberta-disentangled-attention-materialization-natural-v1",
        "status": "COMPLETE",
        "model": str(args.model), "layer": args.layer,
        "target": f"deberta.encoder.layer[{args.layer}].attention.self.query_proj.weight",
        "operator": "disentangled relative-position attention score materialization",
        "candidate": "native DeBERTa c2p/p2c relative score evaluation",
        "reference": "same DeBERTa path with FP32 c2p/p2c score products and native dtype return",
        "input_source": f"real masked-text windows from {args.text_source}",
        "optimizer": {"name": "AdamW first-step proxy", "zero_moments": True, "learning_rate": args.learning_rate},
        "comparison_scope": {"same_model_weights": True, "same_masked_inputs": True, "same_relative_positions": True, "single_changed_boundary": "disentangled c2p/p2c score materialization"},
        "claim_boundary": "One DeBERTa-v3-small checkpoint/layer and declared real-text bank; not a population or loss-quality guarantee.",
        "rows": rows,
        "summary": {
            "state_count": len(rows), "calibration_count": split, "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_reference"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_reference"] for r in rows) / len(rows),
            "write_aligned_mean": sum(aligned) / len(aligned),
            "write_aligned_interval_normal_95": interval(aligned),
            "projection_mean": sum(projections) / len(projections) if projections else None,
            "projection_interval_normal_95": interval(projections) if projections else None,
            "projection_positive": sum(x > 0 for x in projections), "projection_negative": sum(x < 0 for x in projections),
            "loss_difference_interval_normal_95": interval([r["loss_difference"] for r in rows]),
            "calibration_direction_norm": direction_norm,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--text-source", type=Path, default=DEFAULT_TEXT)
    p.add_argument("--tokenizer-json", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--states", type=int, default=16)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
