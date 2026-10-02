#!/usr/bin/env python3
"""Screen the Qwen3 SDPA versus eager attention implementation boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if x.numel() < 2:
        m = float(x.mean())
        return [m, m]
    m = float(x.mean())
    h = 1.96 * float(x.std(unbiased=True)) / (x.numel() ** 0.5)
    return [m - h, m + h]


def ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a)) / max(float(torch.linalg.vector_norm(b)), 1e-30)


def write(g: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    g = g.float()
    return -lr * g / (g.abs() + eps)


def run(args: argparse.Namespace) -> dict[str, Any]:
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank["states"][: args.states]
    device = torch.device(args.device)
    eager = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager").to(device).train()
    sdpa = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).train()
    eager.config.use_cache = False
    sdpa.config.use_cache = False
    target_e = eager.model.layers[args.layer].self_attn.q_proj.weight if args.parameter == "q_proj" else eager.model.layers[args.layer].self_attn.v_proj.weight
    target_s = sdpa.model.layers[args.layer].self_attn.q_proj.weight if args.parameter == "q_proj" else sdpa.model.layers[args.layer].self_attn.v_proj.weight
    rows = []
    effects = []
    for state in states:
        vals = state.get("token_ids", state.get("input_ids"))
        ids = torch.tensor([vals], dtype=torch.long, device=device)
        labels = ids.clone()
        eager.zero_grad(set_to_none=True)
        sdpa.zero_grad(set_to_none=True)
        le = eager(input_ids=ids, labels=labels, use_cache=False).loss
        ls = sdpa(input_ids=ids, labels=labels, use_cache=False).loss
        le.backward(); ls.backward()
        ge = target_e.grad.detach().float().cpu().clone()
        gs = target_s.grad.detach().float().cpu().clone()
        we = write(ge, args.learning_rate, args.eps)
        ws = write(gs, args.learning_rate, args.eps)
        effect = we - ws
        effects.append(effect)
        rows.append({
            "state_id": state.get("state_id", state.get("sequence_id")),
            "loss_difference": float((le-ls).detach().cpu()),
            "gradient_effect_rms_over_reference": ratio(ge-gs, gs),
            "write_effect_rms_over_reference": ratio(effect, ws),
            "write_aligned": float(torch.sum(effect*ws))/max(float(torch.sum(ws*ws)),1e-30),
        })
        del ids, labels, le, ls
        torch.cuda.empty_cache()
    split = len(rows)//2
    direction = torch.stack([x.double() for x in effects[:split]]).mean(0)
    norm = float(torch.linalg.vector_norm(direction))
    proj = []
    if norm:
        direction /= norm
        proj = [float(torch.sum(x.double()*direction)) for x in effects[split:]]
    return {
        "schema": "kernel-analyzer-qwen3-sdpa-eager-natural-probe-v1",
        "status": "COMPLETE", "model": str(args.model), "layer": args.layer, "parameter": args.parameter,
        "operator": "Qwen3 attention backend",
        "candidate": "Qwen3 SDPA implementation", "reference": "same checkpoint with eager attention",
        "input_source": "declared real Qwen3 text input bank",
        "claim_boundary": "One checkpoint, one layer/parameter, and declared bank; candidate-only backend screen, not a closed root cause.",
        "comparison_scope": {"same_model_weights": True, "same_input_ids": True, "single_changed_boundary": "attention backend"},
        "rows": rows,
        "summary": {
            "state_count": len(rows), "calibration_count": split, "confirmation_count": len(rows)-split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_reference"] for r in rows)/len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_reference"] for r in rows)/len(rows),
            "write_aligned_interval_normal_95": interval([r["write_aligned"] for r in rows]),
            "loss_difference_interval_normal_95": interval([r["loss_difference"] for r in rows]),
            "heldout_write_projection_interval_normal_95": interval(proj) if proj else None,
            "heldout_write_projection_positive": sum(x>0 for x in proj), "heldout_write_projection_negative": sum(x<0 for x in proj),
            "calibration_direction_norm": norm,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    p.add_argument("--input-bank", type=Path, default=Path("results/coverage/qwen_seq128_input_bank.json"))
    p.add_argument("--layer", type=int, default=13); p.add_argument("--parameter", choices=("q_proj", "v_proj"), default="q_proj")
    p.add_argument("--states", type=int, default=8); p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--eps", type=float, default=1e-8); p.add_argument("--device", default="cuda:0"); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    out = run(a); a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8"); print(json.dumps(out["summary"], sort_keys=True))


if __name__ == "__main__": main()
