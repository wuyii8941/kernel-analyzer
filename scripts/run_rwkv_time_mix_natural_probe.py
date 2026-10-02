#!/usr/bin/env python3
"""Natural RWKV recurrent time-mix precision probe.

The checkpoint is evaluated on real text.  Candidate and reference share the
same model, tokens, and recurrent boundary; only one declared arithmetic
choice inside RWKV's sequential WKV recurrence is changed.  The probe reports
the actual gradient and first-step AdamW write of a recurrent parameter.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.rwkv import modeling_rwkv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/RWKV/rwkv-4-world-169m")
DEFAULT_TEXT = ROOT / "README.md"
DEFAULT_OUTPUT = ROOT / "results/property/new_problem_group_search_v1/rwkv_time_mix_natural_16_20260920.json"


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean())
    if len(values) < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True)) / math.sqrt(len(values))
    return [mean - half, mean + half]


def write(gradient: torch.Tensor, lr: float = 1e-3, eps: float = 1e-8) -> torch.Tensor:
    g = gradient.float()
    return -lr * g / (g.abs() + eps)


def ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(effect.norm()) / max(float(reference.norm()), 1e-30)


def text_states(tokenizer: Any, source: Path, count: int, max_length: int) -> list[dict[str, torch.Tensor]]:
    lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("empty text source")
    states: list[dict[str, torch.Tensor]] = []
    for index in range(count):
        body = " ".join(lines[index % len(lines) : index % len(lines) + 4])
        if len(body) < 32:
            body = (body + " " + " ".join(lines))[:1024]
        ids = tokenizer(body, max_length=max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        if ids.numel() < 2:
            continue
        states.append({"input_ids": ids, "labels": ids.clone()})
    if len(states) < count:
        raise RuntimeError(f"only built {len(states)} usable states out of {count}")
    return states[:count]


def rwkv_variant(
    time_decay: torch.Tensor,
    time_first: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mode: str,
    state: Any = None,
    return_state: bool = False,
) -> tuple[torch.Tensor, Any]:
    """Copy the reference recurrence while changing one arithmetic boundary."""

    _, seq_length, _ = key.size()
    output = torch.zeros_like(key)
    if state is None:
        num_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
        den_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
        max_state = torch.zeros_like(key[:, 0], dtype=torch.float32) - 1e38
    else:
        num_state, den_state, max_state = state

    if mode in {"decay_fp32", "all_fp32"}:
        decay = -torch.exp(time_decay.float())
    else:
        decay = -torch.exp(time_decay)

    for current_index in range(seq_length):
        current_key = key[:, current_index].float()
        if mode in {"value_fp32", "all_fp32"}:
            current_value = value[:, current_index].float()
        else:
            current_value = value[:, current_index]

        max_for_output = torch.maximum(max_state, current_key + time_first)
        e1 = torch.exp(max_state - max_for_output)
        e2 = torch.exp(current_key + time_first - max_for_output)
        numerator = e1 * num_state + e2 * current_value
        denominator = e1 * den_state + e2
        output[:, current_index] = (numerator / denominator).to(output.dtype)

        max_for_state = torch.maximum(max_state + decay, current_key)
        e1 = torch.exp(max_state + decay - max_for_state)
        e2 = torch.exp(current_key - max_for_state)
        num_state = e1 * num_state + e2 * current_value
        den_state = e1 * den_state + e2
        max_state = max_for_state

    if return_state or state is not None:
        state = [num_state, den_state, max_state]
    return output, state


def install_mode(model: Any, mode_ref: dict[str, str]) -> None:
    attention = model.rwkv.blocks[0].attention
    original = getattr(attention, "_kernel_analyzer_original_forward", attention.forward)
    attention._kernel_analyzer_original_forward = original

    def forward(self, hidden, state=None, use_cache=False):
        receptance, key, value, state = self.extract_key_value(hidden, state=state)
        layer_state = tuple(s[:, :, self.layer_id] for s in state[2:]) if state is not None else None
        if mode_ref["mode"] == "native":
            rwkv, layer_state = modeling_rwkv.rwkv_linear_attention(
                self.time_decay, self.time_first, key, value,
                state=layer_state, return_state=use_cache,
            )
        else:
            rwkv, layer_state = rwkv_variant(
                self.time_decay, self.time_first, key, value,
                mode=mode_ref["mode"], state=layer_state, return_state=use_cache,
            )
        if layer_state is not None:
            state[2][:, :, self.layer_id] = layer_state[0]
            state[3][:, :, self.layer_id] = layer_state[1]
            state[4][:, :, self.layer_id] = layer_state[2]
        return self.output(receptance * rwkv), state

    attention.forward = types.MethodType(forward, attention)


def run(model_path: Path, text_path: Path, count: int, max_length: int, device: str) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16,
    ).to(device).eval()
    model.config.use_cache = False
    target = model.rwkv.blocks[0].attention.time_decay
    states = text_states(tokenizer, text_path, count, max_length)
    base_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    mode_ref = {"mode": "native"}
    install_mode(model, mode_ref)
    rows: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        batch = {key: value.unsqueeze(0).to(device) for key, value in state.items()}
        model.load_state_dict(base_state, strict=True)
        model.zero_grad(set_to_none=True)
        mode_ref["mode"] = "native"
        native = model(**batch)
        native.loss.backward()
        native_grad = target.grad.detach().float().clone()
        native_write = write(native_grad)

        for mode in ("value_fp32", "decay_fp32", "all_fp32"):
            model.load_state_dict(base_state, strict=True)
            model.zero_grad(set_to_none=True)
            mode_ref["mode"] = mode
            reference = model(**batch)
            reference.loss.backward()
            reference_grad = target.grad.detach().float().clone()
            reference_write = write(reference_grad)
            effect = native_write - reference_write
            rows.append({
                "state_id": index,
                "mode": mode,
                "native_loss": float(native.loss.detach()),
                "reference_loss": float(reference.loss.detach()),
                "loss_difference": float(native.loss.detach() - reference.loss.detach()),
                "gradient_effect_rms_over_reference": ratio(native_grad - reference_grad, reference_grad),
                "write_effect_rms_over_reference": ratio(effect, reference_write),
                "write_aligned": float(torch.sum(effect * reference_write)) / max(float(torch.sum(reference_write * reference_write)), 1e-30),
            })
    return {
        "schema": "kernel-analyzer-rwkv-time-mix-natural-v1",
        "status": "COMPLETE_NATURAL_RWKV_TIME_MIX_BOUNDARY",
        "model": str(model_path),
        "operator_family": "rwkv_time_mix_recurrent_attention",
        "target_parameter": "rwkv.blocks.0.attention.time_decay",
        "candidate": "native RWKV sequential WKV recurrence",
        "references": {
            "value_fp32": "only recurrent value materialization changed to FP32",
            "decay_fp32": "only time-decay exponential changed to FP32",
            "all_fp32": "both declared boundaries changed to FP32",
        },
        "input_source": str(text_path),
        "comparison_scope": {"same_model_weights": True, "same_input_ids": True, "same_block": True, "single_changed_boundaries": ["recurrent value materialization", "time-decay exponential"]},
        "rows": rows,
        "summary": {
            "state_count": count,
            "modes": {
                mode: {
                    "write_effect_rms_mean": float(torch.tensor([r["write_effect_rms_over_reference"] for r in rows if r["mode"] == mode], dtype=torch.float64).mean()),
                    "write_aligned_interval_normal_95": interval([r["write_aligned"] for r in rows if r["mode"] == mode]),
                    "loss_difference_interval_normal_95": interval([r["loss_difference"] for r in rows if r["mode"] == mode]),
                    "positive": sum(r["write_aligned"] > 0 for r in rows if r["mode"] == mode),
                    "negative": sum(r["write_aligned"] < 0 for r in rows if r["mode"] == mode),
                } for mode in ("value_fp32", "decay_fp32", "all_fp32")
            },
        },
        "claim_boundary": "One RWKV-4-world checkpoint, block 0, time-decay carrier and declared real-text bank; source/fixed-suite result only, no population or long-run loss claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.model, args.text, args.count, args.max_length, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
