#!/usr/bin/env python3
"""Natural RWKV receptance-sigmoid materialization probe."""

from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/RWKV/rwkv-4-world-169m")
DEFAULT_TEXT = ROOT / "README.md"


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
    states: list[dict[str, torch.Tensor]] = []
    for index in range(count):
        body = " ".join(lines[index % len(lines) : index % len(lines) + 4])
        if len(body) < 32:
            body = (body + " " + " ".join(lines))[:1024]
        ids = tokenizer(body, max_length=max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        if ids.numel() >= 2:
            states.append({"input_ids": ids, "labels": ids.clone()})
    if len(states) < count:
        raise RuntimeError(f"only built {len(states)} usable states out of {count}")
    return states[:count]


def install_mode(model: Any, mode_ref: dict[str, str]) -> None:
    attention = model.rwkv.blocks[0].attention
    original = getattr(attention, "_kernel_analyzer_original_extract_key_value", attention.extract_key_value)
    attention._kernel_analyzer_original_extract_key_value = original

    def extract_key_value(self, hidden, state=None):
        if hidden.size(1) == 1 and state is not None:
            shifted = state[1][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[1][:, :, self.layer_id]
        key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        value = hidden * self.time_mix_value + shifted * (1 - self.time_mix_value)
        receptance_input = hidden * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)
        key = self.key(key)
        value = self.value(value)
        receptance_pre = self.receptance(receptance_input)
        if mode_ref["mode"] == "native":
            receptance = torch.sigmoid(receptance_pre)
        elif mode_ref["mode"] == "fp32":
            receptance = torch.sigmoid(receptance_pre.float()).to(receptance_pre.dtype)
        else:
            raise ValueError(mode_ref["mode"])
        if state is not None:
            state[1][:, :, self.layer_id] = hidden[:, -1]
        return receptance, key, value, state

    attention.extract_key_value = types.MethodType(extract_key_value, attention)


def run(model_path: Path, text_path: Path, count: int, max_length: int, device: str) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16,
    ).to(device).eval()
    model.config.use_cache = False
    target = model.rwkv.blocks[0].attention.receptance.weight
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
        model.load_state_dict(base_state, strict=True)
        model.zero_grad(set_to_none=True)
        mode_ref["mode"] = "fp32"
        reference = model(**batch)
        reference.loss.backward()
        reference_grad = target.grad.detach().float().clone()
        reference_write = write(reference_grad)
        effect = native_write - reference_write
        rows.append({
            "state_id": index,
            "native_loss": float(native.loss.detach()),
            "reference_loss": float(reference.loss.detach()),
            "loss_difference_native_minus_reference": float(native.loss.detach() - reference.loss.detach()),
            "gradient_effect_rms_over_reference": ratio(native_grad - reference_grad, reference_grad),
            "write_effect_rms_over_reference": ratio(effect, reference_write),
            "write_aligned": float(torch.sum(effect * reference_write)) / max(float(torch.sum(reference_write * reference_write)), 1e-30),
        })
    split = count // 2
    return {
        "schema": "kernel-analyzer-rwkv-receptance-sigmoid-natural-v1",
        "status": "COMPLETE_NATURAL_RWKV_RECEPTANCE_SIGMOID_BOUNDARY",
        "model": str(model_path),
        "operator_family": "rwkv_receptance_sigmoid_materialization",
        "target_parameter": "rwkv.blocks.0.attention.receptance.weight",
        "candidate": "native RWKV receptance sigmoid in model dtype",
        "reference": "same RWKV path with only sigmoid evaluated in FP32 then original dtype write-back",
        "input_source": str(text_path),
        "comparison_scope": {"same_model_weights": True, "same_input_ids": True, "same_block": True, "single_changed_boundary": "receptance sigmoid evaluation"},
        "rows": rows,
        "summary": {
            "state_count": count,
            "calibration_count": split,
            "confirmation_count": count - split,
            "write_effect_rms_mean": float(torch.tensor([r["write_effect_rms_over_reference"] for r in rows], dtype=torch.float64).mean()),
            "aligned_write_interval_normal_95": interval([r["write_aligned"] for r in rows]),
            "confirmation_aligned_write_interval_normal_95": interval([r["write_aligned"] for r in rows[split:]]),
            "confirmation_positive": sum(r["write_aligned"] > 0 for r in rows[split:]),
            "confirmation_negative": sum(r["write_aligned"] < 0 for r in rows[split:]),
            "loss_difference_interval_normal_95": interval([r["loss_difference_native_minus_reference"] for r in rows]),
        },
        "claim_boundary": "One RWKV-4-world checkpoint, block 0, receptance carrier and declared real-text bank; source/fixed-suite result only, no population or long-run loss claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model, args.text, args.count, args.max_length, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
