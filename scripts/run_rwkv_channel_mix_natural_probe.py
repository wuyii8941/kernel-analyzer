#!/usr/bin/env python3
"""Natural RWKV channel-mix square-ReLU materialization probe.

The native RWKV feed-forward channel path computes ``square(relu(key(x)))``
in the model dtype.  The probe keeps the recurrent block, inputs and all
other feed-forward operations fixed, and changes only the declared ReLU/square
evaluation boundary before the original dtype write-back.  It reports the
actual gradient and one-step AdamW write of the channel key weight.
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
    if not lines:
        raise RuntimeError("empty text source")
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


def install_mode(model: Any, mode_ref: dict[str, str], block: int) -> None:
    feed_forward = model.rwkv.blocks[block].feed_forward
    original = getattr(feed_forward, "_kernel_analyzer_original_forward", feed_forward.forward)
    feed_forward._kernel_analyzer_original_forward = original

    def forward(self, hidden, state=None):
        if hidden.size(1) == 1 and state is not None:
            shifted = state[0][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[0][:, :, self.layer_id]
        key_input = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        receptance_input = hidden * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)
        key_pre = self.key(key_input)
        mode = mode_ref["mode"]
        if mode == "native":
            key = torch.square(torch.relu(key_pre))
        elif mode == "square_fp32":
            key = torch.square(torch.relu(key_pre).float()).to(key_pre.dtype)
        elif mode == "relu_square_fp32":
            key = torch.square(torch.relu(key_pre.float())).to(key_pre.dtype)
        else:
            raise ValueError(f"unknown mode {mode}")
        value = self.value(key)
        receptance = torch.sigmoid(self.receptance(receptance_input))
        if state is not None:
            state[0][:, :, self.layer_id] = hidden[:, -1]
        return receptance * value, state

    feed_forward.forward = types.MethodType(forward, feed_forward)


def run(model_path: Path, text_path: Path, count: int, max_length: int, device: str, block: int) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16,
    ).to(device).eval()
    model.config.use_cache = False
    target = model.rwkv.blocks[block].feed_forward.key.weight
    states = text_states(tokenizer, text_path, count, max_length)
    base_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    mode_ref = {"mode": "native"}
    install_mode(model, mode_ref, block)
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
        for mode in ("square_fp32", "relu_square_fp32"):
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
                "loss_difference_native_minus_reference": float(native.loss.detach() - reference.loss.detach()),
                "gradient_effect_rms_over_reference": ratio(native_grad - reference_grad, reference_grad),
                "write_effect_rms_over_reference": ratio(effect, reference_write),
                "write_aligned": float(torch.sum(effect * reference_write)) / max(float(torch.sum(reference_write * reference_write)), 1e-30),
            })
    return {
        "schema": "kernel-analyzer-rwkv-channel-mix-natural-v1",
        "status": "COMPLETE_NATURAL_RWKV_CHANNEL_MIX_BOUNDARY",
        "model": str(model_path),
        "operator_family": "rwkv_channel_mix_square_relu",
        "target_parameter": f"rwkv.blocks.{block}.feed_forward.key.weight",
        "candidate": "native RWKV channel-mix square(relu(key)) in model dtype",
        "references": {
            "square_fp32": "native ReLU followed by FP32 square and original dtype write-back",
            "relu_square_fp32": "FP32 ReLU and square followed by original dtype write-back",
        },
        "input_source": str(text_path),
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_block": True,
            "block": block,
            "single_changed_boundaries": ["channel-mix square", "channel-mix ReLU plus square"],
        },
        "rows": rows,
        "summary": {
            "state_count": count,
            "modes": {
                mode: {
                    "write_effect_rms_mean": float(torch.tensor([r["write_effect_rms_over_reference"] for r in rows if r["mode"] == mode], dtype=torch.float64).mean()),
                    "write_aligned_interval_normal_95": interval([r["write_aligned"] for r in rows if r["mode"] == mode]),
                    "loss_difference_interval_normal_95": interval([r["loss_difference_native_minus_reference"] for r in rows if r["mode"] == mode]),
                    "positive": sum(r["write_aligned"] > 0 for r in rows if r["mode"] == mode),
                    "negative": sum(r["write_aligned"] < 0 for r in rows if r["mode"] == mode),
                }
                for mode in ("square_fp32", "relu_square_fp32")
            },
        },
        "claim_boundary": "One RWKV-4-world checkpoint, block 0, channel-mix key carrier and declared real-text bank; source/fixed-suite result only, no population or long-run loss claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model, args.text, args.count, args.max_length, args.device, args.block)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
