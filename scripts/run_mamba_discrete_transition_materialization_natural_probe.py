#!/usr/bin/env python3
"""Natural Mamba probe for discrete state-transition materialization.

Only the computation of exp(A * delta) is changed.  The reference evaluates
the exponent argument in the model dtype, evaluates the exponential, and
casts the result back to FP32 before the unchanged sequential recurrence.
All other Mamba operations, weights, token windows and loss remain shared.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, MambaForCausalLM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/state-spaces/mamba-130m-hf")
DEFAULT_TEXT = ROOT / "docs/root_cause_closure_current.md"


def windows(tokenizer: Any, source: Path, seq_len: int, count: int) -> list[torch.Tensor]:
    tokens = tokenizer(source.read_text(encoding="utf-8"), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    stride = max(1, seq_len * 2)
    need = (count - 1) * stride + seq_len
    if tokens.numel() < need:
        raise RuntimeError(f"text source has {tokens.numel()} tokens, need {need}")
    return [tokens[i * stride : i * stride + seq_len].clone() for i in range(count)]


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if x.numel() < 2:
        value = float(x.mean().item()) if x.numel() else 0.0
        return [value, value]
    mean = float(x.mean().item())
    half = 1.96 * float(x.std(unbiased=True).item()) / math.sqrt(x.numel())
    return [mean - half, mean + half]


def relative(effect: torch.Tensor, reference: torch.Tensor) -> float:
    denom = float(torch.linalg.vector_norm(reference).item())
    return float(torch.linalg.vector_norm(effect).item()) / max(denom, 1e-30)


def write(gradient: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    g = gradient.float()
    return -lr * g / (g.abs() + eps)


def install_transition_variant(mixer: torch.nn.Module, fp32_argument: bool) -> None:
    def slow_forward(
        self: torch.nn.Module,
        input_states: torch.Tensor,
        cache_params: Any = None,
        cache_position: Any = None,
        attention_mask: Any = None,
    ) -> torch.Tensor:
        # This follows the installed Transformers Mamba slow path.  The only
        # changed expression is the materialization of discrete_A.
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        projected_states = self.in_proj(input_states).transpose(1, 2)
        hidden_states, gate = projected_states.chunk(2, dim=1)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)
        if cache_params is not None:
            raise RuntimeError("cache mode is outside this probe")
        ssm_state = torch.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size),
            device=hidden_states.device,
            dtype=dtype,
        )
        hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )
        discrete_time_step = self.dt_proj(time_step)
        discrete_time_step = torch.nn.functional.softplus(discrete_time_step).transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        argument = A[None, :, None, :] * discrete_time_step[:, :, :, None]
        if fp32_argument:
            discrete_A = torch.exp(argument.to(dtype)).float()
        else:
            discrete_A = torch.exp(argument)
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        scan_outputs = []
        for index in range(seq_len):
            ssm_state = discrete_A[:, :, index, :] * ssm_state + deltaB_u[:, :, index, :]
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, index, :].unsqueeze(-1))
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)
        scan_output = scan_output + hidden_states * self.D[None, :, None]
        scan_output = scan_output * self.act(gate)
        return self.out_proj(scan_output.transpose(1, 2))

    mixer.slow_forward = types.MethodType(slow_forward, mixer)


def run_once(
    model: torch.nn.Module,
    ids: torch.Tensor,
    target: torch.nn.Parameter,
    fp32_argument: bool,
) -> tuple[float, torch.Tensor]:
    mixer = model.backbone.layers[0].mixer
    original = mixer.slow_forward
    install_transition_variant(mixer, fp32_argument)
    model.zero_grad(set_to_none=True)
    try:
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
        if target.grad is None:
            raise RuntimeError("target gradient missing")
        return float(loss.detach()), target.grad.detach().float().cpu().clone()
    finally:
        mixer.slow_forward = original


def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bank = windows(tokenizer, args.text_source, args.sequence_length, args.states)
    model = MambaForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16
    ).to(args.device).eval()
    model.config.use_cache = False
    target_name = args.parameter or "backbone.layers.0.mixer.A_log"
    target = dict(model.named_parameters())[target_name]
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for state_id, tokens in enumerate(bank):
        ids = tokens.unsqueeze(0).to(args.device)
        native_loss, native_grad = run_once(model, ids, target, False)
        reference_loss, reference_grad = run_once(model, ids, target, True)
        native_write = write(native_grad, args.learning_rate, args.eps)
        reference_write = write(reference_grad, args.learning_rate, args.eps)
        gradient_effect = reference_grad - native_grad
        write_effect = reference_write - native_write
        effects.append(gradient_effect.double().reshape(-1))
        writes.append(write_effect.double().reshape(-1))
        references.append(native_write.double().reshape(-1))
        effect_flat = write_effect.double().reshape(-1)
        ref_flat = native_write.double().reshape(-1)
        rows.append({
            "state_id": state_id,
            "native_loss": native_loss,
            "fp32_argument_loss": reference_loss,
            "loss_difference_fp32_argument_minus_native": reference_loss - native_loss,
            "gradient_effect_rms_over_native": relative(gradient_effect, native_grad),
            "write_effect_rms_over_native": relative(write_effect, native_write),
            "aligned_write_ratio": float(torch.dot(effect_flat, ref_flat).item())
            / max(float(torch.dot(ref_flat, ref_flat).item()), 1e-30),
        })
        print(json.dumps({"event": "MAMBA_TRANSITION_STATE", "state": state_id}), flush=True)

    split = len(effects) // 2
    direction = torch.stack(effects[:split]).mean(dim=0)
    direction_norm = float(direction.norm().item())
    projections: list[float] = []
    if direction_norm > 0:
        direction = direction / direction_norm
        projections = [float(torch.dot(value, direction).item()) for value in effects[split:]]
    aligned = [row["aligned_write_ratio"] for row in rows[split:]]
    return {
        "schema": "kernel-analyzer-mamba-discrete-transition-materialization-natural-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "Mamba discrete state-transition exp(A*delta) materialization",
        "parameter": target_name,
        "candidate": "native FP32 argument exponential in the sequential Mamba path",
        "reference": "model-dtype exponent argument, exponential, then FP32 recurrence value",
        "input_source": str(args.text_source),
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_sequential_recurrence": True,
            "single_changed_boundary": "discrete transition exp(A*delta)",
        },
        "state_count": len(rows),
        "calibration_count": split,
        "confirmation_count": len(rows) - split,
        "rows": rows,
        "summary": {
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_native"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_native"] for r in rows) / len(rows),
            "aligned_write_mean_confirmation": sum(aligned) / len(aligned),
            "aligned_write_interval_normal_95": interval(aligned),
            "aligned_positive_count": sum(value > 0 for value in aligned),
            "aligned_negative_count": sum(value < 0 for value in aligned),
            "heldout_projection_interval_normal_95": interval(projections),
            "heldout_projection_positive": sum(value > 0 for value in projections),
            "heldout_projection_negative": sum(value < 0 for value in projections),
            "calibration_direction_norm": direction_norm,
        },
        "claim_boundary": "One real Mamba checkpoint, one sequential transition boundary and the declared text bank; not a population or long-run quality claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--text-source", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--parameter", default=None)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
