#!/usr/bin/env python3
"""Single-layer probe for the installed Mamba fused scan implementation.

All non-selected layers use the explicit sequential path.  Only the selected
layer uses the actual ``mamba_ssm`` fused forward, so a positive result is a
local implementation-boundary observation rather than an attribution to the
whole model's fused region.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path
from typing import Any

import torch
from transformers import MambaForCausalLM
from transformers.models.mamba import modeling_mamba


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/state-spaces/mamba-130m-hf")
DEFAULT_BANK = ROOT / "results/coverage/mamba_seq128_input_bank.json"


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if x.numel() < 2:
        value = float(x.mean().item()) if x.numel() else 0.0
        return [value, value]
    mean = float(x.mean().item())
    half = 1.96 * float(x.std(unbiased=True).item()) / math.sqrt(x.numel())
    return [mean - half, mean + half]


def write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    value = gradient.float()
    return -learning_rate * value / (value.abs() + eps)


def relative(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect).item()) / max(float(torch.linalg.vector_norm(reference).item()), 1e-30)


def load_states(path: Path, count: int) -> list[torch.Tensor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    states = payload.get("states", payload if isinstance(payload, list) else None)
    if not states or len(states) < count:
        raise RuntimeError(f"input bank has fewer than {count} states")
    return [torch.tensor(row["token_ids"], dtype=torch.long) for row in states[:count]]


def install_actual_fused(model: MambaForCausalLM, selected_layer: int) -> list[tuple[torch.nn.Module, Any]]:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update

    modeling_mamba.mamba_inner_fn = mamba_inner_fn
    modeling_mamba.selective_scan_fn = selective_scan_fn
    modeling_mamba.selective_state_update = selective_state_update
    modeling_mamba._causal_conv1d_cache = None
    saved: list[tuple[torch.nn.Module, Any]] = []
    for index, block in enumerate(model.backbone.layers):
        mixer = block.mixer
        original = mixer.forward
        method = mixer.cuda_kernels_forward if index == selected_layer else mixer.slow_forward
        mixer.forward = types.MethodType(
            lambda self, hidden_states, cache_params=None, cache_position=None, attention_mask=None, _method=method: _method(
                hidden_states, cache_params, cache_position, attention_mask
            ),
            mixer,
        )
        saved.append((mixer, original))
    return saved


def restore(saved: list[tuple[torch.nn.Module, Any]]) -> None:
    for module, original in saved:
        module.forward = original


def run_once(model: MambaForCausalLM, target: torch.nn.Parameter, ids: torch.Tensor) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    if target.grad is None:
        raise RuntimeError("target gradient is missing")
    return float(loss.detach().cpu()), target.grad.detach().float().cpu().clone()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    states = load_states(args.input_bank, args.states)
    model = MambaForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True).to(args.device)
    model.config.use_cache = False
    target_name = args.parameter or f"backbone.layers.{args.layer}.mixer.out_proj.weight"
    target = dict(model.named_parameters())[target_name]
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    refs: list[torch.Tensor] = []
    # Reference: every layer explicitly sequential.  Candidate: only the
    # selected layer uses the actual installed fused mamba_ssm path.
    for state_id, token_ids in enumerate(states):
        ids = token_ids.unsqueeze(0).to(args.device)
        model.train(False)
        reference_loss, reference_grad = run_once(model, target, ids)
        saved = install_actual_fused(model, args.layer)
        try:
            model.train(True)
            candidate_loss, candidate_grad = run_once(model, target, ids)
        finally:
            restore(saved)
        reference_write = write(reference_grad, args.learning_rate, args.eps)
        candidate_write = write(candidate_grad, args.learning_rate, args.eps)
        write_effect = candidate_write - reference_write
        gradient_effect = candidate_grad - reference_grad
        effects.append(write_effect.double().reshape(-1))
        writes.append(write_effect.double().reshape(-1))
        refs.append(reference_write.double().reshape(-1))
        rows.append({
            "state_id": state_id,
            "reference_loss": reference_loss,
            "candidate_loss": candidate_loss,
            "loss_difference": candidate_loss - reference_loss,
            "gradient_effect_rms_over_reference": relative(gradient_effect, reference_grad),
            "write_effect_rms_over_reference": relative(write_effect, reference_write),
            "aligned_write_ratio": float(torch.dot(write_effect.double().reshape(-1), reference_write.double().reshape(-1)).item())
            / max(float(torch.dot(reference_write.double().reshape(-1), reference_write.double().reshape(-1)).item()), 1e-30),
        })
        del ids, reference_grad, candidate_grad
        torch.cuda.empty_cache()

    calibration_count = min(args.calibration_count, args.states // 2)
    raw = sum(effects[:calibration_count], torch.zeros_like(effects[0]))
    norm = float(torch.linalg.vector_norm(raw).item())
    direction = raw / norm if norm else None
    heldout = rows[calibration_count:]
    aligned = [row["aligned_write_ratio"] for row in heldout]
    projections = [float(torch.dot(value, direction).item()) for value in effects[calibration_count:]] if direction is not None else []
    output = {
        "schema": "kernel-analyzer-mamba-official-local-scan-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "input_bank": str(args.input_bank),
        "layer": args.layer,
        "target_parameter": target_name,
        "candidate": "actual installed mamba_ssm fused forward in selected layer only",
        "reference": "explicit sequential recurrence in every layer",
        "state_count": len(rows),
        "calibration_count": calibration_count,
        "confirmation_count": len(heldout),
        "rows": rows,
        "summary": {
            "write_effect_rms_mean": sum(row["write_effect_rms_over_reference"] for row in rows) / len(rows),
            "heldout_aligned_write_mean": sum(aligned) / len(aligned),
            "heldout_aligned_write_interval_normal_95": interval(aligned),
            "heldout_aligned_positive": sum(value > 0 for value in aligned),
            "heldout_aligned_negative": sum(value < 0 for value in aligned),
            "heldout_frozen_direction_mean": sum(projections) / len(projections) if projections else None,
            "heldout_frozen_direction_interval_normal_95": interval(projections) if projections else None,
            "heldout_frozen_direction_positive": sum(value > 0 for value in projections),
            "heldout_frozen_direction_negative": sum(value < 0 for value in projections),
        },
        "claim_boundary": "A real single-layer fused-versus-sequential boundary on one checkpoint and bank; source mechanism is not closed unless a path-preserving component intervention reproduces the effect.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": output["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--parameter", default=None)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--calibration-count", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
