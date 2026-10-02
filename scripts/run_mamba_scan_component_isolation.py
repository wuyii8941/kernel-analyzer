#!/usr/bin/env python3
"""Source-isolation probe for the Mamba parallel-scan versus recurrence gap.

The probe deliberately keeps the model, token windows, loss, and downstream
parameter fixed.  It compares a sequential recurrence with a parallel scan
in one selected layer and optionally evaluates that layer's x_proj and
dt_proj in FP32 before returning to the original dtype.  The result is a
source-isolation artifact only; a new problem group is promoted only when a
single changed boundary produces a reproducible signed write effect.
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


def first_step_write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    value = gradient.float()
    return -learning_rate * value / (value.abs() + eps)


def relative(effect: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(reference).item())
    return float(torch.linalg.vector_norm(effect).item()) / max(denominator, 1e-30)


def install_linear_precision(mixer: torch.nn.Module, component: str) -> list[tuple[torch.nn.Module, Any]]:
    saved: list[tuple[torch.nn.Module, Any]] = []
    names = ("x_proj", "dt_proj") if component == "both" else (component,)
    for name in names:
        module = getattr(mixer, name)
        original = module.forward

        def fp32_forward(self: torch.nn.Module, value: torch.Tensor, _original: Any = original) -> torch.Tensor:
            bias = self.bias.float() if self.bias is not None else None
            output = torch.nn.functional.linear(value.float(), self.weight.float(), bias)
            return output.to(value.dtype)

        module.forward = types.MethodType(fp32_forward, module)
        saved.append((module, original))
    return saved


def restore_linear_precision(saved: list[tuple[torch.nn.Module, Any]]) -> None:
    for module, original in saved:
        module.forward = original


def configure_recurrence(model: MambaForCausalLM, layer: int, parallel: bool) -> None:
    for index, item in enumerate(model.backbone.layers):
        item.mixer.use_mambapy = bool(parallel and index == layer)


def reassociated_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Differentiable inclusive scan with pairwise reassociation.

    Each pair ``(a_t, b_t)`` represents ``h_t = a_t*h_{t-1}+b_t``.
    Hillis--Steele composition keeps the same recurrence but changes the
    parenthesization, which is the numerical source being tested here.
    """
    prefix_a = a.clone()
    prefix_b = b.clone()
    length = prefix_a.shape[1]
    offset = 1
    while offset < length:
        old_a = prefix_a.clone()
        old_b = prefix_b.clone()
        prefix_a[:, offset:] = old_a[:, offset:] * old_a[:, :-offset]
        prefix_b[:, offset:] = old_b[:, offset:] + old_a[:, offset:] * old_b[:, :-offset]
        offset *= 2
    # The installed Mamba slow path expects the scan state to share the
    # input-dtype representation used by C in the following contraction.
    return prefix_b.to(torch.bfloat16)


def run_once(
    model: MambaForCausalLM,
    target: torch.nn.Parameter,
    ids: torch.Tensor,
    layer: int,
    parallel: bool,
) -> tuple[float, torch.Tensor]:
    configure_recurrence(model, layer, parallel)
    model.train(parallel)
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    if target.grad is None:
        raise RuntimeError("target gradient is missing")
    return float(loss.detach().cpu()), target.grad.detach().float().cpu().clone()


def load_states(bank_path: Path, count: int) -> list[torch.Tensor]:
    payload = json.loads(bank_path.read_text(encoding="utf-8"))
    states = payload.get("states", payload if isinstance(payload, list) else None)
    if not states or len(states) < count:
        raise RuntimeError(f"input bank has fewer than {count} states: {bank_path}")
    return [torch.tensor(row["token_ids"], dtype=torch.long) for row in states[:count]]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this source-isolation probe")
    states = load_states(args.input_bank, args.states)
    model = MambaForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16
    ).to(args.device)
    model.config.use_cache = False
    target_name = args.parameter or f"backbone.layers.{args.layer}.mixer.out_proj.weight"
    target = dict(model.named_parameters())[target_name]

    # Force the installed implementation through the explicit slow path so
    # that the only branch difference is the selected parallel scan.
    saved_globals = {
        name: getattr(modeling_mamba, name, None)
        for name in ("selective_scan_fn", "mamba_inner_fn", "selective_state_update", "pscan")
    }
    for name in ("selective_scan_fn", "mamba_inner_fn", "selective_state_update"):
        if hasattr(modeling_mamba, name):
            setattr(modeling_mamba, name, None)
    # Some Transformers environments do not install mambapy.  Supplying the
    # local reassociated scan keeps the comparison executable without changing
    # the sequential reference or claiming the external package was used.
    modeling_mamba.pscan = reassociated_scan

    calibration_count = min(args.calibration_count, args.states // 2)
    variants = ("native", "x_proj_fp32", "dt_proj_fp32", "both_fp32")
    rows_by_variant: dict[str, list[dict[str, Any]]] = {name: [] for name in variants}
    effects_by_variant: dict[str, list[torch.Tensor]] = {name: [] for name in variants}
    writes_by_variant: dict[str, list[torch.Tensor]] = {name: [] for name in variants}
    refs_by_variant: dict[str, list[torch.Tensor]] = {name: [] for name in variants}
    try:
        for variant in variants:
            component = {
                "native": None,
                "x_proj_fp32": "x_proj",
                "dt_proj_fp32": "dt_proj",
                "both_fp32": "both",
            }[variant]
            layer_module = model.backbone.layers[args.layer].mixer
            saved = install_linear_precision(layer_module, component) if component else []
            try:
                for state_id, tokens in enumerate(states):
                    ids = tokens.unsqueeze(0).to(args.device)
                    reference_loss, reference_grad = run_once(model, target, ids, args.layer, False)
                    candidate_loss, candidate_grad = run_once(model, target, ids, args.layer, True)
                    reference_write = first_step_write(reference_grad, args.learning_rate, args.eps)
                    candidate_write = first_step_write(candidate_grad, args.learning_rate, args.eps)
                    gradient_effect = candidate_grad - reference_grad
                    write_effect = candidate_write - reference_write
                    effects_by_variant[variant].append(write_effect.double().reshape(-1))
                    writes_by_variant[variant].append(write_effect.double().reshape(-1))
                    refs_by_variant[variant].append(reference_write.double().reshape(-1))
                    rows_by_variant[variant].append({
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
            finally:
                restore_linear_precision(saved)
    finally:
        for name, value in saved_globals.items():
            if hasattr(modeling_mamba, name):
                setattr(modeling_mamba, name, value)

    # Freeze a direction from the unmodified recurrence comparison only.
    baseline = effects_by_variant["native"]
    raw_direction = sum(baseline[:calibration_count], torch.zeros_like(baseline[0]))
    direction_norm = float(torch.linalg.vector_norm(raw_direction).item())
    direction = raw_direction / direction_norm if direction_norm else None
    summaries: dict[str, Any] = {}
    for variant in variants:
        rows = rows_by_variant[variant]
        heldout = rows[calibration_count:]
        aligned = [row["aligned_write_ratio"] for row in heldout]
        projections = [float(torch.dot(value, direction).item()) for value in effects_by_variant[variant][calibration_count:]] if direction is not None else []
        summaries[variant] = {
            "write_effect_rms_mean": sum(row["write_effect_rms_over_reference"] for row in rows) / len(rows),
            "heldout_aligned_write_mean": sum(aligned) / len(aligned),
            "heldout_aligned_write_interval_normal_95": interval(aligned),
            "heldout_aligned_positive": sum(value > 0 for value in aligned),
            "heldout_aligned_negative": sum(value < 0 for value in aligned),
            "heldout_frozen_baseline_direction_mean": sum(projections) / len(projections) if projections else None,
            "heldout_frozen_baseline_direction_interval_normal_95": interval(projections) if projections else None,
            "heldout_frozen_baseline_direction_positive": sum(value > 0 for value in projections),
            "heldout_frozen_baseline_direction_negative": sum(value < 0 for value in projections),
        }

    return {
        "schema": "kernel-analyzer-mamba-scan-component-isolation-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "input_bank": str(args.input_bank),
        "layer": args.layer,
        "target_parameter": target_name,
        "candidate": "parallel pscan in selected Mamba layer",
        "reference": "explicit sequential recurrence in all layers",
        "variants": {
            "native": "no source intervention",
            "x_proj_fp32": "selected layer x_proj evaluated in FP32 then cast back",
            "dt_proj_fp32": "selected layer dt_proj evaluated in FP32 then cast back",
            "both_fp32": "both selected projections evaluated in FP32 then cast back",
        },
        "state_count": len(states),
        "calibration_count": calibration_count,
        "confirmation_count": len(states) - calibration_count,
        "direction_frozen_from": "native variant calibration states",
        "summaries": summaries,
        "rows": rows_by_variant,
        "claim_boundary": "A source-isolation experiment for one recurrence layer; a new problem group requires a strictly signed held-out aligned effect and a path-preserving source interpretation. Mixed signs remain a candidate or variance result.",
    }


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
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summaries": result["summaries"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
