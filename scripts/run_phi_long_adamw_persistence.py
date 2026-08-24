#!/usr/bin/env python3
"""Warm-state, multi-horizon direct-persistence experiment for Phi lm_head dX.

This runner deliberately measures only the implementation's same-state direct
effective-update difference.  It first advances the declared parameter and
AdamW moments with the natural implementation, then measures a fresh sequence
of natural-versus-repair update differences without resetting optimizer state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor")

import numpy as np
import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "archive/round1_code/src"):
    sys.path.insert(0, str(path))

from kernel_analyzer.persistence_property import path_statistics_from_gram  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_heldout_lmhead_consequence import adam_delta  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import MMRepair  # noqa: E402


def statistics(
    gram: np.ndarray,
    state_ids: list[str],
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    return path_statistics_from_gram(
        gram,
        state_ids=state_ids,
        max_lag=min(64, len(state_ids) - 1),
        sign_flip_draws=draws,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--state-offset", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=128)
    parser.add_argument("--measurement-steps", type=int, default=512)
    parser.add_argument("--window-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--reset-moments-after-warmup", action="store_true")
    parser.add_argument("--null-draws", type=int, default=4000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gram-output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.state_offset < 0 or args.warmup_steps < 0 or args.measurement_steps < 32:
        raise ValueError("warmup must be nonnegative and measurement needs at least 32 steps")
    if args.window_steps < 16 or args.measurement_steps % args.window_steps:
        raise ValueError("measurement steps must be divisible by the window size")
    if args.null_draws < 100:
        raise ValueError("too few sign-flip null draws")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)

    bank = json.loads(args.input_bank.read_text())
    all_states = bank.get("states", bank.get("records"))
    run_states = args.warmup_steps + args.measurement_steps
    needed = args.state_offset + run_states
    if len(all_states) < needed:
        raise RuntimeError(f"input bank has {len(all_states)} states; index {needed} is required")
    states = all_states[args.state_offset:needed]
    state_ids = [str(row.get("state_id", index)) for index, row in enumerate(states)]
    if len(set(state_ids)) != len(state_ids):
        raise RuntimeError("long-horizon state IDs must be unique")

    configure_candidate_runtime(24_000)
    model = load_model(
        "phi", Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct"), device
    )
    model.eval()
    carrier = model.model.norm.weight
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])

    master = carrier.detach().float().clone()
    first = torch.zeros_like(master)
    second = torch.zeros_like(master)

    def gradient(state: dict[str, Any], *, repair: bool, step_seed: int) -> tuple[str, float, torch.Tensor, Any]:
        with torch.no_grad():
            carrier.copy_(master.to(carrier.dtype))
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        torch.manual_seed(step_seed)
        torch.cuda.manual_seed_all(step_seed)
        model.zero_grad(set_to_none=True)
        observer = MMRepair(modules, "REPAIR_FP32_CAST_BF16") if repair else None
        if observer is None:
            loss = candidate(values)
            loss.backward()
        else:
            with observer:
                loss = candidate(values)
                loss.backward()
        torch.cuda.synchronize(device)
        if carrier.grad is None or not torch.isfinite(carrier.grad).all():
            raise RuntimeError("carrier gradient is absent or nonfinite")
        return (
            tensor_digest(loss),
            float(loss.detach().float().item()),
            carrier.grad.detach().float().clone(),
            observer,
        )

    warmup_losses: list[float] = []
    for index, state in enumerate(states[:args.warmup_steps]):
        step = index + 1
        _, loss_value, grad, _ = gradient(
            state, repair=False, step_seed=40_000 + index
        )
        update, first, second = adam_delta(
            grad,
            first,
            second,
            step,
            learning_rate=args.learning_rate,
            beta1=0.9,
            beta2=0.95,
        )
        master.add_(update)
        warmup_losses.append(loss_value)
        if not args.quiet and (step == 1 or step % 16 == 0):
            print(json.dumps({
                "event": "PHI_LONG_WARMUP",
                "step": step,
                "loss": loss_value,
            }), flush=True)
        del grad, update

    if args.reset_moments_after_warmup:
        first.zero_()
        second.zero_()

    vectors: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    measured_ids: list[str] = []
    measurement_states = states[args.warmup_steps:needed]
    for index, state in enumerate(measurement_states):
        absolute_step = (
            index + 1
            if args.reset_moments_after_warmup
            else args.warmup_steps + index + 1
        )
        state_id = state_ids[args.warmup_steps + index]
        loss_n_digest, loss_value, grad_n, _ = gradient(
            state, repair=False, step_seed=50_000 + index
        )
        loss_r_digest, _, grad_r, observer = gradient(
            state, repair=True, step_seed=50_000 + index
        )
        if loss_n_digest != loss_r_digest:
            raise RuntimeError("backward-only repair changed forward loss")
        if observer is None or observer.calls != 1 or observer.local is None:
            raise RuntimeError("exact Phi repair endpoint was not hit once")

        update_n, next_first, next_second = adam_delta(
            grad_n,
            first,
            second,
            absolute_step,
            learning_rate=args.learning_rate,
            beta1=0.9,
            beta2=0.95,
        )
        update_r, _, _ = adam_delta(
            grad_r,
            first,
            second,
            absolute_step,
            learning_rate=args.learning_rate,
            beta1=0.9,
            beta2=0.95,
        )
        direct = update_n - update_r
        if not torch.isfinite(direct).all():
            raise RuntimeError("direct effective-update difference is nonfinite")
        vectors.append(direct.detach().cpu())
        measured_ids.append(state_id)
        reference_energy = float(torch.dot(update_r.reshape(-1), update_r.reshape(-1)).item())
        reference_relative_alpha = float(
            torch.dot(direct.reshape(-1), update_r.reshape(-1)).item()
            / max(reference_energy, 1e-30)
        )
        rows.append({
            "measurement_step": index + 1,
            "optimizer_step": absolute_step,
            "state_id": state_id,
            "loss": loss_value,
            "direct_update_l2": float(torch.linalg.vector_norm(direct).item()),
            "reference_update_l2": reference_energy ** 0.5,
            "reference_relative_alpha": reference_relative_alpha,
            "endpoint_changed_coordinates": int(observer.local["changed_coordinates"]),
            "endpoint_error_l2": float(observer.local["l2"]),
        })
        master.add_(update_n)
        first, second = next_first, next_second
        if not args.quiet and ((index + 1) == 1 or (index + 1) % 16 == 0):
            print(json.dumps({"event": "PHI_LONG_MEASUREMENT", **rows[-1]}), flush=True)
        del grad_n, grad_r, update_n, update_r, direct

    matrix = torch.stack(vectors).double()
    gram = (matrix @ matrix.T).numpy()
    del matrix, vectors

    horizons: dict[str, Any] = {}
    for horizon in (16, 32, 64, 128, 256, 512):
        if horizon > args.measurement_steps:
            continue
        horizons[str(horizon)] = statistics(
            gram[:horizon, :horizon],
            measured_ids[:horizon],
            seed=20_260_824 + horizon,
            draws=args.null_draws,
        )

    windows: list[dict[str, Any]] = []
    optimizer_step_offset = 0 if args.reset_moments_after_warmup else args.warmup_steps
    for begin in range(0, args.measurement_steps, args.window_steps):
        end = begin + args.window_steps
        result = statistics(
            gram[begin:end, begin:end],
            measured_ids[begin:end],
            seed=20_270_000 + begin,
            draws=args.null_draws,
        )
        windows.append({
            "measurement_steps": [begin + 1, end],
            "optimizer_steps": [optimizer_step_offset + begin + 1, optimizer_step_offset + end],
            "coherence_amplification": result["coherence_amplification"],
            "null_upper_95": result["sign_flip_null"]["upper_95"],
            "one_sided_p": result["sign_flip_null"]["one_sided_p"],
            "above_sign_flip_95": result["above_sign_flip_95"],
            "resultant_l2": result["resultant_l2"],
            "diffusive_scale_l2": result["diffusive_scale_l2"],
        })

    full = statistics(
        gram,
        measured_ids,
        seed=20_280_824,
        draws=args.null_draws,
    )
    late_windows = windows[len(windows) // 2:]
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    eigenvalues = np.sort(eigenvalues)[::-1]
    spectrum_energy = float(eigenvalues.sum())
    effective_rank = float(
        spectrum_energy * spectrum_energy
        / max(float(np.square(eigenvalues).sum()), 1e-30)
    )
    top_spectrum = [
        float(value / max(spectrum_energy, 1e-30))
        for value in eigenvalues[:16]
    ]
    gram_output = args.gram_output or args.output.with_suffix(".gram.npz")
    gram_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(gram_output, direct_update_gram=gram)
    gram_sha256 = hashlib.sha256(gram_output.read_bytes()).hexdigest()
    alphas = np.asarray([row["reference_relative_alpha"] for row in rows], dtype=np.float64)
    payload = {
        "schema": "kernel-analyzer-phi-warm-state-long-direct-persistence-v1",
        "status": "COMPLETE",
        "case_id": "phi4_seq64_lmhead_dx",
        "protocol": {
            "input_bank": str(args.input_bank.resolve()),
            "state_offset": args.state_offset,
            "unique_natural_states": run_states,
            "warmup_steps": args.warmup_steps,
            "measurement_steps": args.measurement_steps,
            "window_steps": args.window_steps,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": args.learning_rate,
                "betas": [0.9, 0.95],
                "epsilon": 1e-8,
                "initial_moments": (
                    "WARMED_ON_NATURAL_IMPLEMENTATION"
                    if not args.reset_moments_after_warmup
                    else "RESET_TO_ZERO_AFTER_PARAMETER_WARMUP"
                ),
                "moments_reset_after_warmup": args.reset_moments_after_warmup,
            },
            "carrier": "model.norm.weight",
            "other_parameters_frozen": True,
            "contrast": "DETERMINISTIC_BF16_MINUS_FP32_CAST_REPAIR",
            "state_path": "NATURAL_IMPLEMENTATION_ADVANCES_COMMON_MASTER_AND_MOMENTS",
        },
        "warmup": {
            "first_loss": warmup_losses[0] if warmup_losses else None,
            "last_loss": warmup_losses[-1] if warmup_losses else None,
            "mean_loss": float(np.mean(warmup_losses)) if warmup_losses else None,
        },
        "horizons": horizons,
        "windows": windows,
        "full": full,
        "measurement_geometry": {
            "kind": "FULL_VECTOR_GRAM",
            "coordinate_space": "model.norm.weight direct effective-update difference",
            "gram_artifact": str(gram_output.resolve().relative_to(ROOT.resolve())),
            "gram_sha256": gram_sha256,
            "gram_shape": [args.measurement_steps, args.measurement_steps],
            "gram_dtype": "float64",
            "effective_rank_participation_ratio": effective_rank,
            "top_16_spectrum_energy_fractions": top_spectrum,
            "fixed_rank_one_projection_used_as_gate": False,
        },
        "reference_relative": {
            "definition": "dot(candidate_minus_repair_update, repair_update) / ||repair_update||^2",
            "mean_alpha": float(alphas.mean()),
            "median_alpha": float(np.median(alphas)),
            "positive_fraction": float(np.mean(alphas > 0.0)),
            "negative_fraction": float(np.mean(alphas < 0.0)),
            "minimum_alpha": float(alphas.min()),
            "maximum_alpha": float(alphas.max()),
        },
        "late_half_windows_above_null": sum(
            bool(row["above_sign_flip_95"]) for row in late_windows
        ),
        "late_half_window_count": len(late_windows),
        "rows": rows,
        "claim_boundary": (
            "Warm-state same-state direct effective-update experiment on one declared "
            "parameter carrier. It tests whether the 32-step signal survives beyond AdamW "
            "startup across fresh natural states. It is not full-parameter training, a "
            "closed-loop candidate-versus-repair trajectory, or training-to-convergence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "event": "PHI_LONG_COMPLETE",
        "output": str(args.output),
        "A": full["coherence_amplification"],
        "null95": full["sign_flip_null"]["upper_95"],
        "late_windows_above_null": payload["late_half_windows_above_null"],
        "late_window_count": payload["late_half_window_count"],
    }), flush=True)


if __name__ == "__main__":
    main()
