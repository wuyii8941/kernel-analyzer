#!/usr/bin/env python3
"""Independently confirm and locally intervene on the Mamba scan screen hit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, MambaForCausalLM

from mamba_scan_screen import canonical_hash, natural_windows


ROOT = Path(__file__).resolve().parents[1]
TARGET = "backbone.layers.23.mixer.x_proj.weight"
TARGET_LAYER = 23


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/state-spaces/mamba-130m-hf"))
    parser.add_argument(
        "--validation-arrow",
        type=Path,
        default=Path(
            "/data1/tzh/cache/huggingface/datasets/Salesforce___wikitext/"
            "wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/"
            "wikitext-validation.arrow"
        ),
    )
    parser.add_argument("--screen", type=Path, default=ROOT / "results/mamba_scan/screen.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/mamba_scan/l23_xproj_confirmation.json")
    parser.add_argument("--direction", type=Path, default=ROOT / "results/mamba_scan/l23_xproj_direction.pt")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--confirmation-states", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_arm(model: MambaForCausalLM, arm: str) -> None:
    for index, layer in enumerate(model.backbone.layers):
        if arm == "REFERENCE":
            layer.mixer.use_mambapy = False
        elif arm == "ALL_PARALLEL":
            layer.mixer.use_mambapy = True
        elif arm == "LOCAL_L23_PARALLEL":
            layer.mixer.use_mambapy = index == TARGET_LAYER
        else:
            raise ValueError(arm)


def execute(
    model: MambaForCausalLM,
    parameter: torch.nn.Parameter,
    input_ids: torch.Tensor,
    arm: str,
) -> dict[str, object]:
    set_arm(model, arm)
    observed: dict[str, torch.Tensor] = {}
    mixer = model.backbone.layers[TARGET_LAYER].mixer

    def pre_hook(_module, args):
        value = args[0]
        value.retain_grad()
        observed["input"] = value

    def forward_hook(_module, _args, output):
        output.retain_grad()
        observed["output"] = output

    pre = mixer.register_forward_pre_hook(pre_hook)
    post = mixer.register_forward_hook(forward_hook)
    model.zero_grad(set_to_none=True)
    try:
        loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        loss.backward()
    finally:
        pre.remove()
        post.remove()
    if parameter.grad is None or observed["input"].grad is None or observed["output"].grad is None:
        raise RuntimeError("actual layer-23 F+B observation incomplete")
    return {
        "loss": float(loss.detach().cpu()),
        "parameter_gradient": parameter.grad.detach().float().cpu().clone(),
        "input": observed["input"].detach().float().cpu().clone(),
        "output": observed["output"].detach().float().cpu().clone(),
        "input_gradient": observed["input"].grad.detach().float().cpu().clone(),
        "output_gradient": observed["output"].grad.detach().float().cpu().clone(),
    }


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)))


def projection(left: torch.Tensor, right: torch.Tensor, direction: torch.Tensor) -> float:
    return float(torch.sum((left - right).double() * direction))


def exact(left: dict[str, object], right: dict[str, object]) -> bool:
    return left["loss"] == right["loss"] and all(
        torch.equal(left[name], right[name])
        for name in ("parameter_gradient", "input", "output", "input_gradient", "output_gradient")
    )


def main() -> None:
    args = parse_args()
    screen = json.loads(args.screen.read_text())
    selected = next(row for row in screen["parameter_rows"] if row["parameter"] == TARGET)
    if not selected["persistent_positive"]:
        raise RuntimeError("frozen discovery candidate no longer passes screen")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    total_states = 8 + args.confirmation_states
    windows = natural_windows(tokenizer, args.validation_arrow, args.seq_len, total_states)
    for state_id in range(8):
        digest = hashlib.sha256(windows[state_id].numpy().tobytes()).hexdigest()
        if digest != screen["state_rows"][state_id]["token_sha256"]:
            raise RuntimeError(f"screen token provenance mismatch at state {state_id}")

    model = MambaForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, local_files_only=True
    ).to(args.device).train()
    model.config.use_cache = False
    parameter = dict(model.named_parameters())[TARGET]

    calibration_deltas = []
    calibration_projections = []
    for state_id in screen["calibration_states"]:
        ids = windows[state_id].unsqueeze(0).to(args.device)
        reference = execute(model, parameter, ids, "REFERENCE")
        candidate = execute(model, parameter, ids, "ALL_PARALLEL")
        calibration_deltas.append(candidate["parameter_gradient"] - reference["parameter_gradient"])
    direction_raw = sum(calibration_deltas)
    direction_norm = torch.linalg.vector_norm(direction_raw.double())
    if direction_norm == 0:
        raise RuntimeError("frozen direction is zero")
    direction = direction_raw.double() / direction_norm
    calibration_projections = [float(torch.sum(delta.double() * direction)) for delta in calibration_deltas]
    expected = selected["calibration_projections"]
    if max(abs(left - right) for left, right in zip(calibration_projections, expected)) > 1e-15:
        raise RuntimeError("reconstructed frozen direction differs from screen")
    args.direction.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "parameter": TARGET,
        "calibration_states": screen["calibration_states"],
        "direction": direction.float(),
        "screen_result_sha256": screen["result_sha256"],
    }, args.direction)

    rows = []
    sham_exact = None
    for state_id in range(8, total_states):
        cpu_ids = windows[state_id]
        ids = cpu_ids.unsqueeze(0).to(args.device)
        reference = execute(model, parameter, ids, "REFERENCE")
        candidate = execute(model, parameter, ids, "ALL_PARALLEL")
        local = execute(model, parameter, ids, "LOCAL_L23_PARALLEL")
        if state_id == 8:
            sham = execute(model, parameter, ids, "ALL_PARALLEL")
            sham_exact = exact(candidate, sham)
        all_projection = projection(candidate["parameter_gradient"], reference["parameter_gradient"], direction)
        local_projection = projection(local["parameter_gradient"], reference["parameter_gradient"], direction)
        rows.append({
            "state_id": state_id,
            "token_sha256": hashlib.sha256(cpu_ids.numpy().tobytes()).hexdigest(),
            "reference_loss": reference["loss"],
            "all_parallel_loss_delta": candidate["loss"] - reference["loss"],
            "local_l23_loss_delta": local["loss"] - reference["loss"],
            "all_parallel_projection": all_projection,
            "local_l23_projection": local_projection,
            "all_parallel_parameter_gradient_rms": float(torch.mean((candidate["parameter_gradient"] - reference["parameter_gradient"]) ** 2).sqrt()),
            "local_l23_parameter_gradient_rms": float(torch.mean((local["parameter_gradient"] - reference["parameter_gradient"]) ** 2).sqrt()),
            "local_same_forward_input_bitwise": torch.equal(local["input"], reference["input"]),
            "local_forward_output_max_abs": max_abs(local["output"], reference["output"]),
            "local_same_output_cotangent_bitwise": torch.equal(local["output_gradient"], reference["output_gradient"]),
            "local_input_gradient_max_abs": max_abs(local["input_gradient"], reference["input_gradient"]),
            "all_finite": all(
                torch.isfinite(value).all().item()
                for arm in (reference, candidate, local)
                for value in arm.values()
                if isinstance(value, torch.Tensor)
            ),
        })
        del reference, candidate, local, ids
        torch.cuda.empty_cache()

    all_values = [row["all_parallel_projection"] for row in rows]
    local_values = [row["local_l23_projection"] for row in rows]
    # Exact one-sided sign-test probability under a symmetric null when all n
    # independent held-out states have the same positive sign.
    all_positive_sign_p = 2.0 ** (-len(all_values)) if all(value > 0 for value in all_values) else None
    local_positive_sign_p = 2.0 ** (-len(local_values)) if all(value > 0 for value in local_values) else None
    output = {
        "schema": "kernel-analyzer-mamba-l23-scan-confirmation-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "dtype": "float32",
        "seq_len": args.seq_len,
        "parameter": TARGET,
        "selected_from_family_size": screen["target_parameter_denominator"],
        "selection_rule": "largest absolute heldout mean among persistent screen directions",
        "calibration_states": screen["calibration_states"],
        "discovery_states": screen["heldout_states"],
        "independent_confirmation_states": list(range(8, total_states)),
        "direction_file": str(args.direction),
        "direction_sha256": hashlib.sha256(args.direction.read_bytes()).hexdigest(),
        "forward_math": {
            "state": "h_t = exp(A * softplus(dt_t)) * h_{t-1} + softplus(dt_t) * B_t * u_t",
            "output": "y_t = C_t^T h_t + D * u_t; mixer_out = out_proj(y_t * silu(gate_t))",
        },
        "actual_backward_math": {
            "state_cotangent": "bar_h_t += C_t * bar_y_t; bar_h_{t-1} += a_t * bar_h_t",
            "local_partials": "bar_a_t += bar_h_t * h_{t-1}; bar_b_t += bar_h_t",
            "binding": "same actual layer-23 mixer input and x_proj parameter; output cotangent equality is measured rather than assumed",
        },
        "arms": {
            "reference": "all 24 layers sequential recurrence",
            "candidate": "all 24 layers mambapy parallel scan",
            "local_intervention": "only layer 23 mambapy parallel scan; layers 0--22 sequential",
        },
        "rows": rows,
        "summary": {
            "all_parallel_positive": sum(value > 0 for value in all_values),
            "all_parallel_negative": sum(value < 0 for value in all_values),
            "all_parallel_mean": sum(all_values) / len(all_values),
            "all_parallel_min": min(all_values),
            "all_parallel_sign_test_p_if_all_positive": all_positive_sign_p,
            "local_l23_positive": sum(value > 0 for value in local_values),
            "local_l23_negative": sum(value < 0 for value in local_values),
            "local_l23_mean": sum(local_values) / len(local_values),
            "local_l23_min": min(local_values),
            "local_l23_sign_test_p_if_all_positive": local_positive_sign_p,
            "holm_threshold_for_original_family": 0.05 / screen["target_parameter_denominator"],
            "candidate_repeat_sham_bitwise_exact": sham_exact,
            "all_inputs_local_bitwise_exact": all(row["local_same_forward_input_bitwise"] for row in rows),
            "all_rows_finite": all(row["all_finite"] for row in rows),
        },
        "tensor_values_saved": False,
        "claim_boundary": "NO_COMPLETE_CASE_UNTIL_DIRECTION_AND_LOCAL_INTERVENTION_BOTH_PASS_AND_LIVE_WEIGHT_ACCUMULATION_IS_SHOWN",
    }
    output["result_sha256"] = canonical_hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "summary": output["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
