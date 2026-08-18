#!/usr/bin/env python3
"""Screen a real Mamba training F+B for sequential-vs-parallel scan bias.

The reference is the Transformers sequential recurrence.  The candidate is
the supported mambapy Blelloch parallel scan.  Both execute the same pretrained
model, natural token windows, loss, and actual autograd backward.  States 0--1
freeze one direction per scan parameter; later states are held out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, MambaForCausalLM


ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--output", type=Path, default=ROOT / "results/mamba_scan/screen.json")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--calibration-states", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def natural_windows(
    tokenizer, arrow: Path, seq_len: int, count: int, stride_multiplier: int = 17
) -> list[torch.Tensor]:
    dataset = Dataset.from_file(str(arrow))
    text = "\n".join(row["text"] for row in dataset if row["text"].strip())
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if stride_multiplier < 1:
        raise ValueError("stride_multiplier must be positive")
    stride = seq_len * stride_multiplier
    required = (count - 1) * stride + seq_len
    if tokens.numel() < required:
        raise RuntimeError(f"validation token bank too short: {tokens.numel()} < {required}")
    return [tokens[index * stride : index * stride + seq_len].clone() for index in range(count)]


def target_parameters(model: MambaForCausalLM) -> dict[str, torch.nn.Parameter]:
    suffixes = (
        ".mixer.A_log",
        ".mixer.D",
        ".mixer.x_proj.weight",
        ".mixer.dt_proj.weight",
        ".mixer.dt_proj.bias",
    )
    return {name: parameter for name, parameter in model.named_parameters() if name.endswith(suffixes)}


def run_arm(
    model: MambaForCausalLM,
    targets: dict[str, torch.nn.Parameter],
    input_ids: torch.Tensor,
    parallel: bool,
) -> tuple[float, dict[str, torch.Tensor]]:
    for layer in model.backbone.layers:
        layer.mixer.use_mambapy = parallel
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    loss.backward()
    gradients = {}
    for name, parameter in targets.items():
        if parameter.grad is None:
            raise RuntimeError(f"missing actual backward gradient: {name}")
        gradients[name] = parameter.grad.detach().float().cpu().clone()
    return float(loss.detach().cpu()), gradients


def main() -> None:
    args = parse_args()
    if args.calibration_states < 1 or args.calibration_states >= args.states:
        raise ValueError("calibration-states must be in [1, states)")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    windows = natural_windows(tokenizer, args.validation_arrow, args.seq_len, args.states)
    model = MambaForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, local_files_only=True
    ).to(args.device).train()
    model.config.use_cache = False
    targets = target_parameters(model)
    if len(targets) != len(model.backbone.layers) * 5:
        raise RuntimeError(f"unexpected scan-parameter denominator: {len(targets)}")

    state_rows = []
    deltas: list[dict[str, torch.Tensor]] = []
    for state_id, cpu_ids in enumerate(windows):
        input_ids = cpu_ids.unsqueeze(0).to(args.device)
        reference_loss, reference = run_arm(model, targets, input_ids, parallel=False)
        candidate_loss, candidate = run_arm(model, targets, input_ids, parallel=True)
        delta = {name: candidate[name] - reference[name] for name in targets}
        deltas.append(delta)
        state_rows.append(
            {
                "state_id": state_id,
                "split": "CALIBRATION" if state_id < args.calibration_states else "HELDOUT",
                "token_sha256": hashlib.sha256(cpu_ids.numpy().tobytes()).hexdigest(),
                "reference_loss": reference_loss,
                "candidate_loss": candidate_loss,
                "loss_delta": candidate_loss - reference_loss,
                "all_finite": all(torch.isfinite(value).all().item() for value in delta.values()),
                "changed_parameters": sum(bool(torch.count_nonzero(value)) for value in delta.values()),
                "global_delta_l2": float(torch.sqrt(sum(torch.sum(value.double() ** 2) for value in delta.values()))),
            }
        )
        del reference, candidate, input_ids
        torch.cuda.empty_cache()

    directions = {}
    for name in targets:
        direction = sum(deltas[index][name] for index in range(args.calibration_states))
        norm = torch.linalg.vector_norm(direction.double())
        if norm > 0:
            directions[name] = direction.double() / norm

    parameter_rows = []
    for name, direction in directions.items():
        projections = [float(torch.sum(deltas[index][name].double() * direction)) for index in range(args.states)]
        heldout = projections[args.calibration_states :]
        parameter_rows.append(
            {
                "parameter": name,
                "numel": direction.numel(),
                "calibration_projections": projections[: args.calibration_states],
                "heldout_projections": heldout,
                "heldout_positive": sum(value > 0 for value in heldout),
                "heldout_negative": sum(value < 0 for value in heldout),
                "heldout_mean": sum(heldout) / len(heldout),
                "heldout_min": min(heldout),
                "heldout_max": max(heldout),
                "persistent_positive": all(value > 0 for value in heldout),
                "persistent_negative": all(value < 0 for value in heldout),
            }
        )
    parameter_rows.sort(key=lambda row: abs(row["heldout_mean"]), reverse=True)

    output = {
        "schema": "kernel-analyzer-mamba-scan-screen-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "model_class": "MambaForCausalLM",
        "dtype": "float32",
        "seq_len": args.seq_len,
        "states": args.states,
        "calibration_states": list(range(args.calibration_states)),
        "heldout_states": list(range(args.calibration_states, args.states)),
        "reference": "Transformers sequential selective-state recurrence",
        "candidate": "Transformers mambapy Blelloch parallel selective scan",
        "mathematical_scan": "h_t = a_t * h_{t-1} + b_t; y_t = C_t^T h_t + D * u_t",
        "actual_backward": "autograd backward of each executed scan program under the same full LM loss",
        "candidate_blind_direction_freeze": True,
        "target_parameter_denominator": len(targets),
        "state_rows": state_rows,
        "parameter_rows": parameter_rows,
        "persistent_positive_count": sum(row["persistent_positive"] for row in parameter_rows),
        "persistent_negative_count": sum(row["persistent_negative"] for row in parameter_rows),
        "tensor_values_saved": False,
        "claim_boundary": "SCREEN_ONLY_NO_BIAS_CASE_WITHOUT_INDEPENDENT_CONFIRMATION_AND_LOCAL_FB_INTERVENTION",
    }
    output["result_sha256"] = canonical_hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "persistent_positive": output["persistent_positive_count"],
        "persistent_negative": output["persistent_negative_count"],
        "top": parameter_rows[:5],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
