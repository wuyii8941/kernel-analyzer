#!/usr/bin/env python3
"""Complete-Gram F+B population test for the new vision LayerNorm candidate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch._inductor.codecache import PyCodeCache
from transformers import AutoProcessor

from qwen_candidate_step import configure_candidate_runtime
from scripts.generated_fp32_observer import GeneratedFP32Observer
from scripts.run_generated_fp32_screen import Gemma3ImageLossStep, load_model, prepare_values
from scripts.runtime_schedule_binding import bind_runtime_schedule


def dot(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name in left:
        a, b = left[name].reshape(-1), right[name].reshape(-1)
        for start in range(0, a.numel(), 1_048_576):
            total += float(torch.dot(a[start:start + 1_048_576], b[start:start + 1_048_576]))
    return total


def sign_flip_p_value(gram: np.ndarray) -> float:
    observed = float(gram.sum() - np.trace(gram))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(gram)):
        vector = np.asarray(signs)
        values.append(float(vector @ gram @ vector - np.trace(gram)))
    return sum(value >= observed - 1e-12 for value in values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    bank = json.loads(args.input_bank.read_text())
    states = bank["states"]
    evaluation = [row for row in states if row["role"] == "SCREENING"]
    if len(evaluation) != 8:
        raise RuntimeError("population protocol requires eight screening states")
    probe = json.loads(args.probe.read_text())
    calibration = next(
        row for row in probe["targets"]
        if row["case_id"] == "gemma_vision_layernorm_backward"
    )
    names = calibration["changed_parameter_gradients"]
    device = torch.device(args.device)
    configure_candidate_runtime(26000)
    model = load_model("gemma3", args.model, device)
    model.gradient_checkpointing_enable()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    candidate = torch.compile(Gemma3ImageLossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm, warm_digests = prepare_values(
        states[2], modality="IMAGE_TEXT", model_path=args.model, device=device, processor=processor,
    )
    model.zero_grad(set_to_none=True); candidate(*warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules)
    inventory = args.work_dir.with_name(f"{args.work_dir.name}_inventory.json.gz")
    campaign_path = args.work_dir.with_name(f"{args.work_dir.name}_campaign.json.gz")
    bind_runtime_schedule(
        modules=modules, work_dir=args.work_dir,
        manifest=args.work_dir.with_name(f"{args.work_dir.name}_manifest.json"),
        inventory=inventory, campaign=campaign_path, architecture="gemma3",
        state=states[2], input_digests=warm_digests, values=warm, modality="IMAGE_TEXT",
        gradient_checkpointing=True, allow_graph_breaks=True,
    )
    with gzip.open(campaign_path, "rt", encoding="utf-8") as handle:
        campaign = json.load(handle)
    choices = sorted(
        (row for row in campaign["rows"] if row["phase"] == "BACKWARD"
         and "add_native_layer_norm_native_layer_norm_backward_view" in row["symbol"]
         and "in_out_ptr0" in row["output_names"]),
        key=lambda row: (row["source_path"], row["source_line"], row["region_id"]),
    )
    if not choices:
        raise RuntimeError("frozen LayerNorm backward target is absent")
    target = choices[0]
    parameters = dict(model.named_parameters())
    missing = set(names) - set(parameters)
    if missing:
        raise RuntimeError(f"calibration parameter coordinates changed: {sorted(missing)}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    baseline_files, state_rows = [], []
    for index, state in enumerate(evaluation):
        values, digests = prepare_values(
            state, modality="IMAGE_TEXT", model_path=args.model, device=device, processor=processor,
        )
        seed = 26000 + index
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True); loss = candidate(*values); loss.backward(); torch.cuda.synchronize(device)
        snapshot = {name: parameters[name].grad.detach().cpu().clone() for name in names}
        path = args.work_dir / f"baseline_{index}.pt"
        torch.save(snapshot, path); baseline_files.append(path)
        state_rows.append({"state_id": state["state_id"], "input_digests": digests, "baseline_loss": float(loss)})
    gram = np.zeros((len(evaluation), len(evaluation)), dtype=np.float64)
    delta_files = []
    observer = GeneratedFP32Observer(
        modules=modules, campaign_rows=[target],
        repair_targets={target["region_id"]: ["in_out_ptr0"]}, allow_unlisted_calls=True,
    )
    with observer:
        for index, state in enumerate(evaluation):
            values, _ = prepare_values(
                state, modality="IMAGE_TEXT", model_path=args.model, device=device, processor=processor,
            )
            seed = 26000 + index
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model.zero_grad(set_to_none=True); loss = candidate(*values); loss.backward(); torch.cuda.synchronize(device)
            baseline = torch.load(baseline_files[index], weights_only=True)
            delta = {
                name: parameters[name].grad.detach().cpu().float() - baseline[name].float()
                for name in names
            }
            state_rows[index]["repair_loss"] = float(loss)
            for previous, path in enumerate(delta_files):
                other = torch.load(path, weights_only=True)
                gram[index, previous] = gram[previous, index] = dot(delta, other)
            gram[index, index] = dot(delta, delta)
            path = args.work_dir / f"delta_{index}.pt"
            torch.save(delta, path); delta_files.append(path)
            baseline_files[index].unlink()
    diagonal = float(np.trace(gram))
    off_diagonal = float(gram.sum() - diagonal)
    amplification = float(np.sqrt(max(float(gram.sum()), 0.0) / diagonal)) if diagonal else 0.0
    payload = {
        "schema": "kernel-analyzer-gemma-vision-layernorm-population-v1",
        "status": "COMPLETE_FULL_PARAMETER_COORDINATE_GRAM",
        "case_id": "gemma_vision_layernorm_backward",
        "target": {key: target[key] for key in ("region_id", "phase", "symbol", "source_path", "source_line")},
        "endpoint": "in_out_ptr0", "calibration_probe_sha256": probe["result_sha256"],
        "declared_parameter_coordinates": names,
        "coordinate_count": sum(parameters[name].numel() for name in names),
        "states": state_rows, "complete_gram": gram.tolist(),
        "diagonal_energy": diagonal, "cross_state_directional_energy": off_diagonal,
        "amplification": amplification, "exact_sign_flip_p_value": sign_flip_p_value(gram),
        "claim_boundary": "Open-loop complete Gram on calibration-frozen parameter reach; persistence is separate.",
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for path in delta_files:
        path.unlink()
    print(json.dumps({key: payload[key] for key in ("coordinate_count", "amplification", "exact_sign_flip_p_value")}, sort_keys=True))


if __name__ == "__main__":
    main()
