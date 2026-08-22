#!/usr/bin/env python3
"""Capture Qwen saved-P formation vectors without retaining large gradients.

The exact saved-P forward/backward wrapper is reused from the frozen trajectory
proof.  Formation is open-loop: all 32 states use the same model and empty
stateless-SGD state, and no arm updates its weights.  Large local/gradient
vectors are temporary float32 files under ``/data1/tzh/cache``; the committed
certificate contains only digests, a 16x16 Gram, and population statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "archive/round1_code/src")]

from kernel_analyzer.bias_formation_v21 import FormationPolicy, summarize_streamed_state_vector_files  # noqa: E402
from kernel_analyzer.seup import adamw_effective_update_delta  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_qwen128_softmax_saved_p_trajectory import CARRIERS, SavedProbabilityRepair  # noqa: E402
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402


MODEL = Path("/data1/tzh/models/Qwen/Qwen3-1.7B")
BANK = ROOT / "results/coverage/qwen_seq128_input_bank.json"
RELEASE = ROOT / "results/coverage/runtime_releases/qwen_seq128_r1"
OUTPUT = ROOT / "results/property/bias_formation/formation/qwen_saved_p_seq128.json"
SPOOL_ROOT = Path("/data1/tzh/cache/bias_formation/qwen_saved_p_seq128")
LR = 1.0e-5


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _common_state(state: dict[str, Any], seed: int, weights_digest: str) -> dict[str, str]:
    input_digest = _digest(state.get("input_ids", state.get("token_ids")))
    rng_digest = _digest({"seed": seed, "cuda_seed": seed})
    optimizer_digest = _digest({"name": "STATELESS_SGD_FP32_MASTER", "learning_rate": LR, "state": "empty"})
    scheduler_digest = _digest({"name": "none"})
    scaler_digest = _digest({"name": "none"})
    return {
        "candidate_weights_digest": weights_digest,
        "repair_weights_digest": weights_digest,
        "candidate_optimizer_digest": optimizer_digest,
        "repair_optimizer_digest": optimizer_digest,
        "candidate_input_digest": input_digest,
        "repair_input_digest": input_digest,
        "candidate_rng_digest": rng_digest,
        "repair_rng_digest": rng_digest,
        "candidate_scheduler_digest": scheduler_digest,
        "repair_scheduler_digest": scheduler_digest,
        "candidate_loss_scaler_digest": scaler_digest,
        "repair_loss_scaler_digest": scaler_digest,
    }


def _run_branch(model: torch.nn.Module, candidate: Any, values: torch.Tensor,
                seed: int, modules: list[Any], mode: str | None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.zero_grad(set_to_none=True)
    observer = SavedProbabilityRepair(modules, mode) if mode else None
    if observer is None:
        loss = candidate(values)
        loss.backward()
    else:
        with observer:
            loss = candidate(values)
            loss.backward()
    torch.cuda.synchronize(values.device)
    parameters = dict(model.named_parameters())
    gradients = {}
    for name in CARRIERS:
        parameter = parameters.get(name)
        if parameter is None or parameter.grad is None:
            raise RuntimeError(f"missing declared Qwen carrier gradient: {name}")
        gradients[name] = parameter.grad.detach().float().cpu().numpy().reshape(-1).copy()
    local = None if observer is None else observer.correction_vector
    return tensor_digest(loss), gradients, local, observer


def _write_vector(root: Path, layer: str, state_id: str, values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    path = root / layer / (hashlib.sha256(state_id.encode()).hexdigest() + ".f32")
    path.parent.mkdir(parents=True, exist_ok=True)
    values.tofile(path)
    return {
        "state_id": state_id,
        "path": str(path),
        "coordinate_count": int(values.size),
        "vector_digest": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        "storage_dtype": "float32",
    }


def _population(rows: dict[str, dict[str, list[dict[str, Any]]]], layer: str,
                partition: str, policy: FormationPolicy) -> dict[str, Any]:
    certificate = summarize_streamed_state_vector_files(
        rows[partition][layer], layer=layer, partition=partition, policy=policy,
        chunk_elements=1_048_576,
    )
    return certificate.as_dict()


def capture(states: list[dict[str, Any]], device_name: str, spool_root: Path, release: Path = RELEASE) -> dict[str, Any]:
    if len(states) != 32:
        raise ValueError("Qwen saved-P v2.1 formation requires exactly 32 frozen states")
    device = torch.device(device_name)
    configure_candidate_runtime(24000)
    model = load_model("qwen", MODEL, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm = torch.tensor([states[0]["input_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    capture_payload = json.loads((release / "capture.json").read_text(encoding="utf-8"))
    validate_release(wrapper_modules(modules), capture_payload)
    weights_digest = _file_digest(MODEL / "model.safetensors.index.json")
    state_ids = [str(row.get("sequence_id", row.get("state_id", i))) for i, row in enumerate(states)]
    if len(set(state_ids)) != 32:
        raise ValueError("Qwen state IDs are not unique")
    policy = FormationPolicy(min_states=16, bootstrap_samples=2000)
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        partition: {layer: [] for layer in ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE")}
        for partition in ("calibration", "confirmation")
    }
    response_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        partition: {layer: [] for layer in ("RESPONSE_EVEN", "RESPONSE_ODD")}
        for partition in ("calibration", "confirmation")
    }
    row_metadata: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        state_id = state_ids[index]
        seed = 24000 + index
        values = torch.tensor([state["input_ids"]], dtype=torch.long, device=device)
        standard_loss, standard_grad, _, _ = _run_branch(model, candidate, values, seed, modules, None)
        sham_loss, sham_grad, _, sham_observer = _run_branch(model, candidate, values, seed, modules, "SHAM")
        repair_loss, repair_grad, local_vector, repair_observer = _run_branch(model, candidate, values, seed, modules, "REPAIR_SAVED_P")
        if standard_loss != sham_loss or any(not np.array_equal(standard_grad[name], sham_grad[name]) for name in CARRIERS):
            raise RuntimeError(f"saved-P matched sham changed the full step at {state_id}")
        if sham_observer is None or sham_observer.changed_coordinates != 0:
            raise RuntimeError(f"saved-P sham changed the endpoint at {state_id}")
        if repair_observer is None or local_vector is None or repair_observer.changed_coordinates == 0:
            raise RuntimeError(f"saved-P repair did not change the exact endpoint at {state_id}")
        gradient_delta = np.concatenate([standard_grad[name] - repair_grad[name] for name in CARRIERS])
        update_delta = -LR * gradient_delta
        # Measure the optimizer response factor from the same common-state
        # gradients without pretending that a BF16 source reflection was
        # physically re-executed.  The reflected gradient is an offline
        # response probe and is explicitly not a source intervention.
        standard_tree = {name: torch.from_numpy(standard_grad[name]) for name in CARRIERS}
        repair_tree = {name: torch.from_numpy(repair_grad[name]) for name in CARRIERS}
        anti_tree = {name: 2.0 * repair_tree[name] - standard_tree[name] for name in CARRIERS}
        zero_first = {name: torch.zeros_like(repair_tree[name]) for name in CARRIERS}
        zero_second = {name: torch.zeros_like(repair_tree[name]) for name in CARRIERS}
        parameter_tree = {
            name: dict(model.named_parameters())[name].detach().float().cpu().reshape(-1)
            for name in CARRIERS
        }
        plus_tree = adamw_effective_update_delta(
            standard_tree, repair_tree, zero_first, zero_second, parameter_tree,
            step=1, learning_rate=LR, betas=(0.9, 0.95), epsilon=1e-8, weight_decay=0.0,
        )
        minus_tree = adamw_effective_update_delta(
            anti_tree, repair_tree, zero_first, zero_second, parameter_tree,
            step=1, learning_rate=LR, betas=(0.9, 0.95), epsilon=1e-8, weight_decay=0.0,
        )
        response_even = np.concatenate([
            (0.5 * (plus_tree[name] + minus_tree[name])).numpy().reshape(-1)
            for name in CARRIERS
        ])
        response_odd = np.concatenate([
            (0.5 * (plus_tree[name] - minus_tree[name])).numpy().reshape(-1)
            for name in CARRIERS
        ])
        partition = "calibration" if index < 16 else "confirmation"
        rows[partition]["LOCAL_ENDPOINT"].append(_write_vector(spool_root, "local", state_id, local_vector))
        rows[partition]["PARAMETER_GRADIENT"].append(_write_vector(spool_root, "gradient", state_id, gradient_delta))
        rows[partition]["EFFECTIVE_UPDATE"].append(_write_vector(spool_root, "update", state_id, update_delta))
        response_rows[partition]["RESPONSE_EVEN"].append(
            _write_vector(spool_root, "response_even", state_id, response_even)
        )
        response_rows[partition]["RESPONSE_ODD"].append(
            _write_vector(spool_root, "response_odd", state_id, response_odd)
        )
        row_metadata.append({
            "state_id": state_id,
            "partition": partition,
            "common_state": _common_state(state, seed, weights_digest),
            "local_coordinates": int(local_vector.size),
            "parameter_coordinates": [*CARRIERS],
            "standard_loss_digest": standard_loss,
            "repair_loss_digest": repair_loss,
            "raw_vectors_retained": False,
        })
        del values, standard_grad, sham_grad, repair_grad, gradient_delta, update_delta
        del standard_tree, repair_tree, anti_tree, zero_first, zero_second, parameter_tree
        del plus_tree, minus_tree, response_even, response_odd
        torch.cuda.empty_cache()
        print(json.dumps({"event": "FORMATION_STATE_COMPLETE", "state": index, "state_id": state_id}, sort_keys=True), flush=True)
    populations = {}
    for partition in rows:
        populations[partition] = {}
        for layer in rows[partition]:
            certificate = _population(rows, layer, partition, policy)
            populations[partition][layer] = certificate
            populations[partition][layer + "_status"] = certificate["status"]
    response_populations = {}
    for partition in response_rows:
        response_populations[partition] = {}
        for layer in response_rows[partition]:
            certificate = _population(response_rows, layer, partition, policy)
            response_populations[partition][layer] = certificate
            response_populations[partition][layer + "_status"] = certificate["status"]
    layer_names = ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE")
    statuses = {partition: {layer: populations[partition][layer]["status"] for layer in layer_names}
                for partition in populations}
    confirmation = statuses["confirmation"]
    first_observed = next((layer for layer in ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE")
                           if confirmation[layer] == "BIASED"), None)
    first_confirmed = None
    prior_centered = True
    for layer in ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE"):
        if first_confirmed is None and prior_centered and confirmation[layer] == "BIASED":
            first_confirmed = layer
        if confirmation[layer] != "CENTERED":
            prior_centered = False
    result = {
        "schema": "kernel-analyzer-bias-formation-certificate-v2_1",
        "case_id": "qwen_saved_p_seq128",
        "status": "COMPLETE",
        "measurement_kind": "candidate_repair_ground_truth",
        "uses_candidate_measurements": True,
        "uses_historical_verdicts": False,
        "verdict_blind": True,
        "state_split": {
            "calibration_state_ids": state_ids[:16], "confirmation_state_ids": state_ids[16:],
            "calibration_count": 16, "confirmation_count": 16,
            "disjoint": True, "both_open_loop_common_state": True,
        },
        "policy": policy.as_dict(),
        "populations": populations,
        "response_populations": response_populations,
        "response_measurement_boundary": "Offline AdamW zero-moment response to a reflected gradient residual; source +/- endpoint representability and source causality are not claimed.",
        "first_confirmed_bias_stage": first_confirmed,
        "first_observed_biased_stage": first_observed,
        "formation_point": "CONFIRMED" if first_confirmed else "UNRESOLVED",
        "trajectory_drift_in_formation": False,
        "missing_rows": [], "unexpected_rows": [],
        "rows": row_metadata,
        "capture_provenance": {
            "runtime_environment": str(release / "environment.json"),
            "release_capture": str(release / "capture.json"),
            "input_bank": "results/coverage/qwen_seq128_input_bank.json",
            "runner": "scripts/capture_qwen_saved_p_bias_formation_v21.py",
            "device": device_name, "raw_vectors_retained": False,
            "weights_digest_scope": "checkpoint_index_manifest",
            "trajectory_drift_in_formation": False,
        },
    }
    result["result_sha256"] = _digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--spool-root", type=Path, default=SPOOL_ROOT)
    parser.add_argument("--release", type=Path, default=RELEASE)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; formation capture is not run")
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    states = bank.get("states", bank.get("records"))
    args.spool_root.mkdir(parents=True, exist_ok=True)
    result = capture(states, args.device, args.spool_root, args.release)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name("." + args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    shutil.rmtree(args.spool_root, ignore_errors=True)
    print(json.dumps({"output": str(args.output), "status": result["status"], "first_confirmed_bias_stage": result["first_confirmed_bias_stage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
