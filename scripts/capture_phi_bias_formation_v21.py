#!/usr/bin/env python3
"""Capture Phi MM local/gradient/update formation vectors under v2.1.

This reuses the already-bound Phi exact release and repair boundary.  It does
not update model weights, and it retains only compact v2.1 Gram certificates.
Raw endpoint vectors exist only in the process and are released after
finalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.bias_formation_v21 import BiasFormationTrace, FormationPolicy  # noqa: E402
from scripts.generated_nontriton_fp32_observer import fp32_external_reference  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import MMRepair  # noqa: E402
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402


LR = 1.0e-4
MODEL = Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct")
BANK = ROOT / "results/coverage/phi4_seq64_input_bank.json"
RELEASE = ROOT / "results/coverage/runtime_releases/phi4_seq64_r1"
OUTPUT = ROOT / "results/property/bias_formation/formation/phi4_lm_head_dx_seq64.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector_value(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float32).reshape(-1)
    return {
        "vector": values.tolist(),
        "coordinate_count": int(values.size),
        "vector_digest": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }


def _common_state(state: dict[str, Any], seed: int, model_digest: str) -> dict[str, str]:
    input_digest = _digest(state.get("token_ids", state.get("input_ids")))
    rng_digest = _digest({"seed": seed, "cuda_seed": seed})
    optimizer_digest = _digest({"name": "STATELESS_SGD_FP32_MASTER", "learning_rate": LR, "state": "empty"})
    scheduler_digest = _digest({"name": "none"})
    loss_scaler_digest = _digest({"name": "none"})
    return {
        "candidate_weights_digest": model_digest,
        "repair_weights_digest": model_digest,
        "candidate_optimizer_digest": optimizer_digest,
        "repair_optimizer_digest": optimizer_digest,
        "candidate_input_digest": input_digest,
        "repair_input_digest": input_digest,
        "candidate_rng_digest": rng_digest,
        "repair_rng_digest": rng_digest,
        "candidate_scheduler_digest": scheduler_digest,
        "repair_scheduler_digest": scheduler_digest,
        "candidate_loss_scaler_digest": loss_scaler_digest,
        "repair_loss_scaler_digest": loss_scaler_digest,
    }


def _run_branch(model: torch.nn.Module, candidate: Any, values: torch.Tensor,
                seed: int, modules: list[Any], mode: str | None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.zero_grad(set_to_none=True)
    observer = MMRepair(modules, mode) if mode else None
    if observer is None:
        loss = candidate(values)
        loss.backward()
    else:
        with observer:
            loss = candidate(values)
            loss.backward()
    torch.cuda.synchronize(values.device)
    norm_parameter = dict(model.named_parameters()).get("model.norm.weight")
    if norm_parameter is None or norm_parameter.grad is None:
        raise RuntimeError("declared Phi carrier model.norm.weight is absent from the gradient")
    norm_grad = norm_parameter.grad.detach().float().cpu().numpy().copy()
    local = None if observer is None else observer.local
    local_vector = None if observer is None else np.asarray(observer.local_vector, dtype=np.float32).copy()
    return tensor_digest(loss), norm_grad, local, local_vector


def capture(states: list[dict[str, Any]], device_name: str) -> dict[str, Any]:
    if len(states) != 32:
        raise ValueError("Phi v2.1 formation requires exactly 32 frozen states")
    device = torch.device(device_name)
    configure_candidate_runtime(24000)
    model = load_model("phi", MODEL, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    capture_payload = json.loads((RELEASE / "capture.json").read_text(encoding="utf-8"))
    validate_release(wrapper_modules(modules), capture_payload)

    # The model is never updated in formation capture.  This digest is the
    # frozen checkpoint manifest used by both counterfactual arms.
    model_digest = _file_digest(MODEL / "model.safetensors.index.json")
    calibration = [str(row.get("state_id", i)) for i, row in enumerate(states[:16])]
    confirmation = [str(row.get("state_id", i + 16)) for i, row in enumerate(states[16:], 16)]
    trace = BiasFormationTrace(
        "phi4_lm_head_dx_seq64", calibration, confirmation,
        FormationPolicy(min_states=16, bootstrap_samples=2000),
    )
    for index, state in enumerate(states):
        state_id = str(state.get("state_id", index))
        seed = 24000 + index
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        standard_loss, standard_grad, _, _ = _run_branch(model, candidate, values, seed, modules, None)
        _, sham_grad, sham_local, sham_vector = _run_branch(model, candidate, values, seed, modules, "SHAM")
        repair_loss, repair_grad, repair_local, repair_vector = _run_branch(
            model, candidate, values, seed, modules, "REPAIR_FP32_CAST_BF16"
        )
        if sham_local is None or sham_vector is None or int(sham_local["changed_coordinates"]) != 0:
            raise RuntimeError("Phi matched sham changed the target endpoint")
        if not np.array_equal(standard_grad, sham_grad):
            raise RuntimeError("Phi matched sham changed model.norm.weight gradient")
        if repair_local is None or repair_vector is None:
            raise RuntimeError("Phi repair did not observe the exact MM endpoint")
        gradient_delta = standard_grad - repair_grad
        update_delta = -LR * gradient_delta
        partition = "calibration" if index < 16 else "confirmation"
        trace.add(
            state_id,
            partition,
            common_state_certificate=_common_state(state, seed, model_digest),
            local_endpoint=_vector_value(repair_vector),
            parameter_gradient=_vector_value(gradient_delta),
            effective_update=_vector_value(update_delta),
            metadata={
                "endpoint": "phi4_seq64:lm_head.input_gradient.mm",
                "optimizer": "STATELESS_SGD_FP32_MASTER",
                "learning_rate": LR,
                "standard_loss_digest": standard_loss,
                "repair_loss_digest": repair_loss,
                "local_coordinates": int(repair_vector.size),
                "parameter_coordinates": "model.norm.weight",
                "raw_vectors_retained": False,
            },
        )
        del values, standard_grad, sham_grad, repair_grad, gradient_delta, update_delta
        torch.cuda.empty_cache()
        print(json.dumps({"event": "FORMATION_STATE_COMPLETE", "state": index, "state_id": state_id}, sort_keys=True), flush=True)
    result = trace.finalize()
    result["capture_provenance"] = {
        "runtime_environment": "results/coverage/runtime_releases/phi4_seq64_r1/environment.json",
        "release_capture": "results/coverage/runtime_releases/phi4_seq64_r1/capture.json",
        "input_bank": "results/coverage/phi4_seq64_input_bank.json",
        "runner": "scripts/capture_phi_bias_formation_v21.py",
        "device": device_name,
        "raw_vectors_retained": False,
        "weights_digest_scope": "checkpoint_index_manifest",
        "trajectory_drift_in_formation": False,
    }
    result["result_sha256"] = _digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; formation capture is not run")
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    states = bank.get("states", bank.get("records"))
    result = capture(states, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name("." + args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "status": result["status"], "first_confirmed_bias_stage": result["first_confirmed_bias_stage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
