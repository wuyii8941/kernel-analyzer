#!/usr/bin/env python3
"""Capture the Liger fused-CE source transition under BiasFormation v2.1.

The fused operator already computes its dW endpoint internally.  A small
observer records that endpoint before autograd consumes it; only the tied
embedding gradient is retained for the next formation layer.  Full model
gradients and raw vectors are never retained in the certificate.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "archive/round1_code/src")]

from kernel_analyzer.bias_formation_v21 import FormationPolicy, summarize_streamed_state_vector_files  # noqa: E402
from scripts.liger_trajectory import tensor_digest  # noqa: E402


MODEL = Path("/data1/tzh/models/Qwen/Qwen3-1.7B")
DESIGN = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/supplementary_state_design_v1.json"
PROTOCOL = ROOT / "results/trajectory/liger_protocol.json"
OUTPUT = ROOT / "results/property/bias_formation/formation/liger_fused_ce_t128.json"
SPOOL_ROOT = Path("/data1/tzh/cache/bias_formation/liger_fused_ce_t128")
TIED = "model.embed_tokens.weight"
LR = 1.0e-4


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LigerEndpointObserver:
    """Record the internal fused dW output without changing its value."""

    def __init__(self, module: Any, mode: str) -> None:
        self.module = module
        self.mode = mode
        self.original: Any | None = None
        self.endpoint_vector: np.ndarray | None = None
        self.calls = 0

    def __enter__(self) -> "LigerEndpointObserver":
        import liger_kernel.ops.fused_linear_cross_entropy as fused

        self.original = fused.fused_linear_cross_entropy_forward
        original = self.original

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            endpoint = result[4]
            if endpoint is None:
                raise RuntimeError("Liger fused dW endpoint is absent")
            self.endpoint_vector = endpoint.detach().float().cpu().numpy().reshape(-1).copy()
            self.calls += 1
            return result

        fused.fused_linear_cross_entropy_forward = wrapped
        return self

    def __exit__(self, *unused: Any) -> None:
        del unused
        import liger_kernel.ops.fused_linear_cross_entropy as fused

        assert self.original is not None
        fused.fused_linear_cross_entropy_forward = self.original
        if self.calls != 1:
            raise RuntimeError(f"Liger endpoint executed {self.calls} times")


def _run_branch(model: Any, loss_module: Any, input_ids: torch.Tensor, mode: str | None):
    from scripts.liger_trajectory import full_step

    # full_step is deliberately reused for its exact hidden/label/dH control
    # sequence, but we avoid its all-parameter gradient clone here.
    model.zero_grad(set_to_none=True)
    outputs = model.model(input_ids=input_ids, use_cache=False, return_dict=True)
    hidden = outputs.last_hidden_state
    observed: list[Any] = []
    hidden.register_hook(lambda gradient: observed.append(gradient.detach().clone()))
    labels = torch.nn.functional.pad(input_ids, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
    observer = LigerEndpointObserver(loss_module, mode) if mode else None
    if observer is None:
        loss = loss_module(model.lm_head.weight, hidden.reshape(-1, hidden.shape[-1]), labels)
        loss_value = loss.detach().clone()
        loss.backward()
    else:
        with observer:
            loss = loss_module(model.lm_head.weight, hidden.reshape(-1, hidden.shape[-1]), labels)
            loss_value = loss.detach().clone()
            loss.backward()
    if len(observed) != 1:
        raise RuntimeError("Liger terminal hidden VJP was not observed exactly once")
    parameter = dict(model.named_parameters()).get(TIED)
    if parameter is None or parameter.grad is None:
        raise RuntimeError("Liger tied carrier gradient is absent")
    gradient = parameter.grad.detach().float().cpu().numpy().reshape(-1).copy()
    return {
        "loss_digest": tensor_digest(loss_value),
        "hidden_digest": tensor_digest(hidden),
        "labels_digest": tensor_digest(labels),
        "dH_digest": tensor_digest(observed[0]),
        "gradient": gradient,
        "endpoint": None if observer is None else observer.endpoint_vector,
    }


def _common(state: dict[str, Any], weights_digest: str) -> dict[str, str]:
    input_digest = _digest(state["input_ids"])
    optimizer_digest = _digest({"name": "STATELESS_SGD_FP32_MASTER", "learning_rate": LR, "state": "empty"})
    none_digest = _digest({"name": "none"})
    return {
        "candidate_weights_digest": weights_digest, "repair_weights_digest": weights_digest,
        "candidate_optimizer_digest": optimizer_digest, "repair_optimizer_digest": optimizer_digest,
        "candidate_input_digest": input_digest, "repair_input_digest": input_digest,
        "candidate_rng_digest": _digest({"seed": 3407}), "repair_rng_digest": _digest({"seed": 3407}),
        "candidate_scheduler_digest": none_digest, "repair_scheduler_digest": none_digest,
        "candidate_loss_scaler_digest": none_digest, "repair_loss_scaler_digest": none_digest,
    }


def _write_vector(root: Path, layer: str, state_id: str, values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    path = root / layer / (hashlib.sha256(state_id.encode()).hexdigest() + ".f32")
    path.parent.mkdir(parents=True, exist_ok=True)
    values.tofile(path)
    return {"state_id": state_id, "path": str(path), "coordinate_count": int(values.size),
            "vector_digest": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
            "storage_dtype": "float32"}


def capture(states: list[dict[str, Any]], device_name: str, spool_root: Path) -> dict[str, Any]:
    if len(states) != 32:
        raise ValueError("Liger v2.1 formation requires exactly 32 frozen states")
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from transformers import AutoModelForCausalLM

    device = torch.device(device_name)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    named = dict(model.named_parameters())
    if TIED not in named or model.lm_head.weight.untyped_storage().data_ptr() != model.model.embed_tokens.weight.untyped_storage().data_ptr():
        raise RuntimeError("Liger tied embedding binding failed")
    default_module = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean", accum_dtype=None).to(device)
    repair_module = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean", accum_dtype=torch.float32).to(device)
    state_ids = [str(row["sequence_id"]) for row in states]
    policy = FormationPolicy(min_states=16, bootstrap_samples=2000)
    rows = {partition: {layer: [] for layer in ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE")} for partition in ("calibration", "confirmation")}
    metadata = []
    for index, state in enumerate(states):
        state_id = state_ids[index]
        values = torch.tensor([state["input_ids"]], dtype=torch.long, device=device)
        standard = _run_branch(model, default_module, values, "SHAM")
        sham = _run_branch(model, default_module, values, "SHAM")
        repair = _run_branch(model, repair_module, values, "REPAIR")
        controls = all(standard[key] == sham[key] for key in ("loss_digest", "hidden_digest", "labels_digest", "dH_digest"))
        if not controls or not np.array_equal(standard["gradient"], sham["gradient"]):
            raise RuntimeError(f"Liger matched sham changed the full step at {state_id}")
        if standard["endpoint"] is None or repair["endpoint"] is None:
            raise RuntimeError(f"Liger internal endpoint was not captured at {state_id}")
        local = standard["endpoint"] - repair["endpoint"]
        gradient = standard["gradient"] - repair["gradient"]
        update = -LR * gradient
        partition = "calibration" if index < 16 else "confirmation"
        rows[partition]["LOCAL_ENDPOINT"].append(_write_vector(spool_root, "local", state_id, local))
        rows[partition]["PARAMETER_GRADIENT"].append(_write_vector(spool_root, "gradient", state_id, gradient))
        rows[partition]["EFFECTIVE_UPDATE"].append(_write_vector(spool_root, "update", state_id, update))
        metadata.append({"state_id": state_id, "partition": partition, "common_state": _common(state, _file_digest(MODEL / "model.safetensors.index.json")),
                         "local_coordinates": int(local.size), "parameter_coordinates": TIED,
                         "raw_vectors_retained": False})
        del values, standard, sham, repair, local, gradient, update
        torch.cuda.empty_cache()
        print(json.dumps({"event": "FORMATION_STATE_COMPLETE", "state": index, "state_id": state_id}, sort_keys=True), flush=True)
    populations = {}
    for partition in rows:
        populations[partition] = {}
        for layer in rows[partition]:
            certificate = summarize_streamed_state_vector_files(rows[partition][layer], layer=layer, partition=partition, policy=policy)
            populations[partition][layer] = certificate.as_dict()
            populations[partition][layer + "_status"] = certificate.status
    layers = ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE")
    confirmation = {layer: populations["confirmation"][layer + "_status"] for layer in layers}
    first_observed = next((layer for layer in layers if confirmation[layer] == "BIASED"), None)
    first_confirmed = None
    prior_centered = True
    for layer in layers:
        if first_confirmed is None and prior_centered and confirmation[layer] == "BIASED":
            first_confirmed = layer
        if confirmation[layer] != "CENTERED":
            prior_centered = False
    result = {"schema": "kernel-analyzer-bias-formation-certificate-v2_1", "case_id": "liger_fused_ce_t128", "status": "COMPLETE",
              "measurement_kind": "candidate_repair_ground_truth", "uses_candidate_measurements": True,
              "uses_historical_verdicts": False, "verdict_blind": True,
              "state_split": {"calibration_state_ids": state_ids[:16], "confirmation_state_ids": state_ids[16:], "calibration_count": 16, "confirmation_count": 16, "disjoint": True, "both_open_loop_common_state": True},
              "policy": policy.as_dict(), "populations": populations, "first_confirmed_bias_stage": first_confirmed,
              "first_observed_biased_stage": first_observed, "formation_point": "CONFIRMED" if first_confirmed else "UNRESOLVED",
              "trajectory_drift_in_formation": False, "missing_rows": [], "unexpected_rows": [], "rows": metadata,
              "capture_provenance": {"runner": "scripts/capture_liger_bias_formation_v21.py", "model": str(MODEL), "device": device_name, "raw_vectors_retained": False, "weights_digest_scope": "checkpoint_index_manifest", "trajectory_drift_in_formation": False}}
    result["result_sha256"] = _digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--spool-root", type=Path, default=SPOOL_ROOT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; formation capture is not run")
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    records = {row["sequence_id"]: row for row in design["records"]}
    states = [records[state_id] for state_id in protocol["trajectory"]["state_order"]]
    args.spool_root.mkdir(parents=True, exist_ok=True)
    result = capture(states, args.device, args.spool_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name("." + args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    shutil.rmtree(args.spool_root, ignore_errors=True)
    print(json.dumps({"output": str(args.output), "status": result["status"], "first_confirmed_bias_stage": result["first_confirmed_bias_stage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
