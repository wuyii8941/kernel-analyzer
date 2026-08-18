#!/usr/bin/env python3
"""Test a transport-pairing intervention for the Phi MM formation candidate."""

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
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "archive/round1_code/src")]

from kernel_analyzer.bias_formation_v21 import FormationPolicy, summarize_state_vectors  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import MMRepair  # noqa: E402
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402


ROOT_MODEL = Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct")
BANK = ROOT / "results/coverage/phi4_seq64_input_bank.json"
RELEASE = ROOT / "results/coverage/runtime_releases/phi4_seq64_r1"
OUTPUT = ROOT / "results/property/bias_formation/interventions/phi4_mm_transport_pairing.json"
CARRIER = "model.norm.weight"
HIDDEN = 3072


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_branch(model: torch.nn.Module, candidate: Any, values: torch.Tensor, modules: list[Any], seed: int,
               mode: str | None, permutation: np.ndarray | None, transport: np.ndarray) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.zero_grad(set_to_none=True)
    observer = None
    if mode is not None:
        permutation_tensor = None if permutation is None else torch.tensor(permutation, dtype=torch.long, device=values.device)
        observer = MMRepair(modules, mode, permutation_tensor)
    if observer is None:
        loss = candidate(values)
        loss.backward()
    else:
        with observer:
            loss = candidate(values)
            loss.backward()
    torch.cuda.synchronize(values.device)
    parameter = dict(model.named_parameters()).get(CARRIER)
    if parameter is None or parameter.grad is None:
        raise RuntimeError("Phi declared carrier gradient was not captured")
    return {
        "loss_digest": tensor_digest(loss),
        "gradient": parameter.grad.detach().float().cpu().numpy().reshape(-1).copy(),
        "transport": transport,
        "observer": observer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    states = json.loads(BANK.read_text(encoding="utf-8"))["states"][16:32]
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model("phi", ROOT_MODEL, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    validate_release(wrapper_modules(modules), json.loads((RELEASE / "capture.json").read_text()))
    natural_gradients: list[np.ndarray] = []
    shuffled_gradients: list[np.ndarray] = []
    natural_local: list[np.ndarray] = []
    shuffled_local_norm_error: list[float] = []
    transport_errors: list[float] = []
    rows = []
    transport_box: list[np.ndarray] = []
    for index, state in enumerate(states):
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        seed = 44000 + index
        # The compiled LossStep may inline the final RMSNorm and omit Python
        # hooks after its warm graph is cached. Capture the analytic RMSNorm
        # transport factor once through an eager, no-grad forward instead.
        transport_box.clear()
        norm = model.model.norm
        def pre_hook(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
            raw = inputs[0].detach().float()
            factor = raw * torch.rsqrt(raw.square().mean(dim=-1, keepdim=True) + float(norm.variance_epsilon))
            transport_box.append(factor.reshape(-1, factor.shape[-1]).cpu().numpy().copy())
        hook = norm.register_forward_pre_hook(pre_hook)
        with torch.no_grad():
            model.model(input_ids=values, use_cache=False, return_dict=True)
        hook.remove()
        if len(transport_box) != 1:
            raise RuntimeError("Phi eager RMSNorm transport was not captured exactly once")
        transport = transport_box[0]
        standard = run_branch(model, candidate, values, modules, seed, "SHAM", None, transport)
        sham = run_branch(model, candidate, values, modules, seed, "SHAM", None, transport)
        natural = run_branch(model, candidate, values, modules, seed, "REPAIR_FP32_CAST_BF16", None, transport)
        permutation = np.random.default_rng(51000 + index).permutation(transport.shape[0])
        shuffled = run_branch(model, candidate, values, modules, seed, "REPAIR_PERMUTE", permutation, transport)
        if standard["loss_digest"] != sham["loss_digest"] or not np.array_equal(standard["gradient"], sham["gradient"]):
            raise RuntimeError(f"Phi intervention sham changed the full step at state {index}")
        if natural["observer"] is None or shuffled["observer"] is None or natural["observer"].natural_vector is None:
            raise RuntimeError("Phi intervention did not capture natural local residual")
        local = natural["observer"].natural_vector
        if shuffled["observer"].natural_vector is None:
            raise RuntimeError("Phi shuffled arm did not capture its local residual")
        shuffled_local = shuffled["observer"].natural_vector
        if local.size != shuffled_local.size or not np.isclose(np.linalg.norm(local), np.linalg.norm(shuffled_local), rtol=1e-6, atol=1e-8):
            raise RuntimeError("Phi transport intervention did not preserve local residual norm")
        gradient_delta = standard["gradient"] - natural["gradient"]
        shuffled_delta = standard["gradient"] - shuffled["gradient"]
        transport = standard["transport"]
        if local.size != transport.size:
            raise RuntimeError("Phi endpoint residual and RMSNorm transport coordinates differ")
        local_matrix = local.reshape(-1, HIDDEN)
        predicted = (local_matrix * transport).sum(axis=0)
        transport_error = float(np.linalg.norm(predicted - gradient_delta) / max(np.linalg.norm(gradient_delta), 1e-30))
        natural_gradients.append(gradient_delta)
        shuffled_gradients.append(shuffled_delta)
        natural_local.append(local)
        shuffled_local_norm_error.append(float(abs(np.linalg.norm(local) - np.linalg.norm(shuffled_local)) / max(np.linalg.norm(local), 1e-30)))
        transport_errors.append(transport_error)
        rows.append({
            "state_id": str(state.get("state_id", state.get("sequence_id", index + 16))),
            "permutation_seed": 51000 + index,
            "permutation_rows": int(permutation.size),
            "natural_gradient_delta_l2": float(np.linalg.norm(gradient_delta)),
            "shuffled_gradient_delta_l2": float(np.linalg.norm(shuffled_delta)),
            "local_residual_l2": float(np.linalg.norm(local)),
            "shuffled_local_norm_relative_error": shuffled_local_norm_error[-1],
            "transport_prediction_relative_error": transport_error,
        })
        del values, standard, sham, natural, shuffled, local, shuffled_local, gradient_delta, shuffled_delta
        torch.cuda.empty_cache()
        print(json.dumps({"event": "TRANSPORT_INTERVENTION_STATE_COMPLETE", "state": index}, sort_keys=True), flush=True)
    policy = FormationPolicy(min_states=16, bootstrap_samples=2000)
    natural_cert = summarize_state_vectors(natural_gradients, state_ids=[row["state_id"] for row in rows], layer="PARAMETER_GRADIENT", partition="confirmation", policy=policy).as_dict()
    shuffled_cert = summarize_state_vectors(shuffled_gradients, state_ids=[row["state_id"] for row in rows], layer="PARAMETER_GRADIENT", partition="confirmation", policy=policy).as_dict()
    payload = {
        "schema": "kernel-analyzer-phi-mm-transport-intervention-v1",
        "status": "SUPPORTS_TRANSPORT_ALIGNMENT" if natural_cert["status"] == "BIASED" and shuffled_cert["status"] == "CENTERED" and max(transport_errors) <= 1e-5 else "MEASURED_NO_TRANSPORT_SUPPORT",
        "case_id": "phi4_lm_head_dx_seq64",
        "intervention": "row-permute local MM residual while preserving residual multiset, norm, and RMSNorm transport multiset",
        "matched_sham": "same candidate implementation and no-op endpoint observer",
        "natural_gradient_population": natural_cert,
        "shuffled_gradient_population": shuffled_cert,
        "rows": rows,
        "gates": {
            "natural_gradient_biased": natural_cert["status"] == "BIASED",
            "shuffled_gradient_centered": shuffled_cert["status"] == "CENTERED",
            "local_norm_preserved_every_state": max(shuffled_local_norm_error) <= 1e-6,
            "analytic_transport_matches_natural_gradient": max(transport_errors) <= 1e-5,
            "sixteen_confirmation_states": len(rows) == 16,
        },
        "claim_boundary": "This intervention tests residual/transport pairing for the Phi final-norm VJP. It does not establish a universal transport property or alter the frozen formation labels.",
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name("." + args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "status": payload["status"], "gates": payload["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
