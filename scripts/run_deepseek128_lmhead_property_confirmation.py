#!/usr/bin/env python3
"""Prospective cross-model confirmation on DeepSeek seq128 lm-head dX."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor")

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "archive/round1_code/src"):
    sys.path.insert(0, str(path))

from kernel_analyzer.persistence_property import path_statistics_from_gram, semantic_orbit_statistics_from_gram  # noqa: E402
from kernel_analyzer.reduction_orbit import frozen_permutations  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_qwen256_lmhead_property_confirmation import ShapeObserver  # noqa: E402


LEFT = (128, 151936)
RIGHT = (151936, 4096)
CARRIER = "model.norm.weight"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "results/property/persistence_v1/confirmation/deepseek128_lmhead.json")
    args = parser.parse_args()
    if args.steps not in {2, 16}:
        raise ValueError("only two-step engineering or sixteen-step confirmation is allowed")
    roster = json.loads((ROOT / "results/property/persistence_v1/roster.json").read_text())
    frozen = next(row for row in roster["prospective_confirmation_cases"] if row["case_id"] == "deepseek8b_seq128_lm_head_dx")
    bank = json.loads((ROOT / "results/coverage/deepseek8b_seq128_input_bank.json").read_text())
    states = bank.get("states", bank.get("records"))[:args.steps]
    ids = [str(row.get("state_id", row.get("sequence_id", index))) for index, row in enumerate(states)]
    if ids != frozen["state_ids"][:args.steps]:
        raise RuntimeError("DeepSeek confirmation state order differs")
    device = torch.device(args.device); configure_candidate_runtime(24000)
    model = load_model("deepseek8", Path("/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"), device)
    model.eval(); parameter = dict(model.named_parameters())[CARRIER]
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    tokens0 = states[0].get("token_ids", states[0].get("input_ids"))
    warm = torch.tensor([tokens0], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:]); permutations = frozen_permutations(LEFT[1], 8, 20260820)
    master = parameter.detach().float().clone(); orbit_vectors = []; update_vectors = []; rows = []

    def evaluate(state, index, mode):
        with torch.no_grad(): parameter.copy_(master.to(parameter.dtype))
        tokens = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        torch.manual_seed(24_000 + index); torch.cuda.manual_seed_all(24_000 + index)
        model.zero_grad(set_to_none=True)
        observer = ShapeObserver(modules, mode, permutations, left_shape=LEFT, right_shape=RIGHT)
        with observer:
            loss = candidate(values); loss.backward()
        return tensor_digest(loss), parameter.grad.detach().float().clone(), observer

    for index, state in enumerate(states):
        loss_c, grad_c, observed = evaluate(state, index, "orbit")
        loss_r, grad_r, repaired = evaluate(state, index, "fp32")
        if loss_c != loss_r or observed.orbit is None:
            raise RuntimeError("DeepSeek common-forward or endpoint gate failed")
        orbit = observed.orbit.pop("_residual_vectors"); orbit_vectors.extend(orbit.unbind(0))
        update = (grad_c - grad_r).mul(-1e-4); update_vectors.append(update.cpu())
        master.add_(grad_c, alpha=-1e-4)
        rows.append({"step": index + 1, "state_id": ids[index],
                     "fp32_endpoint_delta_l2": repaired.changed_l2,
                     "update_error_l2": float(torch.linalg.vector_norm(update).item())})
        print(json.dumps({"event": "DEEPSEEK_CONFIRM_STEP", **rows[-1]}), flush=True)
        del grad_c, grad_r, orbit, update, observed, repaired
        torch.cuda.empty_cache()
    orbit_matrix = torch.stack(orbit_vectors).double(); update_matrix = torch.stack(update_vectors).double()
    source = semantic_orbit_statistics_from_gram(
        (orbit_matrix @ orbit_matrix.T).numpy(), state_ids=ids,
        variant_ids=["identity"] + [f"perm_{i:02d}" for i in range(1, 8)],
        default_variant="identity", sign_flip_draws=4000, seed=20260820,
    )
    update = path_statistics_from_gram(
        (update_matrix @ update_matrix.T).numpy(), state_ids=ids,
        sign_flip_draws=4000, seed=20260820,
    )
    prediction = frozen["prediction_frozen_before_values"]
    gates = {
        "orbit_mean_prediction": source["orbit_mean"]["above_sign_flip_95"] is prediction["orbit_mean_temporal_amplification_above_sign_flip_95"],
        "update_prediction": update["above_sign_flip_95"] is prediction["effective_update_temporal_amplification_above_sign_flip_95"],
        "transport_prediction": (update["coherence_amplification"] >= source["default_schedule"]["coherence_amplification"]) is prediction["transport_does_not_destroy_source_persistence"],
    }
    payload = {
        "schema": "kernel-analyzer-deepseek128-prospective-property-confirmation-v1",
        "status": "CONFIRMED" if args.steps == 16 and all(gates.values()) else ("ENGINEERING_DRY_RUN" if args.steps == 2 else "PREDICTION_FAILED"),
        "case_id": frozen["case_id"], "state_ids": ids, "prediction": prediction,
        "prediction_gates": gates, "source_orbit": source, "effective_update": update,
        "rows": rows, "claim_boundary": "Held-out model-family confirmation at one loss-head invocation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
