#!/usr/bin/env python3
"""Directly measure M_t E_pi[epsilon_t,pi] on DeepSeek's held-out lm-head dX."""

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

from kernel_analyzer.persistence_property import semantic_orbit_statistics_from_gram  # noqa: E402
from kernel_analyzer.reduction_orbit import frozen_permutations  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_deepseek128_lmhead_property_confirmation import LEFT, RIGHT, CARRIER  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_qwen256_lmhead_property_confirmation import ShapeObserver  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "results/property/persistence_v1/confirmation/deepseek128_transported_orbit.json")
    args = parser.parse_args()
    if args.steps not in {2, 16}:
        raise ValueError("only two-step engineering or sixteen-step measurement is allowed")
    bank = json.loads((ROOT / "results/coverage/deepseek8b_seq128_input_bank.json").read_text())
    states = bank.get("states", bank.get("records"))[:args.steps]
    ids = [str(row.get("state_id", row.get("sequence_id", index))) for index, row in enumerate(states)]
    device = torch.device(args.device); configure_candidate_runtime(24000)
    model = load_model("deepseek8", Path("/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"), device)
    model.eval(); parameter = dict(model.named_parameters())[CARRIER]
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    tokens0 = states[0].get("token_ids", states[0].get("input_ids"))
    warm = torch.tensor([tokens0], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:]); permutations = frozen_permutations(LEFT[1], 8, 20260820)
    master = parameter.detach().float().clone(); vectors = []; rows = []

    def evaluate(state, index, mode, permutation=None):
        with torch.no_grad(): parameter.copy_(master.to(parameter.dtype))
        tokens = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        torch.manual_seed(24_000 + index); torch.cuda.manual_seed_all(24_000 + index)
        model.zero_grad(set_to_none=True)
        observer = ShapeObserver(
            modules, mode, permutations, left_shape=LEFT, right_shape=RIGHT,
            selected_permutation=permutation,
        )
        with observer:
            loss = candidate(values); loss.backward()
        return tensor_digest(loss), parameter.grad.detach().float().clone()

    for index, state in enumerate(states):
        loss_ref, grad_ref = evaluate(state, index, "fp32")
        state_vectors = []
        for permutation in permutations:
            loss_variant, grad_variant = evaluate(state, index, "permuted", permutation)
            if loss_variant != loss_ref:
                raise RuntimeError("backward reduction orbit changed forward loss")
            state_vectors.append((grad_variant - grad_ref).mul(-1e-4).cpu())
            del grad_variant
        vectors.extend(state_vectors)
        # Advance the measured trajectory using the identity-orbit candidate.
        identity_gradient = grad_ref - state_vectors[0].to(device).div(1e-4)
        master.add_(identity_gradient, alpha=-1e-4)
        rows.append({"step": index + 1, "state_id": ids[index],
                     "orbit_mean_update_l2": float(torch.linalg.vector_norm(torch.stack(state_vectors).mean(0)).item())})
        print(json.dumps({"event": "DEEPSEEK_TRANSPORTED_ORBIT_STEP", **rows[-1]}), flush=True)
        del grad_ref, state_vectors, identity_gradient
        torch.cuda.empty_cache()
    matrix = torch.stack(vectors).double(); gram = matrix @ matrix.T
    statistics = semantic_orbit_statistics_from_gram(
        gram.numpy(), state_ids=ids,
        variant_ids=["identity"] + [f"perm_{i:02d}" for i in range(1, 8)],
        default_variant="identity", sign_flip_draws=4000, seed=20260820,
    )
    payload = {
        "schema": "kernel-analyzer-deepseek-transported-orbit-v1",
        "status": "COMPLETE" if args.steps == 16 else "ENGINEERING_DRY_RUN",
        "case_id": "deepseek8b_seq128_lm_head_dx", "state_ids": ids,
        "statistics": statistics, "rows": rows,
        "claim_boundary": "Direct effective-update orbit; M_t m_t is measured by averaging eight real backward variants.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
