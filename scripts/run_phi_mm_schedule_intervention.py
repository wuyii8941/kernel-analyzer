#!/usr/bin/env python3
"""Break Phi's fixed MM reduction schedule on the real backward endpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor")

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "archive/round1_code/src"):
    sys.path.insert(0, str(path))

from kernel_analyzer.reduction_orbit import frozen_permutations  # noqa: E402
from kernel_analyzer.trajectory_persistence import OrderedVectorPath  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import MMRepair, TARGET_LEFT_SHAPE  # noqa: E402


class ReductionSchedule:
    def __init__(self, modules: list[Any], permutation: torch.Tensor) -> None:
        self.modules = modules; self.permutation = permutation
        self.restores: list[tuple[Any, Any]] = []; self.calls = 0; self.delta_l2 = 0.0

    def __enter__(self) -> "ReductionSchedule":
        seen: set[int] = set()
        for module in self.modules:
            namespace = getattr(module, "extern_kernels", None)
            if namespace is None or id(namespace) in seen:
                continue
            seen.add(id(namespace)); original = namespace.mm

            def wrapped(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
                result = _original(*args, **kwargs)
                if tuple(args[0].shape) != TARGET_LEFT_SHAPE:
                    return result
                actual = kwargs.get("out", result)
                permutation = self.permutation.to(args[0].device)
                replacement = torch.mm(
                    args[0].index_select(1, permutation),
                    args[1].index_select(0, permutation),
                )
                before = actual.detach().float().clone()
                actual.copy_(replacement.to(actual.dtype))
                self.delta_l2 = float(torch.linalg.vector_norm(before - actual.float()).item())
                self.calls += 1
                return result

            namespace.mm = wrapped; self.restores.append((namespace, original))
        return self

    def __exit__(self, *unused: Any) -> None:
        for namespace, original in self.restores:
            namespace.mm = original
        if self.calls != 1:
            raise RuntimeError(f"schedule intervention reached target {self.calls} times")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "results/property/persistence_v1/interventions/phi_schedule.json")
    args = parser.parse_args()
    if not 2 <= args.steps <= 16:
        raise ValueError("steps must be in [2,16]")
    bank = json.loads((ROOT / "results/coverage/phi4_seq64_input_bank.json").read_text())
    states = bank.get("states", bank.get("records"))[16:16 + args.steps]
    device = torch.device(args.device); configure_candidate_runtime(24000)
    model = load_model("phi", Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct"), device)
    model.eval(); start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:]); parameter = model.model.norm.weight
    master = parameter.detach().float().clone(); learning_rate = 1e-3
    permutations = frozen_permutations(TARGET_LEFT_SHAPE[1], args.steps + 1, 20260821)[1:]
    natural_path = OrderedVectorPath(total_steps=args.steps, calibration_steps=max(1, args.steps // 2))
    randomized_path = OrderedVectorPath(total_steps=args.steps, calibration_steps=max(1, args.steps // 2))
    rows = []

    def evaluate(state: dict[str, Any], index: int, mode: str) -> tuple[str, torch.Tensor, float]:
        with torch.no_grad(): parameter.copy_(master.to(parameter.dtype))
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        torch.manual_seed(4407 + index); torch.cuda.manual_seed_all(4407 + index)
        model.zero_grad(set_to_none=True); observer: Any = None
        if mode == "fp32": observer = MMRepair(modules, "REPAIR_FP32_CAST_BF16")
        elif mode == "random_schedule": observer = ReductionSchedule(modules, permutations[index])
        if observer is None:
            loss = candidate(values); loss.backward(); changed = 0.0
        else:
            with observer:
                loss = candidate(values); loss.backward()
            changed = observer.delta_l2 if mode == "random_schedule" else float(observer.local["l2"])
        gradient = parameter.grad.detach().float().clone()
        return tensor_digest(loss), gradient, changed

    for index, state in enumerate(states):
        loss_default, grad_default, _ = evaluate(state, index, "default")
        loss_fp32, grad_fp32, fp32_delta = evaluate(state, index, "fp32")
        loss_random, grad_random, schedule_delta = evaluate(state, index, "random_schedule")
        if len({loss_default, loss_fp32, loss_random}) != 1:
            raise RuntimeError("backward-only schedule intervention changed forward loss")
        natural = (grad_default - grad_fp32).mul(-learning_rate)
        randomized = (grad_random - grad_fp32).mul(-learning_rate)
        natural_path.add(natural); randomized_path.add(randomized)
        master.add_(grad_default, alpha=-learning_rate)
        rows.append({
            "step": index + 1,
            "state_id": str(state.get("state_id", state.get("sequence_id", 16 + index))),
            "natural_update_error_l2": float(torch.linalg.vector_norm(natural).item()),
            "randomized_update_error_l2": float(torch.linalg.vector_norm(randomized).item()),
            "fp32_endpoint_delta_l2": fp32_delta,
            "schedule_endpoint_delta_l2": schedule_delta,
        })
        print(json.dumps({"event": "PHI_SCHEDULE_STEP", **rows[-1]}), flush=True)
        del grad_default, grad_fp32, grad_random, natural, randomized
        torch.cuda.empty_cache()
    natural_summary = natural_path.finalize(); random_summary = randomized_path.finalize()
    payload = {
        "schema": "kernel-analyzer-phi-real-schedule-intervention-v1",
        "status": "COMPLETE" if args.steps == 16 else "ENGINEERING_DRY_RUN",
        "case_id": "phi4_seq64_lmhead_dx", "steps": args.steps,
        "invariant": "joint K-axis permutation preserves exact GW",
        "natural_vs_fp32": natural_summary,
        "random_schedule_vs_fp32": random_summary,
        "amplification_ratio": random_summary["coherence_amplification"] / max(natural_summary["coherence_amplification"], 1e-30),
        "rows": rows,
        "claim_boundary": "Matched real-kernel reduction schedule intervention on the backward dX endpoint.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
