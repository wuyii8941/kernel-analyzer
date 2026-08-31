#!/usr/bin/env python3
"""Prospective confirmation of persistence property on Qwen seq256 lm-head dX."""

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

from kernel_analyzer.persistence_property import path_statistics_from_gram, semantic_orbit_statistics_from_gram  # noqa: E402
from kernel_analyzer.reduction_orbit import frozen_permutations, gemm_reduction_orbit  # noqa: E402
from scripts.generated_nontriton_fp32_observer import fp32_external_reference  # noqa: E402
from scripts.generated_contrast_observer import _source_identity  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402


LEFT = (256, 151936)
RIGHT = (151936, 2048)
CARRIER = "model.norm.weight"


class ShapeObserver:
    def __init__(self, modules: list[Any], mode: str, permutations: list[torch.Tensor],
                 *, left_shape: tuple[int, int] = LEFT,
                 right_shape: tuple[int, int] = RIGHT,
                 selected_permutation: torch.Tensor | None = None,
                 target_sha: str | None = None) -> None:
        self.modules = modules; self.mode = mode; self.permutations = permutations
        self.left_shape = left_shape; self.right_shape = right_shape
        self.selected_permutation = selected_permutation
        self.target_sha = target_sha
        self.restores: list[tuple[Any, Any]] = []; self.calls = 0
        self.orbit: dict[str, Any] | None = None; self.changed_l2 = 0.0
        self.local_vector: torch.Tensor | None = None
        self.repair_vector: torch.Tensor | None = None
        self.seen: list[dict[str, Any]] = []

    def __enter__(self) -> "ShapeObserver":
        seen: set[int] = set()
        for module in self.modules:
            namespace = getattr(module, "extern_kernels", None)
            if namespace is None or id(namespace) in seen:
                continue
            seen.add(id(namespace)); original = namespace.mm

            def wrapped(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
                filename, line, digest = _source_identity()
                if self.target_sha is not None and digest != self.target_sha:
                    return _original(*args, **kwargs)
                result = _original(*args, **kwargs)
                if tuple(args[0].shape) != self.left_shape or tuple(args[1].shape) != self.right_shape:
                    return result
                actual = kwargs.get("out", result); before = actual.detach().float().clone()
                self.seen.append({
                    "source_sha": digest,
                    "source_line": line,
                    "source_file": filename,
                    "left_shape": list(args[0].shape),
                    "right_shape": list(args[1].shape),
                    "output_shape": list(actual.shape),
                })
                if self.mode == "orbit":
                    self.orbit = gemm_reduction_orbit(
                        args[0], args[1], permutations=self.permutations,
                        candidate=torch.mm, return_vectors=True,
                    )
                elif self.mode == "fp32":
                    high = fp32_external_reference("mm", args, kwargs)
                    delivered = high.to(actual.dtype)
                    self.local_vector = (before - delivered.float()).detach().cpu()
                    self.repair_vector = delivered.detach().float().cpu()
                    actual.copy_(delivered)
                    self.changed_l2 = float(torch.linalg.vector_norm(before - actual.float()).item())
                elif self.mode == "sham":
                    self.local_vector = torch.zeros_like(before).detach().cpu()
                    self.repair_vector = before.detach().cpu()
                    actual.copy_(before.to(actual.dtype))
                    self.changed_l2 = 0.0
                elif self.mode == "permuted":
                    if self.selected_permutation is None:
                        raise RuntimeError("permuted endpoint needs a frozen permutation")
                    permutation = self.selected_permutation.to(args[0].device)
                    replacement = torch.mm(
                        args[0].index_select(1, permutation),
                        args[1].index_select(0, permutation),
                    )
                    actual.copy_(replacement.to(actual.dtype))
                    self.changed_l2 = float(torch.linalg.vector_norm(before - actual.float()).item())
                else:
                    raise ValueError(self.mode)
                self.calls += 1
                return result

            namespace.mm = wrapped; self.restores.append((namespace, original))
        return self

    def __exit__(self, *unused: Any) -> None:
        for namespace, original in self.restores:
            namespace.mm = original
        if self.calls != 1:
            raise RuntimeError(f"held-out endpoint executed {self.calls} times")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "results/property/persistence_v1/confirmation/qwen256_lmhead.json")
    args = parser.parse_args()
    if args.steps not in {2, 16}:
        raise ValueError("only engineering two-step or frozen sixteen-step modes are allowed")
    roster = json.loads((ROOT / "results/property/persistence_v1/roster.json").read_text())
    frozen = roster["prospective_confirmation_cases"][0]
    bank = json.loads((ROOT / "results/coverage/qwen_seq256_input_bank.json").read_text())
    states = bank.get("states", bank.get("records"))[:args.steps]
    ids = [str(row.get("state_id", row.get("sequence_id"))) for row in states]
    if ids != frozen["state_ids"][:args.steps]:
        raise RuntimeError("confirmation state order differs from frozen roster")
    device = torch.device(args.device); configure_candidate_runtime(24000)
    model = load_model("qwen", Path("/data1/tzh/models/Qwen/Qwen3-1.7B"), device)
    model.eval(); parameter = dict(model.named_parameters())[CARRIER]
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0].get("token_ids", states[0].get("input_ids"))], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:]); permutations = frozen_permutations(LEFT[1], 8, 20260820)
    master = parameter.detach().float().clone(); learning_rate = 1e-4
    orbit_vectors: list[torch.Tensor] = []; update_vectors: list[torch.Tensor] = []; rows = []

    def evaluate(state: dict[str, Any], index: int, mode: str) -> tuple[str, torch.Tensor, ShapeObserver]:
        with torch.no_grad(): parameter.copy_(master.to(parameter.dtype))
        values = torch.tensor([state.get("token_ids", state.get("input_ids"))], dtype=torch.long, device=device)
        torch.manual_seed(24_000 + index); torch.cuda.manual_seed_all(24_000 + index)
        model.zero_grad(set_to_none=True); observer = ShapeObserver(modules, mode, permutations)
        with observer:
            loss = candidate(values); loss.backward()
        return tensor_digest(loss), parameter.grad.detach().float().clone(), observer

    for index, state in enumerate(states):
        loss_default, grad_default, observed = evaluate(state, index, "orbit")
        loss_fp32, grad_fp32, repaired = evaluate(state, index, "fp32")
        if loss_default != loss_fp32 or observed.orbit is None:
            raise RuntimeError("confirmation forward/common endpoint gate failed")
        orbit = observed.orbit.pop("_residual_vectors"); orbit_vectors.extend(orbit.unbind(0))
        update = (grad_default - grad_fp32).mul(-learning_rate); update_vectors.append(update.cpu())
        master.add_(grad_default, alpha=-learning_rate)
        rows.append({"step": index + 1, "state_id": ids[index],
                     "fp32_endpoint_delta_l2": repaired.changed_l2,
                     "update_error_l2": float(torch.linalg.vector_norm(update).item())})
        print(json.dumps({"event": "QWEN256_CONFIRM_STEP", **rows[-1]}), flush=True)
        del grad_default, grad_fp32, orbit, update, observed, repaired
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
        "schema": "kernel-analyzer-qwen256-prospective-property-confirmation-v1",
        "status": "CONFIRMED" if args.steps == 16 and all(gates.values()) else ("ENGINEERING_DRY_RUN" if args.steps == 2 else "PREDICTION_FAILED"),
        "case_id": frozen["case_id"], "state_ids": ids,
        "prediction": prediction, "prediction_gates": gates,
        "source_orbit": source, "effective_update": update, "rows": rows,
        "claim_boundary": "One held-out invocation/shape confirmation; not cross-model universality.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
