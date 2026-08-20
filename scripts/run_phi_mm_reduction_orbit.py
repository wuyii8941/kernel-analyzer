#!/usr/bin/env python3
"""Measure the real Phi lm-head dX GEMM reduction orbit on an ordered trajectory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor")

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))

from kernel_analyzer.persistence_property import semantic_orbit_statistics_from_gram  # noqa: E402
from kernel_analyzer.reduction_orbit import frozen_permutations, gemm_reduction_orbit  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import load_model  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import TARGET_LEFT_SHAPE  # noqa: E402


class OrbitObserver:
    def __init__(self, modules: list[Any], permutations: list[torch.Tensor]) -> None:
        self.modules = modules; self.permutations = permutations
        self.restores: list[tuple[Any, Any]] = []; self.result: dict[str, Any] | None = None

    def __enter__(self) -> "OrbitObserver":
        seen: set[int] = set()
        for module in self.modules:
            namespace = getattr(module, "extern_kernels", None)
            if namespace is None or id(namespace) in seen:
                continue
            seen.add(id(namespace)); original = namespace.mm

            def wrapped(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
                result = _original(*args, **kwargs)
                if tuple(args[0].shape) == TARGET_LEFT_SHAPE:
                    if self.result is not None:
                        raise RuntimeError("Phi orbit endpoint executed more than once")
                    self.result = gemm_reduction_orbit(
                        args[0], args[1], permutations=self.permutations,
                        candidate=torch.mm, return_vectors=True,
                    )
                return result

            namespace.mm = wrapped; self.restores.append((namespace, original))
        return self

    def __exit__(self, *unused: Any) -> None:
        for namespace, original in self.restores:
            namespace.mm = original
        if self.result is None:
            raise RuntimeError("Phi orbit endpoint was not reached")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--orbit-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=ROOT / "results/property/persistence_v1/orbits/phi_mm.json")
    args = parser.parse_args()
    if not 2 <= args.steps <= 16 or args.orbit_size != 8:
        raise ValueError("protocol requires 2..16 ordered steps and exactly eight orbit members")
    bank = json.loads((ROOT / "results/coverage/phi4_seq64_input_bank.json").read_text())
    states = bank.get("states", bank.get("records"))[16:16 + args.steps]
    device = torch.device(args.device); configure_candidate_runtime(24000)
    model = load_model("phi", Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct"), device)
    model.eval(); start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    # Inductor wrapper source is version-sensitive.  Bind the already proven
    # semantic endpoint by its exact invocation shape instead of rejecting a
    # recompilation solely because wrapper bytes changed.
    permutations = frozen_permutations(TARGET_LEFT_SHAPE[1], args.orbit_size, 20260820)
    vectors: list[torch.Tensor] = []; rows: list[dict[str, Any]] = []
    parameter = model.model.norm.weight
    master = parameter.detach().float().clone()
    for index, state in enumerate(states):
        with torch.no_grad(): parameter.copy_(master.to(parameter.dtype))
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        torch.manual_seed(4407 + index); torch.cuda.manual_seed_all(4407 + index)
        model.zero_grad(set_to_none=True)
        observer = OrbitObserver(modules, permutations)
        with observer:
            loss = candidate(values); loss.backward()
        assert observer.result is not None
        vector = observer.result.pop("_residual_vectors")
        vectors.extend(vector.unbind(0))
        rows.append({
            "step": index + 1,
            "state_id": str(state.get("state_id", state.get("sequence_id", 16 + index))),
            "orbit_mean_l2": observer.result["orbit_mean_l2"],
            "orbit_mean_energy_fraction": observer.result["orbit_mean_energy_fraction"],
        })
        with torch.no_grad(): master.add_(parameter.grad.detach().float(), alpha=-args.learning_rate)
        print(json.dumps({"event": "PHI_ORBIT_STEP", **rows[-1]}), flush=True)
        del values, loss, vector, observer
        torch.cuda.empty_cache()
    matrix = torch.stack(vectors).double(); gram = matrix @ matrix.T
    state_ids = [row["state_id"] for row in rows]
    stats = semantic_orbit_statistics_from_gram(
        gram.numpy(), state_ids=state_ids,
        variant_ids=["identity"] + [f"perm_{i:02d}" for i in range(1, args.orbit_size)],
        default_variant="identity", sign_flip_draws=4000, seed=20260820,
    )
    payload = {
        "schema": "kernel-analyzer-real-phi-mm-reduction-orbit-v1",
        "status": "COMPLETE" if args.steps == 16 else "ENGINEERING_DRY_RUN",
        "case_id": "phi4_seq64_lmhead_dx",
        "endpoint_binding": {"left_shape": list(TARGET_LEFT_SHAPE), "expected_calls_per_step": 1},
        "ordered_evolving_states": True,
        "steps": args.steps, "orbit_size": args.orbit_size,
        "rows": rows, "statistics": stats,
        "claim_boundary": "Forward-local reduction-orbit predictor; not an F+B persistence verdict.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
