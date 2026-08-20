#!/usr/bin/env python3
"""Measure Qwen seq128 v_proj's real MM reduction orbit on an Adam trajectory."""

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

from kernel_analyzer.persistence_property import semantic_orbit_statistics_from_gram  # noqa: E402
from kernel_analyzer.reduction_orbit import frozen_permutations, gemm_reduction_orbit  # noqa: E402
from scripts.generated_contrast_observer import _source_identity  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import load_model  # noqa: E402
from scripts.run_mm_source_aligned_repair import validate_target_call  # noqa: E402
from scripts.run_qwen128_vproj_rounding_persistence import (  # noqa: E402
    CARRIER, CID, TARGET_SHA, adam_arm,
)


class OrbitObserver:
    def __init__(self, modules: list[Any], target_sha: str, permutations: list[torch.Tensor]) -> None:
        self.modules = modules; self.target_sha = target_sha; self.permutations = permutations
        self.restores: list[tuple[Any, Any]] = []; self.result: dict[str, Any] | None = None

    def __enter__(self) -> "OrbitObserver":
        seen: set[int] = set()
        for module in self.modules:
            namespace = getattr(module, "extern_kernels", None)
            if namespace is None or id(namespace) in seen:
                continue
            seen.add(id(namespace)); original = namespace.mm

            def wrapped(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
                _, _, digest = _source_identity(); result = _original(*args, **kwargs)
                if digest == self.target_sha:
                    if self.result is not None:
                        raise RuntimeError("Qwen orbit endpoint executed more than once")
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
            raise RuntimeError("Qwen orbit endpoint was not reached")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--orbit-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "results/property/persistence_v1/orbits/qwen128_vproj.json")
    args = parser.parse_args()
    if not 2 <= args.steps <= 16 or args.orbit_size != 8:
        raise ValueError("protocol requires 2..16 steps and eight orbit members")
    bank = json.loads((ROOT / "results/coverage/qwen_seq128_input_bank.json").read_text())
    states = bank.get("states", bank.get("records"))[:args.steps]
    device = torch.device(args.device); configure_candidate_runtime(24000)
    model = load_model("qwen", Path("/data1/tzh/models/Qwen/Qwen3-1.7B"), device)
    target = dict(model.named_parameters())[CARRIER]
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0].get("token_ids", states[0].get("input_ids"))], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    binding = validate_target_call([module for module, _ in wrapper_modules(modules)], TARGET_SHA)
    reduction_extent = int(target.shape[1])
    permutations = frozen_permutations(reduction_extent, args.orbit_size, 20260820)
    master = target.detach().float().clone(); first = torch.zeros_like(master); second = torch.zeros_like(master)
    vectors: list[torch.Tensor] = []; rows: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        with torch.no_grad(): target.copy_(master.to(target.dtype))
        tokens = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        torch.manual_seed(24000 + index); torch.cuda.manual_seed_all(24000 + index)
        model.zero_grad(set_to_none=True); observer = OrbitObserver(modules, TARGET_SHA, permutations)
        with observer:
            loss = candidate(values); loss.backward()
        assert observer.result is not None and target.grad is not None
        vector = observer.result.pop("_residual_vectors"); vectors.extend(vector.unbind(0))
        update, first, second = adam_arm(
            target.grad.detach().float(), first, second, index + 1,
            learning_rate=1e-5, beta1=0.9, beta2=0.95, epsilon=1e-8,
        )
        master.add_(update)
        rows.append({
            "step": index + 1,
            "state_id": str(state.get("state_id", state.get("sequence_id", index))),
            "orbit_mean_l2": observer.result["orbit_mean_l2"],
            "orbit_mean_energy_fraction": observer.result["orbit_mean_energy_fraction"],
        })
        print(json.dumps({"event": "QWEN_ORBIT_STEP", **rows[-1]}), flush=True)
        del values, loss, update, vector, observer
        torch.cuda.empty_cache()
    matrix = torch.stack(vectors).double(); gram = matrix @ matrix.T
    stats = semantic_orbit_statistics_from_gram(
        gram.numpy(), state_ids=[row["state_id"] for row in rows],
        variant_ids=["identity"] + [f"perm_{i:02d}" for i in range(1, args.orbit_size)],
        default_variant="identity", sign_flip_draws=4000, seed=20260820,
    )
    payload = {
        "schema": "kernel-analyzer-real-qwen-vproj-reduction-orbit-v1",
        "status": "COMPLETE" if args.steps == 16 else "ENGINEERING_DRY_RUN",
        "case_id": "qwen128_vproj_mm", "candidate_id": CID,
        "runtime_binding": binding, "ordered_evolving_states": True,
        "steps": args.steps, "orbit_size": args.orbit_size, "rows": rows,
        "statistics": stats,
        "claim_boundary": "Forward-local reduction-orbit predictor; not an F+B persistence verdict.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
