#!/usr/bin/env python3
"""Replace Phi dX deterministic MM error by stochastic unbiased materialization."""

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

from kernel_analyzer.precision import source_aligned_mm_output  # noqa: E402
from kernel_analyzer.trajectory_persistence import OrderedVectorPath  # noqa: E402
from scripts.generated_nontriton_fp32_observer import fp32_external_reference  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import MMRepair, TARGET_LEFT_SHAPE  # noqa: E402


class StochasticMM:
    def __init__(self, modules: list[Any], seed: int) -> None:
        self.modules = modules; self.seed = seed; self.restores: list[tuple[Any, Any]] = []
        self.calls = 0; self.source_l2 = 0.0

    def __enter__(self) -> "StochasticMM":
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
                actual = kwargs.get("out", result); before = actual.detach().clone()
                high = fp32_external_reference("mm", args, kwargs)
                generator = torch.Generator(device=actual.device).manual_seed(self.seed)
                repaired = source_aligned_mm_output(before, high, "JOINT", generator=generator)
                actual.copy_(repaired.delivered)
                self.source_l2 = float(torch.linalg.vector_norm(actual.float() - high).item())
                self.calls += 1
                return result

            namespace.mm = wrapped; self.restores.append((namespace, original))
        return self

    def __exit__(self, *unused: Any) -> None:
        for namespace, original in self.restores:
            namespace.mm = original
        if self.calls != 1:
            raise RuntimeError(f"stochastic intervention reached target {self.calls} times")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--sr-repeats", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "results/property/persistence_v1/interventions/phi_sr.json")
    args = parser.parse_args()
    if not 2 <= args.steps <= 16 or args.sr_repeats != 4:
        raise ValueError("protocol requires 2..16 steps and four SR repeats")
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
    natural_path = OrderedVectorPath(total_steps=args.steps, calibration_steps=max(1, args.steps // 2))
    sr_paths = [OrderedVectorPath(total_steps=args.steps, calibration_steps=max(1, args.steps // 2)) for _ in range(args.sr_repeats)]
    rows = []

    def evaluate(state: dict[str, Any], index: int, mode: str, repeat: int = 0) -> tuple[str, torch.Tensor, float]:
        with torch.no_grad(): parameter.copy_(master.to(parameter.dtype))
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        torch.manual_seed(4407 + index); torch.cuda.manual_seed_all(4407 + index)
        model.zero_grad(set_to_none=True); observer: Any = None
        if mode == "fp32": observer = MMRepair(modules, "REPAIR_FP32_CAST_BF16")
        elif mode == "sr": observer = StochasticMM(modules, 8_100_000 + repeat * 1000 + index)
        if observer is None:
            loss = candidate(values); loss.backward(); source_l2 = 0.0
        else:
            with observer:
                loss = candidate(values); loss.backward()
            source_l2 = observer.source_l2 if mode == "sr" else float(observer.local["repair_vs_fp32_l2"])
        return tensor_digest(loss), parameter.grad.detach().float().clone(), source_l2

    for index, state in enumerate(states):
        loss_default, grad_default, _ = evaluate(state, index, "default")
        loss_fp32, grad_fp32, _ = evaluate(state, index, "fp32")
        natural = (grad_default - grad_fp32).mul(-learning_rate); natural_path.add(natural)
        sr_l2 = []
        for repeat in range(args.sr_repeats):
            loss_sr, grad_sr, source_l2 = evaluate(state, index, "sr", repeat)
            if loss_sr != loss_default or loss_fp32 != loss_default:
                raise RuntimeError("backward-only SR intervention changed forward loss")
            delta = (grad_sr - grad_fp32).mul(-learning_rate)
            sr_paths[repeat].add(delta); sr_l2.append(source_l2)
            del grad_sr, delta
        master.add_(grad_default, alpha=-learning_rate)
        rows.append({"step": index + 1, "state_id": str(state.get("state_id", index)),
                     "natural_update_error_l2": float(torch.linalg.vector_norm(natural).item()),
                     "sr_source_l2": sr_l2})
        print(json.dumps({"event": "PHI_SR_STEP", **rows[-1]}), flush=True)
        del grad_default, grad_fp32, natural
        torch.cuda.empty_cache()
    natural = natural_path.finalize(); repeats = [path.finalize() for path in sr_paths]
    sr_a = [row["coherence_amplification"] for row in repeats]
    payload = {
        "schema": "kernel-analyzer-phi-real-sr-intervention-v1",
        "status": "COMPLETE" if args.steps == 16 else "ENGINEERING_DRY_RUN",
        "case_id": "phi4_seq64_lmhead_dx", "steps": args.steps, "sr_repeats": args.sr_repeats,
        "natural_vs_fp32": natural, "sr_vs_fp32": repeats,
        "sr_amplification_mean": sum(sr_a) / len(sr_a),
        "sr_to_natural_amplification_ratio": (sum(sr_a) / len(sr_a)) / max(natural["coherence_amplification"], 1e-30),
        "rows": rows,
        "claim_boundary": "Real backward endpoint, unbiased stochastic BF16 materialization relative to FP32 MM.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
