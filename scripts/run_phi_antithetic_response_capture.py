#!/usr/bin/env python3
"""Capture exact/representable +/- MM residual responses for the Phi case.

The target is the already closed Phi lm-head dX MM boundary.  The script
reuses the frozen compiler release and records only the declared final-norm
gradient carrier plus the target endpoint residual.  A negative residual is
accepted only when its BF16 materialization is within the predeclared
representability floor; otherwise the state is marked unresolved.
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
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "archive/round1_code/src")]

from kernel_analyzer.bias_formation_v21 import FormationPolicy, summarize_state_vectors  # noqa: E402
from scripts.generated_contrast_observer import _source_identity  # noqa: E402
from scripts.generated_nontriton_fp32_observer import fp32_external_reference  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import MMRepair, TARGET_LEFT_SHAPE, TARGET_SHA  # noqa: E402
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402


MODEL = Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct")
BANK = ROOT / "results/coverage/phi4_seq64_input_bank.json"
DEFAULT_RELEASE = ROOT / "results/coverage/runtime_releases/phi4_seq64_r1"
DEFAULT_OUTPUT = ROOT / "results/property/joint_bias_formation_v1/phi_antithetic_response_capture.json"
CARRIER = "model.norm.weight"
REPRESENTABILITY_FLOOR = 0.0


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class AntitheticMMRepair(MMRepair):
    """Extend the existing target observer with exact/representable +/- arms."""

    def __init__(self, modules: list[Any], mode: str) -> None:
        super().__init__(modules, mode)
        if mode not in {"REPAIR_FP32_CAST_BF16", "ANTITHETIC_PLUS", "ANTITHETIC_MINUS"}:
            raise ValueError(mode)
        self.representability_error = None
        self.delivered_vector = None

    def __enter__(self) -> "AntitheticMMRepair":
        seen = set()
        for module in self.modules:
            namespace = getattr(module, "extern_kernels", None)
            if namespace is None or id(namespace) in seen:
                continue
            seen.add(id(namespace))
            original = namespace.mm

            def wrapped(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
                _, _, source_digest = _source_identity()
                if source_digest != TARGET_SHA and tuple(args[0].shape) != TARGET_LEFT_SHAPE:
                    return _original(*args, **kwargs)
                result = _original(*args, **kwargs)
                candidate = kwargs.get("out", result)
                if not isinstance(candidate, torch.Tensor):
                    raise RuntimeError("target MM has no tensor output")
                before = candidate.detach().clone()
                # Use the same higher-precision external reference as the
                # closed Phi repair.  Calling torch.mm here would simply
                # replay the BF16 path and make the natural residual zero.
                replacement = fp32_external_reference("mm", args, kwargs)
                cast = replacement.to(candidate.dtype)
                natural_delta = before.float() - cast.float()
                if self.mode == "REPAIR_FP32_CAST_BF16":
                    delivered = cast
                elif self.mode == "ANTITHETIC_PLUS":
                    delivered = (cast.float() + natural_delta).to(candidate.dtype)
                else:
                    target = cast.float() - natural_delta
                    delivered = target.to(candidate.dtype)
                    self.representability_error = float(
                        torch.linalg.vector_norm(delivered.float() - target).item()
                        / max(torch.linalg.vector_norm(natural_delta).item(), 1e-30)
                    )
                self.natural_vector = natural_delta.detach().float().cpu().numpy().reshape(-1).copy()
                self.delivered_vector = (delivered.float() - cast.float()).detach().cpu().numpy().reshape(-1).copy()
                candidate.copy_(delivered)
                delta = before.float() - delivered.float()
                self.local = {
                    "coordinates": int(delta.numel()),
                    "changed_coordinates": int(torch.count_nonzero(delta).item()),
                    "l2": float(torch.linalg.vector_norm(delta).item()),
                    "natural_l2": float(torch.linalg.vector_norm(natural_delta).item()),
                    "representability_error": self.representability_error,
                }
                self.local_vector = delta.detach().float().cpu().numpy().reshape(-1)
                self.calls += 1
                return result

            namespace.mm = wrapped
            self.restores.append((namespace, original))
        return self


def run_branch(model: torch.nn.Module, candidate: Any, values: torch.Tensor,
               modules: list[Any], mode: str) -> dict[str, Any]:
    torch.manual_seed(46000)
    torch.cuda.manual_seed_all(46000)
    model.zero_grad(set_to_none=True)
    observer = AntitheticMMRepair(modules, mode)
    with observer:
        loss = candidate(values)
        loss.backward()
    torch.cuda.synchronize(values.device)
    parameter = dict(model.named_parameters()).get(CARRIER)
    if parameter is None or parameter.grad is None:
        raise RuntimeError("Phi declared carrier gradient is absent")
    return {
        "loss_digest": tensor_digest(loss),
        "gradient": parameter.grad.detach().float().cpu().numpy().reshape(-1).copy(),
        "observer": observer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()
    if args.states < 1 or args.states > 16:
        raise ValueError("states must be in [1,16]")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use the host GPU")
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    states = bank.get("states", bank.get("records"))[:args.states]
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model("phi", MODEL, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    validate_release(wrapper_modules(modules), json.loads((args.release_dir / "capture.json").read_text()))
    rows = []
    plus_vectors = []
    minus_vectors = []
    for index, state in enumerate(states):
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        repair = run_branch(model, candidate, values, modules, "REPAIR_FP32_CAST_BF16")
        plus = run_branch(model, candidate, values, modules, "ANTITHETIC_PLUS")
        minus = run_branch(model, candidate, values, modules, "ANTITHETIC_MINUS")
        plus_delta = plus["gradient"] - repair["gradient"]
        minus_delta = minus["gradient"] - repair["gradient"]
        plus_vectors.append(plus_delta)
        minus_vectors.append(minus_delta)
        representability = minus["observer"].representability_error
        state_id = str(state.get("state_id", state.get("sequence_id", index)))
        rows.append({
            "state_id": state_id,
            "plus_local_l2": plus["observer"].local["l2"],
            "minus_local_l2": minus["observer"].local["l2"],
            "natural_local_l2": repair["observer"].local["l2"],
            "plus_gradient_delta_l2": float(np.linalg.norm(plus_delta)),
            "minus_gradient_delta_l2": float(np.linalg.norm(minus_delta)),
            "representability_error": representability,
            "exact_antithetic": representability is not None and representability <= REPRESENTABILITY_FLOOR,
            "plus_loss_equals_repair": plus["loss_digest"] == repair["loss_digest"],
            "minus_loss_equals_repair": minus["loss_digest"] == repair["loss_digest"],
        })
        del values, repair, plus, minus, plus_delta, minus_delta
        torch.cuda.empty_cache()
        print(json.dumps({"event": "PHI_ANTITHETIC_STATE_COMPLETE", "state": index, "state_id": state_id}, sort_keys=True), flush=True)
    exact = all(row["exact_antithetic"] for row in rows)
    plus_even = [0.5 * (p + m) for p, m in zip(plus_vectors, minus_vectors)]
    plus_odd = [0.5 * (p - m) for p, m in zip(plus_vectors, minus_vectors)]
    policy = FormationPolicy(min_states=min(16, args.states), bootstrap_samples=2000)
    state_ids = [row["state_id"] for row in rows]
    even_cert = summarize_state_vectors(
        plus_even, state_ids=state_ids, layer="RESPONSE_EVEN",
        partition="engineering", policy=policy,
    ).as_dict()
    odd_cert = summarize_state_vectors(
        plus_odd, state_ids=state_ids, layer="RESPONSE_ODD",
        partition="engineering", policy=policy,
    ).as_dict()
    payload = {
        "schema": "kernel-analyzer-phi-antithetic-response-v1",
        "status": "COMPLETE" if exact else "UNRESOLVED_REPRESENTABILITY",
        "case_id": "phi4_lm_head_dx_seq64",
        "state_count": len(rows),
        "representability_floor": REPRESENTABILITY_FLOOR,
        "exact_antithetic_all_states": exact,
        "rows": rows,
        "response_even_population": even_cert,
        "response_odd_population": odd_cert,
        "claim_boundary": "Engineering capture only; no causal intervention or universal property claim. A nonzero BF16 reflection error fails closed.",
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": payload["status"], "exact_antithetic_all_states": exact}, sort_keys=True))


if __name__ == "__main__":
    main()
