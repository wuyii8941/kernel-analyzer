#!/usr/bin/env python3
"""Same-protocol Phi source intervention under evolving cold-start AdamW."""

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

from kernel_analyzer.persistence_property import CompleteTreeGramPath  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_heldout_lmhead_consequence import adam_delta  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import MMRepair  # noqa: E402
from scripts.run_phi_mm_sr_intervention import StochasticMM  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, choices=(2, 16, 32), required=True)
    parser.add_argument("--sr-repeats", type=int, default=4, choices=(4,))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--state-path", choices=("natural", "repair"), default="natural",
        help="Which arm advances the shared master/moments between matched states.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    bank = json.loads((ROOT / "results/coverage/phi4_seq64_input_bank.json").read_text())
    states = bank.get("states", bank.get("records"))[:args.steps]
    if len(states) != args.steps:
        raise RuntimeError("frozen Phi state bank is incomplete")
    state_ids = [str(row.get("state_id", index)) for index, row in enumerate(states)]
    configure_candidate_runtime(24_000)
    model = load_model("phi", Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct"), device)
    model.eval(); carrier = model.model.norm.weight
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    # Wrapper byte hashes are intentionally not used here: recompilation may
    # change generated source without changing this uniquely shaped endpoint.
    # Every intervention arm below must instead hit the exact target once.

    initial = carrier.detach().float().clone()
    master = initial.clone(); first = torch.zeros_like(initial); second = torch.zeros_like(initial)
    paths = {
        "natural": CompleteTreeGramPath(total_steps=args.steps, max_resident_bytes=64 << 20),
        "sham": CompleteTreeGramPath(total_steps=args.steps, max_resident_bytes=64 << 20),
    }
    for repeat in range(args.sr_repeats):
        paths[f"sr_{repeat}"] = CompleteTreeGramPath(total_steps=args.steps, max_resident_bytes=64 << 20)
    rows = []

    def gradient(state: dict[str, Any], mode: str, index: int, repeat: int = 0) -> tuple[str, torch.Tensor, Any]:
        with torch.no_grad(): carrier.copy_(master.to(carrier.dtype))
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        seed = 24_000 + index
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        observer: Any = None
        if mode == "repair": observer = MMRepair(modules, "REPAIR_FP32_CAST_BF16")
        elif mode == "sham": observer = MMRepair(modules, "SHAM")
        elif mode == "sr": observer = StochasticMM(modules, 8_100_000 + repeat * 1000 + index)
        if observer is None:
            loss = candidate(values); loss.backward()
        else:
            with observer:
                loss = candidate(values); loss.backward()
        torch.cuda.synchronize(device)
        if carrier.grad is None or not torch.isfinite(carrier.grad).all():
            raise RuntimeError("carrier gradient is missing or nonfinite")
        return tensor_digest(loss), carrier.grad.detach().float().clone(), observer

    for index, state in enumerate(states):
        step = index + 1
        loss_n, grad_n, _ = gradient(state, "natural", index)
        loss_r, grad_r, _ = gradient(state, "repair", index)
        loss_s, grad_s, sham_observer = gradient(state, "sham", index)
        if loss_n != loss_r or loss_n != loss_s:
            raise RuntimeError("backward-only intervention changed forward loss")
        if sham_observer.local["changed_coordinates"] != 0 or not torch.equal(grad_n, grad_s):
            raise RuntimeError("matched sham did not reproduce the natural arm")
        update_n, next_first, next_second = adam_delta(
            grad_n, first, second, step, learning_rate=1e-4, beta1=0.9, beta2=0.95,
        )
        update_r, _, _ = adam_delta(
            grad_r, first, second, step, learning_rate=1e-4, beta1=0.9, beta2=0.95,
        )
        natural = update_n - update_r
        paths["natural"].add({"carrier": natural.cpu()})
        paths["sham"].add({"carrier": natural.cpu()})
        sr_norms = []
        for repeat in range(args.sr_repeats):
            loss_sr, grad_sr, _ = gradient(state, "sr", index, repeat)
            if loss_sr != loss_n:
                raise RuntimeError("stochastic rounding changed forward loss")
            update_sr, _, _ = adam_delta(
                grad_sr, first, second, step, learning_rate=1e-4, beta1=0.9, beta2=0.95,
            )
            delta = update_sr - update_r
            paths[f"sr_{repeat}"].add({"carrier": delta.cpu()})
            sr_norms.append(float(torch.linalg.vector_norm(delta).item()))
        rows.append({
            "step": step, "state_id": state_ids[index],
            "natural_update_error_l2": float(torch.linalg.vector_norm(natural).item()),
            "sr_update_error_l2": sr_norms,
        })
        if args.state_path == "natural":
            master.add_(update_n); first, second = next_first, next_second
        else:
            master.add_(update_r)
            _, first, second = adam_delta(
                grad_r, first, second, step, learning_rate=1e-4, beta1=0.9, beta2=0.95,
            )
        if not args.quiet:
            print(json.dumps({"event": "PHI_ADAMW_SR_STEP", **rows[-1]}), flush=True)

    metrics = {
        name: path.finalize(state_ids=state_ids, sign_flip_draws=4000,
                            seed=20_260_824 + offset)
        for offset, (name, path) in enumerate(paths.items())
    }
    natural_a = float(metrics["natural"]["coherence_amplification"])
    sr_a = [float(metrics[f"sr_{repeat}"]["coherence_amplification"])
            for repeat in range(args.sr_repeats)]
    payload = {
        "schema": "kernel-analyzer-phi-adamw-sr-intervention-v1",
        "status": "COMPLETE" if args.steps == 32 else "ENGINEERING_GATE",
        "case_id": "phi4_seq64_lmhead_dx", "steps": args.steps,
        "protocol": {
            "optimizer": {"name": "AdamW", "learning_rate": 1e-4,
                          "betas": [0.9, 0.95], "epsilon": 1e-8,
                          "initial_moments": "ZERO_THEN_EVOLVED_NORMALLY"},
            "state_order": state_ids, "carrier": "model.norm.weight",
            "state_path": (
                "NATURAL_CANDIDATE_ADVANCES_MASTER_AND_MOMENTS"
                if args.state_path == "natural"
                else "FP32_CAST_REPAIR_ADVANCES_MASTER_AND_MOMENTS"
            ),
            "arms": ["DETERMINISTIC_BF16", "FP32_CAST_REPAIR", "NO_OP_SHAM",
                     "STOCHASTIC_ROUNDING_X4"],
        },
        "metrics": metrics, "natural_A": natural_a,
        "sr_A": sr_a, "sr_A_mean": sum(sr_a) / len(sr_a), "rows": rows,
        "claim_boundary": (
            "This is a common-state direct effective-update intervention under the same "
            "cold-start AdamW protocol as the retained Phi direct-persistence result. It is "
            "not a full candidate-versus-repair feedback trajectory."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
