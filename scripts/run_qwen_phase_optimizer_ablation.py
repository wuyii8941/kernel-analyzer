#!/usr/bin/env python3
"""Use genuine natural-training phase snapshots for a same-state response probe.

This is deliberately not a live 32-step trajectory.  Each phase uses its own
weights, input batch and AdamW moments from the natural training run.  The
candidate and repair are compared at that exact phase state, and no moments
from another phase are installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch._inductor.codecache import PyCodeCache

from kernel_analyzer.reduction_orbit import frozen_permutations
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime
from scripts.run_generated_fp32_screen import load_model
from scripts.run_qwen256_lmhead_property_confirmation import ShapeObserver
from scripts.run_heldout_lmhead_consequence import adam_delta


def vector_stats(values: list[torch.Tensor]) -> dict[str, Any]:
    if not values:
        return {"status": "ABSTAIN_NO_VECTORS"}
    matrix = torch.stack([value.reshape(-1).double() for value in values])
    total = matrix.sum(0)
    energy = float(torch.sum(matrix * matrix).item())
    resultant = float(torch.linalg.vector_norm(total).item())
    return {
        "status": "COMPLETE",
        "states": len(values),
        "resultant_l2": resultant,
        "path_energy_sqrt": energy ** 0.5,
        "coherence_amplification": resultant / max(energy ** 0.5, 1e-30),
        "state_ids_are_disjoint": True,
    }


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--token-stream", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--states-per-phase", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--carrier",
        default="model.layers.0.self_attn.q_proj.weight",
        help="Parameter carrier with a genuine AdamW state in the natural snapshots.",
    )
    args = parser.parse_args()
    if args.states_per_phase < 1:
        raise ValueError("states-per-phase must be positive")

    manifest = json.loads(args.bank_manifest.read_text())
    phase_paths = {
        "early": args.phase_dir / "phase_step_0000.pt",
        "middle": args.phase_dir / "phase_step_0008.pt",
        "late": args.phase_dir / "phase_step_0016.pt",
    }
    missing = [name for name, path in phase_paths.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing genuine phase snapshots: {missing}")

    token_path = args.token_stream or (args.phase_dir.parent / "tokens.int32")
    token_count = int(manifest["token_stream"]["token_count"])
    seq_len = int(manifest["protocol"]["seq_len"])
    batch_size = int(manifest["protocol"]["batch_size"])
    tokens = np.memmap(token_path, dtype=np.int32, mode="r", shape=(token_count,))
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model("qwen", args.model, device)
    model.train()
    parameters = dict(model.named_parameters())
    carrier_name = args.carrier
    if carrier_name not in parameters:
        raise KeyError(f"unknown carrier {carrier_name!r}")
    carrier = parameters[carrier_name]
    carrier_index = next(
        index for index, (name, _) in enumerate(model.named_parameters())
        if name == carrier_name
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
        weight_decay=0.0, foreach=False,
    )

    # Compile once at the natural training shape.  The repair observer only
    # changes the declared lm-head GEMM and counts the target call.
    first_ids = torch.tensor(
        [tokens[:seq_len].tolist()], dtype=torch.long, device=device,
    )
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    model.zero_grad(set_to_none=True)
    candidate(first_ids).backward()
    torch.cuda.synchronize(device)
    # Re-resolve the parameter after compilation.  Some torch versions wrap
    # parameter objects while building the compiled callable; using the
    # pre-compile object can make a valid loaded optimizer state look absent.
    parameters = dict(model.named_parameters())
    carrier = parameters[carrier_name]
    modules = list(PyCodeCache.modules[start:])
    vocab_size = int(model.config.vocab_size)
    hidden_size = int(getattr(model.config, "hidden_size", 2048))
    permutations = frozen_permutations(vocab_size, 2, 20260820)

    # The fixed-shape observer is used for the natural seq_len=1024 graph.
    observer_shape = ((seq_len, vocab_size), (vocab_size, hidden_size))
    def gradient(master: torch.Tensor, ids: torch.Tensor, repair: bool) -> torch.Tensor:
        with torch.no_grad():
            carrier.copy_(master.to(carrier.dtype))
        model.zero_grad(set_to_none=True)
        if repair:
            observer = ShapeObserver(
                modules, "fp32", permutations,
                left_shape=observer_shape[0], right_shape=observer_shape[1],
            )
            with observer:
                candidate(ids).backward()
            if observer.calls != 1:
                raise RuntimeError(f"expected one lm-head repair call, saw {observer.calls}")
        else:
            candidate(ids).backward()
        torch.cuda.synchronize(device)
        if carrier.grad is None:
            raise RuntimeError("declared carrier gradient is absent")
        return carrier.grad.detach().float().clone()

    phase_results: dict[str, Any] = {}
    all_state_ids: list[str] = []
    for phase, snapshot_path in phase_paths.items():
        snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        if snapshot.get("schema") != "kernel-analyzer-long-horizon-resume-v1":
            raise RuntimeError(f"{phase}: invalid natural snapshot schema")
        model.load_state_dict(snapshot["model"], strict=True)
        # Read the serialized state by the stable parameter index.  This avoids
        # a torch.compile/nightly identity quirk where optimizer.load_state_dict
        # can expose an equivalent state under a different Python Parameter key.
        serialized_state = snapshot["optimizer"]["state"]
        opt_state = serialized_state.get(carrier_index)
        if opt_state is None and int(snapshot["step"]) == 0:
            # Step zero is a genuine cold-start state: no optimizer update has
            # happened yet, so the real moments are exactly zero rather than
            # missing data.
            opt_state = {
                "step": 0,
                "exp_avg": torch.zeros_like(carrier, device=device),
                "exp_avg_sq": torch.zeros_like(carrier, device=device),
            }
        if opt_state is None:
            opt_state = serialized_state.get(str(carrier_index))
        if opt_state is None:
            raise RuntimeError(
                f"{phase}: carrier {carrier_name} has no captured AdamW moments; "
                "choose a parameter present in the natural optimizer state"
            )
        opt_state = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in opt_state.items()
        }
        first = opt_state["exp_avg"].detach().clone()
        second = opt_state["exp_avg_sq"].detach().clone()
        prior_step = int(opt_state["step"].item() if torch.is_tensor(opt_state["step"]) else opt_state["step"])
        # Use the batch immediately following the recorded natural phase.  The
        # late phase uses the final available batch, never a fabricated state.
        phase_step = int(snapshot["step"])
        start_token = min(phase_step * batch_size * seq_len, token_count - args.states_per_phase * seq_len)
        phase_ids = []
        state_ids = []
        for index in range(args.states_per_phase):
            begin = start_token + index * seq_len
            phase_ids.append(torch.tensor([tokens[begin:begin + seq_len].tolist()], dtype=torch.long, device=device))
            state_ids.append(f"qwen-natural-{phase}-{phase_step:04d}-{index:02d}")
        all_state_ids.extend(state_ids)
        base = carrier.detach().float().clone()
        gradient_vectors: list[torch.Tensor] = []
        sgd_vectors: list[torch.Tensor] = []
        adam_vectors: list[torch.Tensor] = []
        reset_vectors: list[torch.Tensor] = []
        rows = []
        for index, (ids, state_id) in enumerate(zip(phase_ids, state_ids)):
            gc = gradient(base, ids, False)
            gr = gradient(base, ids, True)
            difference = gc - gr
            sgd = -args.learning_rate * difference
            candidate_update, _, _ = adam_delta(
                gc, first, second, prior_step + 1, learning_rate=args.learning_rate,
            )
            repair_update, _, _ = adam_delta(
                gr, first, second, prior_step + 1, learning_rate=args.learning_rate,
            )
            reset_candidate, _, _ = adam_delta(
                gc, torch.zeros_like(first), torch.zeros_like(second), 1,
                learning_rate=args.learning_rate,
            )
            reset_repair, _, _ = adam_delta(
                gr, torch.zeros_like(first), torch.zeros_like(second), 1,
                learning_rate=args.learning_rate,
            )
            gradient_vectors.append(difference.cpu())
            sgd_vectors.append(sgd.cpu())
            adam_vectors.append((candidate_update - repair_update).cpu())
            reset_vectors.append((reset_candidate - reset_repair).cpu())
            rows.append({
                "state_id": state_id,
                "natural_phase_step": phase_step,
                "optimizer_prior_step": prior_step,
                "gradient_l2": float(torch.linalg.vector_norm(difference).item()),
                "captured_adamw_l2": float(torch.linalg.vector_norm(candidate_update - repair_update).item()),
            })
            del gc, gr, difference, sgd, candidate_update, repair_update
        phase_results[phase] = {
            "snapshot": str(snapshot_path),
            "snapshot_sha256": digest_file(snapshot_path),
            "natural_step": phase_step,
            "optimizer_prior_step": prior_step,
            "state_ids": state_ids,
            "arms": {
                "gradient_difference": vector_stats(gradient_vectors),
                "stateless_sgd": vector_stats(sgd_vectors),
                "captured_adamw_moments": vector_stats(adam_vectors),
                "moment_reset_each_step": vector_stats(reset_vectors),
            },
            "rows": rows,
            "claim_boundary": "Each phase uses its own natural weights, inputs and AdamW moments; this is a same-state response probe, not a live persistence trajectory.",
        }
        del snapshot, first, second, base, phase_ids
        torch.cuda.empty_cache()

    if len(all_state_ids) != len(set(all_state_ids)):
        raise RuntimeError("natural phase state IDs are not disjoint")
    result = {
        "schema": "kernel-analyzer-direct-persistence-v4-natural-phase-response-v1",
        "status": "COMPLETE_GENUINE_EARLY_MIDDLE_LATE_PHASES",
        "case_id": "qwen3_1p7b_lmhead_dx_natural_phase_response",
        "model": str(args.model),
        "carrier": carrier_name,
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "betas": [0.9, 0.95], "weight_decay": 0.0},
        "phase_results": phase_results,
        "state_ids_disjoint": True,
        "claim_boundary": "Natural early/middle/late weights, inputs and optimizer moments are measured without cross-phase mixing. This establishes phase-conditioned response sensitivity; it is not a universal persistence conclusion.",
    }
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
