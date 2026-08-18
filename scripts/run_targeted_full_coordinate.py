#!/usr/bin/env python3
"""Run a full-coordinate T1 pilot for one frozen generated callsite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from scripts.analyze_generated_fp32_screen import bootstrap, u_statistic  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import (  # noqa: E402
    file_digest, gradient_digest, load_model, tensor_digest,
)
from scripts.targeted_external_intervention import TargetedExternalIntervention  # noqa: E402


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def nondegenerate_bootstrap_counts(states: int, draws: int, seed: int) -> np.ndarray:
    """Return exactly ``draws`` cluster resamples containing >=2 source states."""

    rng = np.random.default_rng(seed)
    accepted = []
    while sum(row.shape[0] for row in accepted) < draws:
        needed = draws - sum(row.shape[0] for row in accepted)
        samples = rng.integers(0, states, size=(max(needed * 2, 32), states))
        valid = np.any(samples != samples[:, :1], axis=1)
        samples = samples[valid][:needed]
        if samples.size == 0:
            continue
        counts = np.zeros((samples.shape[0], states), dtype=np.float64)
        np.add.at(
            counts,
            (np.repeat(np.arange(samples.shape[0]), states), samples.reshape(-1)),
            1.0,
        )
        accepted.append(counts)
    return np.concatenate(accepted, axis=0)[:draws]


def validate_release(modules: list[tuple[Any, str]], capture: dict[str, Any]) -> None:
    observed = []
    for module, phase in modules:
        observed.append((phase.upper(), hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()))
    expected = [(row["phase"], row["sha256"]) for row in capture["modules"]]
    if observed != expected:
        raise RuntimeError(
            "recompiled wrapper bytes differ from frozen release; create and screen a new release"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument(
        "--queue", type=Path,
        default=ROOT / "results/coverage/bias_candidate_queue.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    args = parser.parse_args()
    if args.states < 2 or args.repeat != 2:
        raise ValueError("T1 requires at least two states and exactly two repeats")
    queue = json.loads(args.queue.read_text())
    matches = [row for row in queue["candidates"] if row["candidate_id"] == args.candidate_id]
    if len(matches) != 1:
        raise RuntimeError("candidate ID is absent or non-unique")
    selected = matches[0]
    target = selected["exact_generated_call"]
    capture = json.loads((args.release_dir / "capture.json").read_text())
    if capture["result_sha256"] != json.loads((args.release_dir / "capture.json").read_text())["result_sha256"]:
        raise RuntimeError("release capture changed while reading")
    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if len(states) < args.states:
        raise RuntimeError("input bank is shorter than requested T1 population")
    if file_digest(args.input_bank) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("input bank does not match the frozen runtime release")

    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model("mamba", args.model, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm_tokens = states[0].get("token_ids", states[0].get("input_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    raw_modules = list(PyCodeCache.modules[start:])
    wrappers = wrapper_modules(raw_modules)
    validate_release(wrappers, capture)

    rows = []
    state_errors = []
    for state_index, state in enumerate(states[: args.states]):
        state_id = str(state.get("sequence_id", state.get("state_id", state_index)))
        tokens = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        seed = 24000 + state_index
        repeat_rows = []
        baseline_identity = None
        for repeat in range(args.repeat):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model.zero_grad(set_to_none=True)
            observer = TargetedExternalIntervention(
                modules=raw_modules, target=target, mode="OBSERVE",
            )
            with observer:
                loss = candidate(values)
                loss.backward()
            torch.cuda.synchronize(device)
            identity = {
                "loss": tensor_digest(loss),
                "gradients": gradient_digest(model),
            }
            if baseline_identity is None:
                baseline_identity = identity
            elif identity != baseline_identity:
                raise RuntimeError(f"runtime instability in state {state_id}")
            summary = observer.summary()
            repeat_rows.append({"repeat": repeat, "identity": identity, "target": summary})
        left = np.asarray(repeat_rows[0]["target"]["signed_error"], dtype=np.float64)
        right = np.asarray(repeat_rows[1]["target"]["signed_error"], dtype=np.float64)
        if not np.array_equal(left, right):
            raise RuntimeError(f"target error changed across repeats in state {state_id}")
        state_errors.append((left + right) / 2.0)
        rows.append({
            "state_id": state_id,
            "token_ids_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            "repeats": repeat_rows,
        })
        del values
        torch.cuda.empty_cache()
        partial = {
            "schema": "kernel-analyzer-targeted-full-coordinate-t1-v1",
            "status": "RUNNING",
            "candidate_id": args.candidate_id,
            "states_complete": len(rows),
            "rows": rows,
        }
        write(args.output, partial)
        print(json.dumps({"event": "STATE_COMPLETE", "state": state_id}), flush=True)
    errors = np.stack(state_errors)
    counts = nondegenerate_bootstrap_counts(args.states, args.bootstrap_draws, 14031)
    statistic = u_statistic(errors)
    confidence = bootstrap(errors, counts)
    output = {
        "schema": "kernel-analyzer-targeted-full-coordinate-t1-v1",
        "status": "COMPLETE_FULL_COORDINATE_T1_PILOT" if args.states < 32 else "COMPLETE_FULL_COORDINATE_T1",
        "candidate_id": args.candidate_id,
        "queue_sha256": queue["result_sha256"],
        "release_capture_sha256": capture["result_sha256"],
        "input_bank_sha256": file_digest(args.input_bank),
        "states": args.states,
        "repeats": args.repeat,
        "coordinates": int(errors.shape[1]),
        "cross_state_inner_product_u": statistic,
        "cluster_bootstrap_95": confidence,
        "directional_positive": confidence["lower_95"] > 0.0,
        "runtime_repeat_exact": True,
        "rows": rows,
        "gates": {
            "exact_frozen_wrapper_identity": True,
            "all_coordinates_observed": int(errors.shape[1]) == 1536,
            "runtime_repeat_exact": True,
            "directional_t1": confidence["lower_95"] > 0.0,
            "independent_32_state_population": args.states == 32,
        },
        "claim_boundary": (
            "T1 only. A positive pilot is not a case and does not establish generated backward "
            "binding, causal repair, a complete carrier, or accumulation."
        ),
    }
    output["result_sha256"] = canonical_hash(output)
    write(args.output, output)
    print(json.dumps({
        "event": "T1_COMPLETE", "states": args.states, "coordinates": int(errors.shape[1]),
        "u": statistic, "lower_95": confidence["lower_95"],
    }), flush=True)


if __name__ == "__main__":
    main()
