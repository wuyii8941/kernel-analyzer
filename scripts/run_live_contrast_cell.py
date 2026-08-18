#!/usr/bin/env python3
"""Run complete-coordinate precision/optimization contrasts for one cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.streaming import direction_certificate_from_vector_files  # noqa: E402
from scripts.generated_contrast_observer import BatchedGeneratedContrastObserver  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import (  # noqa: E402
    file_digest, gradient_digest, load_model, tensor_digest,
)
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402


ARCHITECTURES = {
    "qwen": "qwen3_1p7b",
    "mamba": "mamba_130m",
    "phi": "phi4_mini_3p8b",
    "deepseek8": "deepseek_r1_0528_qwen3_8b",
}

MINIMUM_FREE_BYTES = 50 * 1024**3


def directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.iterdir() if path.is_file())


def enforce_disk_reserve(temp: Path, requested_states: int, completed_states: int) -> Mapping[str, int]:
    """Fail closed if retained vectors are projected to consume the reserve."""
    free = shutil.disk_usage(temp.parent).free
    retained = directory_bytes(temp)
    projected_remaining = 0
    if completed_states:
        bytes_per_state = (retained + completed_states - 1) // completed_states
        projected_remaining = bytes_per_state * max(0, requested_states - completed_states)
    if free - projected_remaining < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            "live contrast projected storage violates 50 GiB reserve: "
            f"free={free}, retained={retained}, projected_remaining={projected_remaining}"
        )
    return {
        "free_bytes": free,
        "retained_bytes": retained,
        "projected_remaining_bytes": projected_remaining,
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
    }


def runtime_environment() -> dict[str, str]:
    import transformers
    import triton

    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "transformers_version": transformers.__version__,
        "triton_version": triton.__version__,
    }


def bind_runtime_environment(release: Path, observed: Mapping[str, str]) -> None:
    """Bind an old source-only release after its wrapper bytes validate exactly."""
    path = release / "environment.json"
    payload = {
        "schema": "kernel-analyzer-runtime-environment-v1",
        "status": "BOUND_AFTER_EXACT_WRAPPER_SOURCE_VALIDATION",
        "environment": dict(observed),
    }
    payload["result_sha256"] = canonical(payload)
    if path.exists():
        expected = json.loads(path.read_text())
        if expected != payload:
            raise RuntimeError("runtime environment differs from frozen release")
        return
    write(path, payload)


def validate_declared_environment(release: Path, observed: Mapping[str, str]) -> None:
    path = release / "environment.json"
    if not path.exists():
        return
    expected = json.loads(path.read_text()).get("environment")
    if expected != dict(observed):
        raise RuntimeError("runtime environment differs from frozen release")


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class RepeatSpool:
    def __init__(self, root: Path, state_id: str, repeat: int, chunk_elements: int) -> None:
        self.root = root
        self.state_id = state_id
        self.repeat = repeat
        self.chunk_elements = chunk_elements
        self.records: dict[tuple[str, str], dict[str, Any]] = {}

    def sink(
        self, candidate_id: str, contrast: str, endpoint: str,
        tensor: torch.Tensor, metadata: Mapping[str, Any],
    ) -> None:
        key = (candidate_id, contrast)
        if key in self.records:
            raise RuntimeError("contrast emitted more than once: %s" % (key,))
        safe = hashlib.sha256((candidate_id + "\0" + contrast + "\0" + self.state_id).encode()).hexdigest()
        path = self.root / f"{safe}.r{self.repeat}.f32"
        digest = hashlib.sha256()
        finite = True
        count = 0
        maximum = 0.0
        flat = tensor.detach().reshape(-1)
        with path.open("wb") as handle:
            for start in range(0, flat.numel(), self.chunk_elements):
                values = flat[start:start + self.chunk_elements].float().cpu().numpy()
                finite = finite and bool(np.isfinite(values).all())
                maximum = max(maximum, float(np.max(np.abs(values))) if values.size else 0.0)
                encoded = values.tobytes(order="C")
                handle.write(encoded)
                digest.update(encoded)
                count += values.size
        constant_zero = maximum == 0.0 and finite
        if constant_zero:
            path.unlink()
        self.records[key] = {
            "path": None if constant_zero else str(path),
            "storage_dtype": "float32", "constant_zero": constant_zero,
            "sha256": digest.hexdigest(), "coordinates": count,
            "finite": finite, "max_abs": maximum, "endpoint": endpoint,
            "metadata": dict(metadata),
        }


def contrast_payload(kind: str, metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    if kind == "PRECISION":
        return {
            "kind": "PRECISION_SAME_SEMANTICS",
            "low_dtype": metadata["low_dtype"],
            "high_dtype": metadata["high_dtype"],
            "semantic_boundary_exact": metadata["semantic_boundary_exact"],
            "semantic_program_sha256": metadata["semantic_program_sha256"],
            "low_arm_program_sha256": metadata["low_arm_program_sha256"],
            "high_arm_program_sha256": metadata["high_arm_program_sha256"],
        }
    if kind == "OPTIMIZATION":
        return {
            "kind": "OPTIMIZATION_SAME_DTYPE",
            "candidate_dtype": metadata["candidate_dtype"],
            "reference_dtype": metadata["low_dtype"],
            "semantic_boundary_exact": metadata["semantic_boundary_exact"],
            "candidate_program_sha256": metadata["candidate_program_sha256"],
            "reference_program_sha256": metadata["low_arm_program_sha256"],
        }
    return {
        "kind": "TOTAL_CANDIDATE_LOW_MINUS_REFERENCE_HIGH",
        "candidate_dtype": metadata["candidate_dtype"],
        "reference_dtype": metadata["high_dtype"],
        "semantic_boundary_exact": metadata["semantic_boundary_exact"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), required=True)
    parser.add_argument("--sequence-length", type=int, choices=(64, 128, 256), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--queue", type=Path, default=ROOT / "results/coverage/bias_candidate_queue.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, default=Path("/data1/tzh/cache/kernel_analyzer_contrasts"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    parser.add_argument("--chunk-elements", type=int, default=1_048_576)
    parser.add_argument("--keep-temporary-vectors", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.repeat != 2 or args.states < 2:
        raise ValueError("live contrast requires two repeats and at least two states")
    if shutil.disk_usage(args.temp_root.parent).free < MINIMUM_FREE_BYTES:
        raise RuntimeError("less than 50 GiB free under contrast temporary root")

    queue = json.loads(args.queue.read_text())
    targets = [
        row for row in queue["candidates"]
        if row["architecture"] == ARCHITECTURES[args.architecture]
        and int(row["sequence_length"]) == args.sequence_length
        and row["claim"] == "PENDING_EXHAUSTIVE_FULL_COORDINATE_AND_FB_BINDING"
    ]
    if not targets:
        raise RuntimeError("cell has no pending exhaustive targets")
    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if len(states) < args.states:
        raise RuntimeError("input bank is shorter than requested population")
    capture = json.loads((args.release_dir / "capture.json").read_text())
    validate_declared_environment(args.release_dir, runtime_environment())
    if file_digest(args.input_bank) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("input bank does not bind to the frozen release")

    campaign_id = canonical({
        "architecture": args.architecture, "sequence_length": args.sequence_length,
        "queue_sha256": queue["result_sha256"], "capture_sha256": capture["result_sha256"],
        "states": args.states, "repeat": args.repeat,
    })[:20]
    temp = args.temp_root / campaign_id
    temp.mkdir(parents=True, exist_ok=True)
    partial = {
        "schema": "kernel-analyzer-live-contrast-cell-v1", "status": "RUNNING",
        "architecture": args.architecture, "sequence_length": args.sequence_length,
        "campaign_id": campaign_id, "queue_sha256": queue["result_sha256"],
        "release_capture_sha256": capture["result_sha256"],
        "input_bank_sha256": file_digest(args.input_bank),
        "candidate_ids": [row["candidate_id"] for row in targets], "states": {},
        "runtime_environment": runtime_environment(),
    }
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        if existing.get("campaign_id") != campaign_id:
            raise RuntimeError("output belongs to another campaign")
        if existing.get("status") == "COMPLETE_LIVE_FULL_COORDINATE_CONTRASTS":
            print(json.dumps({"event": "ALREADY_COMPLETE", "output": str(args.output)}))
            return
        partial = existing

    partial["storage_preflight"] = enforce_disk_reserve(
        temp, args.states, len(partial["states"])
    )

    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend="inductor",
        fullgraph=not bool(capture.get("allow_graph_breaks", False)), dynamic=False,
    )
    warm_tokens = states[0].get("token_ids", states[0].get("input_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    validate_release(wrapper_modules(modules), capture)
    bind_runtime_environment(args.release_dir, runtime_environment())

    for index, state in enumerate(states[:args.states]):
        state_id = str(state.get("sequence_id", state.get("state_id", index)))
        prior = partial["states"].get(state_id)
        if prior and all(row.get("constant_zero") or Path(row["path"]).is_file()
                         for row in prior["vectors"].values()):
            continue
        partial["storage_preflight"] = enforce_disk_reserve(
            temp, args.states, len(partial["states"])
        )
        tokens = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        seed = 24000 + index
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        baseline_loss = candidate(values); baseline_loss.backward(); torch.cuda.synchronize(device)
        baseline = {"loss": tensor_digest(baseline_loss), "gradients": gradient_digest(model)}
        repeats = []
        for repeat in range(args.repeat):
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model.zero_grad(set_to_none=True)
            spool = RepeatSpool(temp, state_id, repeat, args.chunk_elements)
            observer = BatchedGeneratedContrastObserver(modules=modules, targets=targets, sink=spool.sink)
            with observer:
                loss = candidate(values); loss.backward()
            torch.cuda.synchronize(device); observer.validate()
            identity = {"loss": tensor_digest(loss), "gradients": gradient_digest(model)}
            if identity != baseline:
                raise RuntimeError("contrast observer changed model endpoints: %s" % state_id)
            repeats.append(spool.records)
        vectors = {}
        for key, left in repeats[0].items():
            right = repeats[1].get(key)
            if right is None or left["sha256"] != right["sha256"]:
                raise RuntimeError("contrast repeat mismatch: %s/%s" % (state_id, key))
            if right.get("path"):
                Path(right["path"]).unlink(missing_ok=True)
            vectors["\0".join(key)] = left
        for target in targets:
            precision_key = target["candidate_id"] + "\0PRECISION"
            total_key = target["candidate_id"] + "\0TOTAL"
            precision_row, total_row = vectors[precision_key], vectors[total_key]
            if precision_row["sha256"] == total_row["sha256"]:
                if total_row.get("path") and total_row["path"] != precision_row.get("path"):
                    Path(total_row["path"]).unlink(missing_ok=True)
                total_row["path"] = precision_row.get("path")
                total_row["constant_zero"] = precision_row.get("constant_zero", False)
                total_row["shared_with"] = precision_key
        partial["states"][state_id] = {"endpoint_identity": baseline, "vectors": vectors}
        partial["storage_preflight"] = enforce_disk_reserve(
            temp, args.states, len(partial["states"])
        )
        write(args.output, partial)
        del values
        torch.cuda.empty_cache()
        print(json.dumps({"event": "STATE_COMPLETE", "state": state_id}), flush=True)

    results = []
    for target in targets:
        for kind in ("PRECISION", "OPTIMIZATION", "TOTAL"):
            metadata = None; finite = True; maximum = 0.0; coordinates = None
            vector_rows = []
            for state_id, state in partial["states"].items():
                row = state["vectors"][target["candidate_id"] + "\0" + kind]
                vector_rows.append({**row, "state_id": state_id})
                metadata = row["metadata"]; finite = finite and row["finite"]
                maximum = max(maximum, row["max_abs"]); coordinates = row["coordinates"]
            certificate = direction_certificate_from_vector_files(
                vector_rows, chunk_elements=args.chunk_elements,
                bootstrap_draws=args.bootstrap_draws, seed=14031,
            )
            results.append({
                "candidate_id": target["candidate_id"], "contrast_axis": kind,
                "contrast": contrast_payload(kind, metadata or {}),
                "coordinates": coordinates, "finite": finite, "max_abs": maximum,
                "repeat_exact": True, "direction": certificate,
                "t1_eligible": kind != "TOTAL" and finite and maximum > 0.0
                and certificate["cluster_bootstrap_95"]["lower_95"] > 0.0,
            })
    output = {
        **{key: value for key, value in partial.items() if key != "states"},
        "status": "COMPLETE_LIVE_FULL_COORDINATE_CONTRASTS",
        "state_ids": list(partial["states"]), "state_count": len(partial["states"]),
        "repeats": 2, "results": results,
        "claim_boundary": "T1 contrasts only; no row is a case before exact F+B binding and T2-T4.",
    }
    output["result_sha256"] = canonical(output)
    write(args.output, output)
    if not args.keep_temporary_vectors:
        shutil.rmtree(temp)
    print(json.dumps({"event": "CELL_COMPLETE", "candidates": len(targets),
                      "t1_positive_arms": sum(row["t1_eligible"] for row in results)}))


if __name__ == "__main__":
    main()
