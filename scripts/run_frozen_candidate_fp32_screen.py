#!/usr/bin/env python3
"""Freeze one actually compiled candidate and screen all its compute calls.

Unlike the legacy sharded runners, this program never recompiles between the
Triton and non-Triton arms.  The exact warmed wrapper sources are copied,
inventoried, and measured in the same process.  A resumed run is accepted only
when the newly warmed wrapper bytes equal the frozen release byte-for-byte.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/coverage/generated_fp32_protocol.json"
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))
sys.path.insert(0, str(ROOT))

from scripts.generated_fp32_observer import GeneratedFP32Observer  # noqa: E402
from scripts.generated_nontriton_fp32_observer import (  # noqa: E402
    GeneratedNonTritonFP32Observer,
)
from scripts.frozen_state_checkpoint import (  # noqa: E402
    load_state_checkpoints,
    state_checkpoint_path,
    write_gzip,
)
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import (  # noqa: E402
    file_digest,
    gradient_digest,
    load_model,
    tensor_digest,
)


AOT_KIND = re.compile(r"# AOT ID: \['\d+_(forward|backward|inference)'\]")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def wrapper_modules(modules: list[Any]) -> list[tuple[Any, str]]:
    rows = []
    for module in modules:
        source = Path(module.__file__).resolve()
        match = AOT_KIND.search(source.read_text(errors="ignore")[:512])
        if match is None:
            continue
        kind = match.group(1)
        rows.append((module, "forward" if kind == "inference" else kind))
    if not rows or not {phase for _, phase in rows} >= {"forward", "backward"}:
        raise RuntimeError("warmed candidate lacks complete forward/backward wrappers")
    return rows


def freeze_or_validate_release(
    *, modules: list[tuple[Any, str]], release: Path, architecture: str,
    input_bank: Path, state: dict[str, Any], allow_graph_breaks: bool,
) -> tuple[Path, Path]:
    trace = release / "trace"
    capture_path = release / "capture.json"
    inventory_path = release / "inventory.json.gz"
    campaign_path = release / "campaign.json.gz"
    source_rows = []
    phase_counts = {"forward": 0, "backward": 0}
    for ordinal, (module, phase) in enumerate(modules):
        source = Path(module.__file__).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        segment = phase_counts[phase]
        phase_counts[phase] += 1
        target = trace / f"model__{ordinal}_{phase}_segment{segment}_executed" / "output_code.py"
        source_rows.append({
            "phase": phase.upper(), "segment": segment,
            "execution_ordinal": ordinal, "sha256": digest,
            "captured_source": str(target.resolve()), "executed_source": str(source),
        })
    if capture_path.exists():
        capture = json.loads(capture_path.read_text())
        expected = [row["sha256"] for row in capture["modules"]]
        observed = [row["sha256"] for row in source_rows]
        if observed != expected:
            raise RuntimeError("recompiled wrapper bytes differ from frozen runtime release")
        return inventory_path, campaign_path

    if release.exists() and any(release.iterdir()):
        raise RuntimeError("partial runtime release exists without a capture manifest")
    trace.mkdir(parents=True, exist_ok=False)
    for row, (module, _) in zip(source_rows, modules):
        target = Path(row["captured_source"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(module.__file__).resolve(), target)
    tokens = state.get("token_ids", state.get("input_ids"))
    capture = {
        "schema": "kernel-analyzer-in-process-frozen-candidate-v1",
        "status": "COMPLETE_EXACT_EXECUTED_FORWARD_BACKWARD_SOURCE_CAPTURE",
        "architecture": architecture,
        "allow_graph_breaks": allow_graph_breaks,
        "input": {
            "state": 0,
            "state_id": state.get("sequence_id", state.get("state_id", "0")),
            "sequence_length": len(tokens),
            "token_ids_sha256": canonical_hash(tokens),
            "input_bank_sha256": file_digest(input_bank),
        },
        "modules": source_rows,
        "runtime_environment": runtime_environment(),
        "phase_module_counts": {key.upper(): value for key, value in phase_counts.items()},
        "same_process_measurement_required": True,
    }
    capture["result_sha256"] = canonical_hash(capture)
    release.mkdir(parents=True, exist_ok=True)
    capture_path.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n")
    subprocess.run([
        sys.executable, str(ROOT / "scripts/build_current_qwen_generated_inventory.py"),
        "--trace-dir", str(trace), "--capture", str(capture_path),
        "--output", str(inventory_path),
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/build_generated_fp32_campaign.py"),
        "--inventory", str(inventory_path), "--output", str(campaign_path),
    ], cwd=ROOT, check=True)
    return inventory_path, campaign_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("qwen", "mamba", "phi", "deepseek8"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-states", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--metric-chunk-elements", type=int, default=1_048_576)
    parser.add_argument("--allow-graph-breaks", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if len(states) != args.expected_states:
        raise RuntimeError("frozen input population changed")
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend="inductor",
        fullgraph=not args.allow_graph_breaks, dynamic=False,
    )
    warm_tokens = states[0].get("token_ids", states[0].get("input_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    wrappers = wrapper_modules(modules)
    inventory_path, campaign_path = freeze_or_validate_release(
        modules=wrappers, release=args.release_dir, architecture=args.architecture,
        input_bank=args.input_bank, state=states[0],
        allow_graph_breaks=args.allow_graph_breaks,
    )
    with gzip.open(inventory_path, "rt", encoding="utf-8") as handle:
        inventory = json.load(handle)
    with gzip.open(campaign_path, "rt", encoding="utf-8") as handle:
        campaign = json.load(handle)
    inventory_rows = inventory["runtime_call_audit"]["rows"]
    campaign_rows = campaign["rows"]
    triton_path = args.release_dir / "triton_screen.json.gz"
    nontriton_path = args.release_dir / "nontriton_screen.json.gz"
    joint_path = args.release_dir / "joint_screen_checkpoint.json.gz"
    joint_state_dir = args.release_dir / "joint_state_checkpoints"
    common = {
        "architecture": args.architecture,
        "model": str(args.model.resolve()),
        "input_bank_sha256": file_digest(args.input_bank),
        "protocol_sha256": protocol["protocol_sha256"],
        "shard_index": 0, "shard_count": 1, "repeat": args.repeat,
        "release_capture_sha256": json.loads((args.release_dir / "capture.json").read_text())["result_sha256"],
    }
    if joint_path.exists():
        with gzip.open(joint_path, "rt", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        triton_payload = checkpoint["triton"]
        nontriton_payload = checkpoint["nontriton"]
        if triton_payload["release_capture_sha256"] != common["release_capture_sha256"]:
            raise RuntimeError("screen binds another runtime release")
    elif triton_path.exists() or nontriton_path.exists():
        if not (triton_path.exists() and nontriton_path.exists()):
            raise RuntimeError("completed joint screen has only one arm")
        with gzip.open(triton_path, "rt", encoding="utf-8") as handle:
            triton_payload = json.load(handle)
        with gzip.open(nontriton_path, "rt", encoding="utf-8") as handle:
            nontriton_payload = json.load(handle)
    else:
        triton_payload = {
            **common, "schema": "kernel-analyzer-generated-fp32-screen-v1",
            "status": "RUNNING", "campaign_sha256": campaign["result_sha256"],
            "states": {},
        }
        nontriton_payload = {
            **common, "schema": "kernel-analyzer-generated-nontriton-fp32-screen-v1",
            "status": "RUNNING", "inventory_sha256": inventory["result_sha256"],
            "states": {},
        }
    load_state_checkpoints(
        directory=joint_state_dir,
        release_capture_sha256=common["release_capture_sha256"],
        triton_payload=triton_payload,
        nontriton_payload=nontriton_payload,
    )
    for state_index, state in enumerate(states):
        state_id = str(state.get("sequence_id", state.get("state_id", state_index)))
        left_done = len(triton_payload["states"].get(state_id, {}).get("repeats", []))
        right_done = len(nontriton_payload["states"].get(state_id, {}).get("repeats", []))
        if left_done == right_done == args.repeat:
            continue
        if left_done or right_done:
            raise RuntimeError("joint state resume is not atomic")
        tokens = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        seed = 24000 + state_index
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        baseline_loss = candidate(values); baseline_loss.backward(); torch.cuda.synchronize(device)
        baseline = {"loss": tensor_digest(baseline_loss), "gradients": gradient_digest(model)}
        triton_repeats, nontriton_repeats = [], []
        for repeat in range(args.repeat):
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            triton = GeneratedFP32Observer(
                modules=modules, campaign_rows=campaign_rows,
                sample_size=args.sample_size,
                metric_chunk_elements=args.metric_chunk_elements,
            )
            nontriton = GeneratedNonTritonFP32Observer(
                modules=modules, inventory_rows=inventory_rows,
                sample_size=args.sample_size,
                metric_chunk_elements=args.metric_chunk_elements,
            )
            model.zero_grad(set_to_none=True)
            with triton, nontriton:
                loss = candidate(values); loss.backward()
            torch.cuda.synchronize(device)
            observed = {"loss": tensor_digest(loss), "gradients": gradient_digest(model)}
            if observed != baseline:
                raise RuntimeError(f"joint observer perturbed full step: {state_id}")
            triton_summary, nontriton_summary = triton.summary(), nontriton.summary()
            if triton_summary["status"] != "COMPLETE_ALL_TRITON_FP32_REPLAY":
                raise RuntimeError(f"Triton runtime denominator incomplete: {state_id}")
            if nontriton_summary["status"] != "COMPLETE_RUNTIME_NONTRITON_FP32_REPLAY_WITH_STATIC_DISPOSITION":
                raise RuntimeError(f"non-Triton runtime denominator incomplete: {state_id}")
            triton_repeats.append({"repeat": repeat, "summary": triton_summary})
            nontriton_repeats.append({"repeat": repeat, "summary": nontriton_summary})
        token_hash = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        triton_payload["states"][state_id] = {"token_ids_sha256": token_hash, "repeats": triton_repeats}
        nontriton_payload["states"][state_id] = {"token_ids_sha256": token_hash, "repeats": nontriton_repeats}
        # Journal only the new state.  Triton and non-Triton observations stay
        # atomic, while large campaigns avoid rewriting every earlier record.
        joint_state_dir.mkdir(parents=True, exist_ok=True)
        write_gzip(state_checkpoint_path(joint_state_dir, state_id), {
            "schema": "kernel-analyzer-joint-frozen-candidate-state-v1",
            "release_capture_sha256": common["release_capture_sha256"],
            "state_id": state_id,
            "triton_state": triton_payload["states"][state_id],
            "nontriton_state": nontriton_payload["states"][state_id],
        })
        print(json.dumps({"event": "STATE_COMPLETE", "state": state_id,
                          "triton": len(campaign_rows),
                          "nontriton_actual": len(nontriton_summary["records"])}), flush=True)
    triton_payload["status"] = "COMPLETE_SHARD_ALL_TRITON_FP32_REPLAY"
    nontriton_payload["status"] = "COMPLETE_SHARD_ALL_NONTRITON_FP32_REPLAY"
    write_gzip(triton_path, triton_payload); write_gzip(nontriton_path, nontriton_payload)
    if joint_path.exists():
        joint_path.unlink()
    if joint_state_dir.exists():
        shutil.rmtree(joint_state_dir)
    print(json.dumps({"event": "RELEASE_COMPLETE", "release": str(args.release_dir)}))


if __name__ == "__main__":
    main()
