#!/usr/bin/env python3
"""Capture source for the exact uninstrumented Inductor modules just executed."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import torch
from torch._inductor.codecache import PyCodeCache
from transformers import AutoModelForCausalLM, MambaForCausalLM
from transformers.models.mamba import modeling_mamba

from qwen_candidate_step import LossStep, configure_candidate_runtime
from run_qwen_current_triton_references import gradient_digest, tensor_digest


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture", choices=("qwen", "mamba", "phi", "deepseek8", "generic"), default="qwen"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--allow-graph-breaks", action="store_true",
        help="Capture every actually compiled segment when model control flow prevents one full graph.",
    )
    parser.add_argument(
        "--repeat", type=int, default=2,
        help="Use one run for state-specific schedule capture; numerical repeat stability is measured by the replay screen.",
    )
    args = parser.parse_args()
    if args.repeat not in {1, 2}:
        raise ValueError("schedule capture repeat must be one or two")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("output directory must be absent or empty")

    bank = json.loads(args.input_bank.read_text())
    records = bank.get("states", bank.get("records"))
    record = records[args.state]
    token_ids = record.get("input_ids", record.get("token_ids"))
    if token_ids is None:
        raise RuntimeError("selected input record has no token IDs")
    device = torch.device("cuda:0")
    configure_candidate_runtime(24000 + args.state)
    if args.architecture == "mamba":
        modeling_mamba.selective_scan_fn = None
        modeling_mamba.mamba_inner_fn = None
        modeling_mamba.selective_state_update = None
        model = MambaForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
        )
    model = model.to(device).train()
    model.config.use_cache = False
    start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend="inductor", fullgraph=not args.allow_graph_breaks, dynamic=False
    )
    values = torch.tensor([token_ids], dtype=torch.long, device=device)
    runs = []
    for repeat in range(args.repeat):
        model.zero_grad(set_to_none=True)
        loss = candidate(values)
        loss.backward()
        torch.cuda.synchronize(device)
        runs.append({"loss": tensor_digest(loss), "gradients": gradient_digest(model)})
    loaded_modules = list(PyCodeCache.modules[start:])
    wrapper_modules = []
    non_wrapper_modules = []
    for module in loaded_modules:
        source = Path(module.__file__).resolve()
        header = source.read_text(errors="ignore")[:512]
        match = re.search(r"# AOT ID: \['\d+_(forward|backward|inference)'\]", header)
        if match is None:
            non_wrapper_modules.append({
                "source": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "reason": "PYCODECACHE_KERNEL_OR_HELPER_WITHOUT_AOT_WRAPPER_ID",
            })
            continue
        wrapper_modules.append((module, match.group(1)))
    if not wrapper_modules:
        raise RuntimeError("no executed AOT wrapper modules were captured")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    phase_counts = {"forward": 0, "backward": 0}
    for ordinal, (module, aot_kind) in enumerate(wrapper_modules):
        source = Path(module.__file__).resolve()
        phase = "forward" if aot_kind == "inference" else aot_kind
        segment = phase_counts[phase]
        phase_counts[phase] += 1
        target = args.output_dir / (
            f"torchinductor/model__{ordinal}_{phase}_segment{segment}_executed/output_code.py"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({
            "phase": phase.upper(), "segment": segment, "execution_ordinal": ordinal,
            "aot_kind": aot_kind,
            "executed_source": str(source),
            "captured_source": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        })
    if not phase_counts["forward"] or not phase_counts["backward"]:
        raise RuntimeError(f"executed wrapper phases are incomplete: {phase_counts}")
    payload = {
        "schema": "kernel-analyzer-executed-inductor-schedule-v1",
        "status": "COMPLETE_EXACT_EXECUTED_FORWARD_BACKWARD_SOURCE_CAPTURE",
        "architecture": args.architecture,
        "backend": "inductor", "trace_enabled": False,
        "allow_graph_breaks": args.allow_graph_breaks,
        "input": {
            "state": args.state,
            "state_id": record.get("sequence_id", record.get("state_id", str(args.state))),
            "sequence_length": len(token_ids),
            "token_ids_sha256": digest(token_ids),
            "input_bank_sha256": hashlib.sha256(args.input_bank.read_bytes()).hexdigest(),
        },
        "repeat_stable": (
            runs[0] == runs[1] if len(runs) == 2 else "DEFERRED_TO_REPLAY_SCREEN"
        ),
        "runs": runs, "modules": rows,
        "phase_module_counts": {key.upper(): value for key, value in phase_counts.items()},
        "non_wrapper_pycodecache_modules": non_wrapper_modules,
    }
    payload["result_sha256"] = digest(payload)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.manifest), "repeat_stable": payload["repeat_stable"]}))


if __name__ == "__main__":
    main()
