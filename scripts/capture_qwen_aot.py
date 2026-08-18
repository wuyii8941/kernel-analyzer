#!/usr/bin/env python3
"""Capture supported LM AOTAutograd F+B graphs and their runtime identity bridge."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoModelForCausalLM, MambaForCausalLM
from transformers.models.mamba import modeling_mamba

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aot_capture import AOTForwardBackwardCapture  # noqa: E402


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def gradient_digest(model: torch.nn.Module) -> str:
    combined = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        combined.update(name.encode())
        if parameter.grad is None:
            combined.update(b"NONE")
        else:
            value = parameter.grad.detach().contiguous().cpu()
            combined.update(value.view(torch.uint8).numpy().tobytes())
    return combined.hexdigest()


def load_model(architecture: str, model_path: Path, device: str) -> torch.nn.Module:
    if architecture == "mamba":
        modeling_mamba.selective_scan_fn = None
        modeling_mamba.mamba_inner_fn = None
        modeling_mamba.selective_state_update = None
        model = MambaForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
        )
    model = model.to(device).train()
    model.config.use_cache = False
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=("qwen", "mamba", "moe", "phi", "deepseek8"),
        default="qwen",
    )
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preserve-aot-aten",
        action="store_true",
        help="Capture the same undecomposed AOT ATen program used by the proof-ID compile.",
    )
    parser.add_argument(
        "--allow-graph-breaks",
        action="store_true",
        help="Retain unavoidable eager data-dependent control while capturing every AOT segment.",
    )
    parser.add_argument(
        "--inductor-partition",
        action="store_true",
        help=(
            "Capture the exact default-decomposition AOT partition selected by "
            "Inductor, while replacing only the lower compiler with the exact "
            "runtime-identity interpreter."
        ),
    )
    args = parser.parse_args()
    if args.inductor_partition and args.preserve_aot_aten:
        parser.error(
            "--inductor-partition and --preserve-aot-aten are mutually exclusive: "
            "the former must retain Inductor's default decomposition"
        )

    bank = json.loads(args.input_bank.read_text())
    records = bank.get("states", bank.get("records"))
    if records is None:
        raise RuntimeError("input bank must contain states or records")
    record = records[args.state]
    token_ids = record.get("token_ids", record.get("input_ids"))
    if token_ids is None:
        raise RuntimeError("state lacks token ids")

    torch.manual_seed(24000 + args.state)
    torch.cuda.manual_seed_all(24000 + args.state)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.architecture, args.model, args.device)
    inputs = torch.tensor([token_ids], dtype=torch.long, device=args.device)

    execution_seed = 24000 + args.state
    torch.manual_seed(execution_seed)
    torch.cuda.manual_seed_all(execution_seed)
    model.zero_grad(set_to_none=True)
    baseline_loss = model(
        input_ids=inputs, labels=inputs, use_cache=False, return_dict=False
    )[0]
    baseline_loss.backward()
    baseline_loss_value = baseline_loss.detach().clone()
    baseline_gradient = gradient_digest(model)
    model.zero_grad(set_to_none=True)

    class LossStep(torch.nn.Module):
        def __init__(self, subject: torch.nn.Module) -> None:
            super().__init__()
            self.subject = subject

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.subject(
                input_ids=values,
                labels=values,
                use_cache=False,
                return_dict=False,
            )[0]

    capture = AOTForwardBackwardCapture()
    capture_backend = (
        capture.inductor_partition_backend()
        if args.inductor_partition
        else capture.backend(
            decompositions={} if args.preserve_aot_aten else None
        )
    )
    compiled = torch.compile(
        LossStep(model),
        backend=capture_backend,
        fullgraph=not args.allow_graph_breaks,
        dynamic=False,
    )
    # Observation validity is repeat stability of the *same captured AOT
    # program*.  Eager-vs-AOT bitwise identity is deliberately not the gate:
    # a default decomposition or functionalization can change finite-
    # precision execution order, which is a candidate-added numerical effect
    # to retain rather than an instrumentation failure.
    candidate_runs = []
    candidate_loss_values = []
    for repeat in range(2):
        torch.manual_seed(execution_seed)
        torch.cuda.manual_seed_all(execution_seed)
        model.zero_grad(set_to_none=True)
        candidate_loss = compiled(inputs)
        capture.bind_user_outputs(candidate_loss)
        candidate_loss.register_hook(capture.bind_user_cotangent)
        candidate_loss.backward()
        candidate_loss_values.append(candidate_loss.detach().clone())
        candidate_runs.append({
            "repeat": repeat,
            "loss": float(candidate_loss.detach()),
            "gradient_digest": gradient_digest(model),
        })
    capture_payload = capture.as_dict()
    stable = {
        "repeat_loss_exact": bool(torch.equal(
            candidate_loss_values[0], candidate_loss_values[1]
        )),
        "repeat_all_parameter_gradient_digest_exact": (
            candidate_runs[0]["gradient_digest"]
            == candidate_runs[1]["gradient_digest"]
        ),
    }
    eager_comparison = {
        "loss_exact": bool(torch.equal(
            candidate_loss_values[0], baseline_loss_value
        )),
        "all_parameter_gradient_digest_exact": (
            candidate_runs[0]["gradient_digest"] == baseline_gradient
        ),
        "is_observation_validity_gate": False,
    }
    bridge_gates = capture_payload["cross_phase_runtime_bridge"]["gates"]
    if all(stable.values()) and all((
        bridge_gates["all_forward_outputs_resolved"],
        bridge_gates["all_backward_inputs_resolved"],
    )):
        capture_status = "COMPLETE_AOT_FB_CAPTURE"
    elif all(stable.values()) and args.allow_graph_breaks:
        capture_status = "COMPLETE_EXECUTION_PARTIAL_CROSS_SEGMENT_BRIDGE"
    else:
        capture_status = "INVALID_OBSERVATION"
    payload = {
        "schema": "kernel-analyzer-architecture-aot-runtime-identity-v1",
        "status": capture_status,
        "architecture": args.architecture,
        "model": str(args.model.resolve()),
        "input": {
            "state": args.state,
            "sequence_length": len(token_ids),
            "token_ids_sha256": digest(token_ids),
        },
        "rng_comparability": {
            "execution_seed": execution_seed,
            "reset_before_baseline_and_candidate": True,
        },
        "preserve_aot_aten": args.preserve_aot_aten,
        "aot_partition": (
            "EXACT_INDUCTOR_DEFAULT_DECOMPOSITION_AND_PARTITION"
            if args.inductor_partition
            else "STANDALONE_AOT_AUTOGRAD_PARTITION"
        ),
        "aot_capture_mode": (
            "SEGMENTED_WITH_EXPLICIT_EAGER_CONTROL_BOUNDARIES"
            if args.allow_graph_breaks else "SINGLE_FULLGRAPH"
        ),
        "observation_stability": stable,
        "candidate_runs": candidate_runs,
        "eager_comparison": eager_comparison,
        "capture": capture_payload,
        "gates": {
            "forward_graph_present": capture_payload["phase_graph_counts"]["FORWARD"] >= 1,
            "backward_graph_present": capture_payload["phase_graph_counts"]["BACKWARD"] >= 1,
            "single_fullgraph_required": not args.allow_graph_breaks,
            "runtime_cross_phase_identity_only": capture_payload["cross_phase_runtime_bridge"]["gates"]["identity_pairing_only"],
            "all_forward_outputs_resolved": capture_payload["cross_phase_runtime_bridge"]["gates"]["all_forward_outputs_resolved"],
            "all_backward_inputs_resolved": capture_payload["cross_phase_runtime_bridge"]["gates"]["all_backward_inputs_resolved"],
            "actual_inductor_default_aot_partition": args.inductor_partition,
        },
        "claim_boundary": (
            (
                "This is the actual default-decomposition AOT forward/backward program supplied "
                "by Inductor. "
                if args.inductor_partition
                else "This is a standalone AOTAutograd forward/backward program. "
            )
            + "It proves the captured program and its runtime cross-phase "
            "tensor identity inside each captured segment. Explicit eager graph-break control "
            "remains outside AOT and must stay unresolved in the eager-to-AOT bridge. It does not "
            "by itself bind eager dispatch invocations or Inductor regions."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    if args.output.suffix == ".gz":
        with gzip.open(args.output, "wb", compresslevel=6) as handle:
            handle.write(encoded)
    else:
        args.output.write_bytes(encoded)
    print(json.dumps({
        "output": str(args.output.resolve().relative_to(ROOT)),
        "status": payload["status"],
        "phase_graph_counts": capture_payload["phase_graph_counts"],
        "gates": payload["gates"],
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
