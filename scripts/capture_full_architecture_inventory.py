#!/usr/bin/env python3
"""Capture one complete supported LM loss F+B dispatcher atlas.

The denominator is execution-derived.  Mamba is deliberately run through the
Transformers slow recurrence so every arithmetic operation crosses the ATen
dispatcher; the separate fused-screen artifacts remain candidate evidence.
"""

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
from transformers import AutoModelForCausalLM, Gemma3ForConditionalGeneration, MambaForCausalLM
try:
    from transformers import Gemma4ForConditionalGeneration
except ImportError:  # Older environments can still run all non-Gemma-4 cells.
    Gemma4ForConditionalGeneration = None
from transformers.models.mamba import modeling_mamba


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.op_inventory import (  # noqa: E402
    build_full_step_coverage_certificate,
    observe_full_forward_backward_step,
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gradient_digest(model: torch.nn.Module) -> tuple[str, int, int]:
    combined = hashlib.sha256()
    present = 0
    elements = 0
    for name, parameter in sorted(model.named_parameters()):
        combined.update(name.encode())
        if parameter.grad is None:
            combined.update(b"NONE")
            continue
        present += 1
        elements += parameter.grad.numel()
        value = parameter.grad.detach().contiguous().cpu()
        combined.update(value.view(torch.uint8).numpy().tobytes())
        del value
    return combined.hexdigest(), present, elements


def deterministic(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def load_model(
    architecture: str, model_path: Path, device: torch.device, shard_gpus: int,
):
    if architecture == "mamba":
        # Make the reference program explicit.  If any fast-path global remains
        # active, MambaMixer can bypass the dispatcher through a C++ extension.
        modeling_mamba.selective_scan_fn = None
        modeling_mamba.mamba_inner_fn = None
        modeling_mamba.selective_state_update = None
        model = MambaForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, local_files_only=True
        )
        implementation = "transformers_explicit_recurrence_bfloat16"
    elif architecture == "gemma3":
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="eager",
        )
        implementation = "transformers_eager_gemma3_conditional_generation_bfloat16"
    elif architecture == "gemma4":
        if Gemma4ForConditionalGeneration is None:
            raise RuntimeError("Gemma 4 requires a Transformers build with Gemma4 support")
        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="eager",
        )
        implementation = "transformers_eager_gemma4_conditional_generation_bfloat16"
    else:
        load_kwargs: dict[str, Any] = {}
        if shard_gpus > 1:
            load_kwargs["device_map"] = "balanced"
            load_kwargs["max_memory"] = {
                index: "42GiB" for index in range(shard_gpus)
            }
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="eager",
            trust_remote_code=False,
            **load_kwargs,
        )
        implementation = {
            "qwen": "transformers_eager_qwen_bfloat16",
            "moe": "transformers_eager_granite_moe_bfloat16",
            "phi": "transformers_eager_phi4_mini_bfloat16",
            "deepseek8": "transformers_eager_deepseek_r1_qwen3_8b_bfloat16_sharded",
            "generic": "transformers_eager_generic_causal_lm_bfloat16",
        }[architecture]
    if shard_gpus == 1:
        model = model.to(device)
    model = model.train()
    model.config.use_cache = False
    return model, implementation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture", choices=(
            "qwen", "mamba", "moe", "phi", "deepseek8", "generic",
            "gemma3", "gemma4",
        ),
        required=True,
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--without-sequence-numbers", action="store_true")
    parser.add_argument("--shard-gpus", type=int, default=1)
    args = parser.parse_args()

    bank = json.loads(args.input_bank.read_text())
    bank_rows = bank.get("states", bank.get("records"))
    if bank_rows is None:
        raise RuntimeError("input bank must contain states or records")
    bank_row = bank_rows[args.state]
    token_ids = bank_row.get("token_ids", bank_row.get("input_ids"))
    if token_ids is None:
        raise RuntimeError("input-bank row must contain token_ids or input_ids")
    ids_cpu = torch.tensor(token_ids, dtype=torch.long)
    observed_token_hash = hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest()
    expected_token_hash = bank_row.get("token_sha256")
    if expected_token_hash is not None and observed_token_hash != expected_token_hash:
        raise RuntimeError("input-bank token digest mismatch")
    input_ids = ids_cpu.unsqueeze(0).to(args.device)

    deterministic(24000 + args.state)
    model, implementation = load_model(
        args.architecture, args.model, torch.device(args.device), args.shard_gpus
    )

    model.zero_grad(set_to_none=True)
    deterministic(24000 + args.state)
    baseline = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    if baseline.loss is None:
        raise RuntimeError("baseline loss absent")
    baseline_loss = baseline.loss.detach().clone()
    baseline.loss.backward()
    baseline_digest, baseline_gradients, baseline_elements = gradient_digest(model)
    del baseline
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    observed: dict[str, torch.Tensor] = {}

    def loss_closure() -> torch.Tensor:
        deterministic(24000 + args.state)
        output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        if output.loss is None:
            raise RuntimeError("observed loss absent")
        observed["loss"] = output.loss
        return output.loss

    trace = observe_full_forward_backward_step(
        loss_closure=loss_closure,
        endpoint_closure=lambda: {
            "loss": observed["loss"],
            **{
                f"parameter_gradient.{name}": parameter.grad
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            },
        },
        model=model,
        capture_autograd_sequence_numbers=not args.without_sequence_numbers,
        retain_forward_outputs_for_origin_binding=not args.without_sequence_numbers,
    )
    observed_digest, observed_gradients, observed_elements = gradient_digest(model)
    loss_exact = bool(torch.equal(observed["loss"].detach(), baseline_loss))
    gradients_exact = observed_digest == baseline_digest
    stable = loss_exact and gradients_exact
    certificate = build_full_step_coverage_certificate(
        trace,
        observation_stable=stable,
        subject=f"{args.architecture} one natural teacher-forced loss forward/backward step",
        implementation_id=implementation,
    )
    payload = {
        "schema": "kernel-analyzer-full-architecture-invocation-inventory-v1",
        "status": certificate["status"],
        "architecture": args.architecture,
        "scope": "one complete natural loss forward/backward step",
        "model": {
            "path": str(args.model.resolve()),
            "config_sha256": file_hash(args.model / "config.json"),
        },
        "input": {
            "state_id": args.state,
            "sequence_length": ids_cpu.numel(),
            "token_sha256": observed_token_hash,
            "input_bank_sha256": file_hash(args.input_bank),
        },
        "implementation": implementation,
        "observation_stability": {
            "loss_exact": loss_exact,
            "all_parameter_gradient_digest_exact": gradients_exact,
            "baseline_parameter_gradients": baseline_gradients,
            "observed_parameter_gradients": observed_gradients,
            "baseline_gradient_elements": baseline_elements,
            "observed_gradient_elements": observed_elements,
        },
        "coverage": certificate,
        "trace": trace.as_dict(),
        "claim_boundary": {
            "supported": "complete execution-derived eager ATen invocation denominator for the full model F+B step",
            "not_supported": [
                "candidate-region binding",
                "per-invocation finite-precision correctness",
                "directional-bias verdict",
                "property induction",
            ],
        },
    }
    payload["result_sha256"] = canonical_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    if args.output.suffix == ".gz":
        with gzip.open(args.output, "wb", compresslevel=6) as handle:
            handle.write(encoded)
    else:
        args.output.write_bytes(encoded)
    print(json.dumps({
        "output": str(args.output),
        "status": payload["status"],
        "events": len(trace.events),
        "phase_counts": certificate["denominator"]["phase_counts"],
        "unique_overloads": certificate["denominator"]["unique_overloads"],
        "observation_stable": stable,
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
