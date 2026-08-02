#!/usr/bin/env python3
"""Capture every ATen invocation in one Qwen3-VL multimodal F+B step."""

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
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.round2_vl_smoke import prepare_step, sha256_file  # noqa: E402
from scripts.op_inventory import (  # noqa: E402
    build_full_step_coverage_certificate,
    observe_full_forward_backward_step,
)


def gradient_digest(model: Any) -> tuple[str, dict[str, str]]:
    combined = hashlib.sha256()
    parameters: dict[str, str] = {}
    for name, parameter in sorted(model.named_parameters()):
        if parameter.grad is None:
            digest = "NONE"
        else:
            value = parameter.grad.detach().contiguous().cpu()
            digest = hashlib.sha256(
                value.view(torch.uint8).numpy().tobytes()
            ).hexdigest()
        parameters[name] = digest
        combined.update(name.encode("utf-8"))
        combined.update(digest.encode("ascii"))
    return combined.hexdigest(), parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--capture-autograd-sequence-numbers", action="store_true")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    inputs, labels, input_metadata = prepare_step(
        processor,
        args.image,
        width=args.width,
        height=args.height,
    )
    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.config.use_cache = False
    model.eval()
    device_inputs = {key: value.to(device) for key, value in inputs.items()}
    device_labels = labels.to(device)

    model.zero_grad(set_to_none=True)
    baseline = model(
        **device_inputs,
        labels=device_labels,
        use_cache=False,
        return_dict=True,
    )
    if baseline.loss is None:
        raise RuntimeError("baseline loss is absent")
    baseline_loss = baseline.loss.detach().clone()
    baseline.loss.backward()
    baseline_digest, baseline_parameter_digests = gradient_digest(model)
    model.zero_grad(set_to_none=True)
    del baseline

    observed: dict[str, torch.Tensor] = {}

    def loss_closure() -> torch.Tensor:
        result = model(
            **device_inputs,
            labels=device_labels,
            use_cache=False,
            return_dict=True,
        )
        if result.loss is None:
            raise RuntimeError("observed loss is absent")
        observed["loss"] = result.loss
        return result.loss

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
        capture_autograd_sequence_numbers=args.capture_autograd_sequence_numbers,
    )
    observed_digest, observed_parameter_digests = gradient_digest(model)
    loss_exact = bool(torch.equal(observed["loss"].detach(), baseline_loss))
    gradient_exact = observed_digest == baseline_digest
    parameter_exact = observed_parameter_digests == baseline_parameter_digests
    stable = loss_exact and gradient_exact and parameter_exact
    certificate = build_full_step_coverage_certificate(
        trace,
        observation_stable=stable,
        subject="Qwen3-VL-2B natural multimodal teacher-forced loss step",
        implementation_id="torch_eager_bfloat16_attention_eager",
    )
    payload = {
        "schema": "kernel-analyzer.round2-vl-all-op-inventory.v1",
        "status": certificate["status"],
        "scope": "one natural multimodal loss forward/backward step",
        "model": {
            "path": str(args.model.resolve()),
            "config_sha256": sha256_file(args.model / "config.json"),
        },
        "input": {
            **input_metadata,
            "image_path": str(args.image.resolve()),
            "image_sha256": sha256_file(args.image),
        },
        "observation_stability": {
            "loss_exact": loss_exact,
            "gradient_digest_exact": gradient_exact,
            "all_parameter_digests_exact": parameter_exact,
        },
        "coverage": certificate,
        "trace": trace.as_dict(),
        "claim_boundary": {
            "supported": "complete execution-derived eager ATen denominator",
            "not_yet_supported": [
                "complete per-invocation mathematical derivation",
                "numerical bias verdict",
                "candidate implementation correctness",
                "property induction",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    if args.output.suffix == ".gz":
        with gzip.open(args.output, "wb", compresslevel=6) as handle:
            handle.write(encoded)
    else:
        args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "events": len(trace.events),
                "forward": certificate["denominator"]["phase_counts"].get(
                    "FORWARD", 0
                ),
                "backward": certificate["denominator"]["phase_counts"].get(
                    "BACKWARD", 0
                ),
                "unique_overloads": certificate["denominator"]["unique_overloads"],
                "observation_stable": stable,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
