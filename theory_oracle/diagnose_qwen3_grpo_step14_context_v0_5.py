#!/usr/bin/env python
"""Diagnose execution-context parity for the frozen Qwen step-14 scorer.

This is deliberately a diagnostic, not an Oracle result.  Each invocation runs
one context variant in a fresh process and compares the complete 512-token eager
scorer tensor with the frozen in-Trainer observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from forkcert.config import load_config
from forkcert.io import read_jsonl
from forkcert.logprob_runner import configure_determinism, load_hf_path
from scripts.phase0_grpo_train import CompileAudit, make_tracking_backend
from scripts.phase6_twin_training import path_config
from theory_oracle.qwen3_grpo_branch_repair_oracle import select_batch


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_batch(tokenizer: Any, samples: list[dict[str, Any]], device: str):
    import torch

    prompts = [[int(x) for x in row["prompt_ids"]] for row in samples]
    responses = [[int(x) for x in row["response_ids"]] for row in samples]
    max_prompt = max(map(len, prompts))
    max_response = max(map(len, responses))
    pad_id = int(tokenizer.pad_token_id)
    ids, masks = [], []
    for prompt, response in zip(prompts, responses, strict=True):
        left = max_prompt - len(prompt)
        ids.append([pad_id] * left + prompt + response)
        masks.append([0] * left + [1] * (len(prompt) + len(response)))
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
        max_response,
        [token for response in responses for token in response],
    )


def score(model: Any, input_ids: Any, attention_mask: Any, response_length: int):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from trl.trainer.utils import selective_log_softmax

    with sdpa_kernel(SDPBackend.MATH):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=response_length + 1,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :]
        logits = logits[:, -response_length:, :]
        logits.div_(1.0)
        values = selective_log_softmax(logits, input_ids[:, -response_length:])
    return values.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=(
            "outer",
            "accelerate",
            "accelerate_outer",
            "accelerate_compile",
            "accelerate_compile_history",
        ),
        required=True,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--bank-tokens", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")

    import torch
    from accelerate import Accelerator
    from accelerate.utils.operations import convert_outputs_to_fp32

    configure_determinism(20260720)
    evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    event = evaluation["first_stable_event_for_one_step_followup"]
    all_samples = read_jsonl(args.samples)
    samples, states, target = select_batch(all_samples, read_jsonl(args.states), event)
    cfg = replace(path_config(load_config(args.config), "path_ref"), compile_model=False)
    tokenizer, model = load_hf_path(cfg)
    accelerator = None
    if args.variant.startswith("accelerate"):
        accelerator = Accelerator(mixed_precision="fp16")
        model = accelerator.prepare_model(model)
    else:
        model.forward = convert_outputs_to_fp32(model.forward)
    input_ids, attention_mask, response_length, token_ids = build_batch(
        tokenizer, samples, cfg.device
    )
    expected_ids = [int(row["token_id"]) for row in states]
    if token_ids != expected_ids:
        raise RuntimeError("token alignment mismatch")

    compile_audit = CompileAudit()
    if args.variant in {"accelerate_compile", "accelerate_compile_history"}:
        model = torch.compile(model, backend=make_tracking_backend(compile_audit))

    def measured():
        if args.variant in {"outer", "accelerate_outer"}:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return score(model, input_ids, attention_mask, response_length)
        return score(model, input_ids, attention_mask, response_length)

    if args.variant == "accelerate_compile_history":
        for rollout in range(4):
            prior = [
                row for row in all_samples
                if int(row["metadata"]["rollout_batch"]) == rollout
            ]
            prior_ids, prior_mask, prior_response, _ = build_batch(
                tokenizer, prior, cfg.device
            )
            historical = score(model, prior_ids, prior_mask, prior_response)
            del historical
            # Reproduce the parameter-version transition that occurred between
            # successive online scans without changing any parameter value.
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(0.0)
    if args.variant in {"accelerate_compile", "accelerate_compile_history"}:
        warm = measured()
        del warm
    first = measured()
    second = measured()
    expected_rows = [
        row for row in read_jsonl(args.bank_tokens)
        if str(row["state_id"]) == str(event["state_id"])
    ]
    compiled_variant = args.variant in {"accelerate_compile", "accelerate_compile_history"}
    expected_field = "logp_alt_first" if compiled_variant else "logp_ref_first"
    expected_hash = (
        "b3cdc03d205a1df320a423ebf5fb326945498aa5732fe3e5b22cb8566a276ce9"
        if compiled_variant
        else "ab27446d8506e839f751221a49e74ecb78290827838f931060a475a2b7606562"
    )
    expected = torch.tensor(
        [float(row[expected_field]) for row in expected_rows],
        dtype=first.dtype,
        device=first.device,
    )
    delta = first.detach() - expected
    payload = {
        "schema_version": "forkcert.qwen3-grpo-step14-context-diagnostic.v0.5",
        "claim_status": "DIAGNOSTIC_ONLY",
        "variant": args.variant,
        "accelerate_native_amp": None if accelerator is None else bool(accelerator.native_amp),
        "accelerate_mixed_precision": None if accelerator is None else str(accelerator.mixed_precision),
        "accelerate_forward_wrapped": bool(
            hasattr(model, "_original_forward") and hasattr(model.forward, "__wrapped__")
        ),
        "shape": list(first.shape),
        "first_sha256": tensor_sha256(first),
        "second_sha256": tensor_sha256(second),
        "self_exact": bool(torch.equal(first, second)),
        "native_shape_sha256": tensor_sha256(first.reshape(4, 128)),
        "expected_sha256": expected_hash,
        "expected_hash_exact": tensor_sha256(first.reshape(4, 128)) == expected_hash,
        "bank_value_exact": bool(torch.equal(first.detach(), expected)),
        "different_value_count": int(torch.count_nonzero(delta)),
        "max_abs_difference": float(delta.float().abs().max()),
        "mean_signed_difference": float(delta.float().mean()),
        "target_flat_index": target,
        "target_observed": float(first[target].detach()),
        "target_expected": float(
            event["logp_alt"] if compiled_variant else event["logp_ref"]
        ),
        "target_exact": float(first[target].detach())
        == float(event["logp_alt"] if compiled_variant else event["logp_ref"]),
        "compile_audit": {
            "backend_compiles": compile_audit.backend_compiles,
            "runtime_invocations": compile_audit.runtime_invocations,
            "graph_code_sha256": compile_audit.graph_code_sha256,
            "graph_node_counts": compile_audit.graph_node_counts,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
