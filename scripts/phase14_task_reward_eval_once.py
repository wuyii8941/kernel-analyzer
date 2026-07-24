#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def numeric_reward(text: str, expected: float) -> tuple[float, float | None, bool]:
    predictions = NUMBER_RE.findall(text)
    if not predictions:
        return -1.0, None, False
    try:
        predicted = float(predictions[-1].replace(",", ""))
    except ValueError:
        return -1.0, None, False
    error = abs(predicted - expected)
    scale = max(1.0, abs(expected))
    exact = math.isclose(predicted, expected, rel_tol=1e-6, abs_tol=1e-6)
    return 1.0 / (1.0 + error / scale) + (1.0 if exact else 0.0), predicted, exact


def arithmetic_prompt(index: int) -> tuple[str, float]:
    start = 7 + index
    added = 3 + (index % 11)
    removed = 1 + (index % 5)
    expected = float(start + added - removed)
    prompt = (
        "Solve the problem. Show concise reasoning and end with the numeric answer.\n\n"
        f"A box starts with {start} items, receives {added}, then gives away {removed}. How many remain?"
    )
    return prompt, expected


def main() -> None:
    parser = argparse.ArgumentParser(description="One deterministic held-out task-reward evaluation arm.")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="data/phase9_policy_step14_pre")
    parser.add_argument("--start-index", type=int, default=64)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda")
    model.eval()
    examples = []
    for index in range(args.start_index, args.start_index + args.count):
        prompt, expected = arithmetic_prompt(index)
        examples.append({"dataset_index": index, "prompt": prompt, "expected": expected})
    rows = []
    for offset in range(0, len(examples), args.batch_size):
        batch = examples[offset : offset + args.batch_size]
        encoded = tokenizer(
            [row["prompt"] for row in batch],
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        prompt_width = encoded["input_ids"].shape[1]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16), sdpa_kernel(SDPBackend.MATH):
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        for example, completion, token_ids in zip(
            batch, completions, generated[:, prompt_width:].cpu().tolist(), strict=True
        ):
            reward, predicted, exact = numeric_reward(completion, example["expected"])
            rows.append(
                {
                    **example,
                    "completion": completion,
                    "completion_token_ids": token_ids,
                    "predicted": predicted,
                    "reward": reward,
                    "exact": exact,
                }
            )
    encoded_outputs = json.dumps(
        [[row["dataset_index"], row["completion_token_ids"]] for row in rows],
        separators=(",", ":"),
    ).encode("utf-8")
    payload = {
        "schema_version": "forkcert.task_reward_eval_once.v1",
        "arm": args.arm,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "autocast_dtype": "float16",
            "attention_backend": "sdpa_math",
        },
        "evaluation": {
            "start_index": args.start_index,
            "count": args.count,
            "max_new_tokens": args.max_new_tokens,
            "greedy": True,
            "mean_numeric_reward": sum(row["reward"] for row in rows) / len(rows),
            "exact_count": sum(row["exact"] for row in rows),
            "missing_number_count": sum(row["predicted"] is None for row in rows),
            "completion_ids_sha256": hashlib.sha256(encoded_outputs).hexdigest(),
        },
        "rows": rows,
        "claim_scope": (
            "Held-out evaluation using the same numeric reward definition as Phase 0. A reward difference is task-level "
            "evidence for this synthetic arithmetic task; equality does not prove general harmlessness."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
