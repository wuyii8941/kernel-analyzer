#!/usr/bin/env python
"""Corrected Trainer-realization wrapper for the Qwen GRPO branch repair."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import theory_oracle.qwen3_grpo_branch_repair_oracle as base
from forkcert.logprob_runner import attention_backend_context, precision_context


def trainer_response_logps_with_grad(
    tokenizer: Any, model: Any, config: Any, samples: list[dict[str, Any]]
):
    import torch
    from trl.trainer.utils import selective_log_softmax

    prompt_ids = [[int(value) for value in sample["prompt_ids"]] for sample in samples]
    response_ids = [[int(value) for value in sample["response_ids"]] for sample in samples]
    max_prompt = max(map(len, prompt_ids))
    max_response = max(map(len, response_ids))
    if any(len(values) != max_response for values in response_ids):
        raise ValueError("corrected frozen follow-up requires equal response lengths")
    pad_id = int(tokenizer.pad_token_id)
    batch, masks = [], []
    for prompt, response in zip(prompt_ids, response_ids, strict=True):
        left = max_prompt - len(prompt)
        batch.append([pad_id] * left + prompt + response)
        masks.append([0] * left + [1] * (len(prompt) + len(response)))
    input_ids = torch.tensor(batch, dtype=torch.long, device=config.device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=config.device)
    with attention_backend_context(config), precision_context(config):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=max_response + 1,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :]
        logits = logits[:, -max_response:, :]
        logits.div_(1.0)
        completion_ids = input_ids[:, -max_response:]
        logps = selective_log_softmax(logits, completion_ids)
    return logps.reshape(-1), [token for response in response_ids for token in response]


original_run_arm = base.run_arm


def anchor_validated_run_arm(*args: Any, **kwargs: Any):
    result = original_run_arm(*args, **kwargs)
    name = str(args[0])
    event = args[5]
    expected = float(event["logp_ref"] if name == "A_reference" else event["logp_alt"])
    result["expected_anchor_logp"] = expected
    result["anchor_logp_exact"] = result["target_logp"] == expected
    if not result["anchor_logp_exact"]:
        raise RuntimeError(
            f"{name} endpoint realization mismatch: {result['target_logp']} != {expected}"
        )
    return result


def main() -> None:
    base.batch_response_logps_with_grad = trainer_response_logps_with_grad
    base.run_arm = anchor_validated_run_arm
    base.main()
    out_index = sys.argv.index("--out") + 1
    out = Path(sys.argv[out_index])
    payload = json.loads(out.read_text())
    payload["schema_version"] = "forkcert.qwen3-grpo-one-step-branch-repair.v0.2"
    payload["contract"] = str(
        Path(__file__).with_name(
            "QWEN3_GRPO_ONE_STEP_BRANCH_REPAIR_CONTRACT_V0_2_2026-07-17.md"
        ).resolve()
    )
    payload["trainer_realization_anchor_parity"] = all(
        bool(arm.get("anchor_logp_exact")) for arm in payload["arms"]
    )
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
