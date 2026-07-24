#!/usr/bin/env python
"""Trainer-parity, branch-functional Qwen GRPO one-step repair."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import theory_oracle.qwen3_grpo_branch_repair_oracle as base
from theory_oracle.qwen3_grpo_branch_repair_oracle_v0_2 import (
    trainer_response_logps_with_grad,
)


_scoring_hashes: list[str] = []


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def recorded_trainer_response_logps_with_grad(
    tokenizer: Any, model: Any, config: Any, samples: list[dict[str, Any]]
):
    logps, token_ids = trainer_response_logps_with_grad(tokenizer, model, config, samples)
    _scoring_hashes.append(_tensor_sha256(logps))
    return logps, token_ids


def branch_functional_surrogate_loss(
    torch: Any,
    logps: Any,
    old: Any,
    advantages: Any,
    eps: float,
    target: int,
    forced_clip: bool | None,
):
    ratio = torch.exp(logps - old)
    clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    objectives = torch.minimum(unclipped, clipped)
    if forced_clip is not None:
        if forced_clip:
            boundary = 1.0 + eps if float(advantages[target]) > 0.0 else 1.0 - eps
            # The clipped branch is a constant function of the selected logp.
            # The zero-multiplied term keeps the objective attached to the graph
            # while fixing its derivative to exactly zero.
            forced = logps[target] * 0.0 + boundary * advantages[target]
        else:
            forced = unclipped[target]
        objectives = torch.cat(
            (objectives[:target], forced.unsqueeze(0), objectives[target + 1 :])
        )
    return -objectives.mean(), ratio, clipped_ratio


_original_run_arm = base.run_arm


def validated_run_arm(*args: Any, **kwargs: Any):
    _scoring_hashes.clear()
    result = _original_run_arm(*args, **kwargs)
    name = str(args[0])
    event = args[5]
    expected = float(event["logp_ref"] if name == "A_reference" else event["logp_alt"])
    result["expected_anchor_logp"] = expected
    result["anchor_logp_exact"] = result["target_logp"] == expected
    result["scoring_call_logps_sha256"] = list(_scoring_hashes)
    result["measured_logps_sha256"] = _scoring_hashes[-1]
    result["scoring_self_stable"] = len(set(_scoring_hashes)) == 1
    if not result["anchor_logp_exact"]:
        raise RuntimeError(
            f"{name} endpoint realization mismatch: {result['target_logp']} != {expected}"
        )
    if not result["scoring_self_stable"]:
        raise RuntimeError(f"{name} scoring calls were not self-stable")
    return result


def main() -> None:
    base.batch_response_logps_with_grad = recorded_trainer_response_logps_with_grad
    base.selected_surrogate_loss = branch_functional_surrogate_loss
    base.run_arm = validated_run_arm
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    contract = str(
        Path(__file__).with_name(
            "QWEN3_GRPO_ONE_STEP_BRANCH_REPAIR_CONTRACT_V0_3_2026-07-17.md"
        ).resolve()
    )
    try:
        base.main()
        payload = json.loads(out.read_text(encoding="utf-8"))
        by_name = {arm["arm"]: arm for arm in payload["arms"]}
        b_arm, c_arm = by_name["B_candidate"], by_name["C_branch_repair"]
        full_bc_identity = (
            b_arm["measured_logps_sha256"] == c_arm["measured_logps_sha256"]
            and b_arm["compile_audit"]["graph_hashes"]
            == c_arm["compile_audit"]["graph_hashes"]
            and b_arm["compile_audit"]["graph_nodes"]
            == c_arm["compile_audit"]["graph_nodes"]
        )
        zero_gradient_gate = c_arm["target_logp_loss_gradient"] == 0.0
        if not full_bc_identity:
            raise RuntimeError("B/C full scoring or compiled graph identity mismatch")
        if not zero_gradient_gate:
            raise RuntimeError("C did not realize the flat reference clipped branch")

        payload["schema_version"] = "forkcert.qwen3-grpo-one-step-branch-repair.v0.3"
        payload["contract"] = contract
        payload["status"] = "MECHANICALLY_VALID_PENDING_INDEPENDENT_AUDIT"
        payload["trainer_realization_anchor_parity"] = all(
            bool(arm["anchor_logp_exact"]) for arm in payload["arms"]
        )
        payload["full_batch_bc_scoring_identity"] = full_bc_identity
        payload["reference_branch_functional_gate"] = zero_gradient_gate
        payload["reference_directed_repair_effect_l2"] = (
            payload["distances"]["A_B"]["l2"] - payload["distances"]["A_C"]["l2"]
        )
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception as error:
        failure = {
            "schema_version": "forkcert.qwen3-grpo-one-step-branch-repair.v0.3",
            "contract": contract,
            "status": "INVALID",
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
