#!/usr/bin/env python
"""Verify whether a saved parameter delta losslessly reconstructs a post-state.

This is a construction gate for the common-evaluator smoke test.  It does not
measure an implementation effect and it does not make a population claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qwen3_grpo_natural_transition_v0_2 import json_sha256, named_tensor_hashes, sha256_file


SCHEMA_VERSION = "forkcert.qwen3-poststate-reconstruction-check.v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--transition-result", required=True)
    parser.add_argument("--parameter-updates", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir).resolve()
    result_path = Path(args.transition_result).resolve()
    updates_path = Path(args.parameter_updates).resolve()
    out_path = Path(args.out).resolve()

    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    transition = json.loads(result_path.read_text(encoding="utf-8"))
    recorded_artifact = transition["vector_artifacts"]["parameter_updates"]
    observed_update_sha256 = sha256_file(updates_path)

    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir,
        dtype=torch.float32,
        trust_remote_code=False,
        attn_implementation="sdpa",
        local_files_only=True,
        device_map="cpu",
    )
    named_parameters = list(model.named_parameters())
    named_buffers = list(model.named_buffers())
    _, observed_pre_digest = named_tensor_hashes(named_parameters)
    _, observed_pre_buffer_digest = named_tensor_hashes(named_buffers)

    parameter_names = [name for name, _ in named_parameters]
    with safe_open(updates_path, framework="pt", device="cpu") as handle:
        update_names = list(handle.keys())
        missing_updates = sorted(set(parameter_names) - set(update_names))
        unexpected_updates = sorted(set(update_names) - set(parameter_names))
        if missing_updates or unexpected_updates:
            raise ValueError(
                f"update key mismatch: missing={missing_updates[:5]}, "
                f"unexpected={unexpected_updates[:5]}"
            )
        with torch.no_grad():
            for name, parameter in named_parameters:
                update = handle.get_tensor(name)
                if update.shape != parameter.shape or update.dtype != parameter.dtype:
                    raise ValueError(
                        f"update metadata mismatch for {name}: "
                        f"update={tuple(update.shape)}/{update.dtype}, "
                        f"parameter={tuple(parameter.shape)}/{parameter.dtype}"
                    )
                parameter.add_(update)

    reconstructed_hashes, reconstructed_digest = named_tensor_hashes(named_parameters)
    _, reconstructed_buffer_digest = named_tensor_hashes(named_buffers)
    reconstructed_hashes_sha256 = json_sha256(reconstructed_hashes)

    checks: dict[str, bool] = {
        "update_artifact_sha256_exact": observed_update_sha256 == recorded_artifact["sha256"],
        "pre_parameter_digest_exact": (
            observed_pre_digest == transition["pre_state"]["parameter_digest"]
        ),
        "pre_buffer_digest_exact": (
            observed_pre_buffer_digest == transition["pre_state"]["buffer_digest"]
        ),
        "reconstructed_post_parameter_digest_exact": (
            reconstructed_digest == transition["post_state"]["parameter_digest"]
        ),
        "reconstructed_post_parameter_hash_set_digest_exact": (
            reconstructed_hashes_sha256
            == transition["post_state"]["parameter_hashes_sha256"]
        ),
        "reconstructed_post_buffer_digest_exact": (
            reconstructed_buffer_digest == transition["post_state"]["buffer_digest"]
        ),
    }
    valid = all(checks.values())
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "verdict": "LOSSLESS_RECONSTRUCTION" if valid else "RECONSTRUCTION_MISMATCH",
        "inputs": {
            "snapshot_dir": str(snapshot_dir),
            "transition_result": str(result_path),
            "transition_result_sha256": sha256_file(result_path),
            "parameter_updates": str(updates_path),
            "parameter_updates_sha256": observed_update_sha256,
        },
        "checks": checks,
        "observed": {
            "pre_parameter_digest": observed_pre_digest,
            "pre_buffer_digest": observed_pre_buffer_digest,
            "reconstructed_post_parameter_digest": reconstructed_digest,
            "reconstructed_post_parameter_hashes_sha256": reconstructed_hashes_sha256,
            "reconstructed_post_buffer_digest": reconstructed_buffer_digest,
        },
        "expected": {
            "pre_state": transition["pre_state"],
            "post_state": transition["post_state"],
        },
        "interpretation": (
            "The saved update is sufficient to reconstruct the recorded post-parameter state exactly."
            if valid
            else "The saved update is not a lossless representation of the recorded post-parameter state; "
            "a common-evaluator study must save or regenerate exact post parameters."
        ),
        "nonclaims": [
            "this construction check is not evidence of population bias",
            "this construction check is not a correctness oracle",
            "this construction check is not operator attribution",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "checks": checks}, indent=2))
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
