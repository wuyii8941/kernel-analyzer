#!/usr/bin/env python
"""Post-reveal audit for the Qwen compiler-gradient development case.

This case has a public issue witness but no independently verified upstream
fix/patch artifact in this repository.  The audit therefore records issue
agreement and deliberately cannot upgrade the result to external-patch
coverage.  Keeping that gate in code prevents stale reports from silently
turning a development case into a localization-accuracy benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-report", required=True, type=Path)
    parser.add_argument("--buggy-run", required=True, type=Path)
    parser.add_argument("--fixed-run", required=True, type=Path)
    parser.add_argument(
        "--external-patch-manifest",
        type=Path,
        help="optional independently verified patch manifest; absent means issue agreement only",
    )
    parser.add_argument("--negative-linear-run", type=Path)
    parser.add_argument("--dynamo-eager-run", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    blind = json.loads(args.blind_report.read_text())
    buggy = json.loads(args.buggy_run.read_text())
    fixed = json.loads(args.fixed_run.read_text())
    external_patch = (
        json.loads(args.external_patch_manifest.read_text())
        if args.external_patch_manifest
        else None
    )
    negative = json.loads(args.negative_linear_run.read_text()) if args.negative_linear_run else None
    dynamo_eager = json.loads(args.dynamo_eager_run.read_text()) if args.dynamo_eager_run else None
    targets = [str(node.get("target", "")) for node in blind.get("candidate_operation_inventory", [])]
    operation_covers_mm = any("mm" in target for target in targets)
    silent_loss = not buggy["endpoint"]["requires_grad"] and not buggy["endpoint"]["has_grad_fn"]
    fixed_explicit = fixed["endpoint"]["has_grad_fn"] and not fixed["endpoint"]["backward_succeeds"]
    result = {
        "schema_version": "forkcert.qwen3-compiler-grad-case-score.v0.1",
        "case_id": blind["case_id"],
        "blind_observation": blind["oracle"],
        "post_reveal_issue_context": {
            "issue": "https://github.com/pytorch/pytorch/issues/181581",
            "mechanism": "AOTAutograd/torch.compile loses higher-order gradient metadata for selected operations",
            "independent_patch_manifest": external_patch,
            "independent_patch_verified": external_patch is not None,
        },
        "scoring": {
            "blind_semantic_effect_reproduced": bool(blind["oracle"]["semantic_disagreement"]),
            "numeric_delta_would_miss_it": bool(blind["oracle"]["numeric_equal"]),
            "candidate_operation_inventory_covers_mm": operation_covers_mm,
            "buggy_run_shows_silent_metadata_loss": silent_loss,
            "fixed_run_is_explicit_not_silent": fixed_explicit,
            "linear_negative_control_preserves_metadata": bool(
                negative is not None
                and negative["endpoint"]["requires_grad"]
                and negative["endpoint"]["has_grad_fn"]
            ),
            "same_version_dynamo_eager_preserves_metadata": bool(
                dynamo_eager is not None
                and dynamo_eager["endpoint"]["requires_grad"]
                and dynamo_eager["endpoint"]["has_grad_fn"]
            ),
            "stage_localization_supports_aot_autograd": bool(
                dynamo_eager is not None
                and dynamo_eager["endpoint"]["requires_grad"]
                and not buggy["endpoint"]["requires_grad"]
            ),
            "claim_level": (
                "COMPILER_OPERATION_CANDIDATE_WITH_EXTERNAL_PATCH_COVERAGE"
                if operation_covers_mm and silent_loss and fixed_explicit and external_patch is not None
                else "BLIND_STAGE_OPERATION_CANDIDATE_WITH_POST_REVEAL_ISSUE_AGREEMENT"
                if operation_covers_mm and silent_loss
                else "SEMANTIC_EFFECT_ONLY"
            ),
        },
        "limitations": [
            "isolated Qwen3 projection boundary case, not a complete Qwen3 training run",
            "the public issue provides mechanism context but is not an independently verified patch artifact",
            "a newer-run comparison is a regression observation, not external-patch coverage",
            "operation coverage does not prove a unique compiler pass or source line",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
