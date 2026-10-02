#!/usr/bin/env python3
"""Read-only checks for research-document links, figures and protected evidence.

This is a maintenance check, not verification of experimental execution or a
bias theorem. It writes no reports, caches or experiment files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCS = (
    "README.md", "results/README.md", "docs/README.md",
    "docs/current_mainline.md", "docs/method.md", "docs/claims.md",
    "docs/root_cause_closure_current.md", "docs/novelty_positioning.md",
    "docs/liger_single_boundary_collapse_experiment.md",
    "docs/system.md",
    "docs/numerical_coverage_execution.md", "docs/statistics_experiment_alignment.md",
    "docs/population_inference_contract.md",
    "docs/observed_kernel_catalog_v2.md",
    "docs/bias_checker_triton_examples_20260914.md",
    "docs/optimizer_update_family_audit.md",
    "docs/adamw8bit_residual_structure_20260913.md",
    "docs/liger_language_mechanism_followup.md",
    "docs/source_aligned_repair.md",
    "docs/l23_qproj_tile.md",
    "docs/effective_antithetic_symmetry.md",
    "docs/fused_rotary_position_scaling_audit.md",
)

RETIRED_DUPLICATE_DOCS = (
    "direct_persistence_evidence.md",
    "direct_persistence_heldout.md",
    "direct_persistence_optimizer.md",
    "direct_persistence_screen.md",
    "extended_unified_profiles.md",
    "five_case_training_bias_profile_v2.md",
    "prospective_training_bias_profiles.md",
    "training_bias_profile_v2.md",
    "unified_measurement_round.md",
    "three_mechanism_profiles.md",
    "operator_candidate_screening_20260914.md",
    "denominator.md",
    "population_exceedance_inference.md",
    "gemma_llama_operator_scan.md",
    "unmeasured_triton_family_frontier.md",
    "training_numerical_analysis_v2.md",
    "liger_fp32_order_training_20260915.md",
    "qwen_saved_p_declared_trajectory_20260915.md",
    "liger_silu_long_horizon_recheck.md",
    "gemma_rms_order_intervention_20260914.md",
)
LINK = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\n]+)\)")


def check_links() -> list[str]:
    errors = []
    for name in CURRENT_DOCS:
        path = ROOT / name
        if not path.is_file():
            continue  # check_doc_contract reports missing entry points.
        for match in LINK.finditer(path.read_text(encoding="utf-8")):
            target = match.group(1).strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith("#"):
                continue
            dest = path.parent / unquote(parsed.path)
            if not dest.exists():
                errors.append(f"Missing link in {name}: {target}")
    return errors


def check_doc_contract() -> list[str]:
    errors = []
    for name in CURRENT_DOCS:
        if not (ROOT / name).is_file():
            errors.append(f"Missing current document: {name}")
    for name in RETIRED_DUPLICATE_DOCS:
        if (ROOT / "docs" / name).exists():
            errors.append(f"Retired duplicate document is present: docs/{name}")
    return errors


def check_figures() -> list[str]:
    errors = []
    result = json.loads((ROOT / (
        "results/property/single_point_collapse_v2/full_10000_summary.json"
    )).read_text())
    loss = result["validation_loss"]
    params = result["final_parameters"]
    expected = (
        f'{loss["difference_at_4096"]:+.5f}',
        f'{loss["difference_at_10000"]:+.5f}',
        f'{100 * params["relative_l2_difference_at_4096"]:.2f}%',
        f'{100 * params["relative_l2_difference_at_10000"]:.2f}%',
    )
    # The current-mainline page is intentionally concise; detailed Liger
    # figures live in the single case report.  Check both maintained entry
    # points so shortening the mainline cannot make a real result stale.
    figure_text = "\n".join(
        (ROOT / name).read_text().replace("−", "-")
        for name in (
            "docs/current_mainline.md",
            "docs/root_cause_closure_current.md",
            "docs/liger_single_boundary_collapse_experiment.md",
        )
    )
    for number in expected:
        if number not in figure_text:
            errors.append(f"Liger figure missing or stale in maintained case report: {number}")

    five = json.loads((ROOT / (
        "results/property/training_bias_profile_v2/five_case_summary.json"
    )).read_text())
    for name, case in five["cases"].items():
        if case["optimizer"]["moments"] != "ZERO_AT_EVERY_INPUT_STATE":
            errors.append(f"Recheck documented five-case moment policy: {name}")

    legacy = json.loads((ROOT / (
        "results/property/declared_persistent_4096/all_bias_case_audit.json"
    )).read_text())
    case_ledger = (ROOT / "docs/root_cause_closure_current.md").read_text()
    for text in (
        f'{legacy["unique_matrix_case_count"]} 个主矩阵 ID',
        f'{legacy["case_count"]} 行',
        f'{legacy["final_case_count"]} 行旧 bias+loss 标签',
    ):
        if text not in case_ledger:
            errors.append(f"Legacy audit count differs from case ledger: {text}")

    return errors


def check_protected(base: str) -> list[str]:
    # git diff includes both staged and unstaged tracked changes relative to base.
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", base, "--"], cwd=ROOT, text=True,
    ).splitlines()
    generated = (
        "results/mainline_case_roles.json",
        "results/property/training_numerical_analysis_v1/",
    )
    return [f"Protected content changed since {base}: {name}" for name in changed
            if (name.startswith("results/") and name != "results/README.md"
                and name != generated[0] and not name.startswith(generated[1]))
            or name == "docs/talk_beyond_tolerance.md"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Optional pre-cleanup commit for evidence protection")
    args = parser.parse_args()
    errors = check_doc_contract() + check_links() + check_figures()
    if args.base:
        errors.extend(check_protected(args.base))
    for error in errors:
        print(error)
    if errors:
        return 1
    print(f"OK: {len(CURRENT_DOCS)} current documents; referenced figures agree.")
    if args.base:
        print("OK: tracked experiment results and the user-maintained talk are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
