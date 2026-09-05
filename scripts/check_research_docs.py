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
    "README.md", "PROJECT.md", "results/README.md", "docs/README.md",
    "docs/current_mainline.md", "docs/method.md", "docs/claims.md",
    "docs/case_evidence_map.md", "docs/novelty_positioning.md",
    "docs/liger_single_boundary_collapse_experiment.md",
    "docs/liger_silu_long_horizon_recheck.md",
    "docs/mainline_cleanup_20260905.md",
    "docs/training_numerical_analysis_v1.md",
)
LINK = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\n]+)\)")


def check_links() -> list[str]:
    errors = []
    for name in CURRENT_DOCS:
        path = ROOT / name
        for match in LINK.finditer(path.read_text(encoding="utf-8")):
            target = match.group(1).strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith("#"):
                continue
            dest = path.parent / unquote(parsed.path)
            if not dest.exists():
                errors.append(f"Missing link in {name}: {target}")
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
    mainline = (ROOT / "docs/current_mainline.md").read_text().replace("−", "-")
    for number in expected:
        if number not in mainline:
            errors.append(f"Liger figure missing or stale in mainline: {number}")

    five = json.loads((ROOT / (
        "results/property/training_bias_profile_v2/five_case_summary.json"
    )).read_text())
    for name, case in five["cases"].items():
        if case["optimizer"]["moments"] != "ZERO_AT_EVERY_INPUT_STATE":
            errors.append(f"Recheck documented five-case moment policy: {name}")

    legacy = json.loads((ROOT / (
        "results/property/declared_persistent_4096/all_bias_case_audit.json"
    )).read_text())
    evidence_map = (ROOT / "docs/case_evidence_map.md").read_text()
    for text in (
        f'{legacy["unique_matrix_case_count"]} 个主矩阵 ID',
        f'{legacy["case_count"]} 行',
        f'{legacy["final_case_count"]} 行旧 bias+loss 标签',
    ):
        if text not in evidence_map:
            errors.append(f"Legacy audit count differs from evidence map: {text}")

    analysis = json.loads((ROOT / (
        "results/property/training_numerical_analysis_v1/summary.json"
    )).read_text())
    generated = (ROOT / "docs/training_numerical_analysis_v1.md").read_text()
    for row in analysis["rows"]:
        if row["fixed_suite_total_rms"] is not None:
            number = f'{100 * row["fixed_suite_total_rms"]:.3f}%'
            if number not in generated:
                errors.append(f"Generated analysis report is stale: {row['case_id']} {number}")

    utility = json.loads((ROOT / (
        "results/property/training_numerical_analysis_v1/training_utility_summary.json"
    )).read_text())
    utility_gap = f'{utility["validation_loss"]["candidate_minus_reference"]:+.5f}'
    if utility_gap not in generated:
        errors.append(f"Generated training-utility result is stale: {utility_gap}")
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
    errors = check_links() + check_figures()
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
