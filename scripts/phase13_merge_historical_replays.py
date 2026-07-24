#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independent historical issue replays.")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--out", default="results/phase13_historical_bug_replays.json")
    parser.add_argument("--report", default="reports/phase13_historical_bug_replays.md")
    args = parser.parse_args()
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.run]
    by_case: dict[str, list[dict]] = {}
    for run in runs:
        by_case.setdefault(run["case"], []).append(run)
    cases = []
    for name, values in sorted(by_case.items()):
        environments = {json.dumps(row["environment"], sort_keys=True) for row in values}
        eager_hashes = {row["outputs"]["eager"]["sha256"] for row in values}
        inductor_hashes = {
            row["outputs"]["inductor"]["sha256"]
            for row in values
            if "inductor" in row["outputs"]
        }
        all_wrong_result = all(row["upstream_wrong_result_reproduced"] for row in values)
        all_fail_closed = all(row.get("upstream_bug_fail_closed", False) for row in values)
        cases.append(
            {
                "case": name,
                "issue": values[0]["issue"],
                "url": values[0]["url"],
                "independent_runs": len(values),
                "same_environment": len(environments) == 1,
                "eager_self_exact": len(eager_hashes) == 1,
                "inductor_self_exact": len(inductor_hashes) == 1 if inductor_hashes else None,
                "all_runs_reproduce_upstream_wrong_result": all_wrong_result,
                "all_runs_fail_closed": all_fail_closed,
                "status": "wrong_result_reproduced" if all_wrong_result else "fail_closed" if all_fail_closed else "mixed",
                "max_abs_deltas": [
                    row["comparisons"]["eager_vs_inductor"]["max_abs_delta"] for row in values
                    if row["comparisons"]["eager_vs_inductor"] is not None
                ],
                "argmax_fork_all_runs": (
                    all(row["comparisons"]["eager_vs_inductor"]["argmax_fork"] for row in values)
                    if all(row["comparisons"]["eager_vs_inductor"] is not None for row in values)
                    else None
                ),
                "top16_candidate_set_fork_all_runs": (
                    all(row["comparisons"]["eager_vs_inductor"]["top16_candidate_set_fork"] for row in values)
                    if all(row["comparisons"]["eager_vs_inductor"] is not None for row in values)
                    else None
                ),
            }
        )
    payload = {
        "schema_version": "forkcert.historical_bug_replays.v1",
        "cases": cases,
        "claim_scope": "Independent replay of already reported upstream bugs; no new-bug discovery claim.",
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Phase 13 Historical Bug Replays",
        "",
        "## Objective",
        "",
        "Test ForkCert-adjacent decision signals on independently documented upstream wrong-result bugs.",
        "",
        "## Results",
        "",
        "| Case | Independent runs | Status | Eager self exact | Inductor self exact | Max delta | Argmax fork | top-16 set fork |",
        "|---|---:|---|---|---|---:|---|---|",
    ]
    for case in cases:
        max_delta = f"{max(case['max_abs_deltas']):.6g}" if case["max_abs_deltas"] else "n/a"
        lines.append(
            f"| [{case['issue']}]({case['url']}) | {case['independent_runs']} | "
            f"{case['status']} | {case['eager_self_exact']} | "
            f"{case['inductor_self_exact']} | {max_delta} | "
            f"{case['argmax_fork_all_runs']} | {case['top16_candidate_set_fork_all_runs']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            payload["claim_scope"],
            "",
            "These operator-level output decisions are replay gates. They are not PPO clipping, sampling, reward, or task-level consequences.",
            "",
            "## Artifacts",
            "",
            f"- `{args.out}`",
            "- `results/phase13_historical_bug_replays/`",
            "- `scripts/phase13_historical_bug_replay_once.py`",
            "",
        ]
    )
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "out": args.out, "report": args.report}, indent=2))


if __name__ == "__main__":
    main()
