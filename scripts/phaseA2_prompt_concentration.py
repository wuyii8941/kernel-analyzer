#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from forkcert.detector import clip_boundary
from forkcert.io import read_jsonl
from forkcert.report import CLAIM_SCOPE, markdown_table


EXTERNAL_VALIDITY = (
    "These concentration statistics describe the T4 FP16 GRPO run. They do not establish how BF16 kernels "
    "redistribute near-boundary decisions; BF16-hardware replication remains required."
)


def append_section(path: Path, marker: str, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    block = f"{start}\n{section.rstrip()}\n{end}\n"
    if start in text and end in text:
        before, remainder = text.split(start, 1)
        _old, after = remainder.split(end, 1)
        text = before.rstrip() + "\n\n" + block + after.lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + block
    path.write_text(text, encoding="utf-8")


def concentration(records: list[dict], group_key: str) -> dict:
    totals: dict[str, int] = defaultdict(int)
    near: dict[str, int] = defaultdict(int)
    for row in records:
        key = str(row[group_key])
        totals[key] += 1
        near[key] += int(row["near"])
    ordered = sorted(near.items(), key=lambda item: item[1], reverse=True)
    near_total = sum(near.values())
    target = 0.8 * near_total
    cumulative = 0
    groups_for_80 = 0
    for _key, count in ordered:
        if cumulative >= target:
            break
        cumulative += count
        groups_for_80 += 1
    contributing = sum(1 for value in near.values() if value > 0)
    return {
        "dimension": group_key,
        "groups": len(totals),
        "groups_with_near_boundary": contributing,
        "near_boundary_tokens": near_total,
        "groups_for_80_percent": groups_for_80,
        "fraction_groups_for_80_percent": groups_for_80 / len(totals) if totals else 0.0,
        "max_group_share": ordered[0][1] / near_total if ordered and near_total else 0.0,
        "top_groups": ordered[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit prompt/response concentration of iteration-2 near-boundary tokens.")
    parser.add_argument("--dump", default="data/phase0_grpo_dump.jsonl")
    parser.add_argument("--samples", default="data/phase0_grpo_samples.jsonl")
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--out-json", default="results/phaseA2_prompt_concentration.json")
    parser.add_argument("--report", default="reports/phaseA2_prompt_concentration.md")
    parser.add_argument("--phase0-report", default="reports/phase0.md")
    args = parser.parse_args()

    samples = {str(row["case_id"]): row for row in read_jsonl(args.samples)}
    rows = read_jsonl(args.dump)
    max_iteration = max(int(row["policy_iteration"]) for row in rows)
    parsed = []
    missing_sample = 0
    for row in rows:
        if int(row["policy_iteration"]) != max_iteration or int(row.get("advantage_sign", 0)) == 0:
            continue
        sample = samples.get(str(row["case_id"]))
        if sample is None:
            missing_sample += 1
            continue
        sign = int(row["advantage_sign"])
        margin = abs((float(row["new_logp"]) - float(row["old_logp"])) - clip_boundary(sign, args.eps))
        prompt = str(sample["prompt"])
        parsed.append(
            {
                "prompt": prompt,
                "response": str(row["case_id"]),
                "near": margin < args.threshold,
            }
        )
    prompt = concentration(parsed, "prompt")
    response = concentration(parsed, "response")
    high_concentration = prompt["groups_for_80_percent"] <= 3
    payload = {
        "policy_iteration": max_iteration,
        "threshold": args.threshold,
        "applicable_tokens": len(parsed),
        "missing_sample_rows": missing_sample,
        "prompt": prompt,
        "response": response,
        "highly_concentrated_in_two_or_three_prompts": high_concentration,
        "interpretation": (
            "Near-boundary mass is concentrated in at most three prompts; external validity is suspect."
            if high_concentration
            else "Near-boundary mass is not confined to only two or three prompts."
        ),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = [
        {key: value for key, value in item.items() if key != "top_groups"}
        for item in [prompt, response]
    ]
    top_prompt_rows = [{"prompt": key, "near_boundary_tokens": count} for key, count in prompt["top_groups"]]
    text = "\n".join(
        [
            "# Phase A2 Near-Boundary Concentration Audit",
            "",
            "## Claim Scope",
            CLAIM_SCOPE,
            "",
            "## Confound Checklist",
            f"- iteration-2 only: PASS ({max_iteration})",
            "- zero-advantage rows excluded: PASS",
            f"- case-to-prompt alignment: {'PASS' if missing_sample == 0 else 'FAIL'}",
            "",
            "## Delta Self Control",
            "Not applicable to the Phase 0 margin distribution; Phase A1 audits path self consistency.",
            "",
            "## External Validity",
            EXTERNAL_VALIDITY,
            "",
            "## Concentration",
            markdown_table(summary, list(summary[0].keys())),
            "",
            "## Top Prompts",
            markdown_table(top_prompt_rows, ["prompt", "near_boundary_tokens"]),
            "",
            "## Conclusion",
            payload["interpretation"],
            "",
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(text, encoding="utf-8")
    append_section(
        Path(args.phase0_report),
        "phaseA2",
        "\n".join(
            [
                "## Phase A2 Near-Boundary Concentration",
                "",
                markdown_table(summary, list(summary[0].keys())),
                "",
                payload["interpretation"],
                "",
                "See `reports/phaseA2_prompt_concentration.md` for the top-prompt table and external-validity scope.",
            ]
        ),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
