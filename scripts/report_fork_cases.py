#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.confounds import checklist_passed_for_claim, infer_confound_checklist
from forkcert.io import read_jsonl
from forkcert.report import CLAIM_SCOPE, markdown_table


def one_line_math(cert: dict) -> str:
    side = "positive advantage: boundary=log(1+eps)" if cert["advantage_sign"] > 0 else "negative advantage: boundary=log(1-eps)"
    return (
        f"{side}; old_logp={cert['old_logp']:.8g}, logp_ref={cert['logp_ref']:.8g}, "
        f"logp_alt={cert['logp_alt']:.8g}, margin={cert['clip_margin']:.8g}, "
        f"delta={cert['logprob_delta']:.8g}, clip_ref={cert['clip_ref']}, clip_alt={cert['clip_alt']}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate minimal actual-fork case report with confound checklist.")
    parser.add_argument("--certificates", default="results/phase4_certificates.jsonl")
    parser.add_argument("--out", default="reports/fork_cases.md")
    parser.add_argument("--max-cases", type=int, default=20)
    args = parser.parse_args()

    rows = read_jsonl(args.certificates)
    actual = [row for row in rows if row.get("actual_fork")]
    actual = sorted(actual, key=lambda row: (float(row.get("logprob_delta", 0.0)), float(row.get("clip_margin", 0.0))))[: args.max_cases]

    lines = ["# Fork Case Report", "", "## Claim Scope", CLAIM_SCOPE, "", "## Summary"]
    lines.append(f"- certificates: {len(rows)}")
    lines.append(f"- actual_fork_cases: {len([row for row in rows if row.get('actual_fork')])}")
    lines.append(f"- reported_cases: {len(actual)}")
    if not actual:
        lines.extend(["", "_No actual fork cases found._"])
    for idx, cert in enumerate(actual, start=1):
        checklist = infer_confound_checklist(cert)
        claim_ready = checklist_passed_for_claim(checklist)
        lines.extend(
            [
                "",
                f"## Case {idx}: {cert.get('case_id')} token {cert.get('token_index')}",
                "",
                f"- claim_ready_without_manual_review: {claim_ready}",
                f"- region: {cert.get('region')}",
                f"- token_id: {cert.get('token_id')}",
                f"- token_text: {json.dumps(cert.get('token_text'), ensure_ascii=False)}",
                f"- path_ref: {cert.get('path_ref')}",
                f"- path_alt: {cert.get('path_alt')}",
                "",
                "### One-Line Math",
                one_line_math(cert),
                "",
                "### Confound Checklist",
                markdown_table([item.to_json_dict() for item in checklist], ["name", "status", "evidence"]),
                "",
                "### Certificate",
                "```json",
                json.dumps(cert, indent=2, ensure_ascii=False, sort_keys=True),
                "```",
            ]
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(actual)} reported cases")


if __name__ == "__main__":
    main()
