from __future__ import annotations

from pathlib import Path
from typing import Any


CLAIM_SCOPE = (
    "Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it "
    "into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal "
    "bound; raw numerical mismatch alone is not a claim."
)


def write_phase_report(
    path: str | Path,
    *,
    title: str,
    summary: str,
    confound_checklist: dict[str, bool | str],
    delta_self_summary: str,
    sections: dict[str, str],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## Claim Scope", CLAIM_SCOPE, "", "## Confound Checklist"]
    for key, value in confound_checklist.items():
        mark = "PASS" if value is True else "FAIL" if value is False else str(value)
        lines.append(f"- {key}: {mark}")
    lines.extend(["", "## Delta Self Control", delta_self_summary, "", "## Summary", summary])
    for heading, body in sections.items():
        lines.extend(["", f"## {heading}", body])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)
