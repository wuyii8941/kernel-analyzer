#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from html import escape
from collections import defaultdict
from pathlib import Path

from forkcert.report import markdown_table, write_phase_report
from forkcert.stats import percentile


THRESHOLDS = [1e-4, 1e-3, 1e-2, 5e-2]


def boundary(sign: int, eps: float) -> float:
    if sign > 0:
        return math.log1p(eps)
    if sign < 0:
        return math.log1p(-eps)
    raise ValueError("zero advantage sign has no clipping boundary")


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_key(row: dict) -> str:
    epoch = row.get("epoch", "na")
    minibatch = row.get("minibatch", row.get("minibatch_index", "na"))
    return f"epoch={epoch},minibatch={minibatch}"


def group_tuple(row: dict) -> tuple[int, int]:
    epoch = row.get("rollout_batch", row.get("epoch", -1))
    minibatch = row.get("policy_iteration", row.get("minibatch", row.get("minibatch_index", -1)))
    try:
        return int(epoch), int(minibatch)
    except (TypeError, ValueError):
        return -1, -1


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    n = len(ordered)

    out = {
        "n": n,
        "p0.1": percentile(ordered, 0.1),
        "p1": percentile(ordered, 1),
        "p5": percentile(ordered, 5),
        "p50": percentile(ordered, 50),
    }
    for t in THRESHOLDS:
        out[f"P(margin<{t:g})"] = float(sum(1 for value in ordered if value < t) / n) if n else 0.0
    return out


def write_histogram_svg(by_iteration: dict[int, list[float]], path: Path) -> None:
    iterations = sorted(by_iteration)
    panel_width = 760
    panel_height = 190
    width = 840
    height = 70 + panel_height * max(1, len(iterations))
    left = 65
    top = 45
    bins = [-12.0 + 0.5 * index for index in range(27)]  # 1e-12 through 1e1
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0;fill:#111827}.axis{stroke:#6b7280;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}</style>',
        '<text x="20" y="25" font-size="16" font-weight="700">GRPO clipping-margin distribution by policy iteration</text>',
    ]
    if not iterations:
        lines.append('<text x="20" y="55" font-size="13">No non-zero-advantage margins.</text>')
    for panel, iteration in enumerate(iterations):
        values = by_iteration[iteration]
        counts = [0] * (len(bins) - 1)
        for value in values:
            log_value = math.log10(max(float(value), 1e-12))
            index = min(max(int((log_value - bins[0]) / 0.5), 0), len(counts) - 1)
            counts[index] += 1
        peak = max(counts, default=1) or 1
        y0 = top + panel * panel_height
        plot_height = 125
        bar_width = panel_width / len(counts)
        lines.extend(
            [
                f'<text x="{left}" y="{y0}" font-size="13" font-weight="700">policy_iteration={escape(str(iteration))} (n={len(values)})</text>',
                f'<line class="axis" x1="{left}" y1="{y0 + plot_height + 10}" x2="{left + panel_width}" y2="{y0 + plot_height + 10}"/>',
            ]
        )
        for index, count in enumerate(counts):
            bar_height = plot_height * count / peak
            x = left + index * bar_width
            y = y0 + plot_height + 10 - bar_height
            lines.append(
                f'<rect x="{x + 0.5:.2f}" y="{y:.2f}" width="{max(bar_width - 1, 0.5):.2f}" height="{bar_height:.2f}" fill="{colors[panel % len(colors)]}"/>'
            )
        for exponent in [-12, -9, -6, -3, 0]:
            x = left + (exponent - bins[0]) / (bins[-1] - bins[0]) * panel_width
            lines.append(f'<line class="grid" x1="{x:.2f}" y1="{y0 + 10}" x2="{x:.2f}" y2="{y0 + plot_height + 10}"/>')
            lines.append(f'<text x="{x:.2f}" y="{y0 + plot_height + 28}" text-anchor="middle" font-size="10">1e{exponent}</text>')
        lines.append(f'<text x="{left + panel_width / 2}" y="{y0 + plot_height + 45}" text-anchor="middle" font-size="11">clip_margin (log scale)</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 margin histogram from rollout/minibatch JSONL dumps.")
    parser.add_argument("--input", required=True, help="JSONL with new_logp, old_logp, advantage or advantage_sign.")
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--out-json", default="results/phase0_margin_summary.json")
    parser.add_argument("--report", default="reports/phase0.md")
    parser.add_argument("--metadata", default=None, help="Optional training metadata JSON; defaults beside the input dump.")
    parser.add_argument("--histogram-svg", default="reports/phase0_margin_hist.svg")
    parser.add_argument("--fail-on-downgrade", action="store_true", help="Exit non-zero after writing outputs if late-minibatch near-boundary mass fails the Go gate.")
    parser.add_argument("--require-real-training", action="store_true", help="Reject synthetic PPO-style probes as canonical Phase 0 evidence.")
    args = parser.parse_args()

    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    by_iteration: dict[int, list[float]] = defaultdict(list)
    skipped = 0
    input_rows = load_rows(Path(args.input))
    metadata_path = Path(args.metadata) if args.metadata else Path(args.input).with_suffix(".metadata.json")
    training_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    deterministic_warnings = (training_metadata or {}).get("deterministic_warn_messages")
    environment = (training_metadata or {}).get("environment") or {}
    deterministic_valid = (
        environment.get("deterministic_algorithms") is True
        and environment.get("deterministic_warn_only") is True
        and environment.get("cudnn_benchmark") is False
        and environment.get("cublas_workspace_config") == ":4096:8"
        and environment.get("pythonhashseed") == "0"
    )
    training_kinds = sorted({str(row.get("training_kind", "unknown")) for row in input_rows})
    advantage_sources = sorted({str(row.get("advantage_source", "unknown")) for row in input_rows})
    old_logp_sources = sorted({str(row.get("old_logp_source", "unknown")) for row in input_rows})
    for row in input_rows:
        sign = row.get("advantage_sign")
        if sign is None:
            adv = float(row["advantage"])
            sign = 1 if adv > 0 else -1 if adv < 0 else 0
        sign = int(sign)
        if sign == 0:
            skipped += 1
            continue
        margin = abs((float(row["new_logp"]) - float(row["old_logp"])) - boundary(sign, args.eps))
        key = group_tuple(row)
        grouped[key].append(margin)
        by_iteration[key[1]].append(margin)

    summaries = []
    all_values = []
    max_iteration = max(by_iteration, default=-1)
    for key, values in sorted(grouped.items()):
        all_values.extend(values)
        row = {"group": f"rollout={key[0]},iteration={key[1]}", **summarize(values)}
        summaries.append(row)
    overall = {"group": "overall", **summarize(all_values)} if all_values else {"group": "overall", "n": 0}
    iteration_summaries = [
        {"group": f"policy_iteration={iteration}", **summarize(values)}
        for iteration, values in sorted(by_iteration.items())
    ]
    late_values = by_iteration.get(max_iteration, [])
    late = (
        {"group": f"late_policy_iteration={max_iteration}", **summarize(late_values)}
        if late_values
        else {"group": "late_policy_iteration=none", "n": 0}
    )
    histogram_path = Path(args.histogram_svg)
    write_histogram_svg(by_iteration, histogram_path)

    real_training = (
        training_kinds == ["trl_grpo"]
        and advantage_sources == ["trl_group_normalized_rewards"]
        and old_logp_sources == ["trl_old_per_token_logps"]
    )
    payload = {
        "overall": overall,
        "late_minibatches": late,
        "policy_iterations": iteration_summaries,
        "groups": summaries,
        "skipped_zero_advantage": skipped,
        "eps": args.eps,
        "provenance": {
            "training_kinds": training_kinds,
            "advantage_sources": advantage_sources,
            "old_logp_sources": old_logp_sources,
            "canonical_real_training": real_training,
        },
        "determinism": {
            "metadata_present": training_metadata is not None,
            "warn_messages_recorded": deterministic_warnings is not None,
            "warn_message_count": len(deterministic_warnings or []),
            "settings_verified": deterministic_valid,
        },
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    go = late.get("P(margin<0.01)", 0.0) >= 0.001 and (real_training or not args.require_real_training)
    recommendation = (
        "GO: late-minibatch near-boundary mass is sufficient for clipping fork scan."
        if go
        else "DOWNGRADE: clipping near-boundary mass is low; continue with coverage certification or Phase 7 candidates."
    )
    columns = ["group", "n", "p0.1", "p1", "p5", "p50"] + [f"P(margin<{t:g})" for t in THRESHOLDS]
    write_phase_report(
        args.report,
        title="Phase 0 Margin Histogram",
        confound_checklist={
            "fixed_response_tokens": "N/A for margin dump",
            "real_old_logp_present": old_logp_sources == ["trl_old_per_token_logps"],
            "real_grpo_training": real_training,
            "zero_advantage_excluded_as_not_applicable": True,
            "zero_advantage_rows_excluded": skipped,
            "deterministic_env_recorded": deterministic_valid,
            "warn_only_messages_recorded": deterministic_warnings is not None,
            "late_minibatch_gate_used": True,
        },
        delta_self_summary="N/A in Phase 0; delta_self is checked in Phase 1.",
        summary=recommendation,
        sections={
            "Overall": markdown_table([overall], columns),
            "Late Minibatches": markdown_table([late], columns),
            "By Policy Iteration": markdown_table(iteration_summaries, columns),
            "By Minibatch": markdown_table(summaries, columns),
            "Margin Histogram": f"![Phase 0 margin histogram]({histogram_path.name})",
            "Deterministic Warnings": (
                "\n".join(f"- {message}" for message in deterministic_warnings)
                if deterministic_warnings
                else "_None recorded._"
            ),
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if (args.fail_on_downgrade and not go) or (args.require_real_training and not real_training):
        print(
            "Phase 0 downgrade gate failed: require canonical TRL GRPO provenance and "
            "late-policy-iteration P(margin<1e-2) >= 0.1%.",
            file=sys.stderr,
        )
        raise SystemExit(20)


if __name__ == "__main__":
    main()
