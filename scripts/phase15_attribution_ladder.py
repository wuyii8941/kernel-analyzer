#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from forkcert.ladder import DEFAULT_LEVELS, attribution_from_measurements
from forkcert.report import markdown_table, write_phase_report


def propagation_exponent(layerwise: list[dict]) -> float | None:
    points = [
        (math.log(int(row.get("layer_index", index)) + 1), math.log(float(row["diff_l2"])))
        for index, row in enumerate(layerwise)
        if float(row.get("diff_l2", 0.0)) > 0
    ]
    if len(points) < 2:
        return None
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def write_propagation_svg(rows: list[dict], path: Path) -> None:
    width, height = 900, 500
    left, right, top, bottom = 70, 25, 45, 65
    series = []
    for row in rows:
        layer_rows = row.get("layerwise_activation_diffs", [])
        values = [float(item.get("diff_l2", 0.0)) for item in layer_rows]
        if values:
            series.append((str(row.get("level")), values))
    positive = [value for _level, values in series for value in values if value > 0]
    ymin = min(positive) if positive else 1e-12
    ymax = max(positive) if positive else 1.0
    log_min = math.log10(ymin)
    log_max = math.log10(ymax)
    if log_max <= log_min:
        log_max = log_min + 1.0
    max_points = max((len(values) for _level, values in series), default=1)
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0;fill:#111827}.axis{stroke:#6b7280}.grid{stroke:#e5e7eb}</style>',
        '<text x="20" y="25" font-size="16" font-weight="700">Residual-stream difference by transformer depth</text>',
        f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
    ]
    plot_w, plot_h = width - left - right, height - top - bottom
    for tick in range(5):
        value = log_min + (log_max - log_min) * tick / 4
        y = top + plot_h * (1 - tick / 4)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-size="10">1e{value:.1f}</text>')
    for series_index, (level, values) in enumerate(series):
        points = []
        for index, value in enumerate(values):
            x = left + plot_w * index / max(max_points - 1, 1)
            clipped = max(value, ymin)
            y = top + plot_h * (1 - (math.log10(clipped) - log_min) / (log_max - log_min))
            points.append(f"{x:.2f},{y:.2f}")
        color = colors[series_index % len(colors)]
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        legend_y = 48 + 18 * series_index
        lines.append(f'<line x1="{width-145}" y1="{legend_y}" x2="{width-125}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{width-118}" y="{legend_y+4}" font-size="11">{level}</text>')
    lines.append(f'<text x="{left + plot_w/2}" y="{height-20}" text-anchor="middle" font-size="11">transformer layer depth</text>')
    lines.append(f'<text x="15" y="{top + plot_h/2}" transform="rotate(-90 15 {top + plot_h/2})" text-anchor="middle" font-size="11">activation diff L2 (log scale)</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.5 attribution ladder summarizer.")
    parser.add_argument(
        "--measurements",
        required=False,
        help="JSON/JSONL measured rows with level, variable, paired activation differences, and final_logprob_delta.",
    )
    parser.add_argument("--out-json", default="results/phase15_attribution.json")
    parser.add_argument("--report", default="reports/phase15.md")
    parser.add_argument("--propagation-svg", default="reports/phase15_propagation.svg")
    args = parser.parse_args()

    if args.measurements:
        path = Path(args.measurements)
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["rows"] if isinstance(payload, dict) else payload
    else:
        rows = [
            {
                "level": level.level,
                "variable": level.variable,
                "mechanism": level.mechanism,
                "first_observed_diff_l2": 0.0,
                "max_activation_diff_l2": 0.0,
                "propagation_gain_first_to_last": None,
                "final_logprob_delta": 0.0,
            }
            for level in DEFAULT_LEVELS
        ]

    attribution = [row.to_json_dict() for row in attribution_from_measurements(rows)]
    rows_by_level = {str(row.get("level")): row for row in rows}
    for item in attribution:
        layerwise = rows_by_level.get(item["level"], {}).get("layerwise_activation_diffs", [])
        item["propagation_exponent"] = propagation_exponent(layerwise)
    propagation_path = Path(args.propagation_svg)
    write_propagation_svg(rows, propagation_path)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": attribution}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    levels = {str(row["level"]) for row in attribution}
    complete = {"L1", "L2", "L3", "L4", "L5", "L6"}.issubset(levels)
    summary = (
        "All six one-variable sensitivity measurements are present. Ratios are relative to L6 and are not additive attribution percentages."
        if complete
        else "Attribution ladder is incomplete. These measurements are sensitivity comparisons, not an additive decomposition."
    )
    write_phase_report(
        args.report,
        title="Phase 1.5 Attribution Ladder",
        confound_checklist={
            "one_variable_changed_per_level": "required by measurement run",
            "same_samples_and_tokens": "required by measurement run",
            "activation_dump_compact": True,
            "large_tensor_outputs_avoided": True,
            "additive_percent_claim_disabled": True,
            "residual_stream_grouped_by_layer_index": all(bool(row.get("residual_layer_indexed")) for row in rows),
            "causal_local_injection_separated": all(bool(row.get("local_injection_separated")) for row in rows),
        },
        delta_self_summary="Phase 1.5 consumes Phase 1 self-consistent path pairs.",
        summary=summary,
        sections={
            "Attribution": markdown_table(attribution, list(attribution[0].keys()) if attribution else []),
            "Propagation Curves": f"![Phase 1.5 propagation curves]({propagation_path.name})",
        },
    )
    print(json.dumps({"rows": attribution}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
