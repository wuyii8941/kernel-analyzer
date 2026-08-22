#!/usr/bin/env python3
"""Compare the 16-step local persistence triage score with simple baselines.

The primary cohort is the frozen 12-row residual-nonzero, parameter-reachable
screen-negative audit.  Labels are the independent 32-step local sign-flip
verdicts.  This keeps optimizer, runner, and consequence protocol aligned.  A
single historical Phi source-persistent row is reported separately and is not
pooled into the primary AUROC.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import random
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/joint_bias_formation_v1"
HOTSPOT = ROOT / "results/property/bias_formation/hotspot_search"


def load(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = math.fsum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def stratified_bootstrap_auc(
    labels: list[bool], scores: list[float], *, draws: int = 10_000, seed: int = 20260822
) -> list[float] | None:
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        return None
    generator = random.Random(seed)
    values = []
    for _ in range(draws):
        sampled_positive = [generator.choice(positive) for _ in positive]
        sampled_negative = [generator.choice(negative) for _ in negative]
        values.append(auc(
            [True] * len(sampled_positive) + [False] * len(sampled_negative),
            sampled_positive + sampled_negative,
        ))
    values.sort()
    return [values[int(0.025 * draws)], values[int(0.975 * draws) - 1]]


def operating_point(labels: list[bool], scores: list[float], threshold: float) -> dict[str, Any]:
    predictions = [score > threshold for score in scores]
    tp = sum(prediction and label for prediction, label in zip(predictions, labels))
    fp = sum(prediction and not label for prediction, label in zip(predictions, labels))
    fn = sum(not prediction and label for prediction, label in zip(predictions, labels))
    tn = sum(not prediction and not label for prediction, label in zip(predictions, labels))
    return {
        "threshold_rule": f"score>{threshold}",
        "flagged": tp + fp,
        "total": len(labels),
        "flag_rate": (tp + fp) / len(labels),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "recall": tp / max(tp + fn, 1),
        "miss_rate": fn / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def split_top_level(text: str) -> list[str]:
    parts, start, depth = [], 0, 0
    for index, char in enumerate(text):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def triton_reduction_extent(source: str, symbol: str, exact_endpoint: str) -> int:
    definition = re.search(r"def\s+" + re.escape(symbol) + r"\(([^\n]*)\):", source)
    if definition is None:
        raise RuntimeError(f"missing Triton definition: {symbol}")
    signature = [value.strip().split(" ")[0].split(":")[0]
                 for value in split_top_level(definition.group(1))]
    node = exact_endpoint.rsplit(":", 1)[-1].replace("backward_g1__", "")
    call_text = None
    for comment in re.finditer(r"# Topologically Sorted Source Nodes:.*" + re.escape(node) + r".*", source):
        tail = source[comment.end():comment.end() + 4000]
        call = re.search(re.escape(symbol) + r"\.run\(([^\n]*)\)", tail)
        if call is not None:
            call_text = call.group(1)
            break
    if call_text is None:
        calls = list(re.finditer(re.escape(symbol) + r"\.run\(([^\n]*)\)", source))
        if not calls:
            raise RuntimeError(f"missing Triton call: {symbol}")
        call_text = calls[-1].group(1)
    arguments = split_top_level(call_text)
    extents = []
    for index, name in enumerate(signature):
        if re.fullmatch(r"r\d*_numel", name):
            value = arguments[index]
            if not re.fullmatch(r"\d+", value):
                raise RuntimeError(f"dynamic reduction extent for {symbol}: {value}")
            extents.append(int(value))
    result = math.prod(extents) if extents else 1
    return int(result)


def external_mm_reduction_extent(source: str, scheduler_node: str) -> int:
    lines = source.splitlines()
    matches = [
        index for index, line in enumerate(lines)
        if "extern_kernels.mm(" in line and f"out={scheduler_node}" in line
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one external MM call for {scheduler_node}, got {len(matches)}")
    index = matches[0]
    call = lines[index]
    body = call[call.index("extern_kernels.mm(") + len("extern_kernels.mm("):]
    body = body.rsplit(", out=", 1)[0]
    arguments = split_top_level(body)
    if len(arguments) != 2:
        raise RuntimeError("external MM does not have two matrix operands")

    def shape(argument: str) -> tuple[int, int]:
        embedded = re.search(r"reinterpret_tensor\([^,]+,\s*\((\d+),\s*(\d+)\)", argument)
        if embedded:
            return int(embedded.group(1)), int(embedded.group(2))
        name = argument.strip()
        pattern = re.compile(
            r"assert_size_stride\(" + re.escape(name) + r",\s*\((\d+),\s*(\d+)\)"
        )
        for prior in reversed(lines[:index]):
            match = pattern.search(prior)
            if match:
                return int(match.group(1)), int(match.group(2))
        raise RuntimeError(f"cannot bind MM operand shape: {argument}")

    left, right = shape(arguments[0]), shape(arguments[1])
    if left[1] != right[0]:
        raise RuntimeError(f"MM reduction mismatch: {left} x {right}")
    return left[1]


def bind_reduction_extent(consequence: dict[str, Any], plan_case: dict[str, Any],
                          plan: dict[str, Any], hotspot_rows: list[dict[str, Any]]) -> dict[str, Any]:
    architecture = consequence["architecture"]
    model = "phi4" if architecture == "phi" else architecture
    matches = [
        row for row in hotspot_rows
        if row["model"] == model
        and int(row["sequence_length"]) == int(plan["sequence_length"])
        and row["task_id"] == plan_case["task_id"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"hotspot binding is not unique for {consequence['case_id']}")
    binding = matches[0]
    release = ROOT / consequence["release"]
    source_path = release / "trace" / binding["region_source_path"]
    source = source_path.read_text(encoding="utf-8")
    symbol = binding["region_symbol"]
    if symbol.startswith("triton_"):
        extent = triton_reduction_extent(source, symbol, binding["exact_aot_endpoint_id"])
        method = "TRITON_RNUMEL_FROM_EXACT_EXECUTED_CALL"
    elif symbol == "mm":
        tasks = load(release / "same_dtype_tasks.json.gz")
        task = next(row for row in tasks["rows"] if row["task_id"] == plan_case["task_id"])
        scheduler = task["compiler_origin_rows"][0]["scheduler_node"]
        extent = external_mm_reduction_extent(source, scheduler)
        method = "EXTERNAL_MM_K_FROM_EXACT_EXECUTED_CALL"
    else:
        raise RuntimeError(f"unsupported reduction baseline symbol: {symbol}")
    return {
        "reduction_extent": extent,
        "reduction_binding_method": method,
        "implementation_kind": binding["implementation_kind"],
        "symbol": symbol,
        "source": str(source_path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE / "oracle_baselines")
    args = parser.parse_args()
    consequence_summary = load(BASE / "consequence_summary.json")
    rms = load(BASE / "rms_persistence/rms_persistence.json")
    rms_by_case = {
        (row["architecture"], row["case_id"]): float(row["local_rms"])
        for row in rms["live_trajectory_sample"]["rows"]
    }
    hotspots = load(HOTSPOT / "multishape_backward_hotspots.json")["rows"]
    rows = []
    for summary_row in consequence_summary["rows"]:
        consequence = load(ROOT / summary_row["source"])
        plan = load(ROOT / consequence["case_plan"])
        plan_case = next(row for row in plan["cases"] if row["case_id"] == consequence["case_id"])
        reduction = bind_reduction_extent(consequence, plan_case, plan, hotspots)
        rows.append({
            "case_id": consequence["case_id"],
            "architecture": consequence["architecture"],
            "family": plan_case["family"],
            "label_32step_local_persistent": bool(
                summary_row["levels"]["local"]["above_sign_flip_95"]
            ),
            "oracle_prefix16_local_A": float(
                summary_row["prefix16"]["local"]["coherence_amplification"]
            ),
            "local_effective_update_rms": rms_by_case[
                (consequence["architecture"], consequence["case_id"])
            ],
            "candidate_dtype_is_bfloat16": 1.0,
            **reduction,
        })

    labels = [bool(row["label_32step_local_persistent"]) for row in rows]
    features = {
        "prefix16_local_persistence_oracle": [row["oracle_prefix16_local_A"] for row in rows],
        "local_effective_update_rms": [row["local_effective_update_rms"] for row in rows],
        "dtype_is_bfloat16": [row["candidate_dtype_is_bfloat16"] for row in rows],
        "compiled_reduction_extent": [float(row["reduction_extent"]) for row in rows],
    }
    comparisons = {}
    for index, (name, values) in enumerate(features.items()):
        value = auc(labels, values)
        comparisons[name] = {
            "auroc": value,
            "stratified_bootstrap_95": stratified_bootstrap_auc(
                labels, values, seed=20260822 + index
            ),
            "danger_orientation": "larger_score_is_riskier",
        }

    primary_operating_point = operating_point(
        labels, features["prefix16_local_persistence_oracle"], 1.0
    )
    payload = {
        "schema": "kernel-analyzer-oracle-baseline-comparison-v1",
        "status": "COMPLETE_RETROSPECTIVE_FROZEN_12_ROW_COMPARISON",
        "cohort": {
            "rows": len(rows),
            "positives": sum(labels),
            "negatives": len(rows) - sum(labels),
            "selection": "frozen residual-nonzero parameter-reachable screen-negative sample",
            "label": "independent 32-step local level above its complete sign-flip 95% null",
        },
        "comparisons": comparisons,
        "operating_point": {
            "name": "DIFFUSIVE_BOUNDARY_A_GT_1",
            **primary_operating_point,
        },
        "rows": rows,
        "historical_headline_sensitivity": {
            "case_id": "phi4_seq64_lmhead_dx",
            "prefix16_operator_A": 3.1839773160203646,
            "full32_operator_A": 4.487687419828846,
            "passes_A_gt_1_triage": True,
            "pooled_into_primary_auroc": False,
            "reason": "different SGD carrier-scale protocol; retained as external sensitivity only",
        },
        "claim_boundary": (
            "This is a retrospective AUROC on the frozen 12-row audit, with only one local "
            "positive. It tests prefix ranking and miss rate, not universal Oracle accuracy. "
            "The dtype feature is constant by design. Reduction extent is bound to each exact "
            "executed kernel call."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt

        names = list(comparisons)
        values = [comparisons[name]["auroc"] for name in names]
        figure, axis = plt.subplots(figsize=(8.5, 4.6))
        axis.bar(range(len(names)), values, color=["#c44e52", "#4c72b0", "#55a868", "#8172b2"])
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set_ylim(0, 1)
        axis.set_ylabel("AUROC (larger means riskier)")
        axis.set_xticks(range(len(names)), [
            "16-step Oracle", "local RMS", "dtype", "reduction extent"
        ], rotation=20, ha="right")
        axis.set_title("Predicting 32-step local persistence (frozen n=12 audit)")
        figure.tight_layout()
        figure.savefig(args.output_dir / "comparison.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    print(json.dumps({
        "status": payload["status"],
        "oracle_auroc": comparisons["prefix16_local_persistence_oracle"]["auroc"],
        "miss_rate": primary_operating_point["miss_rate"],
        "flag_rate": primary_operating_point["flag_rate"],
    }))


if __name__ == "__main__":
    main()
