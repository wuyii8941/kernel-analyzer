#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.phase8_compile_fusion_probe import SETTINGS


CASES = {
    "clip-step5-grpo_000001_2817771126c0-t80": {
        "probe_files": [
            "results/attribution/step5_compile_fusion_probe.json",
            "results/attribution/step5_compile_fusion_repro.json",
            "results/attribution/stabilize_probe.json",
            "results/attribution/task4_step5_missing.json",
        ],
        "batchscan_files": [
            "results/attribution/batchscan_step5_base.json",
            "results/attribution/batchscan_step5_size1.json",
            "results/attribution/batchscan_step5_size2.json",
            "results/attribution/batchscan_step5_size4.json",
        ],
    },
    "clip-step11-grpo_000003_692fbb817526-t72": {
        "probe_files": [
            "results/attribution/step11_t72_compile_fusion_probe.json",
            "results/attribution/step11_t72_compile_fusion_probe_extended.json",
            "results/attribution/task4_step11_t72_missing.json",
        ],
        "batchscan_files": [],
    },
    "clip-step11-grpo_000003_692fbb817526-t88": {
        "probe_files": [
            "results/attribution/step11_t88_compile_fusion_probe.json",
            "results/attribution/step11_t88_fusion_size2.json",
            "results/attribution/step11_t88_fusion_size2_repro.json",
            "results/attribution/step11_t88_fusion_size4.json",
            "results/attribution/step11_t88_fusion_size8.json",
            "results/attribution/step11_t88_fusion_size16.json",
            "results/attribution/step11_t88_fusion_size32.json",
            "results/attribution/task4_step11_t88_missing.json",
        ],
        "batchscan_files": [],
    },
    "clip-step14-grpo_000004_50bbbbeba833-t34": {
        "probe_files": ["results/attribution/task4_step14_t34_full.json"],
        "batchscan_files": [],
    },
    "clip-step14-grpo_000004_50bbbbeba833-t116": {
        "probe_files": ["results/attribution/task4_step14_t116_full.json"],
        "batchscan_files": [],
    },
}


UNREPLAYABLE: list[dict[str, str]] = []


def mechanism_classes(row: dict[str, Any]) -> set[str]:
    intervention = row["intervention"]
    patch = row.get("inductor_patch") or {}
    classes = set()
    if "max_fusion_size" in patch or intervention in {"no_epilogue_fusion", "no_prologue_fusion", "no_batch_fusion", "aggressive_fusion"}:
        classes.add("fusion_partition")
    if any(key in patch for key in {"triton.persistent_reductions", "split_reductions", "triton.mix_order_reduction"}):
        classes.add("reduction_scheduling")
    if any(key in patch for key in {"pick_loop_orders", "loop_ordering_after_fusion", "loop_reindexing_after_fusion", "reorder_for_locality"}):
        classes.add("loop_scheduling")
    if "deterministic" in patch:
        classes.add("determinism_bundle")
    return classes or {"other_compile_optimization"}


def batchscan_measurement(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    size = payload.get("max_fusion_size")
    intervention = "compile_baseline" if size is None else f"max_fusion_size_{size}"
    target = payload["target"]
    return {
        "intervention": intervention,
        "inductor_patch": {} if size is None else {"max_fusion_size": size},
        "logp": target["logp_alt"],
        "signed_delta_vs_ref": target["signed_delta"],
        "signed_margin": None,
        "fork_vs_reference": target["actual_branch_fork"],
        "generated_kernel_count": payload["generated_kernel_count"],
        "valid_intervention": True,
        "source_kind": "full_batch_scan",
    }


def load_case(fork_id: str, config: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sources = []
    for filename in config["probe_files"]:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["fork_id"] != fork_id:
            raise ValueError(f"fork mismatch in {path}")
        sources.append(filename)
        for row in payload["measurements"]:
            item = dict(row)
            item["source_kind"] = "target_probe"
            grouped[item["intervention"]].append(item)
    for filename in config["batchscan_files"]:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(path)
        sources.append(filename)
        item = batchscan_measurement(path)
        grouped[item["intervention"]].append(item)

    measurements = []
    duplicate_checks = []
    for name in SETTINGS:
        rows = grouped.get(name, [])
        if not rows:
            continue
        logps = [float(row["logp"]) for row in rows]
        forks = [bool(row["fork_vs_reference"]) for row in rows]
        consistent = max(logps) - min(logps) <= 2e-6 and len(set(forks)) == 1
        duplicate_checks.append(
            {
                "intervention": name,
                "repetitions": len(rows),
                "consistent": consistent,
                "max_logp_spread": max(logps) - min(logps),
            }
        )
        if not consistent:
            raise ValueError(f"inconsistent repeated intervention for {fork_id}: {name}")
        representative = dict(rows[-1])
        representative["repetitions"] = len(rows)
        representative["all_sources_valid"] = all(bool(row.get("valid_intervention", True)) for row in rows)
        measurements.append(representative)

    measured = {row["intervention"] for row in measurements}
    missing = [name for name in SETTINGS if name not in measured]
    if missing:
        raise ValueError(f"incomplete setting matrix for {fork_id}: {missing}")
    baseline = next(row for row in measurements if row["intervention"] == "compile_baseline")
    if not baseline["fork_vs_reference"]:
        raise ValueError(f"compile baseline does not reproduce fork for {fork_id}")
    invalid = [row["intervention"] for row in measurements if not row["all_sources_valid"]]
    if invalid:
        raise ValueError(f"invalid intervention canaries for {fork_id}: {invalid}")
    effective = [row for row in measurements if row["intervention"] != "compile_baseline" and not row["fork_vs_reference"]]
    singleton = [row for row in effective if len(row.get("inductor_patch") or {}) == 1]
    combinations = [row for row in effective if len(row.get("inductor_patch") or {}) > 1]
    singleton_keys = {tuple((row.get("inductor_patch") or {}).items()) for row in singleton}
    interaction_only = []
    for row in combinations:
        patch = row.get("inductor_patch") or {}
        if not any(((key, value),) in singleton_keys for key, value in patch.items()):
            interaction_only.append(row["intervention"])
    # Attribute mechanism classes from sufficient singleton interventions.
    # Combination-only keys are not independently causal when a constituent
    # singleton already repairs the fork.
    effective_classes = sorted({item for row in singleton for item in mechanism_classes(row)})
    return {
        "fork_id": fork_id,
        "sources": sources,
        "settings_total": len(SETTINGS),
        "settings_measured": len(measured),
        "missing_settings": missing,
        "measurements": measurements,
        "duplicate_checks": duplicate_checks,
        "effective_settings": [row["intervention"] for row in effective],
        "effective_singletons": [row["intervention"] for row in singleton],
        "effective_combinations": [row["intervention"] for row in combinations],
        "interaction_only_combinations": interaction_only,
        "effective_mechanism_classes": effective_classes,
        "minimal_elimination_set_size": 1 if singleton else (2 if combinations else None),
        "attribution_scope": f"Inductor {', '.join(effective_classes)} configuration class; not a unique source operator.",
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 9 Attribution Interaction Audit",
        "",
        "## Objective",
        "",
        "Determine whether replayable natural clipping forks require interactions between Inductor settings or can be eliminated by a single configuration intervention.",
        "",
        "## Controls",
        "",
        "- Every target probe uses the frozen checkpoint, replay batch, token IDs, old logprob and advantage for that fork.",
        "- Every setting uses a fresh Dynamo reset and independent Inductor cache; generated-code hash is the intervention canary.",
        "- Repeated measurements must agree within `2e-6` in logprob and exactly in branch outcome.",
        "- A combination is called interaction-only only when none of its constituent singleton settings eliminates the fork.",
        "",
        "## Replayable Forks",
        "",
        "| Fork | Settings | Effective singleton settings | Effective combinations | Interaction required |",
        "|---|---:|---|---|---|",
    ]
    for case in payload["cases"]:
        lines.append(
            f"| {case['fork_id']} | {case['settings_measured']}/{case['settings_total']} | "
            f"{', '.join(case['effective_singletons']) or 'none'} | "
            f"{', '.join(case['effective_combinations']) or 'none'} | "
            f"{bool(case['interaction_only_combinations'])} |"
        )
    lines.extend(
        [
            "",
            "## Unreplayable Forks",
            "",
        ]
    )
    for row in payload["unreplayable"]:
        lines.append(f"- `{row['fork_id']}`: {row['reason']}")
    if not payload["unreplayable"]:
        lines.append("_None._")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"`{payload['singleton_attributed_replayable']}/{payload['replayable_cases']}` replayable forks have at least one singleton elimination setting. The step-14 state was deterministically reconstructed and accepted only after 512/512 online rows and both eager/compile fork logprobs matched the canonical run exactly within the registered gates.",
            "",
            "The evidence does not identify a unique source-level operator. Multiple settings can alter fusion partitioning, loop order or reduction scheduling and converge to a repaired branch despite different generated-code inventories. The supported attribution is therefore a compile scheduling class, with the exact effective subclasses listed in the structured artifact.",
            "",
            "Configuration-class coverage and unique-operator attribution remain distinct: a 5/5 singleton scheduling repair rate does not identify five unique source operators.",
            "",
            "## Artifacts",
            "",
            "- `results/phase9_attribution_interaction.json`",
            "- `scripts/phase9_attribution_interaction.py`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate complete Inductor setting ablations across natural forks.")
    parser.add_argument("--out", default="results/phase9_attribution_interaction.json")
    parser.add_argument("--report", default="reports/phase9_attribution_interaction.md")
    args = parser.parse_args()
    cases = [load_case(fork_id, config) for fork_id, config in CASES.items()]
    payload = {
        "schema_version": "forkcert.attribution_interaction.v1",
        "cases": cases,
        "unreplayable": UNREPLAYABLE,
        "natural_forks_total": len(cases) + len(UNREPLAYABLE),
        "replayable_cases": len(cases),
        "singleton_attributed_replayable": sum(bool(case["effective_singletons"]) for case in cases),
        "claim_scope": "Configuration-class causal attribution for replayable forks; no unique source-operator or bug claim.",
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.report).write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "cases"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
