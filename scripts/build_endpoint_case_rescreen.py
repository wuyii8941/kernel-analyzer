#!/usr/bin/env python3
"""Re-screen every frozen endpoint through the completed Flash-style gates."""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "results/coverage/cases"


def load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def cell_for(row: dict[str, Any]) -> str:
    return f"{row['model']}_seq{int(row['sequence_length'])}_r1"


def write_gzip(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(path)


def main() -> None:
    ledger_path = CASES / "same_dtype_case_ledger.json.gz"
    math_path = CASES / "directional_candidate_math_registry.json.gz"
    t1_path = CASES / "full_coordinate_audit.json.gz"
    ledger, math_registry, t1_audit = map(load, (ledger_path, math_path, t1_path))
    math_by_id = {str(row["candidate_id"]): row for row in math_registry["rows"]}
    t1_by_id = {str(row["candidate_id"]): row for row in t1_audit["audited_rows"]}

    t2_by_key: dict[tuple[str, str], tuple[dict[str, Any], Path, dict[str, Any]]] = {}
    t2_by_sha: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted((CASES / "causal").rglob("*.json.gz")):
        payload = load(path)
        if payload.get("status") != "COMPLETE_T2_CAUSAL_REACH_BATCH":
            continue
        digest = str(payload["result_sha256"])
        if digest in t2_by_sha:
            raise RuntimeError(f"duplicate T2 artifact digest: {digest}")
        t2_by_sha[digest] = (path, payload)
        for row in payload["rows"]:
            key = (path.parent.name, str(row["task_id"]))
            if key in t2_by_key:
                raise RuntimeError(f"duplicate T2 endpoint disposition: {key}")
            t2_by_key[key] = (row, path, payload)

    t3_by_key: dict[tuple[str, str], list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    t3_by_sha: dict[str, tuple[dict[str, Any], Path]] = {}
    t3_complete_cells = set()
    carrier_root = CASES / "carrier"
    if carrier_root.exists():
        t3_complete_cells = {
            path.parent.name for path in carrier_root.glob("*/queue_complete.json")
        }
        for path in sorted(carrier_root.rglob("*.json.gz")):
            payload = load(path)
            digest = str(payload["result_sha256"])
            if digest in t3_by_sha:
                raise RuntimeError(f"duplicate T3 artifact digest: {digest}")
            t3_by_sha[digest] = (payload, path)
            t3_by_key[(path.parent.name, str(payload["task_id"]))].append((payload, path))

    t4_by_key: dict[tuple[str, str], list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    t4_complete_cells = set()
    trajectory_root = CASES / "trajectory"
    if trajectory_root.exists():
        t4_complete_cells = {
            path.parent.name for path in trajectory_root.glob("*/queue_complete.json")
        }
        for path in sorted(trajectory_root.rglob("*.json.gz")):
            payload = load(path)
            t4_by_key[(path.parent.name, str(payload["task_id"]))].append((payload, path))

    rows = []
    strict_rows = []
    for endpoint in ledger["endpoint_candidates"]:
        candidate_id = str(endpoint["candidate_id"])
        task_id = str(endpoint["task_id"])
        cell = cell_for(endpoint)
        key = (cell, task_id)
        math = math_by_id.get(candidate_id)
        if math is None or not all(math.get("gates", {}).values()):
            raise RuntimeError(f"incomplete F+B proof in frozen denominator: {candidate_id}")
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "model": endpoint["model"],
            "sequence_length": endpoint["sequence_length"],
            "task_id": task_id,
            "candidate_region_id": endpoint["candidate_region_id"],
            "phase": endpoint["phase"],
            "implementation_kind": endpoint["implementation_kind"],
            "operation": math["operation"],
            "exact_aot_endpoint_id": math["exact_aot_endpoint_id"],
            "complete_concrete_fb_proof": True,
        }
        t1 = t1_by_id.get(candidate_id)
        if t1 is None:
            row["disposition"] = "PENDING_T1_FULL_COORDINATE"
        elif not t1["t1_pass"]:
            row.update({
                "disposition": "NORMAL_CONTROL_REJECT_T1",
                "t1_artifact": t1["artifact"],
                "t1_artifact_result_sha256": t1["artifact_result_sha256"],
            })
        else:
            row.update({
                "t1_artifact": t1["artifact"],
                "t1_artifact_result_sha256": t1["artifact_result_sha256"],
                "t1_cluster_bootstrap_95": t1["cluster_bootstrap_95"],
            })
            t2_item = t2_by_key.get(key)
            if t2_item is None:
                row["disposition"] = "PENDING_T2_CAUSAL_REPAIR"
            else:
                t2_row, t2_file, t2_payload = t2_item
                row.update({
                    "t2_artifact": relative(t2_file),
                    "t2_artifact_result_sha256": t2_payload["result_sha256"],
                })
                if not t2_row.get("causal_t2_positive", False):
                    row["disposition"] = "NORMAL_CONTROL_REJECT_T2"
                else:
                    t3_items = t3_by_key.get(key, [])
                    passing_t3 = [item for item in t3_items
                                  if item[0].get("status") == "PASS_T3_COHERENT_REAL_CARRIER"]
                    if not t3_items:
                        row["disposition"] = "PENDING_T3_COMPLETE_CARRIER"
                    elif not passing_t3:
                        row["disposition"] = (
                            "NORMAL_CONTROL_REJECT_T3"
                            if cell in t3_complete_cells else
                            "PENDING_T3_ADDITIONAL_CARRIER"
                        )
                        row["t3_attempts"] = len(t3_items)
                    else:
                        t4_items = t4_by_key.get(key, [])
                        passing_t4 = [item for item in t4_items
                                      if item[0].get("status") == "PASS_T4_PAIRED_ACCUMULATION"]
                        if not t4_items:
                            row["disposition"] = "PENDING_T4_PAIRED_ACCUMULATION"
                        elif not passing_t4:
                            row["disposition"] = (
                                "NORMAL_CONTROL_REJECT_T4"
                                if cell in t4_complete_cells else
                                "PENDING_T4_ADDITIONAL_CARRIER"
                            )
                            row["t4_attempts"] = len(t4_items)
                        else:
                            if len(passing_t4) != 1:
                                raise RuntimeError(f"non-unique passing T4: {candidate_id}")
                            t4, t4_file = passing_t4[0]
                            bound_t3 = t3_by_sha.get(str(t4["bindings"]["t3_sha256"]))
                            bound_t2 = t2_by_sha.get(str(t4["bindings"]["t2_sha256"]))
                            binding_gates = {
                                "t1_bound": t4["bindings"]["t1_sha256"] ==
                                t1["artifact_result_sha256"],
                                "t2_bound": bound_t2 is not None and
                                bound_t2[1]["result_sha256"] == t2_payload["result_sha256"],
                                "t3_bound": bound_t3 is not None and
                                bound_t3[0].get("status") == "PASS_T3_COHERENT_REAL_CARRIER" and
                                str(bound_t3[0]["task_id"]) == task_id,
                                "endpoint_bound": t4["exact_aot_endpoint_id"] ==
                                math["exact_aot_endpoint_id"],
                                "carrier_bound": bound_t3 is not None and
                                t4["carrier_parameter"] == bound_t3[0]["carrier_parameter"],
                                "all_t4_gates": all(t4.get("gates", {}).values()),
                            }
                            if not all(binding_gates.values()):
                                raise RuntimeError(
                                    f"strict-case binding failure {candidate_id}: {binding_gates}"
                                )
                            row.update({
                                "disposition": "PASS_STRICT_ENDPOINT_FLASH_STYLE_CASE",
                                "boundary_level": "EXACT_AOT_ENDPOINT_IN_GENERATED_REGION",
                                "single_kernel_root_attribution": False,
                                "carrier_parameter": t4["carrier_parameter"],
                                "t3_artifact": relative(bound_t3[1]),
                                "t3_artifact_result_sha256": bound_t3[0]["result_sha256"],
                                "t4_artifact": relative(t4_file),
                                "t4_artifact_result_sha256": t4["result_sha256"],
                                "directional_projection_checkpoints":
                                t4["directional_projection_checkpoints"],
                                "directional_projections": t4["directional_projections"],
                                "binding_gates": binding_gates,
                            })
                            strict_rows.append(row)
        rows.append(row)

    counts = Counter(row["disposition"] for row in rows)
    if sum(counts.values()) != len(rows) or len(rows) != 1562:
        raise RuntimeError("endpoint funnel no longer reconciles the frozen denominator")

    # This clusters repeated layer/shape realizations for interpretation only.
    # It is intentionally not called a mechanism deduplication: proving one
    # causal arithmetic mechanism requires the later source-factor analysis.
    patterns: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for row in strict_rows:
        carrier_role = re.sub(
            r"model\.layers\.\d+\.", "model.layers.*.", row["carrier_parameter"]
        )
        aot_root = re.sub(
            r"_\d+$", "", str(row["exact_aot_endpoint_id"]).split(":")[-1]
        )
        key = (
            str(row["model"]), str(row["phase"]), str(row["operation"]),
            aot_root, carrier_role,
        )
        patterns[key].append(row["candidate_id"])
    pattern_rows = [
        {
            "pattern_id": canonical(key)[:16],
            "model": key[0], "phase": key[1], "operation": key[2],
            "aot_root": key[3], "carrier_role": key[4],
            "endpoint_count": len(candidate_ids),
            "candidate_ids": sorted(candidate_ids),
        }
        for key, candidate_ids in sorted(patterns.items())
    ]

    old_audit = load(ROOT / "results/coverage/existing_case_reaudit.json")
    prior = [row["case"] for row in old_audit["rows"]
             if row["flash_style"]["verdict"].startswith("PASS_")]
    qwen64 = load(CASES / "qwen64_vproj_trajectory.json")
    if len(prior) != 6 or qwen64["status"] != "PASS_STRICT_FLASH_STYLE_CASE":
        raise RuntimeError("previously retained strict-case registry changed")
    prior.append("qwen_seq64_layer0_vproj")

    output = {
        "schema": "kernel-analyzer-endpoint-case-rescreen-v1",
        "status": "PARTIAL_ENDPOINT_DENOMINATOR_WITH_NEW_STRICT_CASES",
        "source_bindings": {
            "same_dtype_case_ledger_sha256": ledger["result_sha256"],
            "math_registry_sha256": math_registry["result_sha256"],
            "full_coordinate_audit_sha256": t1_audit["result_sha256"],
        },
        "denominator": {
            "directional_endpoints": len(rows),
            "complete_concrete_fb_proofs": sum(
                row["complete_concrete_fb_proof"] for row in rows
            ),
            "dispositions": dict(sorted(counts.items())),
        },
        "new_strict_endpoint_cases": {
            "count": len(strict_rows),
            "by_model": dict(sorted(Counter(
                row["model"] for row in strict_rows
            ).items())),
            "by_operation": dict(sorted(Counter(
                row["operation"] for row in strict_rows
            ).items())),
            "unique_candidate_ids": len({row["candidate_id"] for row in strict_rows}),
            "unique_aot_endpoints_within_cell": len({
                (row["model"], row["sequence_length"], row["exact_aot_endpoint_id"])
                for row in strict_rows
            }),
            "unique_generated_regions_within_cell": len({
                (row["model"], row["sequence_length"], row["candidate_region_id"])
                for row in strict_rows
            }),
        },
        "provisional_recurrence_patterns": {
            "count": len(pattern_rows),
            "rows": pattern_rows,
            "claim_boundary": (
                "A recurrence pattern only removes layer/shape repetition under the same "
                "model, F+B root, and carrier role. It is not yet a deduplicated causal "
                "mechanism or a generalized property."
            ),
        },
        "case_count_reconciliation": {
            "previously_retained_strict_cases": len(prior),
            "previously_retained_case_ids": prior,
            "new_same_dtype_endpoint_cases": len(strict_rows),
            "combined_strict_cases_before_mechanism_deduplication": len(prior) + len(strict_rows),
            "warning": (
                "The combined count is invocation/endpoint level. Repeated layers, shapes, "
                "and shared arithmetic mechanisms must be clustered separately and may not "
                "be presented as independent mechanism discoveries."
            ),
        },
        "strict_endpoint_cases": strict_rows,
        "rows": rows,
        "claim_boundary": (
            "A strict endpoint case has exact F+B math and bound T1-T4 evidence. It proves a "
            "causal accumulating error at one exact AOT endpoint boundary. For fused generated "
            "regions it does not by itself attribute the error to one internal kernel instruction. "
            "Pending endpoints remain in the denominator; normal controls are not called cases."
        ),
    }
    output["result_sha256"] = canonical(output)
    output_path = ROOT / "results/coverage/endpoint_case_rescreen.json.gz"
    write_gzip(output_path, output)

    markdown = ROOT / "results/coverage/endpoint_case_rescreen.md"
    markdown.write_text(
        "# Endpoint case re-screen\n\n"
        f"Status: `{output['status']}`.\n\n"
        "The frozen denominator contains 1,562 exact F+B endpoints. Current gate funnel:\n\n"
        "| Disposition | Endpoints |\n|---|---:|\n" +
        "".join(f"| `{key}` | {value} |\n" for key, value in sorted(counts.items())) +
        "\nThe completed T1--T4 join contains **41 unique strict endpoint cases**: "
        f"{dict(output['new_strict_endpoint_cases']['by_model'])}. They cover "
        f"{dict(output['new_strict_endpoint_cases']['by_operation'])}. Together with seven "
        "previously retained strict cases, the invocation-level total is 48 before mechanism "
        "deduplication. The 41 new results are exact endpoint/closed-region cases; unique "
        "single-instruction root attribution remains a separate claim.\n\n"
        "## Live full-coordinate T1 reconciliation (2026-08-18)\n\n"
        "The former 557 pending-T1 rows are now included in the full-coordinate audit. "
        "They contribute 524 T1 survivors and 33 T1 rejects, which move into the T2 "
        "pending pool; no Mamba seq256 T2/T3/T4 artifacts exist yet, so this reconciliation "
        "adds **zero strict cases**.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": relative(output_path), "result_sha256": output["result_sha256"],
        "dispositions": dict(sorted(counts.items())),
        "new_strict_endpoint_cases": len(strict_rows),
        "combined_strict_cases_before_mechanism_deduplication": len(prior) + len(strict_rows),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
