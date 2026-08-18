#!/usr/bin/env python3
"""Build the F+B-level Signed Transport Coherence study population.

T4 is intentionally absent.  The target is the T3 verdict: whether a local,
causal endpoint effect reaches a complete parameter-gradient carrier with a
stable common direction across frozen natural states.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
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


def t3_passes() -> dict[tuple[str, str], list[tuple[dict[str, Any], Path]]]:
    passed: dict[tuple[str, str], list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for path in sorted((CASES / "carrier").rglob("*.json.gz")):
        payload = load(path)
        if payload.get("status") == "PASS_T3_COHERENT_REAL_CARRIER":
            passed[(path.parent.name, str(payload["task_id"]))].append((payload, path))
    return passed


def observed_t3_label(
    endpoint: dict[str, Any], passing: list[tuple[dict[str, Any], Path]],
) -> tuple[str, str]:
    if passing:
        return "COHERENT_F_B_BIAS", "OBSERVED_T3_COHERENT_COMPLETE_CARRIER"
    disposition = str(endpoint["disposition"])
    normal = {
        "NORMAL_CONTROL_REJECT_T1": "NORMAL_REFERENCE_NO_LOCAL_DIRECTIONAL_DIFFERENCE",
        "NORMAL_CONTROL_REJECT_T2": "NORMAL_REFERENCE_NO_CAUSAL_ENDPOINT_EFFECT",
        "NORMAL_CONTROL_REJECT_T3": "NORMAL_REFERENCE_NO_COHERENT_F_B_CARRIER",
    }
    if disposition in normal:
        return "NORMAL_REFERENCE", normal[disposition]
    return "UNRESOLVED", "UNRESOLVED_PENDING_REQUIRED_GATE"


def main() -> None:
    rescreen_path = ROOT / "results/coverage/endpoint_case_rescreen.json.gz"
    math_path = CASES / "directional_candidate_math_registry.json.gz"
    rescreen = load(rescreen_path)
    math_registry = load(math_path)
    math_by_id = {str(row["candidate_id"]): row for row in math_registry["rows"]}
    passing_t3 = t3_passes()

    rows = []
    queue_rows = []
    for endpoint in rescreen["rows"]:
        candidate_id = str(endpoint["candidate_id"])
        math = math_by_id[candidate_id]
        key = (cell_for(endpoint), str(endpoint["task_id"]))
        evidence = passing_t3.get(key, [])
        role, label = observed_t3_label(endpoint, evidence)
        rows.append({
            "candidate_id": candidate_id,
            "grouping_metadata": {
                "model": endpoint["model"],
                "sequence_length": endpoint["sequence_length"],
                "phase": endpoint["phase"],
                "mathematical_operation": endpoint["operation"],
                "implementation_kind": endpoint["implementation_kind"],
            },
            "exact_aot_endpoint_id": endpoint["exact_aot_endpoint_id"],
            "complete_concrete_f_b_proof": bool(endpoint["complete_concrete_fb_proof"]),
            "math_proof_row_sha256": math["row_sha256"],
            "observed_label": {"role": role, "verdict": label},
            "label_evidence": [
                {"artifact": relative(path), "result_sha256": payload["result_sha256"]}
                for payload, path in evidence
            ],
            "predictor_status": "PENDING_SIGNED_EVENT_FACTOR_REPLAY",
        })
        queue_rows.append({
            "candidate_id": candidate_id,
            "task_id": endpoint["task_id"],
            "exact_aot_endpoint_id": endpoint["exact_aot_endpoint_id"],
            "math_proof_row_sha256": math["row_sha256"],
            "proof_owner_ids": math["proof_owner_ids"],
            "task_plan_sha256": math["task_plan_sha256"],
            "study_partition": (
                "RETROSPECTIVE_PROPERTY_DEVELOPMENT" if role != "UNRESOLVED"
                else "PROSPECTIVE_AFTER_PROPERTY_FREEZE"
            ),
            "required": [
                "REFERENCE_OPERANDS_FOR_EVERY_DECLARED_ARITHMETIC_EVENT",
                "DECLARED_DTYPE_ACCUMULATION_AND_ROUNDING_SCHEDULE",
                "SIGNED_LOCAL_EVENT_ERRORS_WITHOUT_CANDIDATE_TENSOR_VALUES",
                "COMPLETE_EVENT_TO_PARAMETER_GRADIENT_F_B_TRANSPORT",
                "EXACT_AFFINE_IDENTITY_OR_NONLINEAR_REMAINDER_BOUND",
                "SEMANTIC_PRESERVING_ARITHMETIC_INTERVENTION_AFTER_PROPERTY_FREEZE",
            ],
        })

    counts = Counter(row["observed_label"]["role"] for row in rows)
    verdict_counts = Counter(row["observed_label"]["verdict"] for row in rows)
    if len(rows) != int(rescreen["denominator"]["directional_endpoints"]):
        raise RuntimeError("property population does not preserve the endpoint denominator")
    if any(not row["complete_concrete_f_b_proof"] for row in rows):
        raise RuntimeError("property population contains an incomplete F+B proof")

    coherent = [row for row in rows
                if row["observed_label"]["role"] == "COHERENT_F_B_BIAS"]
    operation_counts = Counter(
        row["grouping_metadata"]["mathematical_operation"] for row in coherent
    )
    model_counts = Counter(row["grouping_metadata"]["model"] for row in coherent)

    output = {
        "schema": "kernel-analyzer-signed-transport-property-population-v2",
        "status": "NO_REFERENCE_ONLY_PROPERTY_CLAIM_YET",
        "target": {
            "unit": "ONE_EXACT_FORWARD_PLUS_ACTUAL_BACKWARD_ENDPOINT",
            "verdict_layer": "T3_COHERENT_COMPLETE_PARAMETER_GRADIENT_CARRIER",
            "t4_used_as_label_or_predictor": False,
            "definition": (
                "Signed Transport Coherence: schedule-derived signed arithmetic event "
                "errors transported through complete reference-only F+B derivatives have a "
                "common cross-state component above the nonlinear remainder and frozen "
                "reference margin."
            ),
        },
        "source_bindings": {
            "endpoint_rescreen_sha256": rescreen["result_sha256"],
            "math_registry_sha256": math_registry["result_sha256"],
        },
        "population": {
            "endpoint_count": len(rows),
            "role_counts": dict(sorted(counts.items())),
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "all_endpoints_retained": True,
            "representative_sampling": False,
        },
        "descriptive_only": {
            "coherent_endpoint_operation_counts": dict(sorted(operation_counts.items())),
            "coherent_endpoint_model_counts": dict(sorted(model_counts.items())),
            "identity_fields_are_not_predictors": True,
        },
        "hypothesis": {
            "id": "SIGNED_TRANSPORT_COHERENCE",
            "formula": {
                "event_error": "epsilon_s,e = Q_e(f_e(x_s,e)) - f_e(x_s,e)",
                "transported_error": "v_s = sum_e J_s,e epsilon_s,e",
                "amplitude": "A = E_s[||v_s||^2]",
                "directional_energy": "D = ||E_s[v_s]||^2",
                "concentration": "C = D / A",
                "sufficient_bound": "sqrt(D) - E_s[rho_s] > frozen_reference_margin",
            },
            "flash_attention_special_case": (
                "epsilon_T=(delta_lp-delta_hp)[T], "
                "J_T=alpha*(PK)[T]^T*X[T]"
            ),
            "claim_status": "UNRESOLVED_PENDING_REFERENCE_SCHEDULE_FACTOR_REPLAY",
        },
        "leakage_policy": {
            "allowed_predictor_inputs": [
                "reference_operands", "declared_arithmetic_schedule",
                "analytic_f_b_transport", "reference_only_margin",
            ],
            "forbidden_predictor_inputs": [
                "candidate_tensor_values", "observed_t1_t2_t3_t4_values",
                "historical_verdicts", "operator_model_module_identity",
            ],
            "observed_labels_are_evaluation_only": True,
        },
        "rows": rows,
        "claim_boundary": (
            "This artifact fixes the correct F+B/T3 study population and an executable "
            "property interface. It does not claim that Signed Transport Coherence is already "
            "supported: event-level source factors and reference-only transports remain to be "
            "computed prospectively. T4 is neither a label nor a predictor."
        ),
    }
    output["result_sha256"] = canonical(output)
    result_path = ROOT / "results/property/hypothesis_matrix.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")

    queue = {
        "schema": "kernel-analyzer-signed-transport-factor-queue-v1",
        "source_population_sha256": output["result_sha256"],
        "endpoint_count": len(queue_rows),
        "all_endpoints_queued": True,
        "representative_sampling": False,
        "rows": queue_rows,
    }
    queue["result_sha256"] = canonical(queue)
    queue_path = ROOT / "results/property/signed_transport_queue.json.gz"
    write_gzip(queue_path, queue)

    print(json.dumps({
        "output": relative(result_path),
        "queue": relative(queue_path),
        "status": output["status"],
        "role_counts": output["population"]["role_counts"],
        "result_sha256": output["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
