#!/usr/bin/env python3
"""Build the frozen Bias Formation Map population without assigning formation labels.

The endpoint rows come from the already-frozen hypothesis population.  Known
strict/anchor cases are appended as semantic-region rows because they are not
necessarily one invocation in that endpoint census.  Legacy T1--T4/SEUP
evidence is retained as provenance only; every formation field stays PENDING.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HYPOTHESIS = ROOT / "results/property/hypothesis_matrix.json"
ROSTER = ROOT / "results/property/bias_formation_v2_1/roster_bound.json"
REAUDIT = ROOT / "results/coverage/existing_case_reaudit.json"
OUT = ROOT / "results/property/bias_formation/bias_population.csv"

FORMATION_FIELDS = {
    "formation_local": "PENDING",
    "formation_parameter_gradient": "PENDING",
    "formation_effective_update": "PENDING",
    "first_confirmed_bias_stage": "PENDING",
    "formation_label_source": "NOT_MEASURED",
}


def _text(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _base_row():
    return {
        "population_id": "",
        "population_kind": "",
        "case_id": "",
        "candidate_id": "",
        "model": "",
        "sequence_length": "",
        "phase": "",
        "mathematical_operation": "",
        "implementation_kind": "",
        "exact_endpoint_or_region": "",
        "proof_unit_bound": "",
        "legacy_observed_role": "",
        "legacy_observed_verdict": "",
        "legacy_flash_verdict": "",
        "legacy_generalizable_verdict": "",
        "legacy_evidence": "",
        "legacy_evidence_only": "true",
        "claim_boundary": "Formation has not been measured; legacy labels are not formation labels.",
        **FORMATION_FIELDS,
    }


def build_rows():
    hypothesis = json.loads(HYPOTHESIS.read_text(encoding="utf-8"))
    roster = json.loads(ROSTER.read_text(encoding="utf-8"))
    reaudit = json.loads(REAUDIT.read_text(encoding="utf-8"))

    rows = []
    seen = set()

    # The 1,562-row endpoint population is retained exactly, including normal
    # references and unresolved rows.  It is the denominator, not a label set.
    for item in hypothesis["rows"]:
        candidate_id = _text(item["candidate_id"])
        population_id = "endpoint::" + candidate_id
        if population_id in seen:
            raise ValueError("duplicate endpoint population id: %s" % population_id)
        seen.add(population_id)
        grouping = item.get("grouping_metadata", {})
        observed = item.get("observed_label", {})
        row = _base_row()
        row.update(
            {
                "population_id": population_id,
                "population_kind": "ENDPOINT_UNIT",
                "candidate_id": candidate_id,
                "model": _text(grouping.get("model")),
                "sequence_length": _text(grouping.get("sequence_length")),
                "phase": _text(grouping.get("phase")),
                "mathematical_operation": _text(grouping.get("mathematical_operation")),
                "implementation_kind": _text(grouping.get("implementation_kind")),
                "exact_endpoint_or_region": _text(item.get("exact_aot_endpoint_id")),
                "proof_unit_bound": _text(item.get("complete_concrete_f_b_proof")),
                "legacy_observed_role": _text(observed.get("role")),
                "legacy_observed_verdict": _text(observed.get("verdict")),
                "legacy_evidence": "hypothesis_matrix.json; math_proof_row_sha256="
                + _text(item.get("math_proof_row_sha256")),
            }
        )
        rows.append(row)

    # Append known strict/anchor cases.  These rows are semantic-case records,
    # not extra endpoint discoveries, and are intentionally not deduplicated
    # against invocation rows by fuzzy names.
    known = {}
    for item in roster.get("cases", []):
        known[_text(item.get("case_id"))] = {
            "case": item,
            "legacy": None,
        }
    # Legacy reports used longer names for five of the same semantic cases.
    # Canonicalize those names before appending, so the population does not
    # count one endpoint twice merely because an audit renamed it.
    legacy_case_alias = {
        "liger_fused_linear_ce_dw": "liger_fused_ce_t128",
        "phi4_seq64_lmhead_dx_mm": "phi4_lm_head_dx_seq64",
        "qwen128_layer27_softmax_saved_state": "qwen_saved_p_seq128",
        "qwen128_layer0_vproj_output": "qwen_vproj_seq128",
        "qwen3vl_silu_backward_decomposition": "qwen3vl_silu_seq160",
    }
    for item in reaudit.get("rows", []):
        raw_case_id = _text(item.get("case"))
        case_id = legacy_case_alias.get(raw_case_id, raw_case_id)
        item = dict(item)
        item["case"] = raw_case_id
        if case_id not in known:
            known[case_id] = {"case": None, "legacy": item}
        else:
            known[case_id]["legacy"] = item

    for case_id in sorted(known):
        item = known[case_id]["case"] or {}
        legacy = known[case_id]["legacy"] or {}
        population_id = "case::" + case_id
        if population_id in seen:
            raise ValueError("duplicate case population id: %s" % population_id)
        seen.add(population_id)
        row = _base_row()
        flash = legacy.get("flash_style", {})
        general = legacy.get("generalizable_bias", {})
        gates = flash.get("gates", {})
        evidence = list(item.get("artifacts", []))
        evidence.extend(
            [
                "roster_bound.json",
                "feasibility=" + _text(item.get("feasibility")),
                "source_status=" + _text(item.get("source_status")),
            ]
        )
        if legacy:
            evidence.append("existing_case_reaudit.json")
        row.update(
            {
                "population_id": population_id,
                "population_kind": "KNOWN_STRICT_CASE",
                "case_id": case_id,
                "model": _text(item.get("model")),
                "sequence_length": _text(item.get("sequence_length")),
                "exact_endpoint_or_region": _text(
                    item.get("endpoint_or_region_id") or legacy.get("case")
                ),
                "proof_unit_bound": _text(item.get("proof_unit_id")),
                "legacy_observed_role": _text(item.get("role")),
                "legacy_observed_verdict": _text(general.get("verdict")),
                "legacy_flash_verdict": _text(flash.get("verdict")),
                "legacy_generalizable_verdict": _text(general.get("verdict")),
                "legacy_evidence": ";".join(evidence),
                "claim_boundary": "Known strict/anchor case retained for the denominator; formation stages remain PENDING until v2.1 capture.",
            }
        )
        rows.append(row)

    rows.sort(key=lambda r: r["population_id"])
    return rows


def write(rows):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temp = OUT.with_name("." + OUT.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(OUT)


if __name__ == "__main__":
    result = build_rows()
    write(result)
    endpoint_count = sum(r["population_kind"] == "ENDPOINT_UNIT" for r in result)
    case_count = sum(r["population_kind"] == "KNOWN_STRICT_CASE" for r in result)
    print(json.dumps({"output": str(OUT), "rows": len(result), "endpoint_rows": endpoint_count, "known_case_rows": case_count}, sort_keys=True))
