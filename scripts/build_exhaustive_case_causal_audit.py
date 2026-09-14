"""Build a four-question review for every current measured or historical record."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "results/property/numerical_coverage_v1/observed_kernel_catalog_with_signature_measurements_v3.json.gz"
COVERAGE = ROOT / "results/property/numerical_coverage_v1/coverage_with_grouped_softmax_embedding_v2.json"
FAMILY = ROOT / "results/property/numerical_coverage_v1/family_execution_audit_20260907_all_three_complete.json"
FAMILY_MAP = ROOT / "results/property/numerical_coverage_v1/family_measurement_inventory_merge_manifest_v1.json"
HISTORICAL = ROOT / "results/property/declared_persistent_4096/all_bias_case_audit.json"
ROLES = ROOT / "results/mainline_case_roles.json"
REAUDIT = ROOT / "results/coverage/existing_case_reaudit.json"


def load(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream)


def artifact(path_text):
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return None
    return json.loads(path.read_bytes())


def fixed_rms(analysis, raw):
    value = (analysis or {}).get("bias_analysis", {}).get("fixed_suite_total_rms")
    if value is not None:
        return value
    values = (raw or {}).get("original_coordinate_statistics", {}).get("ADAMW_UPDATE", [])
    confirmation = set(map(str, (raw or {}).get("confirmation_state_ids", [])))
    states = list(map(str, (raw or {}).get("state_ids", [])))
    if confirmation and len(values) == len(states):
        values = [row for state, row in zip(states, values) if state in confirmation]
    effect = sum(row.get("effect_energy", 0.0) for row in values)
    repair = sum(row.get("repair_energy", 0.0) for row in values)
    return (effect / repair) ** .5 if repair > 0 else None


def catalog_inputs():
    coverage = load(COVERAGE)
    old = {}
    for row in coverage["records"]:
        evidence = row.get("measurement_evidence") or {}
        if evidence.get("raw_artifact"):
            old[(str(Path(row["release"]).resolve()), row["task_id"])] = evidence

    mapping = {row["audit_family"]: row["release_name"] for row in load(FAMILY_MAP)["mappings"]}
    family = {}
    for group in load(FAMILY)["families"]:
        release = mapping[group["name"]]
        for row in group["records"]:
            family[(release, row["task_id"])] = row

    return old, family


def catalog_rows():
    old, family = catalog_inputs()
    seen, rows = set(), []
    for position in load(CATALOG)["positions"]:
        if position.get("support_status") != "VALID_MEASUREMENT_COMPLETED":
            continue
        identity = (position.get("canonical_release", position["release"]), position["task_id"])
        if identity in seen:
            continue
        seen.add(identity)
        release_name = Path(identity[0]).name
        evidence = old.get((str(Path(position["release"]).resolve()), position["task_id"]))
        if evidence:
            raw = artifact(evidence.get("raw_artifact"))
            analysis = evidence.get("analysis")
            evidence_path = evidence.get("raw_artifact")
        else:
            evidence = family.get((release_name, position["task_id"]))
            if evidence:
                raw = artifact(evidence.get("raw_artifact"))
                analysis = evidence.get("analysis")
                evidence_path = evidence.get("raw_artifact")
            else:
                evidence = position.get("triton_signature_measurement_evidence") or {}
                analysis = artifact(evidence.get("analysis_artifact"))
                analysis_path = Path(evidence.get("analysis_artifact", ""))
                raw_path = analysis_path.parent / "raw" / (evidence.get("case_id", "") + ".json")
                raw = artifact(str(raw_path))
                evidence_path = str(raw_path) if raw else evidence.get("analysis_artifact")
        rms = fixed_rms(analysis, raw)
        scope = (raw or {}).get("reference_comparison_scope")
        if rms == 0:
            assessment = "OBSERVED_IDENTITY_ON_FIXED_SUITE_NO_NONZERO_BIAS_TO_EXPLAIN"
            why = "The stored update difference is zero on the declared fixed suite; this does not prove identity outside it."
        elif rms is None:
            assessment = "MEASUREMENT_RECORD_PRESENT_BUT_NUMERICAL_SUMMARY_NOT_RECOVERED"
            why = "No nonzero-bias explanation can be made from the recovered metadata."
        else:
            assessment = "NONZERO_FIXED_SUITE_DIFFERENCE_ROOT_AND_NONZERO_MEAN_NOT_ESTABLISHED"
            why = ("A nonzero fixed-suite total difference was measured. Total energy does not prove a nonzero mean, "
                   "and the reference substitution does not isolate a low-level source unless separately stated.")
        code = {"implementation_kind": position.get("implementation_kind"), "phase": position.get("phase"),
                "symbol": position.get("symbol"), "formal_pointer": position.get("formal_pointer"),
                "task_id": position.get("task_id"), "semantic_endpoint": position.get("exact_semantic_endpoint_id"),
                "carrier": position.get("carrier"), "reference_source": position.get("reference_candidates")}
        intervention = {
            "contrast_id": (raw or {}).get("contrast_id"), "reference_comparison_scope": scope,
            "interpretation": ("Same-input output substitution" if (scope or {}).get("same_local_operands") else
                               "Region/endpoint substitution which may include upstream differences"),
            "source_isolation": (scope or {}).get("single_kernel_source_attribution", "NOT_DECLARED"),
        }
        rows.append({"audit_id": "catalog:" + release_name + ":" + position["task_id"],
                     "source_kind": "MEASURED_POSITION", "case_id": (raw or {}).get("case_id", evidence.get("case_id")),
                     "operator_family": position.get("operator_family"), "code_location": code,
                     "numerical_error": {"comparison": scope, "fixed_suite_update_rms": rms},
                     "why_nonzero_bias": why, "intervention": intervention,
                     "assessment": assessment, "evidence": evidence_path})
    if len(rows) != 551:
        raise ValueError(f"expected 551 catalog rows, got {len(rows)}")
    return rows


def historical_rows():
    rows = []
    for index, record in enumerate(load(HISTORICAL)["rows"]):
        direct_name = record.get("long_direct", {}).get("artifact")
        consequence_name = record.get("paired_consequence", {}).get("artifact")
        direct = artifact(direct_name)
        consequence = artifact(consequence_name)
        metadata = direct or consequence or {}
        protocol = metadata.get("protocol", {})
        contrast = protocol.get("contrast") if isinstance(protocol, dict) else None
        label = record.get("final_label", "")
        if "UNRESOLVED" in label or "PENDING" in label:
            assessment = "HISTORICAL_UNRESOLVED"
        elif "BIAS" in label:
            assessment = "HISTORICAL_DIRECTION_OR_OUTCOME_LABEL_SOURCE_CAUSE_NOT_PROVED_BY_THIS_ROW"
        else:
            assessment = "HISTORICAL_CONTROL_OR_OUTCOME_RECORD"
        code = {"operator_or_region": record.get("operator_or_region"),
                "runtime_release": metadata.get("runtime_release", metadata.get("release")),
                "runner": metadata.get("runner"), "carrier": metadata.get("carrier"),
                "case_plan": metadata.get("case_plan")}
        rows.append({"audit_id": f"historical:{index}", "source_kind": "HISTORICAL_MATRIX_ROW",
                     "case_id": record.get("matrix_case_id", record["case"]), "code_location": code,
                     "numerical_error": {"historical_formation_description": record.get("formation_path"),
                                         "explicit_contrast": contrast},
                     "why_nonzero_bias": ("This row records an earlier formation description and/or trajectory result; "
                                           "it does not itself derive a nonzero source mean."),
                     "intervention": {"explicit_contrast": contrast,
                                      "direct_evidence": direct_name if direct else "MISSING_OR_NOT_RUN",
                                      "training_consequence": consequence_name if consequence else "MISSING_OR_NOT_RUN",
                                      "claim_boundary": metadata.get("claim_boundary")},
                     "assessment": assessment,
                     "evidence": [name for name, value in ((direct_name, direct), (consequence_name, consequence)) if value]})
    return rows


def role_rows():
    rows = []
    for index, record in enumerate(load(ROLES)["cases"]):
        code = dict(record.get("actual_backend_evidence") or {})
        code["carrier"] = record.get("carrier")
        rows.append({"audit_id": f"role:{index}", "source_kind": "MAINLINE_ROLE_RECORD",
                     "case_id": record["case_id"], "code_location": code,
                     "numerical_error": {"semantic_family": record.get("semantic_family"),
                                         "measurement_geometry": record.get("measurement_geometry")},
                     "why_nonzero_bias": "A mainline role assignment is not a mathematical proof of a nonzero source mean.",
                     "intervention": {"artifact": record.get("artifact"),
                                      "formation_artifact": record.get("formation_artifact"),
                                      "claim_boundary": record.get("claim_boundary")},
                     "assessment": "ROLE_AND_EVIDENCE_POINTER_REQUIRES_CASE_SPECIFIC_CAUSAL_REVIEW",
                     "evidence": [x for x in (record.get("artifact"), record.get("formation_artifact")) if x]})
    return rows


def reaudit_rows():
    rows = []
    for index, record in enumerate(load(REAUDIT)["rows"]):
        gates = record.get("flash_style", {}).get("gates", {})
        repair = gates.get("F2_CAUSAL_REPAIR", {})
        rows.append({"audit_id": f"reaudit:{index}", "source_kind": "LEGACY_CASE_REAUDIT",
                     "case_id": record["case"], "code_location": {"mechanism_level": record.get("mechanism_level")},
                     "numerical_error": {"legacy_mechanism_level": record.get("mechanism_level")},
                     "why_nonzero_bias": ("The legacy gate records a causal-repair verdict, but this summary row does not "
                                           "by itself establish the physical source or population mean."),
                     "intervention": {"legacy_gate": repair, "flash_style_verdict": record.get("flash_style", {}).get("verdict"),
                                      "generalizable_bias_verdict": record.get("generalizable_bias", {}).get("verdict")},
                     "assessment": "LEGACY_CAUSAL_ASSERTION_RETAINED_UNDERLYING_EVIDENCE_MUST_CONTROL_CLAIM",
                     "evidence": repair.get("evidence")})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        parser.error("choose a new repository output")
    rows = catalog_rows() + historical_rows() + role_rows() + reaudit_rows()
    required = ("code_location", "numerical_error", "why_nonzero_bias", "intervention", "assessment")
    if len(rows) != 866 or any(any(key not in row for key in required) for row in rows):
        raise ValueError("exhaustive audit invariant failed")
    result = {"schema": "exhaustive-case-causal-audit-v1", "status": "EVERY_SOURCE_RECORD_CLASSIFIED",
              "record_count": len(rows), "counts_by_source": dict(Counter(r["source_kind"] for r in rows)),
              "counts_by_assessment": dict(Counter(r["assessment"] for r in rows)), "rows": rows,
              "scope": ("Every source record has the four requested fields. This is exhaustive over source records, "
                        "not a claim that overlapping rows are independent cases or that every root cause is known.")}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"records": len(rows), "sources": result["counts_by_source"],
                      "assessments": result["counts_by_assessment"]}))


if __name__ == "__main__":
    main()
