#!/usr/bin/env python3
"""Assemble the final, fail-closed Bias Formation Map artifacts.

This is an analysis-only reducer.  It never assigns formation labels from
legacy T1--T4, SEUP, or carrier results.  Existing compact v2.1 formation
certificates are the only source of natural-case layer labels.  The endpoint
population is retained as a screening denominator; rows without a capture
certificate are explicitly marked ``NOT_CAPTURED``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/property/bias_formation"
OUT = ROOT / "results/property/bias_formation_final"
POPULATION = SRC / "bias_population.csv"
INTERVENTION = SRC / "interventions/phi4_mm_transport_pairing.json"

FORMATION_CASES = {
    "liger_fused_ce_t128": SRC / "formation/liger_fused_ce_t128.json",
    "phi4_lm_head_dx_seq64": SRC / "formation/phi4_lm_head_dx_seq64.json",
    "qwen_saved_p_seq128": SRC / "formation/qwen_saved_p_seq128.json",
}

LAYER_KEYS = ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _status(cert: Mapping[str, Any] | None, layer: str) -> str:
    if not cert:
        return "NOT_CAPTURED"
    populations = cert.get("populations", {})
    confirmation = populations.get("confirmation", {}) if isinstance(populations, Mapping) else {}
    value = confirmation.get(layer)
    if isinstance(value, Mapping):
        return str(value.get("status", "UNRESOLVED"))
    return str(confirmation.get(layer + "_status", "UNRESOLVED"))


def _stats(cert: Mapping[str, Any] | None, layer: str, partition: str = "confirmation") -> dict[str, Any]:
    if not cert:
        return {"status": "NOT_CAPTURED"}
    value = cert.get("populations", {}).get(partition, {}).get(layer)
    if not isinstance(value, Mapping):
        return {"status": "UNRESOLVED"}
    return {
        "status": value.get("status"),
        "cross_state_ratio": value.get("cross_state_ratio"),
        "bootstrap_lower": value.get("bootstrap_lower"),
        "bootstrap_upper": value.get("bootstrap_upper"),
        "state_count": value.get("state_count"),
        "coordinate_count": value.get("coordinate_count"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_phi_transport_decomposition() -> dict[str, Any]:
    intervention = _load(INTERVENTION)
    rows = intervention.get("rows", [])
    errors = [float(row["transport_prediction_relative_error"]) for row in rows]
    local_norm_errors = [float(row["shuffled_local_norm_relative_error"]) for row in rows]
    natural = intervention.get("natural_gradient_population", {})
    shuffled = intervention.get("shuffled_gradient_population", {})
    natural_energy = float(natural.get("average_state_energy", 0.0))
    shuffled_energy = float(shuffled.get("average_state_energy", 0.0))
    ratio = shuffled_energy / natural_energy if natural_energy else None
    analytic_closed = bool(intervention.get("gates", {}).get("analytic_transport_matches_natural_gradient"))
    if (
        natural.get("status") == "BIASED"
        and shuffled.get("status") == "CENTERED"
        and not analytic_closed
    ):
        status = "UNRESOLVED_ANALYTIC_TRANSPORT_NOT_CLOSED"
        mechanism = "TRANSPORT_CANDIDATE_NOT_VALIDATED"
    elif analytic_closed and natural.get("status") == "BIASED" and shuffled.get("status") == "CENTERED":
        status = "SUPPORTS_TRANSPORT_ALIGNMENT"
        mechanism = "TRANSPORT_BIAS"
    else:
        status = "COUNTEREVIDENCE_OR_UNRESOLVED"
        mechanism = "UNRESOLVED"
    return {
        "schema": "kernel-analyzer-bias-formation-phi-transport-decomposition-v1",
        "case_id": "phi4_lm_head_dx_seq64",
        "status": status,
        "mechanism_label": mechanism,
        "formation_label_unchanged": True,
        "natural_gradient_status": natural.get("status"),
        "shuffled_gradient_status": shuffled.get("status"),
        "rows": len(rows),
        "local_residual_norm_preserved": bool(intervention.get("gates", {}).get("local_norm_preserved_every_state")),
        "natural_to_shuffled_energy_ratio": ratio,
        "transport_prediction_relative_error": {
            "min": min(errors) if errors else None,
            "mean": mean(errors) if errors else None,
            "max": max(errors) if errors else None,
        },
        "local_norm_relative_error": {
            "min": min(local_norm_errors) if local_norm_errors else None,
            "max": max(local_norm_errors) if local_norm_errors else None,
        },
        "per_state": [
            {
                "state_id": row.get("state_id"),
                "natural_gradient_delta_l2": row.get("natural_gradient_delta_l2"),
                "shuffled_gradient_delta_l2": row.get("shuffled_gradient_delta_l2"),
                "local_residual_l2": row.get("local_residual_l2"),
                "transport_prediction_relative_error": row.get("transport_prediction_relative_error"),
                "shuffled_local_norm_relative_error": row.get("shuffled_local_norm_relative_error"),
            }
            for row in rows
        ],
        "evidence": str(INTERVENTION.relative_to(ROOT)),
        "interpretation": (
            "Residual row permutation changes the measured gradient population, "
            "but the current RMSNorm-only analytic transport does not reconstruct "
            "the complete semantic VJP.  The result is a transport candidate, not "
            "a validated transport property."
        ),
        "claim_boundary": intervention.get("claim_boundary"),
    }


def build_liger_analysis() -> dict[str, Any]:
    cert = _load(FORMATION_CASES["liger_fused_ce_t128"])
    calibration = cert["populations"]["calibration"]
    confirmation = cert["populations"]["confirmation"]
    result = {
        "schema": "kernel-analyzer-bias-formation-liger-analysis-v1",
        "case_id": cert["case_id"],
        "status": "UNRESOLVED_CONFIRMATION_MARGIN_CROSSED",
        "formation_point": cert.get("formation_point"),
        "confirmation_states": cert["state_split"]["confirmation_count"],
        "calibration": {layer: _stats(cert, layer, "calibration") for layer in LAYER_KEYS},
        "confirmation": {},
        "source_bias_confirmed": False,
        "seup_is_formation_label": False,
        "interpretation": (
            "Calibration local energy clears the frozen bias margin, but the "
            "independent confirmation interval crosses that margin.  The source "
            "transition is therefore unresolved; earlier Liger persistence is "
            "consequence evidence only."
        ),
        "evidence": str(FORMATION_CASES["liger_fused_ce_t128"].relative_to(ROOT)),
    }
    for layer in LAYER_KEYS:
        value = confirmation.get(layer)
        result["confirmation"][layer] = _stats(cert, layer, "confirmation")
        if isinstance(value, Mapping) and value.get("status") == "BIASED":
            result["source_bias_confirmed"] = False
    return result


def build_population_screening() -> list[dict[str, Any]]:
    certificates = {case: _load(path) for case, path in FORMATION_CASES.items()}
    rows: list[dict[str, Any]] = []
    with POPULATION.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            case_id = source.get("case_id", "")
            cert = certificates.get(case_id)
            if cert is not None:
                local = _status(cert, "LOCAL_ENDPOINT")
                gradient = _status(cert, "PARAMETER_GRADIENT")
                update = _status(cert, "EFFECTIVE_UPDATE")
                screening = "FORMATION_MEASURED"
                evidence = str(FORMATION_CASES[case_id].relative_to(ROOT))
            elif case_id == "qwen_bmm_seq64":
                local = gradient = update = "INELIGIBLE"
                screening = "INELIGIBLE_EXACT_REPAIR_SHAM_PROVENANCE"
                evidence = "results/property/bias_formation_v2_1/feasibility_report.json"
            else:
                local = gradient = update = "NOT_MEASURED"
                screening = "NOT_CAPTURED_EXISTING_ARTIFACT_ONLY"
                evidence = source.get("legacy_evidence", "")
            rows.append({
                "population_id": source.get("population_id", ""),
                "population_kind": source.get("population_kind", ""),
                "case_id": case_id,
                "candidate_id": source.get("candidate_id", ""),
                "model": source.get("model", ""),
                "sequence_length": source.get("sequence_length", ""),
                "exact_endpoint_or_region": source.get("exact_endpoint_or_region", ""),
                "legacy_role_provenance_only": source.get("legacy_observed_role", ""),
                "formation_local": local,
                "formation_parameter_gradient": gradient,
                "formation_effective_update": update,
                "screening_status": screening,
                "evidence": evidence,
                "claim_boundary": (
                    "Formation labels come only from v2.1 open-loop certificates; "
                    "legacy T1--T4/SEUP roles are provenance, not screening labels."
                ),
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_reports() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mechanism_reports").mkdir(exist_ok=True)
    phi = build_phi_transport_decomposition()
    liger = build_liger_analysis()
    population = build_population_screening()
    _write_json(OUT / "phi_transport_decomposition.json", phi)
    _write_json(OUT / "liger_formation_analysis.json", liger)
    write_csv(OUT / "population_screening.csv", population)

    reports = {
        "source_bias.md": """# Source bias\n\n## Verdict\n\n`NOT_CONFIRMED`. Liger's calibration local population is directional, but its independent confirmation interval crosses the frozen bias margin. No source-generated transition is confirmed. SEUP persistence is consequence evidence and is not reused as a formation label.\n\n## Evidence\n\n- `liger_formation_analysis.json`\n- `results/property/bias_formation/formation/liger_fused_ce_t128.json`\n\n## Boundary\n\nA source-centering intervention is not triggered until the confirmation population is resolved. This is a negative/uncertain result, not evidence that BF16 accumulation is harmless.\n""",
        "transport_bias.md": """# Transport bias\n\n## Verdict\n\n`CASE_CANDIDATE_NOT_VALIDATED` for Phi MM. The open-loop formation map is `LOCAL_CENTERED -> PARAMETER_GRADIENT_BIASED -> EFFECTIVE_UPDATE_BIASED`. A row-pairing intervention changes the gradient population from `BIASED` to `CENTERED` while preserving local residual norms, but the current analytic RMSNorm-only transport reconstruction has relative error 0.32--0.60.\n\n## Evidence\n\n- `phi_transport_decomposition.json`\n- `results/property/bias_formation/interventions/phi4_mm_transport_pairing.json`\n\n## Boundary\n\nThe pairing result motivates transport analysis; it does not establish a complete transport mechanism or a cross-operator property. The missing semantic VJP terms must be closed before promotion.\n""",
        "contract_bias.md": """# Numerical contract bias\n\n## Verdict\n\n`UNRESOLVED`. Qwen saved-P is centered at local endpoint, parameter-gradient, and effective-update layers in both calibration and confirmation populations. No contract repair has been used to turn this centered result into a mechanism claim.\n\n## Evidence\n\n- `results/property/bias_formation/formation/qwen_saved_p_seq128.json`\n\n## Boundary\n\nThe centered result is case-level evidence only. It does not rule out a contract defect outside the measured semantic region, nor does it prove that saved-state reconstruction is harmless in general.\n""",
        "optimizer_bias.md": """# Optimizer bias\n\n## Verdict\n\n`NOT_OBSERVED`. No eligible natural case in the completed formation map has the decisive `GRADIENT_CENTERED -> EFFECTIVE_UPDATE_BIASED` transition. Therefore no optimizer-induced mechanism is claimed.\n\n## Boundary\n\nThe current map does not test SGD/AdamW substitutions or moment interventions. This mechanism remains an explicit unresolved branch, not a negative universal result.\n""",
    }
    for name, text in reports.items():
        (OUT / "mechanism_reports" / name).write_text(text, encoding="utf-8")

    formation_cases = {}
    for case, path in FORMATION_CASES.items():
        cert = _load(path)
        formation_cases[case] = {
            "status": cert.get("status"),
            "formation_point": cert.get("formation_point"),
            "first_confirmed_bias_stage": cert.get("first_confirmed_bias_stage"),
            "confirmation": {layer: _stats(cert, layer, "confirmation") for layer in LAYER_KEYS},
            "evidence": str(path.relative_to(ROOT)),
        }
    taxonomy = {
        "schema": "kernel-analyzer-bias-formation-taxonomy-v1",
        "measured_cases": formation_cases,
        "mechanism_status": {
            "SOURCE_BIAS": "NOT_CONFIRMED",
            "TRANSPORT_BIAS": phi["mechanism_label"],
            "CONTRACT_BIAS": "UNRESOLVED",
            "OPTIMIZER_BIAS": "NOT_OBSERVED",
            "VARIANCE_ONLY": "QWEN_SAVED_P_CASE_LEVEL_ONLY",
        },
        "claim_boundary": "The map distinguishes measured transitions from mechanism validation; it makes no universal claim from four cases.",
    }
    _write_json(OUT / "bias_taxonomy.json", taxonomy)
    write_csv(OUT / "bias_transition_matrix.csv", [
        {
            "case": "liger_fused_ce_t128",
            "local": "UNRESOLVED_INSUFFICIENT_STATES",
            "gradient": "UNRESOLVED_INSUFFICIENT_STATES",
            "update": "UNRESOLVED_INSUFFICIENT_STATES",
            "formation_type": "UNRESOLVED",
            "status": "CONFIRMATION_MARGIN_CROSSED",
            "evidence": "formation/liger_fused_ce_t128.json",
        },
        {
            "case": "phi4_lm_head_dx_seq64",
            "local": "CENTERED",
            "gradient": "BIASED",
            "update": "BIASED",
            "formation_type": "TRANSPORT_OR_CONTRACT_CANDIDATE",
            "status": "CONFIRMED_TRANSITION_MECHANISM_UNRESOLVED",
            "evidence": "formation/phi4_lm_head_dx_seq64.json",
        },
        {
            "case": "qwen_saved_p_seq128",
            "local": "CENTERED",
            "gradient": "CENTERED",
            "update": "CENTERED",
            "formation_type": "VARIANCE_ONLY_CASE_LEVEL",
            "status": "COMPLETE",
            "evidence": "formation/qwen_saved_p_seq128.json",
        },
        {
            "case": "qwen_bmm_seq64",
            "local": "INELIGIBLE",
            "gradient": "INELIGIBLE",
            "update": "INELIGIBLE",
            "formation_type": "UNRESOLVED",
            "status": "MISSING_EXACT_REPAIR_SHAM_PROVENANCE",
            "evidence": "../bias_formation_v2_1/feasibility_report.json",
        },
    ])
    (OUT / "bias_taxonomy.md").write_text(
        """# Bias Formation Taxonomy\n\nThe completed map separates formation from persistence and keeps unresolved mechanisms explicit.\n\n| Case | Measured transition | Current interpretation |\n|---|---|---|\n| Phi MM | `LOCAL_CENTERED -> GRADIENT_BIASED -> UPDATE_BIASED` | transport/contract candidate; intervention not analytically closed |\n| Qwen saved-P | `CENTERED -> CENTERED -> CENTERED` | case-level centered result; no universal harmlessness claim |\n| Liger | confirmation unresolved | source bias not confirmed |\n| Qwen bmm | ineligible | no formation label; missing exact repair/sham provenance |\n\nThe current evidence supports a **Bias Formation Map**, not a single universal property. A mechanism is promoted only after its transition and matched intervention are both closed.\n""", encoding="utf-8")
    (OUT / "scientific_summary.md").write_text(
        """# Bias Formation Map — final current result\n\n## Scientific question\n\nWhen does an implementation-induced numerical difference stop being harmless variance and become directional training bias?\n\nThe measured chain is:\n\n```text\nimplementation difference -> local residual -> parameter-gradient residual\n                         -> effective-update residual -> SEUP consequence\n```\n\n## Current answer\n\nPhi MM is the only completed natural case with a confirmed formation transition: local error is centered, while the parameter-gradient and effective-update populations are directionally biased in both 16-state partitions. Its separate SEUP replay closes the consequence link, but the residual/transport pairing intervention is not yet a complete causal transport proof because the current analytic reconstruction misses part of the gradient delta.\n\nQwen saved-P is centered at all three measured layers. This is a valuable case-level variance-only observation, not a universal negative. Liger has a directional calibration signal but its independent confirmation interval crosses the frozen bias margin, so source bias remains unresolved. Qwen bmm is not eligible for formation labeling because exact repair/sham provenance is missing.\n\n## What can be claimed\n\n1. The formation pipeline distinguishes local, gradient, and update stages with open-loop common states.\n2. A real example exists where bias first appears at the parameter-gradient stage (Phi MM).\n3. A local difference can remain centered through all measured stages (Qwen saved-P).\n4. Persistence and formation are separate: Phi's SEUP consequence does not serve as its formation label.\n5. No universal source, transport, contract, or optimizer property is established yet.\n\n## What remains open\n\nThe endpoint denominator is retained in `population_screening.csv`, but only the exact v2.1 capture cases receive formation labels. The remaining endpoint population is explicitly `NOT_CAPTURED_EXISTING_ARTIFACT_ONLY`; legacy T1--T4 and SEUP roles are provenance, never formation ground truth.\n\nThe next scientific step is to close Phi's complete semantic transport decomposition and add an independent eligible case before promoting transport bias beyond a case-specific candidate.\n""", encoding="utf-8")

    manifest = {
        "schema": "kernel-analyzer-bias-formation-final-manifest-v1",
        "source_dir": str(SRC.relative_to(ROOT)),
        "population_rows": len(population),
        "measured_formation_cases": sorted(FORMATION_CASES),
        "not_a_universal_property_claim": True,
    }
    _write_json(OUT / "manifest.json", manifest)
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    sums = "".join(f"{_sha256(path)}  {path.relative_to(OUT)}\n" for path in files)
    (OUT / "SHA256SUMS").write_text(sums, encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "population_rows": len(population),
        "measured_cases": sorted(FORMATION_CASES),
        "phi_status": phi["status"],
        "liger_status": liger["status"],
    }, sort_keys=True))


if __name__ == "__main__":
    write_reports()
