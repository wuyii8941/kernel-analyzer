"""Fail-closed T1 -> T4 orchestration and resumable artifact production."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .api import (
    BLOCKED,
    FAIL,
    PASS,
    TIERS,
    AnalysisContext,
    AnalysisReport,
    AnalysisSpec,
    CaseCertificate,
    TierEvidence,
)
from .artifacts import digest, write_json, write_json_gz
from .render import render_case, render_mathematics
from .property import validate_predictor_features


def _file_sha256(path: Path) -> str:
    digest_value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest()


def _code_provenance(value: Any) -> Mapping[str, Any]:
    """Fingerprint plugin code and retained evidence used by a run.

    Resume is unsafe when it keys only on human-readable candidate IDs.  This
    compact traversal records class source and every Path-valued artifact in
    provider/backend configuration without attempting to serialize models or
    tensors.
    """

    seen = set()

    def encode(item: Any) -> Any:
        if isinstance(item, Path):
            resolved = item.resolve()
            return {
                "path": str(resolved),
                "exists": resolved.exists(),
                "sha256": _file_sha256(resolved) if resolved.is_file() else None,
            }
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        if isinstance(item, Mapping):
            return {str(key): encode(val) for key, val in sorted(item.items(), key=lambda x: str(x[0]))}
        if isinstance(item, (list, tuple)):
            return [encode(val) for val in item]
        marker = id(item)
        if marker in seen:
            return {"recursive": True, "class": item.__class__.__qualname__}
        seen.add(marker)
        target = item if inspect.isfunction(item) or inspect.isclass(item) else item.__class__
        source = inspect.getsourcefile(target)
        payload = {
            "class": "%s.%s" % (item.__class__.__module__, item.__class__.__qualname__),
            "source": str(Path(source).resolve()) if source else None,
            "source_sha256": _file_sha256(Path(source).resolve()) if source and Path(source).is_file() else None,
        }
        if hasattr(item, "__dict__"):
            payload["configuration"] = {
                key: encode(val)
                for key, val in sorted(vars(item).items())
                if not callable(val) and not key.startswith("_")
            }
        return payload

    return encode(value)


def _unit_id(row: Mapping[str, Any]) -> str:
    value = row.get("unit_id", row.get("proof_unit_id"))
    if not value:
        raise ValueError("proof unit lacks a stable ID")
    return str(value)


def _blocked(reason: str) -> TierEvidence:
    return TierEvidence(status=BLOCKED, reason=reason)


def _tier_payload(rows: Mapping[str, TierEvidence]) -> Mapping[str, Any]:
    return {unit_id: row.as_dict() for unit_id, row in sorted(rows.items())}


def _load_tier(path: Path, expected_ids: Sequence[str]) -> Dict[str, TierEvidence]:
    data = json.loads(path.read_text())
    if data.get("input_unit_ids") != list(expected_ids):
        raise ValueError("stale stage input set: %s" % path)
    return {unit_id: TierEvidence(**row) for unit_id, row in data["rows"].items()}


def _classification(tiers: Mapping[str, TierEvidence]) -> str:
    if all(tiers[tier].status == PASS for tier in TIERS):
        return "COMPLETE_DIRECTIONAL_ACCUMULATION_CASE"
    if tiers["T1_LOCAL"].status != PASS:
        return "NO_LOCAL_DIFFERENCE_OR_UNRESOLVED"
    if tiers["T2_CAUSAL"].status != PASS:
        return "LOCAL_SCREEN_ONLY"
    if tiers["T3_COHERENT"].status != PASS:
        return "CAUSAL_NONCOHERENT"
    return "COHERENT_SINGLE_STEP"


class Analyzer:
    """Run candidate plugins while enforcing ordered, non-substitutable gates."""

    schema = "kernel-analyzer-run-v1"

    def analyze(self, spec: AnalysisSpec, resume: bool = True) -> AnalysisReport:
        identity = {
            "subject": spec.subject,
            "states": [(state.state_id, state.role) for state in spec.states],
            "candidates": [candidate.candidate_id for candidate in spec.candidates],
            "metadata": dict(spec.metadata),
            "reference_provenance": _code_provenance(spec.reference),
            "candidate_provenance": [
                _code_provenance(candidate) for candidate in spec.candidates
            ],
            "model_factory_provenance": _code_provenance(spec.model_factory),
            "step_builder_provenance": _code_provenance(spec.step_builder),
        }
        identity_sha256 = digest(identity)
        run_id = "run-" + identity_sha256[:16]
        run_dir = spec.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        report_path = run_dir / "report.json"
        if resume and report_path.exists():
            data = json.loads(report_path.read_text())
            if data.get("run_identity_sha256") != identity_sha256:
                raise ValueError("stale report identity: %s" % report_path)
            certificates = [CaseCertificate(
                case_id=row["case_id"],
                proof_unit_id=row["proof_unit_id"],
                candidate_id=row["candidate_id"],
                natural=bool(row["natural"]),
                cause_axis=row.get("cause_axis", "UNATTRIBUTED"),
                classification=row["classification"],
                tiers={tier: TierEvidence(**row["tiers"][tier]) for tier in TIERS},
            ) for row in data["case_certificates"]]
            return AnalysisReport(
                run_id=data["run_id"], status=data["status"], subject=data["subject"],
                proof_unit_count=int(data["proof_unit_count"]),
                unresolved_proof_units=int(data["unresolved_proof_units"]),
                candidate_summaries=data["candidate_summaries"],
                case_certificates=certificates, artifact_dir=run_dir,
            )

        reference = spec.reference.analyze(spec, run_dir)
        units = list(reference.proof_units)
        unit_ids = [_unit_id(row) for row in units]
        target_ids = [_unit_id(row) for row in reference.case_targets]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("reference proof-unit IDs are not unique")
        if set(unit_ids) & set(target_ids) or len(target_ids) != len(set(target_ids)):
            raise ValueError("case-target IDs must be unique and separate from the denominator")
        measurement_target_ids = unit_ids + target_ids
        write_json_gz(run_dir / "proof_units.json.gz", {
            "subject": spec.subject,
            "census": dict(reference.census),
            "templates": list(reference.templates),
            "unresolved": list(reference.unresolved),
            "case_targets": list(reference.case_targets),
            "proof_units": units,
        })
        render_mathematics(run_dir / "mathematics.md", reference.templates)

        context = AnalysisContext(spec=spec, reference=reference, run_dir=run_dir)
        cases = []
        summaries: Dict[str, Mapping[str, Any]] = {}
        for backend in spec.candidates:
            stage_dir = run_dir / "stages" / backend.candidate_id
            stage_dir.mkdir(parents=True, exist_ok=True)
            census_path = stage_dir / "census.json"
            if resume and census_path.exists():
                census_data = json.loads(census_path.read_text())
                from .api import CandidateCensus
                census = CandidateCensus(
                    candidate_id=backend.candidate_id,
                    runtime_regions=census_data["runtime_regions"],
                    status=census_data["status"], metadata=census_data["metadata"],
                )
            else:
                census = backend.census(context)
                write_json(census_path, {
                    "status": census.status,
                    "runtime_regions": list(census.runtime_regions),
                    "metadata": dict(census.metadata),
                })
            property_path = stage_dir / "PROPERTY_SIGNED_TRANSPORT.json"
            if resume and property_path.exists():
                property_predictions = _load_tier(
                    property_path, measurement_target_ids
                )
            else:
                predictor = getattr(backend, "predict_signed_transport", None)
                if predictor is None:
                    property_predictions = {
                        unit_id: TierEvidence(
                            status="UNRESOLVED",
                            reason="backend has no reference/schedule factor provider",
                        )
                        for unit_id in measurement_target_ids
                    }
                else:
                    property_predictions = dict(
                        predictor(context, units + list(reference.case_targets))
                    )
                if set(property_predictions) != set(measurement_target_ids):
                    missing = sorted(set(measurement_target_ids) - set(property_predictions))
                    extra = sorted(set(property_predictions) - set(measurement_target_ids))
                    raise ValueError(
                        "property predictor must account for every F+B unit; "
                        "missing=%s extra=%s" % (missing, extra)
                    )
                for unit_id, prediction in property_predictions.items():
                    validate_predictor_features(
                        prediction.evidence.get("predictor_inputs", {})
                    )
                write_json(property_path, {
                    "schema": "kernel-analyzer-signed-transport-predictions-v1",
                    "rows": _tier_payload(property_predictions),
                    "input_unit_ids": measurement_target_ids,
                    "candidate_tensor_values_read": False,
                    "t4_used_as_label_or_predictor": False,
                })
            t1_path = stage_dir / "T1_LOCAL.json"
            if resume and t1_path.exists():
                t1 = _load_tier(t1_path, measurement_target_ids)
            else:
                t1 = dict(backend.measure_local(context, units))
                write_json(t1_path, {"rows": _tier_payload(t1),
                                     "input_unit_ids": measurement_target_ids})
            unknown = set(t1) - (set(unit_ids) | set(target_ids))
            if unknown:
                raise ValueError("T1 evidence references unknown proof units: %s" % sorted(unknown))
            t1_pass = sorted(unit_id for unit_id, row in t1.items() if row.status == PASS)
            t2_path = stage_dir / "T2_CAUSAL.json"
            if resume and t2_path.exists():
                t2 = _load_tier(t2_path, t1_pass)
            else:
                t2 = dict(backend.intervene_causally(context, t1_pass)) if t1_pass else {}
                write_json(t2_path, {"rows": _tier_payload(t2), "input_unit_ids": t1_pass})
            t2_pass = sorted(unit_id for unit_id in t1_pass if t2.get(unit_id, _blocked("")).status == PASS)
            t3_path = stage_dir / "T3_COHERENT.json"
            if resume and t3_path.exists():
                t3 = _load_tier(t3_path, t2_pass)
            else:
                t3 = dict(backend.confirm_coherence(context, t2_pass)) if t2_pass else {}
                write_json(t3_path, {"rows": _tier_payload(t3), "input_unit_ids": t2_pass})
            t3_pass = sorted(unit_id for unit_id in t2_pass if t3.get(unit_id, _blocked("")).status == PASS)
            t4_path = stage_dir / "T4_ACCUMULATION.json"
            if resume and t4_path.exists():
                t4 = _load_tier(t4_path, t3_pass)
            else:
                t4 = dict(backend.run_trajectory(context, t3_pass)) if t3_pass else {}
                write_json(t4_path, {"rows": _tier_payload(t4), "input_unit_ids": t3_pass})

            backend_cases = []
            for unit_id in sorted(set(t1) | set(t2) | set(t3) | set(t4)):
                tiers = {
                    "T1_LOCAL": t1.get(unit_id, _blocked("not measured")),
                    "T2_CAUSAL": t2.get(unit_id, _blocked("blocked until T1 pass")),
                    "T3_COHERENT": t3.get(unit_id, _blocked("blocked until T2 pass")),
                    "T4_ACCUMULATION": t4.get(unit_id, _blocked("blocked until T3 pass")),
                }
                certificate = CaseCertificate(
                    case_id="%s::%s" % (backend.candidate_id, unit_id),
                    proof_unit_id=unit_id,
                    candidate_id=backend.candidate_id,
                    tiers=tiers,
                    classification=_classification(tiers),
                    natural=all(bool(row.evidence.get("natural", True)) for row in tiers.values()),
                    cause_axis=str(
                        tiers["T1_LOCAL"].evidence.get("cause_axis", "UNATTRIBUTED")
                    ),
                )
                backend_cases.append(certificate)
                if (certificate.classification == "COMPLETE_DIRECTIONAL_ACCUMULATION_CASE"
                        and certificate.natural):
                    cases.append(certificate)
                    write_json(run_dir / "cases" / (hashlib.sha256(
                        certificate.case_id.encode()).hexdigest()[:16] + ".json"), certificate.as_dict())
                    render_case(run_dir / "cases" / (hashlib.sha256(
                        certificate.case_id.encode()).hexdigest()[:16] + ".md"), certificate)

            summaries[backend.candidate_id] = {
                "scope": census.metadata.get("scope", "FULL_FB_DENOMINATOR"),
                "candidate_census_status": census.status,
                "runtime_regions": len(census.runtime_regions),
                "global_fb_units": len(units),
                "total_fb_units": (
                    len(units) if census.metadata.get("scope", "FULL_FB_DENOMINATOR")
                    == "FULL_FB_DENOMINATOR" else None
                ),
                "total_target_units": (
                    len(units) if census.metadata.get("scope", "FULL_FB_DENOMINATOR")
                    == "FULL_FB_DENOMINATOR" else len(t1)
                ),
                "T1_accounted": len(t1),
                "T1_tested": sum(row.status in {"PASS", "FAIL"} for row in t1.values()),
                "T1_unresolved": sum(row.status == "UNRESOLVED" for row in t1.values()),
                "T1_pass": len(t1_pass),
                "T2_pass": len(t2_pass),
                "T2_accounted": len(t2),
                "T2_unresolved": sum(row.status == "UNRESOLVED" for row in t2.values()),
                "T3_pass": len(t3_pass),
                "T3_accounted": len(t3),
                "T3_unresolved": sum(row.status == "UNRESOLVED" for row in t3.values()),
                "T4_pass": sum(row.status == PASS for row in t4.values()),
                "T4_accounted": len(t4),
                "T4_unresolved": sum(row.status == "UNRESOLVED" for row in t4.values()),
                "PROPERTY_accounted": len(property_predictions),
                "PROPERTY_predicted_coherent": sum(
                    row.status == PASS for row in property_predictions.values()
                ),
                "PROPERTY_predicted_normal": sum(
                    row.status == FAIL for row in property_predictions.values()
                ),
                "PROPERTY_unresolved": sum(
                    row.status in {"UNRESOLVED", "BLOCKED"}
                    for row in property_predictions.values()
                ),
                "complete_cases": sum(
                    row.classification == "COMPLETE_DIRECTIONAL_ACCUMULATION_CASE"
                    and row.natural
                    for row in backend_cases
                ),
                "positive_controls": sum(
                    row.classification == "COMPLETE_DIRECTIONAL_ACCUMULATION_CASE"
                    and not row.natural
                    for row in backend_cases
                ),
            }
            summary_row = summaries[backend.candidate_id]
            full_scope = summary_row["scope"] == "FULL_FB_DENOMINATOR"
            summary_row["pipeline_complete"] = (
                (not full_scope or summary_row["T1_accounted"] == len(units))
                and summary_row["T1_unresolved"] == 0
                and summary_row["T2_unresolved"] == 0
                and summary_row["T3_unresolved"] == 0
                and summary_row["T4_unresolved"] == 0
                and (not full_scope or census.status == "CAPTURED_EXECUTION_DERIVED")
            )
            write_json_gz(run_dir / "candidates" / (backend.candidate_id + ".json.gz"), {
                "census": {
                    "status": census.status,
                    "runtime_regions": list(census.runtime_regions),
                    "metadata": dict(census.metadata),
                },
                "certificates": [row.as_dict() for row in backend_cases],
                "signed_transport_predictions": _tier_payload(property_predictions),
            })

        report = AnalysisReport(
            run_id=run_id,
            status=(
                "COMPLETE" if not reference.unresolved
                and all(row["pipeline_complete"] for row in summaries.values())
                else "PARTIAL_FAIL_CLOSED"
            ),
            subject=spec.subject,
            proof_unit_count=len(units),
            unresolved_proof_units=len(reference.unresolved),
            candidate_summaries=summaries,
            case_certificates=cases,
            artifact_dir=run_dir,
        )
        payload = report.as_dict()
        payload["schema"] = self.schema
        payload["run_identity_sha256"] = identity_sha256
        payload["run_identity"] = identity
        payload["report_sha256"] = digest(payload)
        if spec.resources is not None and spec.resources.max_artifact_bytes is not None:
            artifact_bytes = sum(
                path.stat().st_size for path in run_dir.rglob("*") if path.is_file()
            )
            if artifact_bytes > spec.resources.max_artifact_bytes:
                raise RuntimeError("run artifacts exceed the declared resource budget")
        write_json(run_dir / "report.json", payload)
        lines = [
            "# Kernel Analyzer run",
            "",
            "- Subject: `%s`" % spec.subject,
            "- Status: `%s`" % report.status,
            "- F+B proof units: %d" % report.proof_unit_count,
            "- Unresolved mathematical units: %d" % report.unresolved_proof_units,
            "- Complete natural cases: %d" % len(report.case_certificates),
            "",
            "## Candidates",
            "",
        ]
        for candidate_id, row in sorted(summaries.items()):
            lines.append("- `%s`: T1 pass/tested/total %d/%d/%d "
                         "(%d unresolved), T2 %d, T3 %d, T4 %d; "
                         "property predicted/unresolved %d/%d" % (
                candidate_id, row["T1_pass"], row["T1_tested"],
                row["total_target_units"], row["T1_unresolved"], row["T2_pass"],
                row["T3_pass"], row["T4_pass"],
                row["PROPERTY_predicted_coherent"], row["PROPERTY_unresolved"],
            ))
        (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
        return report
