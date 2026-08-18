"""Candidate backend implementations and retained-evidence regression adapter."""

from __future__ import annotations

import json
from pathlib import Path
from abc import ABC, abstractmethod
import math
from typing import Any, Mapping, Sequence

from .api import (
    BLOCKED, FAIL, PASS, UNRESOLVED, AnalysisContext, CandidateBackend, CandidateCensus,
    TierEvidence,
)
from .statistics import coherence_certificate
from .property import (
    SignedTransportState,
    signed_event_transport,
    signed_transport_certificate,
    validate_predictor_features,
)


class CandidateRuntime(ABC):
    """Kernel/backend-specific observation surface; verdicts stay generic."""

    @abstractmethod
    def runtime_census(self, context: AnalysisContext) -> Sequence[Mapping[str, Any]]:
        pass

    def signed_transport_factors(self, context, proof_units):
        """Optional reference/schedule-only property factors for every unit.

        A returned row contains ``predictor_inputs``, ``reference_margin`` and
        frozen state rows.  Each state supplies either ``transported_error`` or
        complete scalar ``event_errors`` plus matching
        ``reference_transport_directions``.  Candidate tensor values and T1--T4
        results are forbidden.
        """
        return {}

    @abstractmethod
    def local_observations(self, context, proof_units):
        """Rows require mapped, finite, repeat_stable, max_abs and natural."""

    @abstractmethod
    def causal_replacements(self, context, proof_unit_ids):
        """Rows require sham_exact, parameter_reached and delta_norm."""

    @abstractmethod
    def confirmation_vectors(self, context, proof_unit_ids):
        """Return fixed complete-carrier vectors on independent states."""

    @abstractmethod
    def paired_trajectories(self, context, proof_unit_ids):
        """Rows require same_weight_contrast, nonzero_grad and live_divergence."""


class NumericalCandidateBackend(CandidateBackend):
    """Generic automatic T1-T4 evaluator over a runtime instrumentation plugin."""

    def __init__(self, candidate_id: str, runtime: CandidateRuntime,
                 bootstrap_samples: int = 2000, hypothesis: str = "raw",
                 hypothesis_family_size: int = 3) -> None:
        if hypothesis not in {"raw", "relative", "factor"}:
            raise ValueError("unsupported frozen T3 hypothesis: %s" % hypothesis)
        self.candidate_id = candidate_id
        self.runtime = runtime
        self.bootstrap_samples = bootstrap_samples
        self.hypothesis = hypothesis
        if hypothesis_family_size < 1:
            raise ValueError("hypothesis_family_size must be positive")
        self.hypothesis_family_size = hypothesis_family_size

    def census(self, context):
        regions = list(self.runtime.runtime_census(context))
        return CandidateCensus(
            candidate_id=self.candidate_id,
            runtime_regions=regions,
            status=(getattr(self.runtime, "census_status", None)
                    or ("CAPTURED_EXECUTION_DERIVED" if regions
                        else "UNRESOLVED_EMPTY_CENSUS")),
            metadata={"scope": getattr(self.runtime, "scope", "FULL_FB_DENOMINATOR")},
        )

    def predict_signed_transport(self, context, proof_units):
        rows = self.runtime.signed_transport_factors(context, proof_units)
        output = {}
        for unit in proof_units:
            unit_id = str(unit.get("unit_id", unit.get("proof_unit_id")))
            row = rows.get(unit_id)
            if not row:
                output[unit_id] = TierEvidence(
                    status=UNRESOLVED,
                    reason="reference/schedule property factors are unavailable",
                )
                continue
            predictor_inputs = dict(row.get("predictor_inputs", {}))
            validate_predictor_features(predictor_inputs)
            if row.get("candidate_tensor_values_read", False):
                raise ValueError("property factors read candidate tensor values")
            if row.get("t4_used_as_label_or_predictor", False):
                raise ValueError("T4 cannot enter the F+B property predictor")
            states = []
            for state in row.get("states", ()):
                if "transported_error" in state:
                    transported = tuple(state["transported_error"])
                else:
                    transported = signed_event_transport(
                        state["event_errors"],
                        state["reference_transport_directions"],
                    )
                states.append(SignedTransportState(
                    state_id=str(state["state_id"]),
                    transported_error=transported,
                    nonlinear_remainder_bound=float(
                        state.get("nonlinear_remainder_bound", 0.0)
                    ),
                ))
            try:
                certificate = signed_transport_certificate(
                    states,
                    reference_margin=float(row["reference_margin"]),
                    bootstrap_samples=self.bootstrap_samples,
                    seed=int(row.get("bootstrap_seed", 0)),
                )
            except (KeyError, TypeError, ValueError) as error:
                output[unit_id] = TierEvidence(
                    status=UNRESOLVED,
                    reason="invalid or incomplete signed transport factors: %s" % error,
                )
                continue
            status = (
                PASS if certificate["status"] == "PREDICTED_COHERENT_F_B_BIAS"
                else UNRESOLVED if certificate["status"].startswith("ABSTAIN_")
                else FAIL
            )
            output[unit_id] = TierEvidence(
                status=status,
                evidence={
                    "predictor_inputs": predictor_inputs,
                    "certificate": certificate,
                    "candidate_tensor_values_read": False,
                    "t4_used_as_label_or_predictor": False,
                },
                reason=("" if status in {PASS, FAIL}
                        else "signed transport certificate abstained"),
            )
        return output

    def measure_local(self, context, proof_units):
        rows = self.runtime.local_observations(context, proof_units)
        output = {}
        for unit_id, row in rows.items():
            if not bool(row.get("observation_available", True)):
                output[unit_id] = TierEvidence(
                    status=UNRESOLVED, evidence=dict(row),
                    reason="no candidate observation for this F+B unit",
                )
                continue
            contrast = dict(row.get("contrast", {}))
            same_dtype_optimization = (
                contrast.get("kind") == "OPTIMIZATION_SAME_DTYPE"
                and contrast.get("candidate_dtype") == contrast.get("reference_dtype")
                and bool(contrast.get("semantic_boundary_exact"))
                and bool(contrast.get("candidate_program_sha256"))
                and bool(contrast.get("reference_program_sha256"))
            )
            precision_same_semantics = (
                contrast.get("kind") == "PRECISION_SAME_SEMANTICS"
                and contrast.get("low_dtype") != contrast.get("high_dtype")
                and bool(contrast.get("low_dtype"))
                and bool(contrast.get("high_dtype"))
                and bool(contrast.get("semantic_boundary_exact"))
                and bool(contrast.get("semantic_program_sha256"))
                and bool(contrast.get("low_arm_program_sha256"))
                and bool(contrast.get("high_arm_program_sha256"))
            )
            contrast_valid = same_dtype_optimization or precision_same_semantics
            cause_axis = (
                "OPTIMIZATION" if same_dtype_optimization else
                "PRECISION" if precision_same_semantics else
                "UNATTRIBUTED"
            )
            direction = dict(row.get("local_direction", {}))
            directional = (
                bool(direction.get("full_coordinates"))
                and bool(direction.get("independent_states"))
                and float(direction.get("cluster_bootstrap_lower_95") or 0.0) > 0.0
            ) or bool(direction.get("analytic_factor_direction_proved"))
            base_valid = (
                bool(row.get("mapped")) and bool(row.get("finite"))
                and bool(row.get("repeat_stable"))
                and float(row.get("max_abs", 0.0)) > 0.0
            )
            passed = base_valid and contrast_valid and directional
            unresolved = base_valid and (not contrast_valid or not directional)
            output[unit_id] = TierEvidence(
                status=PASS if passed else UNRESOLVED if unresolved else FAIL,
                evidence={
                    **dict(row),
                    "engine_gates": {
                        "base_local_observation_valid": base_valid,
                        "same_dtype_optimization_contrast_valid": same_dtype_optimization,
                        "precision_same_semantics_contrast_valid": precision_same_semantics,
                        "attributable_contrast_valid": contrast_valid,
                        "local_direction_proved": directional,
                    },
                    "cause_axis": cause_axis,
                },
                reason=(
                    "" if passed else
                    "attributable precision/optimization contrast or complete local direction is unresolved"
                    if unresolved else
                    "local mapping/finite/repeat/nonzero gate failed"
                ),
            )
        return output

    def intervene_causally(self, context, proof_unit_ids):
        rows = self.runtime.causal_replacements(context, proof_unit_ids)
        output = {}
        for unit_id in proof_unit_ids:
            row = rows.get(unit_id, {})
            if not row:
                output[unit_id] = TierEvidence(
                    status=UNRESOLVED, evidence={},
                    reason="causal replacement has not been executed",
                )
                continue
            passed = (
                bool(row.get("replacement_exact")) and bool(row.get("sham_exact"))
                and bool(row.get("parameter_reached"))
                and bool(row.get("non_target_endpoints_exact"))
                and float(row.get("delta_norm", 0.0)) > 0.0
            )
            output[unit_id] = TierEvidence(
                status=PASS if passed else FAIL, evidence=dict(row),
                reason="" if passed else "replacement/sham/parameter-reach gate failed",
            )
        return output

    def confirm_coherence(self, context, proof_unit_ids):
        rows = self.runtime.confirmation_vectors(context, proof_unit_ids)
        output = {}
        # Bonferroni is conservative and satisfies family-wise control without
        # selecting coordinates or hypotheses after seeing candidate values.
        alpha = 0.05 / max(1, len(proof_unit_ids) * self.hypothesis_family_size)
        for unit_id in proof_unit_ids:
            parent_row = rows.get(unit_id, {})
            hypotheses = parent_row.get("hypotheses", {}) if parent_row else {}
            row = (
                hypotheses.get(self.hypothesis, {})
                if hypotheses else parent_row
            )
            if not row:
                output[unit_id] = TierEvidence(
                    status=UNRESOLVED, evidence={},
                    reason="independent complete-carrier confirmation has not been executed",
                )
                continue
            complete = bool(row.get("complete_coordinates"))
            independent = bool(row.get("independent_states"))
            repeats = bool(row.get("repeat_exact"))
            state_ids = list(row.get("state_ids", ()))
            pilot_ids = set(row.get("pilot_state_ids", ()))
            disjoint = bool(state_ids) and not (set(state_ids) & pilot_ids)
            declared_roles = {state.state_id: state.role for state in context.spec.states}
            states_declared = all(
                declared_roles.get(value) == "CONFIRMATION" for value in state_ids
            ) and all(
                declared_roles.get(value) == "DISCOVERY" for value in pilot_ids
            )
            factor_valid = (
                self.hypothesis != "factor" or bool(row.get("analytic_factorization"))
            )
            relative_valid = (
                self.hypothesis != "relative"
                or row.get("statistic") == "REFERENCE_RELATIVE_GRADIENT_SCALE"
            )
            protocol_ok = (complete and independent and repeats and disjoint
                           and factor_valid and relative_valid and states_declared)
            if protocol_ok and row.get("streaming_complete_gram") is True:
                certificate = dict(row.get("precomputed_certificate", {}))
                if not certificate.get("streamed_complete_gram"):
                    certificate = {"status": "UNRESOLVED_INVALID_PRECOMPUTED_CERTIFICATE"}
            elif protocol_ok:
                certificate = coherence_certificate(
                    row.get("vectors", ()), alpha=alpha,
                    bootstrap_samples=self.bootstrap_samples,
                    seed=int(row.get("seed", 0)),
                )
            else:
                certificate = {"status": "UNRESOLVED_INCOMPLETE_CONFIRMATION_PROTOCOL"}
            passed = certificate["status"] == "PASS"
            output[unit_id] = TierEvidence(
                status=PASS if passed else FAIL,
                evidence={**dict(row), "vectors": "OMITTED_AFTER_STATISTICS",
                          "certificate": certificate,
                          "multiple_testing": "BONFERRONI_FWER",
                          "hypothesis_family_size": self.hypothesis_family_size,
                          "frozen_hypothesis": self.hypothesis,
                          "pilot_confirmation_disjoint": disjoint,
                          "state_roles_declared": states_declared},
                reason="" if passed else certificate["status"],
            )
        return output

    def run_trajectory(self, context, proof_unit_ids):
        rows = self.runtime.paired_trajectories(context, proof_unit_ids)
        output = {}
        for unit_id in proof_unit_ids:
            row = rows.get(unit_id, {})
            if not row:
                output[unit_id] = TierEvidence(
                    status=UNRESOLVED, evidence={},
                    reason="paired live-weight trajectory has not been executed",
                )
                continue
            declared_roles = {state.state_id: state.role for state in context.spec.states}
            trajectory_ids = list(row.get("state_ids", ()))
            states_declared = bool(trajectory_ids) and all(
                declared_roles.get(value) == "TRAJECTORY" for value in trajectory_ids
            )
            passed = (
                bool(row.get("same_weight_contrast"))
                and bool(row.get("nonzero_gradient_contrast"))
                and bool(row.get("live_weight_divergence"))
                and bool(row.get("frozen_paired_trajectory"))
                and states_declared
            )
            output[unit_id] = TierEvidence(
                status=PASS if passed else FAIL, evidence=dict(row),
                reason="" if passed else "paired live-weight trajectory gate failed",
            )
        return output


class RetainedCaseBackend(CandidateBackend):
    """Replay frozen T1-T4 certificates through the new ordered scheduler."""

    def __init__(self, candidate_id: str, audit_path: Path) -> None:
        self.candidate_id = candidate_id
        self.audit_path = audit_path

    def _rows(self):
        return json.loads(self.audit_path.read_text())["rows"]

    def census(self, context: AnalysisContext) -> CandidateCensus:
        return CandidateCensus(
            candidate_id=self.candidate_id,
            runtime_regions=[{
                "region_id": "retained-case::" + row["case"],
                "kind": "CLOSED_RETAINED_CASE_TARGET",
            } for row in self._rows()],
            status="CAPTURED_RETAINED_EXECUTION_EVIDENCE",
            metadata={"source": str(self.audit_path), "scope": "CASE_TARGETS_ONLY"},
        )

    def _tier(self, name: str, allowed: Sequence[str] = ()) -> Mapping[str, TierEvidence]:
        output = {}
        allowed_set = set(allowed)
        dual_track_names = {
            "T1_LOCAL": "F1_COMPLETE_FB",
            "T2_CAUSAL": "F2_CAUSAL_REPAIR",
            "T3_COHERENT": "F3_REAL_CARRIER",
            "T4_ACCUMULATION": "F4_PAIRED_TRAJECTORY",
        }
        for row in self._rows():
            unit_id = "retained-case::" + row["case"]
            if allowed_set and unit_id not in allowed_set:
                continue
            source = (
                row["tiers"][name] if "tiers" in row
                else row["flash_style"]["gates"][dual_track_names[name]]
            )
            source_status = source["status"]
            status = PASS if source_status == "PASS" else FAIL
            output[unit_id] = TierEvidence(
                status=status,
                evidence={
                    "source": source["evidence"],
                    "source_status": source_status,
                    "natural": True,
                },
                reason=source.get("note", ""),
            )
        return output

    def measure_local(self, context, proof_units):
        return self._tier("T1_LOCAL")

    def intervene_causally(self, context, proof_unit_ids):
        return self._tier("T2_CAUSAL", proof_unit_ids)

    def confirm_coherence(self, context, proof_unit_ids):
        return self._tier("T3_COHERENT", proof_unit_ids)

    def run_trajectory(self, context, proof_unit_ids):
        return self._tier("T4_ACCUMULATION", proof_unit_ids)


class FlashControlBackend(CandidateBackend):
    """Bind the retained paper-mechanism positive control to ordered gates."""

    candidate_id = "flash_paper_positive_control"
    unit_id = "control::flash_paper_reference"

    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact

    def _data(self):
        return json.loads(self.artifact.read_text())

    def census(self, context):
        return CandidateCensus(
            candidate_id=self.candidate_id,
            runtime_regions=[{"region_id": self.unit_id, "kind": "PAPER_REFERENCE_CONTROL"}],
            status="CAPTURED_RETAINED_EXECUTION_EVIDENCE",
            metadata={"scope": "POSITIVE_CONTROL"},
        )

    def measure_local(self, context, proof_units):
        data = self._data()
        nonzero = any(row["unstable"]["max_grad_abs"] > 0 for row in data["cases"])
        finite = all(row["unstable"]["nonfinite"] == 0 for row in data["cases"])
        return {self.unit_id: TierEvidence(
            status=PASS if nonzero and finite else FAIL,
            evidence={"source": str(self.artifact), "nonzero": nonzero,
                      "finite": finite, "natural": False},
        )}

    def intervene_causally(self, context, proof_unit_ids):
        data = self._data()
        passed = bool(data["positive_control"]["closed_f_b"])
        return {self.unit_id: TierEvidence(
            status=PASS if passed else FAIL,
            evidence={"fb_closure": data["fb_closure"], "natural": False},
        )} if self.unit_id in proof_unit_ids else {}

    def confirm_coherence(self, context, proof_unit_ids):
        row = self._data()["positive_control"]["v_only_live_weight"]
        passed = row["heldout_positive"] == row["heldout_states"]
        return {self.unit_id: TierEvidence(
            status=PASS if passed else FAIL, evidence={**row, "natural": False},
        )} if self.unit_id in proof_unit_ids else {}

    def run_trajectory(self, context, proof_unit_ids):
        data = self._data()
        row = data["positive_control"]["v_only_live_weight"]
        passed = (
            row["same_initial_weights"] and row["reference_evaluated_at_each_current_state"]
            and len(data["paired_trajectory"]["records"]) > 1
        )
        return {self.unit_id: TierEvidence(
            status=PASS if passed else FAIL,
            evidence={"steps": data["paired_trajectory"]["steps"], "natural": False},
        )} if self.unit_id in proof_unit_ids else {}


class ObservedRegionRuntime(CandidateRuntime):
    """Adapt candidate-blind changed-region observations to automatic T1."""

    census_status = "PARTIAL_CHANGED_REGION_CENSUS"
    scope = "FULL_FB_DENOMINATOR"

    def __init__(self, observation_path: Path, candidate_cell: str,
                 t2_evidence_path: Path | None = None,
                 t3_evidence_path: Path | None = None) -> None:
        self.observation_path = observation_path
        self.candidate_cell = candidate_cell
        self.t2_evidence_path = t2_evidence_path
        self.t3_evidence_path = t3_evidence_path

    def _data(self):
        return json.loads(self.observation_path.read_text())

    def runtime_census(self, context):
        return [{
            "region_id": row["region_id"], "phase": row["phase"],
            "kind": row["kind"], "observed_endpoints": row["endpoint_count"],
        } for row in self._data()["rows"]]

    def local_observations(self, context, proof_units):
        data = self._data()
        by_region = {row["region_id"]: row for row in data["rows"]}
        output = {}
        for unit in proof_units:
            unit_id = unit["unit_id"]
            cell = unit.get("candidate_cells", {}).get(self.candidate_cell, {})
            ids = cell.get("candidate_region_ids", {}).get("ids", ()) or ()
            observed = [by_region[value] for value in ids if value in by_region]
            if not observed:
                output[unit_id] = {
                    "observation_available": False,
                    "candidate_region_ids": list(ids),
                    "natural": True,
                }
                continue
            output[unit_id] = {
                "observation_available": True,
                "mapped": cell.get("mapping_status") == "EXACT_ALL_MEMBERS",
                "finite": all(row["all_finite"] for row in observed),
                "repeat_stable": all(
                    row["all_steps_and_repeats_present"] for row in observed
                ) and bool(data["gates"]["all_worker_observations_stable"]),
                "max_abs": max(float(row["max_abs_max"]) for row in observed),
                "candidate_region_ids": [row["region_id"] for row in observed],
                "contrast": {
                    "kind": "UNRESOLVED_RETAINED_ARTIFACT_LACKS_TYPED_ARM_PROVENANCE",
                    "candidate_dtype": None,
                    "reference_dtype": None,
                    "semantic_boundary_exact": False,
                    "candidate_program_sha256": None,
                    "reference_program_sha256": None,
                },
                "local_direction": {
                    "full_coordinates": False,
                    "independent_states": True,
                    "cluster_bootstrap_lower_95": None,
                    "analytic_factor_direction_proved": False,
                },
                "candidate_values_used_to_select_units": False,
                "natural": True,
            }
        return output

    def causal_replacements(self, context, proof_unit_ids):
        if self.t2_evidence_path is None or not self.t2_evidence_path.exists():
            return {}
        data = json.loads(self.t2_evidence_path.read_text())
        output = {}
        for unit_id in proof_unit_ids:
            row = data.get("unit_rows", {}).get(unit_id)
            # An incomplete many-region intervention is not a negative result;
            # omit it so the generic evaluator reports UNRESOLVED fail-closed.
            if row and row.get("status") != "UNRESOLVED":
                output[unit_id] = dict(row)
        return output

    def confirmation_vectors(self, context, proof_unit_ids):
        if self.t3_evidence_path is None or not self.t3_evidence_path.exists():
            return {}
        data = json.loads(self.t3_evidence_path.read_text())
        return {
            unit_id: dict(data.get("unit_rows", {}).get(unit_id, {}))
            for unit_id in proof_unit_ids
            if data.get("unit_rows", {}).get(unit_id, {}).get("status")
            in {"PASS", "FAIL"}
        }

    def paired_trajectories(self, context, proof_unit_ids):
        return {}
