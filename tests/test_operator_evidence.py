from forkcert.operator_evidence import (
    EvidenceGates,
    allowed_claim_level,
    compare_non_target_context,
    validate_evidence_report,
    production_mediation_interpretation,
)


def gates(**changes):
    values = dict(
        complete_witness=True,
        same_input_local_replay=True,
        local_discrepancy_reproducible=True,
        provenance_complete=True,
        candidate_realization_preserved=True,
        intervention_executed=True,
        oracle_recomputed=True,
        non_target_context_invariant=True,
        lower_level_replay=False,
        first_bad_stage_isolated=False,
        null_controls_valid=True,
    )
    values.update(changes)
    return EvidenceGates(**values)


def test_effect_magnitude_cannot_upgrade_missing_provenance():
    assert allowed_claim_level(gates(provenance_complete=False)) == "INTERVENTION_DEPENDENT_ATTRIBUTION"


def test_changed_candidate_anchor_cannot_be_operator_level():
    assert allowed_claim_level(gates(candidate_realization_preserved=False)) == "INTERVENTION_DEPENDENT_ATTRIBUTION"


def test_vacuous_non_target_context_cannot_be_operator_level():
    # A whole-program or whole-wrapper replacement has no remaining artifact
    # on which to test context invariance.  It is attribution evidence, not an
    # operator-level result merely because an empty comparison is equal.
    assert (
        allowed_claim_level(gates(non_target_context_invariant=False))
        == "INTERVENTION_DEPENDENT_ATTRIBUTION"
    )


def test_cross_level_requires_first_bad_stage_not_just_kernel_replay():
    assert allowed_claim_level(gates(lower_level_replay=True)) == "OPERATOR_LEVEL_EFFECT"
    assert (
        allowed_claim_level(gates(lower_level_replay=True, first_bad_stage_isolated=True))
        == "CROSS_LEVEL_COMPILER_LOCALIZATION"
    )


def test_invalid_null_control_invalidates_all_claims():
    assert allowed_claim_level(gates(null_controls_valid=False)) == "INVALID"


def test_non_target_context_ignores_only_declared_target():
    baseline = {
        "compiler_config_digest": "c",
        "graph_count": 1,
        "graphs": [{"sha256": "g"}],
        "artifacts": [
            {"target_id": "norm", "sha256": "old"},
            {"target_id": "head", "sha256": "same"},
        ],
        "shape_layout_contracts": [],
    }
    intervention = {
        **baseline,
        "artifacts": [
            {"target_id": "norm", "sha256": "new"},
            {"target_id": "head", "sha256": "same"},
        ],
    }
    assert compare_non_target_context(baseline, intervention, ["norm"])["exact"]
    intervention["artifacts"][1]["sha256"] = "changed"
    assert not compare_non_target_context(baseline, intervention, ["norm"])["exact"]


def test_report_claim_is_recomputed_fail_closed():
    evidence = gates(provenance_complete=False)
    report = {
        "schema_version": "forkcert.operator-evidence.v0.1",
        "case_identity": {},
        "region_inventory": [],
        "local_replay": {},
        "provenance": {},
        "intervention": {},
        "oracle": {},
        "gates": evidence.__dict__,
        "allowed_claim_level": "OPERATOR_LEVEL_EFFECT",
        "limitations": ["test"],
    }
    assert validate_evidence_report(report) == [
        "claim level mismatch: reported=OPERATOR_LEVEL_EFFECT expected=INTERVENTION_DEPENDENT_ATTRIBUTION"
    ]


def test_production_and_mediation_are_orthogonal():
    assert "no current" in production_mediation_interpretation(False, False)["interpretation"]
    assert "no observed effect" in production_mediation_interpretation(True, False)["interpretation"]
    assert "upstream discrepancy" in production_mediation_interpretation(False, True)["interpretation"]
    assert "not by itself a root-cause" in production_mediation_interpretation(True, True)["interpretation"]


def test_uninstantiated_chain_cannot_be_interpreted_as_negative():
    row = production_mediation_interpretation(True, None)
    assert row["mediation_observed"] is None
    assert "uninstantiated" in row["interpretation"]
