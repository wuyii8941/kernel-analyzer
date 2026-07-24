from theory_oracle.seeded_fault_conformance_v0_1 import EXPECTED, calibration_rows


def test_seeded_fault_controls_match_preregistered_evidence_matrix():
    rows = calibration_rows()
    observed = {
        row["name"]: (row["production_observed"], row["mediation_observed"])
        for row in rows
    }
    assert observed == EXPECTED


def test_benign_mutation_changes_continuous_value_without_event_disagreement():
    row = next(row for row in calibration_rows() if row["name"] == "benign_continuous_mutation")
    assert row["local_continuous"]["nonzero"] > 0
    assert row["local_continuous"]["disagreement"] == 0
    assert row["mediation_observed"] is False


def test_propagation_only_is_not_a_local_producer():
    row = next(row for row in calibration_rows() if row["name"] == "propagation_only")
    assert row["local_continuous"]["nonzero"] == 0
    assert row["production_observed"] is False
    assert row["mediation_observed"] is True
