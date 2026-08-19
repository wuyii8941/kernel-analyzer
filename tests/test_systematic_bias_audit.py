import copy

import pytest

from scripts.build_systematic_bias_audit import cases
from kernel_analyzer.systematic_bias_audit import validate_audit, validate_case


def test_systematic_audit_has_exactly_eight_unique_cases():
    rows = cases()
    validate_audit(rows)
    assert len(rows) == 8
    assert len({row["case_id"] for row in rows}) == 8
    assert "qwen_layer23_key_materialization" not in {row["case_id"] for row in rows}


def test_global_centered_saved_p_is_not_variance_only():
    row = next(row for row in cases() if row["case_id"] == "qwen_saved_p_seq128")
    assert set(row["formation"]["global"].values()) == {"CENTERED"}
    assert row["mechanism"]["verdict"] != "VARIANCE_ONLY_UNDER_DECLARED_CONDITION"
    assert set(row["formation"]["conditional"].values()) == {"NOT_MEASURED"}


def test_trajectory_cannot_supply_formation_label():
    row = copy.deepcopy(cases()[0])
    row["formation"]["label_source"] = "TRAJECTORY"
    with pytest.raises(ValueError, match="trajectory cannot label formation"):
        validate_case(row)


def test_supported_mechanism_requires_intervention_and_sham():
    row = copy.deepcopy(cases()[0])
    row["mechanism"]["intervention"]["matched_sham_exact"] = False
    with pytest.raises(ValueError, match="lacks intervention/sham"):
        validate_case(row)


def test_qwen128_does_not_join_rounding_source_to_accumulation_trajectory():
    row = next(row for row in cases() if row["case_id"] == "qwen128_vproj_mm")
    assert row["mechanism"]["trajectory_repairs_declared_local_source"] is False
    assert row["mechanism"]["verdict"] == "UNRESOLVED_CONTRAST_MISMATCH"


def test_variance_only_needs_all_three_conditional_nulls():
    row = copy.deepcopy(cases()[0])
    row["mechanism"]["verdict"] = "VARIANCE_ONLY_UNDER_DECLARED_CONDITION"
    with pytest.raises(ValueError, match="three conditional nulls"):
        validate_case(row)
