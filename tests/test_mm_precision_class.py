from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mm_precision_class_has_three_independent_concrete_units() -> None:
    payload = json.loads(
        (ROOT / "results/coverage/cases/mm_precision_mechanism_class.json").read_text()
    )
    assert payload["status"] == "COMPLETE_WITHIN_MM_FAMILY_MECHANISM_CLASS"
    assert payload["gates"]["three_distinct_concrete_fb_units"]
    assert payload["gates"]["three_distinct_models"]
    assert payload["gates"]["forward_and_backward_endpoints_represented"]
    assert payload["gates"]["every_total_direction_ci_lower_positive"]


def test_mm_precision_class_keeps_cross_family_claim_closed() -> None:
    payload = json.loads(
        (ROOT / "results/coverage/cases/mm_precision_mechanism_class.json").read_text()
    )
    assert not payload["gates"]["cross_operator_family_generalization"]
    assert {row["operator_family"] for row in payload["members"]} == {"MM"}
    assert len(payload["mechanism_support"]["kernel"]) >= 2
    assert len(payload["mechanism_support"]["output_rounding"]) >= 2
