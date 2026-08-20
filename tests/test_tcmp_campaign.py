import math

import numpy as np
import pytest

from kernel_analyzer.tcmp_campaign import (
    ModelCampaignSpec,
    ModelCellSpec,
    TCMPDisposition,
    audit_denominator,
    benjamini_hochberg,
    exact_sign_flip_statistics,
    holm_rejections,
)


def test_exact_sign_flip_separates_persistent_and_canceling_paths():
    persistent = np.ones((8, 8))
    canceling_vector = np.array([1.0, -1.0] * 4)
    canceling = np.outer(canceling_vector, canceling_vector)
    positive = exact_sign_flip_statistics(persistent)
    null = exact_sign_flip_statistics(canceling)
    assert positive["coherence_amplification"] == math.sqrt(8)
    assert positive["above_null_95"]
    assert null["coherence_amplification"] == 0.0
    assert not null["above_null_95"]


def test_multiplicity_rules_are_fail_closed():
    values = {"a": 0.001, "b": 0.02, "c": 0.5}
    assert benjamini_hochberg(values, 0.10) == {"a": True, "b": True, "c": False}
    assert holm_rejections(values, 0.05) == {"a": True, "b": True, "c": False}


def test_tcmp_support_requires_orbit_applicability():
    with pytest.raises(ValueError, match="valid semantic orbit"):
        TCMPDisposition("i", "p", "EXACT_REPAIR_ONLY", "TCMP_SUPPORTED")


def test_denominator_never_drops_unresolved_invocations():
    rows = [
        TCMPDisposition("i0", "p0", "TCMP_ORBIT_READY", "NO_DETECTABLE_PERSISTENCE_UNDER_SCREEN"),
        TCMPDisposition("i1", "p1", "UNRESOLVED_BOUNDARY", "UNRESOLVED_PROOF"),
    ]
    report = audit_denominator(["i0", "i1"], rows)
    assert report["status"] == "COMPLETE_DENOMINATOR_DISPOSITION"
    assert report["unresolved_count"] == 1
    assert not report["universal_claim_eligible"]
    missing = audit_denominator(["i0", "i1", "i2"], rows)
    assert missing["status"] == "INCOMPLETE_DENOMINATOR"


def test_campaign_freezes_unique_cells_and_8_16_protocol():
    cells = [ModelCellSpec("m_text128", "owner/model", "d" * 40, "TEXT", 128)]
    campaign = ModelCampaignSpec("c", "deadbeef", cells)
    assert campaign.screening_steps == 8
    assert campaign.confirmation_steps == 16
    with pytest.raises(ValueError, match="unique"):
        ModelCampaignSpec("c", "deadbeef", cells + cells)
