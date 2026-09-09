import pytest

from scripts.prioritize_uncovered_triton_structures import prioritize


def record(symbol, *, status="REFERENCE_ADAPTER_REQUIRED", reduction=False,
           atomic=False, calls=None):
    return {
        "source": f"/data1/tzh/{symbol}.py",
        "symbol": symbol,
        "status": status,
        "operation_inventory": {
            "status": "BODY_INSPECTED",
            "calls": calls or {"tl.load": 1, "tl.store": 1},
            "arithmetic": {"Add": 1},
            "loop_count": 0,
            "has_atomic": atomic,
            "has_reduction": reduction,
            "structural_role": "NUMERICAL_COMPUTATION_REQUIRES_SEMANTIC_REVIEW",
        },
    }


def test_only_uncovered_definitions_are_ranked_without_semantic_claim():
    rows = prioritize([
        record("plain"),
        record("sum", reduction=True, calls={"tl.load": 1, "tl.sum": 1, "tl.store": 1}),
        record("known", status="REFERENCE_TEMPLATE_AVAILABLE"),
    ])
    assert [row["representative_symbol"] for row in rows] == ["sum", "plain"]
    assert all(row["semantic_family_status"] == "REQUIRES_HUMAN_REVIEW" for row in rows)


def test_expression_specific_calls_do_not_split_a_structural_cluster():
    a = record("a", calls={"tl.load": 1, "tl.store": 1, "tmp0.to": 1})
    b = record("b", calls={"tl.load": 1, "tl.store": 1, "tmp9.to": 4})
    rows = prioritize([a, b])
    assert len(rows) == 1
    assert rows[0]["definition_count"] == 2


def test_unseen_structure_is_ranked_before_covered_structure():
    known = record("known", status="REFERENCE_TEMPLATE_AVAILABLE")
    repeated = record("repeated")
    new = record("new", reduction=True, calls={"tl.load": 1, "tl.sum": 1, "tl.store": 1})
    rows = prioritize([known, repeated, new])
    assert rows[0]["representative_symbol"] == "new"
    assert rows[0]["structurally_seen_with_registered_reference"] is False
    repeated_row = next(row for row in rows if row["representative_symbol"] == "repeated")
    assert repeated_row["structurally_seen_with_registered_reference"] is True


def test_obvious_existing_family_name_is_a_hint_not_a_semantic_verdict():
    rows = prioritize([record("triton_fused_softmax_1")])
    assert rows[0]["likely_existing_family_hints"] == ["SOFTMAX"]
    assert rows[0]["semantic_family_status"] == "REQUIRES_HUMAN_REVIEW"


def test_rsqrt_reduction_is_flagged_as_likely_existing_normalization():
    row = record(
        "generic_fused_math",
        reduction=True,
        calls={"tl.load": 1, "tl.sum": 1, "libdevice.rsqrt": 1, "tl.store": 1},
    )
    assert prioritize([row])[0]["likely_existing_family_hints"] == ["NORMALIZATION"]


def test_type_audit_must_match_and_floating_cluster_is_ranked_first():
    integer = record("integer")
    floating = record(
        "floating", reduction=True,
        calls={"tl.load": 1, "tl.sum": 1, "tl.store": 1},
    )
    types = [
        {"source": integer["source"], "symbol": integer["symbol"],
         "type_review": {"status": "INTEGER_BOOLEAN_POINTERS_REQUIRES_CONTROL_REVIEW"}},
        {"source": floating["source"], "symbol": floating["symbol"],
         "type_review": {"status": "FLOATING_POINTERS_REQUIRES_SEMANTIC_REVIEW"}},
    ]
    rows = prioritize([integer, floating], types)
    assert rows[0]["representative_symbol"] == "floating"
    with pytest.raises(ValueError, match="identities differ"):
        prioritize([integer, floating], types[:1])
