import pytest

from scripts.build_direct_parameter_operand_mappings import map_direct_operands


def test_unique_direct_parameter_operand_is_bound_without_claiming_gradient_reach():
    forward = {"nodes": [{
        "name": "forward_g0__embedding", "input_edges": [
            {"source_node": "forward_g0__weight"},
            {"source_node": "forward_g0__indices"},
        ],
    }]}
    binding = {
        "name": "model.embed_tokens.weight", "aliases": ["model.embed_tokens.weight"],
        "shape": [10, 4],
    }
    result = map_direct_operands(
        forward, {"forward_g0__weight": binding},
        {"forward:graph0:forward_g0__embedding"},
    )
    row = result["rows"][0]
    assert row["status"] == "UNIQUE_DIRECT_PARAMETER_OPERAND"
    assert row["parameters"][0]["name"] == "model.embed_tokens.weight"
    assert "NOT_GRADIENT_REACH_PROOF" in row["parameters"][0]["binding_scope"]


def test_ambiguous_parameter_operands_fail_closed():
    forward = {"nodes": [{
        "name": "target", "input_edges": [
            {"source_node": "left"}, {"source_node": "right"},
        ],
    }]}
    by_primal = {
        name: {"name": name, "aliases": [name], "shape": [2, 2]}
        for name in ("left", "right")
    }
    result = map_direct_operands(forward, by_primal, {"forward:graph0:target"})
    assert result["rows"][0]["status"] == "UNRESOLVED_DIRECT_PARAMETER_OPERAND"
    assert result["rows"][0]["parameters"] == []


def test_unknown_endpoint_is_rejected():
    with pytest.raises(ValueError, match="Unique endpoint"):
        map_direct_operands({"nodes": []}, {}, {"forward:graph0:missing"})
