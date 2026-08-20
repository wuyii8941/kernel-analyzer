import ast
import json
from pathlib import Path

from forkcert.generated_compute_dataflow_audit import (
    _is_statically_empty_allocation,
    validate_generated_compute_dataflow_audit,
)


ARTIFACT = Path(
    "results/training_semantic_oracle/qwen3_1p7b/full_step_inventory/"
    "generated_compute_dataflow_audit_v1.json"
)


def test_only_static_zero_numel_allocations_are_boundary_safe() -> None:
    empty = ast.parse("x = empty_strided_cuda((0, 1024), (1024, 1), dtype)").body[0].value
    nonempty = ast.parse("x = empty_strided_cuda((8, 1024), (1024, 1), dtype)").body[0].value
    assert _is_statically_empty_allocation(empty)
    assert not _is_statically_empty_allocation(nonempty)


def test_all_generated_compute_pointer_dataflow_is_exact() -> None:
    artifact = json.loads(ARTIFACT.read_text())
    validate_generated_compute_dataflow_audit(artifact)
    denominator = artifact["denominator"]
    assert denominator["compute_invocations"] == 1447
    assert denominator["kind_counts"] == {
        "DIRECT_ATEN": 1,
        "EXTERN": 760,
        "TRITON": 686,
    }
    assert denominator["direct_tensor_producer_edges"] > 0
    assert len(artifact["rows"]) == 1447
    assert all(
        row["input_tensor_variables"]
        or not row["transitive_runner_abi_input_variables"]
        for row in artifact["rows"]
    )
    assert not artifact["gates"]["forward_vjp_semantic_identity_granted"]
    assert not artifact["gates"]["candidate_correctness_granted"]
    assert not artifact["gates"]["property_generalization_allowed"]


def test_embedding_scatter_is_bound_to_zero_fill_and_cast_by_pointer() -> None:
    artifact = json.loads(ARTIFACT.read_text())
    row = next(
        row
        for row in artifact["rows"]
        if row["kind"] == "DIRECT_ATEN"
    )
    assert row["symbol"] == "index_put_"
    assert row["boundary_witness"]["mutated_target"] == "buf1126"
    producers = {
        edge["tensor_variable"]: edge["producer_region_id"]
        for edge in row["direct_producer_edges"]
    }
    assert producers["buf1126"] == "backward:1"
    assert producers["buf1125"] == "backward:925"
    assert "primals_1" in row["direct_boundary_input_variables"]
    cast = next(
        row
        for row in artifact["rows"]
        if row["region_id"] == "backward:926"
    )
    assert cast["direct_producer_edges"] == [
        {
            "producer_region_id": "backward:direct_aten:0",
            "storage_root_variable": "buf1126",
            "tensor_variable": "buf1126",
        }
    ]
