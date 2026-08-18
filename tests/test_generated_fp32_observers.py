import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.generated_fp32_observer import (
    GeneratedFP32Observer,
    nonfinite_aware_metrics,
    promoted_pointer_arguments,
    validate_compiled_triton_replay_abi,
    validate_typed_triton_reference_abi,
)
from scripts.analyze_generated_fp32_screen import bootstrap, bootstrap_counts, metric_equal


def test_two_state_bootstrap_rejects_degenerate_cluster_draws() -> None:
    counts = bootstrap_counts(states=2, draws=4000, seed=7)
    assert counts.shape == (4000, 2)
    assert np.all(np.count_nonzero(counts, axis=1) == 2)
    assert np.all(counts.sum(axis=1) == 2)


from scripts.typed_triton_reference import fp32_pointer_program
from scripts.generated_nontriton_fp32_observer import fp32_external_reference
from scripts.inductor_buffer_origins import (
    InductorBufferOriginRecorder,
    _node_record,
)
from scripts.same_dtype_semantic_observer import SameDtypeSemanticCandidateObserver
from scripts.frozen_state_checkpoint import (
    load_state_checkpoints,
    state_checkpoint_path,
    write_gzip,
)


def test_promotion_preserves_shared_storage_views() -> None:
    base = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
    left = base[:, :3]
    right = base[:, 1:4]
    _, pointers = promoted_pointer_arguments((left, right))
    assert pointers[0].dtype == torch.float32
    assert pointers[0].untyped_storage().data_ptr() == pointers[1].untyped_storage().data_ptr()
    assert pointers[0].stride() == left.stride()
    assert pointers[1].storage_offset() == right.storage_offset()


def test_promotion_fails_closed_for_cross_dtype_alias() -> None:
    base = torch.arange(8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="cross-dtype aliases"):
        promoted_pointer_arguments((base, base.view(torch.uint8)))


def test_compiled_triton_replay_rejects_pointer_dtype_promotion() -> None:
    kernel = SimpleNamespace(triton_meta={
        "signature": {"in_ptr0": "*bf16", "out_ptr0": "*bf16", "xnumel": "i32"}
    })
    source = torch.arange(8, dtype=torch.bfloat16)
    output = torch.empty_like(source)
    promoted, _ = promoted_pointer_arguments((source, output, 8))
    with pytest.raises(RuntimeError, match="INVALID_REFERENCE_ABI"):
        validate_compiled_triton_replay_abi(kernel, (source, output, 8), promoted)


def test_typed_triton_program_changes_only_float_pointer_abis() -> None:
    source = """
import triton
import triton.language as tl
from torch._inductor.runtime import triton_heuristics
@triton_heuristics.pointwise(
    size_hints={'x': 8}, filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp16',
                               'index_ptr': '*i64', 'xnumel': 'i32'},
                 'constants': {}},
    inductor_meta={})
@triton.jit
def copy_kernel(in_ptr0, out_ptr0, index_ptr, xnumel, XBLOCK: tl.constexpr):
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    mask = x < xnumel
    value = tl.load(in_ptr0 + x, mask)
    tl.store(out_ptr0 + x, value, mask)
"""
    typed, metadata = fp32_pointer_program(source, "copy_kernel")
    assert metadata["changed_float_pointers"] == {
        "in_ptr0": {"from": "*bf16", "to": "*fp32"},
        "out_ptr0": {"from": "*fp16", "to": "*fp32"},
    }
    assert metadata["typed_signature"]["index_ptr"] == "*i64"
    assert "tl.load(in_ptr0 + x, mask)" in typed
    assert metadata["only_pointer_abi_literals_changed"]


def test_typed_triton_reference_accepts_physical_fp32_storage() -> None:
    kernel = SimpleNamespace(triton_meta={
        "signature": {"in_ptr0": "*fp32", "out_ptr0": "*fp32", "xnumel": "i32"}
    })
    source = torch.arange(8, dtype=torch.float32)
    output = torch.empty_like(source)
    validate_typed_triton_reference_abi(kernel, (source, output, 8))


def test_typed_triton_reference_rejects_stale_bf16_abi() -> None:
    kernel = SimpleNamespace(triton_meta={
        "signature": {"in_ptr0": "*bf16", "out_ptr0": "*bf16", "xnumel": "i32"}
    })
    source = torch.arange(8, dtype=torch.float32)
    output = torch.empty_like(source)
    with pytest.raises(RuntimeError, match="TYPED_REFERENCE_ABI_MISMATCH"):
        validate_typed_triton_reference_abi(kernel, (source, output, 8))


def test_inductor_buffer_origin_record_prefers_exact_origin_node() -> None:
    exact = SimpleNamespace(name="ka_f_0042_mul")
    contributors = [exact, SimpleNamespace(name="ka_f_0041_cast")]
    ir_node = SimpleNamespace(
        get_name=lambda: "buf7",
        get_layout=lambda: SimpleNamespace(
            device="cuda:0", dtype=torch.bfloat16,
            size=(1, 64, 2048), stride=(131072, 2048, 1),
        ),
        get_origin_node=lambda: exact,
        get_origins=lambda: contributors,
    )
    scheduler = SimpleNamespace(
        node=ir_node,
        get_name=lambda: "op7",
        get_outputs=lambda: [ir_node],
    )
    row = _node_record("triton_poi_fused_mul_0", scheduler)
    assert row is not None
    assert row["buffer_names"] == ["buf7"]
    assert row["buffer_metadata"]["buf7"]["shape"] == ["1", "64", "2048"]
    assert row["exact_origin_node"] == "ka_f_0042_mul"
    assert row["origin_node_exact"]
    assert row["contributing_origin_nodes"] == [
        "ka_f_0041_cast", "ka_f_0042_mul",
    ]


def test_inductor_buffer_origin_certificate_fails_closed_without_exact_origin() -> None:
    recorder = InductorBufferOriginRecorder()
    recorder._records = [{
        "kernel_name": "kernel", "scheduler_node": "op0",
        "buffer_names": ["buf0"], "exact_origin_node": None,
        "buffer_metadata": {"buf0": None},
        "contributing_origin_nodes": ["a", "b"],
        "origin_node_exact": False, "phase": "UNRESOLVED",
        "is_external": False,
    }]
    result = recorder.certificate()
    assert result["status"] == "PARTIAL_FAIL_CLOSED"
    assert result["denominator"]["buffers_with_exact_origin_node"] == 0


def test_same_dtype_candidate_observer_captures_exact_formal_output() -> None:
    class Kernel:
        triton_meta = {"signature": {
            "in_ptr0": "*bf16", "out_ptr0": "*bf16", "xnumel": "i32",
        }}

        @staticmethod
        def run(source, output, xnumel):
            output.copy_(source)

    module = SimpleNamespace(kernel=Kernel())
    observed = []
    observer = SameDtypeSemanticCandidateObserver(
        modules=[module],
        campaign_rows=[{"symbol": "kernel", "region_id": "forward:0"}],
        task_rows=[{
            "task_id": "forward:0:out_ptr0",
            "candidate_region_id": "forward:0",
            "formal_pointer": "out_ptr0",
            "exact_aot_endpoint_id": "forward:graph0:mul",
        }],
        sink=lambda task_id, tensor, metadata: observed.append(
            (task_id, tensor.clone(), metadata)
        ),
    )
    source = torch.arange(4, dtype=torch.bfloat16)
    output = torch.empty_like(source)
    with observer:
        module.kernel.run(source, output, 4)
    observer.validate()
    assert observed[0][0] == "forward:0:out_ptr0"
    assert torch.equal(observed[0][1], source)
    assert observed[0][2]["exact_aot_endpoint_id"] == "forward:graph0:mul"


def test_same_dtype_observer_does_not_reexecute_theorem_closed_prelude() -> None:
    observer = SameDtypeSemanticCandidateObserver(
        modules=[], campaign_rows=[], task_rows=[], sink=lambda *_args: None,
        inventory_rows=[{
            "category": "COMPUTE",
            "compute_region_id": "forward:device_put",
            "implementation_kind_or_helper_role": "DIRECT_TENSOR_METHOD",
            "source_line_sha256": "copy-line",
        }],
    )
    assert not observer.nontriton_rows
    observer.validate()


def test_nonfinite_streaming_metrics_do_not_hide_geometry() -> None:
    candidate = torch.tensor([1.0, float("nan"), float("inf"), 4.0])
    reference = torch.tensor([1.0, float("nan"), 3.0, 2.0])
    result = nonfinite_aware_metrics(
        candidate, reference, sample_size=4, metric_chunk_elements=2
    )
    assert result["matching_nan"] == 1
    assert result["nonfinite_mismatch"] == 1
    assert not result["exact"]
    assert result["full_value_scan"]


def test_finite_streaming_fast_path_preserves_exact_metrics() -> None:
    candidate = torch.tensor([1.0, 2.5, -4.0, 8.0], dtype=torch.bfloat16)
    reference = torch.tensor([0.5, 2.0, -3.0, 8.0], dtype=torch.bfloat16)
    result = nonfinite_aware_metrics(
        candidate, reference, sample_size=4, metric_chunk_elements=2
    )
    delta = candidate.float() - reference.float()
    assert result["candidate_finite"] and result["reference_finite"]
    assert result["nonfinite_mismatch"] == 0
    assert result["nonzero_elements"] == int(torch.count_nonzero(delta))
    assert result["signed_mean"] == float(delta.double().mean())
    assert result["rms"] == float(delta.double().square().mean().sqrt())
    assert result["max_abs"] == float(delta.abs().max())


def test_external_mm_reference_promotes_floating_storage() -> None:
    left = torch.tensor([[1.25, 2.5]], dtype=torch.bfloat16)
    right = torch.tensor([[3.0], [4.0]], dtype=torch.bfloat16)
    reference = fp32_external_reference("mm", (left, right), {})
    assert reference.dtype == torch.float32
    assert torch.equal(reference, left.float() @ right.float())


def test_repeat_comparison_treats_matching_nonfinite_sketches_as_equal() -> None:
    metric = {
        "exact": False, "nonzero_elements": 0, "signed_mean": 0.0,
        "rms": 0.0, "max_abs": 0.0, "candidate_finite": False,
        "reference_finite": True, "nonfinite_mismatch": 1,
        "directional_error_sketch": {
            "flat_coordinate_indices": [0, 1],
            "signed_delta_values": [0.0, float("nan")],
        },
    }
    assert metric_equal(metric, dict(metric))


def test_vectorized_cluster_bootstrap_is_finite() -> None:
    errors = torch.arange(24, dtype=torch.float64).reshape(6, 4).numpy()
    result = bootstrap(errors, bootstrap_counts(states=6, draws=128, seed=7))
    assert result["lower_95"] <= result["median"] <= result["upper_95"]


def test_triton_observer_binds_exact_embedded_program(tmp_path) -> None:
    program = """\n@triton.jit\ndef kernel(in_ptr0, out_ptr0, xnumel: tl.constexpr):\n    x = tl.load(in_ptr0)\n    tl.store(out_ptr0, x)\n"""
    wrapper = tmp_path / "output_code.py"
    wrapper.write_text(
        "kernel = async_compile.triton('kernel', " + repr(program) + ")\n"
    )
    row = {
        "symbol": "kernel", "region_id": "forward:0",
        "embedded_program_sha256": hashlib.sha256(program.encode()).hexdigest(),
    }
    observer = GeneratedFP32Observer(
        modules=[SimpleNamespace(__file__=str(wrapper))], campaign_rows=[row]
    )
    observer.validate_program_identity()
    row["embedded_program_sha256"] = "0" * 64
    changed = GeneratedFP32Observer(
        modules=[SimpleNamespace(__file__=str(wrapper))], campaign_rows=[row]
    )
    with pytest.raises(RuntimeError, match="differs from frozen campaign"):
        changed.validate_program_identity()


def test_joint_state_checkpoint_is_atomic_and_release_bound(tmp_path) -> None:
    state_id = "state/with unsafe path"
    release_sha = "a" * 64
    directory = tmp_path / "joint_states"
    directory.mkdir()
    path = state_checkpoint_path(directory, state_id)
    write_gzip(path, {
        "schema": "kernel-analyzer-joint-frozen-candidate-state-v1",
        "release_capture_sha256": release_sha,
        "state_id": state_id,
        "triton_state": {"repeats": [{"repeat": 0}]},
        "nontriton_state": {"repeats": [{"repeat": 0}]},
    })
    assert path.parent == directory
    assert "/" not in path.name
    triton = {"states": {}}
    nontriton = {"states": {}}
    load_state_checkpoints(
        directory=directory,
        release_capture_sha256=release_sha,
        triton_payload=triton,
        nontriton_payload=nontriton,
    )
    assert triton["states"][state_id]["repeats"][0]["repeat"] == 0
    assert nontriton["states"][state_id]["repeats"][0]["repeat"] == 0
    with pytest.raises(RuntimeError, match="binds another release"):
        load_state_checkpoints(
            directory=directory,
            release_capture_sha256="b" * 64,
            triton_payload={"states": {}},
            nontriton_payload={"states": {}},
        )
