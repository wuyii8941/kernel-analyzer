from __future__ import annotations

import json
import importlib.metadata
import shutil
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "5_specification"
sys.path[:0] = [str(SPEC), str(SPEC / "corpus")]

import call_binding as binding_backend
from kernel_analyzer import contract_binding, measure
from kernel_analyzer.check import PRODUCER_RECORD, _boolean_task_comparison, contract_binding_context
from kernel_analyzer.contract_binding import ContractBindingError, ContractReference
from kernel_analyzer.reference_eval.storage_effects import InputSnapshot

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _fixture(tmp_path, monkeypatch):
    call = tmp_path / "ce_call.py"
    call.write_text("def call(inputs):\n    raise RuntimeError('test wrapper is not executed here')\n")
    call_ref = f"{call}:call"
    plan = {
        "schema": "triton-call-binding-v1", "id": "test.ce.measure.v1", "status": "reviewed",
        "subject": {"library": "test", "version": "1", "callable": "ce_call.call",
                    "runtime_package": {"distribution": "torch",
                                        "version": importlib.metadata.version("torch")},
                    "boundary": "forward and VJP", "measure_call": call_ref},
        "launch_scope": "all_triton_launches_within_call",
        "tasks": [{
            "id": "ce", "contract": "CE", "profile": "index_unsmoothed",
            "task_statement": "Two-class, unsmoothed, summed class-index cross entropy and its VJP.",
            "input_map": [
                {"target": "/logits", "source": "/tensor_arguments/logits/before",
                 "conversion": "tensor_values"},
                {"target": "/target", "source": "/tensor_arguments/target/before",
                 "conversion": "tensor_values"},
                {"target": "/weights", "source": "/scalars/weights"},
                {"target": "/eps", "source": "/scalars/eps"},
                {"target": "/ignore_index", "source": "/scalars/ignore_index"},
                {"target": "/reduction", "source": "/scalars/reduction"},
                {"target": "/upstream", "source": "/scalars/upstream"},
            ],
            "output_map": [
                {"target": "/loss", "source": "/returns/loss", "measure_output": "loss",
                 "conversion": "tensor_values"},
                {"target": "/gradient", "source": "/returns/gradient", "measure_output": "gradient",
                 "conversion": "tensor_values"},
            ],
        }],
        "state_map": [{"argument": "logits", "effect": "unchanged"},
                      {"argument": "target", "effect": "unchanged"}],
    }
    plan_path = tmp_path / "binding.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    reviewed = tmp_path / "reviewed"
    for directory in ("bindings", "calls", "contracts", "adapters"):
        (reviewed / directory).mkdir(parents=True, exist_ok=True)
    (reviewed / "bindings" / f"{plan['id']}.json").write_bytes(plan_path.read_bytes())
    (reviewed / "calls" / f"{plan['id']}.py").write_bytes(call.read_bytes())
    shutil.copyfile(SPEC / "contracts/CE.yaml", reviewed / "contracts/CE.yaml")
    shutil.copyfile(SPEC / "adapters/CE.py", reviewed / "adapters/CE.py")
    monkeypatch.setattr(binding_backend.registry, "VALIDATED", reviewed)
    return call, call_ref, plan_path, reviewed


def test_admitted_contract_reference_is_derived_from_the_reviewed_plan(tmp_path, monkeypatch):
    _, call_ref, plan_path, reviewed = _fixture(tmp_path, monkeypatch)
    reference = ContractReference(plan_path=plan_path, specification_root=SPEC, call_ref=call_ref,
                                  base_dir=tmp_path, measure_names=["loss", "gradient"],
                                  backend=binding_backend, reviewed_root=reviewed)
    inp = {"logits": torch.tensor([[0.0, 0.0]]), "target": torch.tensor([0]), "weights": None,
           "eps": 0, "ignore_index": -100, "reduction": "sum", "upstream": 1}
    snapshot = InputSnapshot(inp)
    outs = {"loss": torch.tensor(np.log(2), dtype=torch.float64),
            "gradient": torch.tensor([[-0.5, 0.5]], dtype=torch.float64)}
    specs, summary = reference.evaluate(inp, outs, [SimpleNamespace(kernel_name="ce_kernel")], snapshot)
    assert summary["reference_usable_for_bias_measurement"] is True
    assert summary["binding_status"] == "reviewed"
    assert specs["loss"][0] <= np.log(2) <= specs["loss"][1]
    expected = np.array([[-0.5, 0.5]])
    assert np.all(specs["gradient"][0] <= expected)
    assert np.all(expected <= specs["gradient"][1])


def test_specification_root_is_relative_to_the_declaration(tmp_path, monkeypatch):
    _, call_ref, plan_path, reviewed = _fixture(tmp_path, monkeypatch)
    reference = ContractReference(plan_path=plan_path, specification_root="relative-specification",
                                  call_ref=call_ref, base_dir=tmp_path,
                                  measure_names=["loss", "gradient"],
                                  backend=binding_backend, reviewed_root=reviewed)
    assert reference.root == (tmp_path / "relative-specification").resolve()


def test_contract_reference_rejects_a_changed_call_wrapper(tmp_path, monkeypatch):
    call, call_ref, plan_path, reviewed = _fixture(tmp_path, monkeypatch)
    call.write_text(call.read_text() + "# changed after review\n")
    with pytest.raises(ContractBindingError, match="call wrapper differs"):
        ContractReference(plan_path=plan_path, specification_root=SPEC, call_ref=call_ref,
                          base_dir=tmp_path, measure_names=["loss", "gradient"],
                          backend=binding_backend, reviewed_root=reviewed)


def test_contract_reference_rejects_a_different_runtime_package_version(tmp_path, monkeypatch):
    _, call_ref, plan_path, reviewed = _fixture(tmp_path, monkeypatch)
    plan = json.loads(plan_path.read_text())
    plan["subject"]["runtime_package"]["version"] = "not-the-installed-version"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    (reviewed / "bindings" / f"{plan['id']}.json").write_bytes(plan_path.read_bytes())
    with pytest.raises(ContractBindingError, match="bound runtime package version"):
        ContractReference(plan_path=plan_path, specification_root=SPEC, call_ref=call_ref,
                          base_dir=tmp_path, measure_names=["loss", "gradient"],
                          backend=binding_backend, reviewed_root=reviewed)


def test_contract_reference_rejects_a_tensor_hidden_inside_a_scalar_input(tmp_path, monkeypatch):
    _, call_ref, plan_path, reviewed = _fixture(tmp_path, monkeypatch)
    reference = ContractReference(plan_path=plan_path, specification_root=SPEC, call_ref=call_ref,
                                  base_dir=tmp_path, measure_names=["loss", "gradient"],
                                  backend=binding_backend, reviewed_root=reviewed)
    inp = {"logits": torch.tensor([[0.0, 0.0]]), "target": torch.tensor([0]), "weights": None,
           "eps": 0, "ignore_index": -100, "reduction": "sum", "upstream": 1,
           "hidden": [torch.tensor(1.0)]}
    snapshot = InputSnapshot(inp)
    outs = {"loss": torch.tensor(np.log(2), dtype=torch.float64),
            "gradient": torch.tensor([[-0.5, 0.5]], dtype=torch.float64)}
    with pytest.raises(ContractBindingError, match="nested tensor"):
        reference.evaluate(inp, outs, [SimpleNamespace(kernel_name="ce_kernel")], snapshot)


def test_contract_reference_rejects_an_unaccounted_tensor_source(tmp_path, monkeypatch):
    _, call_ref, plan_path, reviewed = _fixture(tmp_path, monkeypatch)
    reference = ContractReference(plan_path=plan_path, specification_root=SPEC, call_ref=call_ref,
                                  base_dir=tmp_path, measure_names=["loss", "gradient"],
                                  backend=binding_backend, reviewed_root=reviewed)
    inp = {"logits": torch.tensor([[0.0, 0.0]]), "target": torch.tensor([0]), "weights": None,
           "eps": 0, "ignore_index": -100, "reduction": "sum", "upstream": 1}
    snapshot = InputSnapshot(inp)
    outs = {"loss": torch.tensor(np.log(2), dtype=torch.float64),
            "gradient": torch.tensor([[-0.5, 0.5]], dtype=torch.float64)}
    context = {"unaccounted_sources_by_output": {"loss": ["L0:hidden_weight"]}}
    with pytest.raises(ContractBindingError, match="not accounted for by the binding plan"):
        reference.evaluate(inp, outs, [SimpleNamespace(kernel_name="ce_kernel")], snapshot,
                           binding_context=context)


def test_contract_binding_context_keeps_only_sources_without_admitted_provenance():
    out = torch.empty(2)
    storage = out.untyped_storage().data_ptr()
    context = contract_binding_context(
        {"out": out},
        {storage: ["L0:hidden_weight", f"L1:copy{PRODUCER_RECORD}copy: from input]"]})
    assert context == {"unaccounted_sources_by_output": {"out": ["L0:hidden_weight"]}}


def test_contract_binding_context_keeps_dispatch_visibility_separate_from_sources():
    from kernel_analyzer.provenance import ENCLOSED_LAUNCHES
    out = torch.empty(2)
    spans = [{"operator": "aten.sort.default", "first_launch": 0, "stop_launch": 9}]
    context = contract_binding_context(
        {"out": out}, {out.untyped_storage().data_ptr(): ["L1:prefix"]},
        {ENCLOSED_LAUNCHES: spans})
    assert context["unaccounted_sources_by_output"] == {"out": ["L1:prefix"]}
    assert context["dispatch_calls_with_untraced_internals"] == spans


def test_binding_refusal_explains_invisible_dispatch_internals(tmp_path, monkeypatch):
    _, call_ref, plan_path, reviewed = _fixture(tmp_path, monkeypatch)
    reference = ContractReference(plan_path=plan_path, specification_root=SPEC, call_ref=call_ref,
                                  base_dir=tmp_path, measure_names=["loss", "gradient"],
                                  backend=binding_backend, reviewed_root=reviewed)
    context = {
        "unaccounted_sources_by_output": {"loss": ["L1:prefix"]},
        "dispatch_calls_with_untraced_internals": [
            {"operator": "aten.sort.default", "first_launch": 0, "stop_launch": 9}],
    }
    with pytest.raises(ContractBindingError) as exc:
        reference.evaluate({}, {}, [], None, binding_context=context)
    message = str(exc.value)
    assert "L1:prefix" in message
    assert "untraced internals" in message
    assert "aten.sort.default" in message
    assert "[0, 9)" in message


def test_contract_reference_rejects_a_draft_plan_before_measurement():
    plan = SPEC / "corpus/call_bindings/liger_cross_entropy.json"
    with pytest.raises(ContractBindingError, match="reviewed binding plan"):
        ContractReference(plan_path=plan, specification_root=SPEC,
                          call_ref="missing.py:call", base_dir=ROOT,
                          measure_names=["loss"], backend=binding_backend)


def test_reviewed_inductor_softmax_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/inductor_softmax.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "pytorch.inductor.softmax.lastdim.v1"
    assert reference.measure_names == ["out"]


def test_reviewed_inductor_cross_entropy_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/inductor_cross_entropy.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "pytorch.inductor.cross_entropy.mean.v1"
    assert reference.measure_names == ["loss", "gradient"]


def test_reviewed_direct_triton_matmul_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_matmul.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.matmul.f32.16x32x16.v1"
    assert reference.measure_names == ["out"]


def test_reviewed_inductor_silu_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/inductor_silu.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "pytorch.inductor.silu.v1"
    assert reference.measure_names == ["out"]


def test_reviewed_direct_triton_conv2d_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_conv2d.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.conv2d.f32.grouped.v1"
    assert reference.measure_names == ["out"]


def test_reviewed_direct_triton_row_sum_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_row_sum.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.row_sum.f32.32x256.v1"
    assert reference.measure_names == ["out"]


def test_reviewed_direct_triton_exp_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_exp.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.exp.f32.1024.v1"
    assert reference.measure_names == ["out"]


def test_reviewed_direct_triton_kl_batchmean_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_kl_batchmean.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.kl_div.probability.batchmean.f32.8x64.v1"
    assert reference.measure_names == ["loss", "grad"]


def test_reviewed_direct_triton_bf16_cast_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_cast_bf16.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.cast.f32_to_bf16.1024.v1"
    assert reference.measure_names == ["out"]
    assert reference.program_reference_mode == "rounding_check"


def test_reviewed_direct_triton_multiply_add_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_multiply_add.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.multiply_add.f32.1024.v1"
    assert reference.measure_names == ["out"]
    assert reference.program_reference_mode == "numerical_difference"


def test_reviewed_direct_triton_transpose_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_transpose.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.transpose.f32.32x64.v1"
    assert reference.measure_names == ["out"]
    assert reference.program_reference_mode == "numerical_difference"


def test_reviewed_direct_triton_boolean_comparison_binding_is_admitted_and_exact():
    declaration_path = SPEC / "corpus/measure_cases/triton_greater_zero.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.greater_zero.f32.1024.v1"
    assert reference.measure_names == ["out"]
    assert reference.output_kinds == {"out": "boolean"}

    x = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float32)
    specs, summary = reference.evaluate(
        {"x": x}, {"out": torch.tensor([False, False, True])},
        [SimpleNamespace(kernel_name="comparison")], InputSnapshot({"x": x}))
    assert np.array_equal(specs["out"][0], np.array([0.0, 0.0, 1.0]))
    assert np.array_equal(specs["out"][1], specs["out"][0])
    assert summary["output_kinds"] == {"out": "boolean"}


def test_reviewed_direct_triton_index_select_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_index_select.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.index_select.f32.128x32.64.v1"
    assert reference.measure_names == ["out"]
    assert reference.output_kinds == {"out": "numeric"}


def test_reviewed_direct_triton_sort_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_sort_unique.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.sort_unique.f32.32x32.v1"
    assert reference.measure_names == ["values", "indices"]
    assert reference.output_kinds == {"values": "numeric", "indices": "numeric"}


def test_reviewed_direct_triton_max_pool2d_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_max_pool2d.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.max_pool2d.f32.2x3x16x16.k2s2.v1"
    assert reference.measure_names == ["out"]
    assert reference.output_kinds == {"out": "numeric"}


def test_reviewed_direct_triton_batch_norm_eval_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_batch_norm_eval.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.batch_norm_eval.f32.2x8x8x8.v1"
    assert reference.measure_names == ["out"]
    assert reference.output_kinds == {"out": "numeric"}


def test_reviewed_direct_triton_dropout_backward_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_dropout_backward.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.dropout_backward.f32.1024.v1"
    assert reference.measure_names == ["out"]
    assert reference.output_kinds == {"out": "numeric"}


def test_reviewed_direct_triton_index_add_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_index_add.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.index_add.f32.128x32.even64.v1"
    assert reference.measure_names == ["out"]
    assert reference.output_kinds == {"out": "numeric"}


def test_reviewed_direct_triton_clip_grad_value_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_clip_grad_value.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.clip_grad_value.f32.4096.c0p75.v1"
    assert reference.measure_names == ["out"]
    assert reference.output_kinds == {"out": "numeric"}


def test_reviewed_direct_triton_sgd_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_sgd.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.sgd.f32.4096.warm_nesterov.v1"
    assert reference.measure_names == ["theta", "b"]
    assert reference.output_kinds == {"theta": "numeric", "b": "numeric"}


def test_reviewed_direct_triton_radam_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_radam.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.radam.f32.4096.warm_rectified.v1"
    assert reference.measure_names == ["theta", "m", "v", "t"]
    assert reference.output_kinds == {
        "theta": "numeric", "m": "numeric", "v": "numeric", "t": "numeric"
    }


def test_reviewed_direct_triton_adafactor_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_adafactor.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.adafactor.f32.32x32.warm_factored.v1"
    assert reference.measure_names == ["theta", "row", "col", "t"]
    assert reference.output_kinds == {
        "theta": "numeric", "row": "numeric", "col": "numeric", "t": "numeric"
    }


def test_reviewed_direct_triton_rprop_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_rprop.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.rprop.f32.4096.warm.v1"
    assert reference.measure_names == ["theta", "prev", "eta", "t"]
    assert reference.output_kinds == {
        "theta": "numeric", "prev": "numeric", "eta": "numeric", "t": "numeric"
    }


def test_reviewed_direct_triton_sdpa_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_sdpa.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.sdpa.f32.b2.hq4.hkv2.l16.s32.e16.causal.v1"
    assert reference.measure_names == ["out"]
    assert reference.output_kinds == {"out": "numeric"}


def test_reviewed_direct_triton_embedding_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_embedding.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.embedding.f32.v64.d16.n128.pad0.freq.v1"
    assert reference.measure_names == ["out", "dweight"]
    assert reference.output_kinds == {"out": "numeric", "dweight": "numeric"}


def test_reviewed_direct_triton_llama_rope_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_llama_rope.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.llama_rope.f32.b2.hq4.hk2.l16.d16.tables.v1"
    assert reference.measure_names == ["q_out", "k_out", "dq", "dk"]
    assert reference.output_kinds == {"q_out": "numeric", "k_out": "numeric", "dq": "numeric", "dk": "numeric"}


def test_reviewed_direct_triton_mixtral_top2_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_mixtral_top2.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.mixtral_top2.f32.t128.e8.d32.v1"
    assert reference.measure_names == ["selected", "weights", "out"]
    assert reference.output_kinds == {"selected": "boolean", "weights": "numeric", "out": "numeric"}


def test_reviewed_direct_triton_bsr_softmax_values_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_bsr_softmax_values.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.bsr_softmax_values.f32.m128.n64.r16.c16.nnz16.v1"
    assert reference.measure_names == ["out"]
    assert reference.output_kinds == {"out": "numeric"}


def test_reviewed_direct_triton_topk_topp_keep_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_topk_topp_keep.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.topk_topp_keep.f32.b64.v64.vllm016.v1"
    assert reference.measure_names == ["keep"]
    assert reference.output_kinds == {"keep": "boolean"}


def test_reviewed_direct_triton_mamba_ssu_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_mamba_ssu.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.mamba_ssu.f32.b8.h4.d32.n8.g2.v1"
    assert reference.measure_names == ["y", "state"]
    assert reference.output_kinds == {"y": "numeric", "state": "numeric"}


def test_reviewed_direct_triton_lora_shrink_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_lora_shrink.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.lora_shrink.f16.t64.l4.r8.k32.s2.v1"
    assert reference.measure_names == ["y"]
    assert reference.output_kinds == {"y": "numeric"}


def test_reviewed_direct_triton_paged_attention_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_paged_attention.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.paged_attention.f32.t5.hq4.hkv2.d16.bs8.window4.v1"
    assert reference.measure_names == ["out"]
    assert reference.output_kinds == {"out": "numeric"}


def test_reviewed_direct_triton_pack_varlen_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_pack_varlen.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.pack_varlen.f16.tq5.tk7.hq4.hkv2.d16.causal.v1"
    assert reference.measure_names == ["out"]
    assert reference.output_kinds == {"out": "numeric"}


def test_boolean_task_comparison_counts_errors_even_when_signed_differences_cancel():
    rows = [{
        "seed": 7,
        "ok_total": np.array([True, True, True, True]),
        "k": np.array([0.0, 1.0, 1.0, 0.0]),
        "f_lo": np.array([0.0, 0.0, 1.0, 1.0]),
        "f_hi": np.array([0.0, 0.0, 1.0, 1.0]),
    }]
    result = _boolean_task_comparison(rows)
    assert result["matches"] == 2
    assert result["mismatches"] == 2
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["signed_truth_rate_difference"] == 0.0


def test_reviewed_direct_triton_mxfp8_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/triton_mxfp8_e5m2_floor.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "triton.mxfp8.e5m2.floor.f32.4x64.v1"
    assert reference.measure_names == ["elements", "scale_codes"]
    assert reference.output_kinds == {"elements": "numeric", "scale_codes": "numeric"}


def test_reviewed_torchao_state4bit_copy_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/torchao_optim_state4bit_copy.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "torchao.optim_state4bit.copy.f32.signed.4096x128.v1"
    assert reference.measure_names == ["codes", "scale"]
    assert reference.output_kinds == {"codes": "numeric", "scale": "numeric"}


def test_reviewed_inductor_layer_norm_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/inductor_layer_norm.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "pytorch.inductor.layer_norm.f32.b16.d128.autograd.v1"
    assert reference.measure_names == ["y", "dx", "dweight", "dbias"]
    assert reference.output_kinds == {
        "y": "numeric", "dx": "numeric", "dweight": "numeric", "dbias": "numeric"
    }


def test_reviewed_inductor_rms_norm_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/inductor_rms_norm.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "pytorch.inductor.rms_norm.f32.b16.d128.autograd.v1"
    assert reference.measure_names == ["y", "dx", "dweight"]
    assert reference.output_kinds == {"y": "numeric", "dx": "numeric", "dweight": "numeric"}


def test_reviewed_inductor_group_norm_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/inductor_group_norm.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "pytorch.inductor.group_norm.f32.n4.c16.h8.w8.g4.autograd.v1"
    assert reference.measure_names == ["y", "dx", "dweight", "dbias"]
    assert reference.output_kinds == {
        "y": "numeric", "dx": "numeric", "dweight": "numeric", "dbias": "numeric"
    }


def test_reviewed_inductor_gelu_erf_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/inductor_gelu_erf.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "pytorch.inductor.gelu_erf.f32.r32.c128.autograd.v1"
    assert reference.measure_names == ["y", "dx"]
    assert reference.output_kinds == {"y": "numeric", "dx": "numeric"}


@pytest.mark.parametrize("name,binding_id", [
    ("relu", "pytorch.inductor.relu.f32.r32.c128.autograd.v1"),
    ("tanh", "pytorch.inductor.tanh.f32.r32.c128.autograd.v1"),
    ("sigmoid", "pytorch.inductor.sigmoid.f32.r32.c128.autograd.v1"),
])
def test_reviewed_inductor_common_activation_bindings_are_admitted_before_measurement(name, binding_id):
    declaration_path = SPEC / f"corpus/measure_cases/inductor_{name}.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == binding_id
    assert reference.measure_names == ["y", "dx"]
    assert reference.output_kinds == {"y": "numeric", "dx": "numeric"}


@pytest.mark.parametrize("loss_name,binding_id", [
    ("mse", "pytorch.inductor.mse_loss_mean.f32.r32.c128.autograd.v1"),
    ("l1", "pytorch.inductor.l1_loss_mean.f32.r32.c128.autograd.v1"),
])
def test_reviewed_inductor_regression_loss_bindings_are_admitted_before_measurement(loss_name, binding_id):
    declaration_path = SPEC / f"corpus/measure_cases/inductor_{loss_name}_loss.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == binding_id
    assert reference.measure_names == ["loss", "dx", "dtarget"]
    assert reference.output_kinds == {
        "loss": "numeric", "dx": "numeric", "dtarget": "numeric"
    }


def test_reviewed_inductor_softmax_backward_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/inductor_softmax_backward.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "pytorch.inductor.softmax_backward.f32.r16.c128.lastdim.v1"
    assert reference.measure_names == ["dx"]
    assert reference.output_kinds == {"dx": "numeric"}


def test_reviewed_inductor_log_softmax_backward_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/inductor_log_softmax_backward.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "pytorch.inductor.log_softmax_backward.f32.r16.c128.lastdim.v1"
    assert reference.measure_names == ["dx"]
    assert reference.output_kinds == {"dx": "numeric"}


def test_reviewed_inductor_logsumexp_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/inductor_logsumexp.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "pytorch.inductor.logsumexp.f32.r16.c128.autograd.v1"
    assert reference.measure_names == ["y", "dx"]
    assert reference.output_kinds == {"y": "numeric", "dx": "numeric"}


def test_reviewed_inductor_gather_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/inductor_gather.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "pytorch.inductor.gather.f32.r32.c64.k24.dim1.v1"
    assert reference.measure_names == ["out"]
    assert reference.output_kinds == {"out": "numeric"}


def test_reviewed_inductor_cumsum_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/inductor_cumsum.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "pytorch.inductor.cumsum.f32.r16.c128.dim1.autograd.v1"
    assert reference.measure_names == ["y", "dx"]
    assert reference.output_kinds == {"y": "numeric", "dx": "numeric"}


def test_reviewed_inductor_var_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/inductor_var.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "pytorch.inductor.var.f32.r16.c128.dim1.correction1.autograd.v1"
    assert reference.measure_names == ["y", "dx"]
    assert reference.output_kinds == {"y": "numeric", "dx": "numeric"}


def test_reviewed_inductor_std_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/inductor_std.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == "pytorch.inductor.std.f32.r16.c128.dim1.correction1.autograd.v1"
    assert reference.measure_names == ["y", "dx"]
    assert reference.output_kinds == {"y": "numeric", "dx": "numeric"}


def test_reviewed_inductor_var_mean_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/inductor_var_mean.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == \
        "pytorch.inductor.var_mean.f32.r16.c128.dim1.correction1.autograd.v1"
    assert reference.measure_names == ["variance", "mean", "dx"]
    assert reference.output_kinds == {
        "variance": "numeric", "mean": "numeric", "dx": "numeric"
    }


def test_reviewed_inductor_vector_norm_binding_is_admitted_before_measurement():
    declaration_path = SPEC / "corpus/measure_cases/inductor_vector_norm.json"
    declaration = json.loads(declaration_path.read_text())
    compare = declaration["compare"]
    reference = ContractReference(
        plan_path=compare["binding_plan"],
        specification_root=compare["specification_root"],
        call_ref=declaration["call"],
        base_dir=declaration_path.parent,
        measure_names=compare["measure"],
        backend=binding_backend,
    )
    assert reference.id == \
        "pytorch.inductor.linalg_vector_norm.f32.r16.c128.dim1.ord2.autograd.v1"
    assert reference.measure_names == ["y", "dx"]
    assert reference.output_kinds == {"y": "numeric", "dx": "numeric"}


def test_admitted_special_value_class_mismatch_is_a_pointwise_contradiction():
    verdict = measure.task_special_value_verdict({
        "elements_with_special_f": 0,
        "k_vs_f_class_mismatch": 3,
        "unbounded_reference_elements": 0,
    }, admitted_contract=True)
    assert verdict["judgment"] == "contradiction"
    assert verdict["mismatched_coordinates"] == 3
    refused = measure.task_special_value_verdict({"k_vs_f_class_mismatch": 3}, admitted_contract=False)
    assert refused["judgment"] == "not established"


@CUDA
def test_measure_uses_the_admitted_contract_reference_end_to_end(tmp_path, monkeypatch):
    call = tmp_path / "triton_ce.py"
    call.write_text('''
import torch
import triton
import triton.language as tl

@triton.jit
def ce_kernel(logits, target, loss, gradient):
    x0 = tl.load(logits)
    x1 = tl.load(logits + 1)
    m = tl.maximum(x0, x1)
    e0 = tl.exp(x0 - m)
    e1 = tl.exp(x1 - m)
    p0 = e0 / (e0 + e1)
    p1 = e1 / (e0 + e1)
    t = tl.load(target)
    # Deliberate task bias: the admitted CE contract has no +0.25 term.
    tl.store(loss, tl.where(t == 0, -tl.log(p0), -tl.log(p1)) + 0.25)
    # Leave the public gradient at a deliberately wrong non-Triton initialization.  K-f must still cover the complete
    # public output even though no K-G_pi decomposition is available for this output.

def make_target(seed, shape, dtype):
    return torch.tensor([seed % 2], dtype=torch.int64, device="cuda")

def call(inputs):
    loss = torch.empty((), dtype=torch.float32, device="cuda")
    gradient = torch.full_like(inputs["logits"], 3.0)
    ce_kernel[(1,)](inputs["logits"], inputs["target"], loss, gradient)
    return {"loss": loss, "gradient": gradient}
'''.lstrip())
    call_ref = f"{call}:call"
    plan = {
        "schema": "triton-call-binding-v1", "id": "test.ce.measure.gpu.v1", "status": "reviewed",
        "subject": {"library": "test", "version": "1", "callable": "triton_ce.call",
                    "runtime_package": {"distribution": "torch",
                                        "version": importlib.metadata.version("torch")},
                    "boundary": "one Triton forward call", "measure_call": call_ref},
        "launch_scope": "all_triton_launches_within_call",
        "tasks": [{
            "id": "ce", "contract": "CE", "profile": "index_unsmoothed",
            "task_statement": "One-row, two-class, unsmoothed, summed class-index cross entropy and its VJP.",
            "input_map": [
                {"target": "/logits", "source": "/tensor_arguments/logits/before",
                 "conversion": "tensor_values"},
                {"target": "/target", "source": "/tensor_arguments/target/before",
                 "conversion": "tensor_values"},
                {"target": "/weights", "source": "/scalars/weights"},
                {"target": "/eps", "source": "/scalars/eps"},
                {"target": "/ignore_index", "source": "/scalars/ignore_index"},
                {"target": "/reduction", "source": "/scalars/reduction"},
                {"target": "/upstream", "source": "/scalars/upstream"},
            ],
            "output_map": [
                {"target": "/loss", "source": "/returns/loss", "measure_output": "loss"},
                {"target": "/gradient", "source": "/returns/gradient", "measure_output": "gradient",
                 "conversion": "tensor_values"},
            ],
        }],
        "state_map": [{"argument": "logits", "effect": "unchanged"},
                      {"argument": "target", "effect": "unchanged"}],
    }
    plan_path = tmp_path / "binding.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    reviewed = tmp_path / "reviewed"
    for directory in ("bindings", "calls", "contracts", "adapters"):
        (reviewed / directory).mkdir(parents=True, exist_ok=True)
    (reviewed / "bindings" / f"{plan['id']}.json").write_bytes(plan_path.read_bytes())
    (reviewed / "calls" / f"{plan['id']}.py").write_bytes(call.read_bytes())
    shutil.copyfile(SPEC / "contracts/CE.yaml", reviewed / "contracts/CE.yaml")
    shutil.copyfile(SPEC / "adapters/CE.py", reviewed / "adapters/CE.py")
    monkeypatch.setattr(binding_backend.registry, "VALIDATED", reviewed)
    real_reference = contract_binding.ContractReference

    def reference_for_test(**kwargs):
        return real_reference(**kwargs, backend=binding_backend, reviewed_root=reviewed)

    monkeypatch.setattr(contract_binding, "ContractReference", reference_for_test)
    declaration = {
        "call": call_ref,
        "inputs": {
            "logits": {"sampler": {"normal": [0, 1]}, "shape": [1, 2], "dtype": "float32"},
            "target": {"state": f"{call}:make_target", "shape": [1], "dtype": "int64"},
            "weights": {"const": None}, "eps": {"const": 0}, "ignore_index": {"const": -100},
            "reduction": {"const": "sum"}, "upstream": {"const": 1},
        },
        "compare": {"mode": "contract", "measure": ["loss", "gradient"],
                    "binding_plan": str(plan_path), "specification_root": str(SPEC)},
        "budget": {"cpu_seconds": 120, "gpu_seconds": 120, "case_timeout": 60, "max_units": 24},
        "units": {"development": 8, "confirmation": 16}, "_base_dir": str(tmp_path),
    }
    level = measure.run(declaration)["levels"][0]
    assert level["status"] == "ok", level
    assert level["reference_kind"] == "admitted_task_specification"
    assert level["semantic_reason"].startswith("admitted task specification")
    assert all(row["reference_usable_for_bias_measurement"] for row in level["contract_bindings"])
    assert level["outputs"]["loss"]["semantic"]["verdict_scope"] == \
        "G_pi_minus_admitted_task_specification"
    assert level["outputs"]["loss"]["total"]["verdict_scope"] == \
        "candidate_minus_admitted_task_specification"
    assert "statistics" in level["outputs"]["loss"]["total"]
    # Binding correctness does not require a t-test verdict: distribution
    # diagnostics can withhold inference on almost constant rounding residuals.
    # Check the planted task difference and complete observations directly.
    assert level["outputs"]["loss"]["semantic"]["reference_difference_excludes_zero"] == 24
    assert level["outputs"]["loss"]["total"]["covers_complete_public_output"] is True
    assert level["outputs"]["loss"]["total"]["scale"]["mean_abs_residual"] == pytest.approx(.25, abs=1e-5)
    assert level["outputs"]["gradient"]["total"]["covers_complete_public_output"] is True
    assert level["outputs"]["gradient"]["reference"]["reference_scope"].startswith("not established")
    assert level["outputs"]["gradient"]["total"]["elements_per_seed"] == 2
    assert "semantic" not in level["outputs"]["gradient"]
    assert level["outputs"]["gradient"]["total"]["scale"]["mean_abs_residual"] == pytest.approx(3., abs=1e-12)


def _unmeasured_position_plan(plan_path, reviewed, basis):
    plan = json.loads(plan_path.read_text())
    entry = {"target": "/gradient", "literal": None}
    if basis is not None:
        entry["basis"] = basis
    plan["tasks"][0]["output_map"][1] = entry
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    (reviewed / "bindings" / f"{plan['id']}.json").write_bytes(plan_path.read_bytes())


def test_declared_unmeasured_position_is_admitted_without_measure_output(tmp_path, monkeypatch):
    """Path A2 rule v2: a return position the task does not define (or the call does not return) is mapped as
    literal null with a basis; it is not measured, and every other output still needs its measure_output."""
    _, call_ref, plan_path, reviewed = _fixture(tmp_path, monkeypatch)
    _unmeasured_position_plan(plan_path, reviewed, "return 1 is not defined by the task (test)")
    reference = ContractReference(plan_path=plan_path, specification_root=SPEC, call_ref=call_ref,
                                  base_dir=tmp_path, measure_names=["loss"],
                                  backend=binding_backend, reviewed_root=reviewed)
    assert list(reference._measure) == ["loss"]


def test_unmeasured_position_needs_a_basis(tmp_path, monkeypatch):
    _, call_ref, plan_path, reviewed = _fixture(tmp_path, monkeypatch)
    _unmeasured_position_plan(plan_path, reviewed, None)
    with pytest.raises(ContractBindingError, match="measure_output"):
        ContractReference(plan_path=plan_path, specification_root=SPEC, call_ref=call_ref,
                          base_dir=tmp_path, measure_names=["loss"],
                          backend=binding_backend, reviewed_root=reviewed)
