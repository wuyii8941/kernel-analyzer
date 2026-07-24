"""A small longitudinal slice for TVM's ONNX ScatterElements reduction case.

This is deliberately an evidence collector, not a bug finder.  It runs the
same ONNX witness through the reference (ONNX Runtime) and one TVM checkout,
records the Relax boundary produced by the frontend, and then runs a narrowly
defined IR repair in which only the ``reduction`` attribute is changed from
the value observed at the boundary to the specification value (``add``).

The TVM fix commit was read while screening this candidate.  Consequently the
report is labelled PATCH_EXCLUDED_REPLAY rather than a blind historical-bug
score.  The repair is useful for checking the pipeline's evidence contract,
but it is not a proof that this is the compiler's unique root cause.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_record(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": _sha(value.tobytes(order="C")),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def make_model() -> onnx.ModelProto:
    # Adding to the existing data is the important semantic witness.  There
    # are no duplicate indices needed: overwrite and reduction=add already
    # differ at every addressed element.
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 4])
    indices = helper.make_tensor_value_info("idx", TensorProto.INT64, [2, 2])
    updates = helper.make_tensor_value_info("updates", TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 4])
    node = helper.make_node(
        "ScatterElements",
        ["X", "idx", "updates"],
        ["Y"],
        axis=0,
        reduction="add",
    )
    graph = helper.make_graph([node], "scatter_reduction_witness", [x, indices, updates], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def inputs() -> dict[str, np.ndarray]:
    return {
        "X": np.ones((4, 4), dtype=np.float32),
        "idx": np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        "updates": np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
    }


def build_relax_repair(relax, tensors):
    x = relax.Var("X", relax.TensorStructInfo((4, 4), "float32"))
    idx = relax.Var("idx", relax.TensorStructInfo((2, 2), "int64"))
    updates = relax.Var("updates", relax.TensorStructInfo((2, 2), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, idx, updates]):
        with bb.dataflow():
            out = bb.emit(relax.op.scatter_elements(x, idx, updates, axis=0, reduction="add"))
            bb.emit_output(out)
        bb.emit_func_output(out)
    return bb.get()


def run_vm(tvm, relax, mod, values):
    exe = relax.build(mod, target="llvm")
    vm = relax.VirtualMachine(exe, device=tvm.cpu())
    args = [tvm.runtime.tensor(values[name], device=tvm.cpu()) for name in ("X", "idx", "updates")]
    return vm["main"](*args).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tvm-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--case-id", default="tvm_scatter_reduction_v0_1")
    parser.add_argument("--case-dir", default=None)
    args = parser.parse_args()

    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    if args.case_dir:
        case_dir = pathlib.Path(args.case_dir)
        model = onnx.load(case_dir / "model.onnx")
        with np.load(case_dir / "input.npz") as packed:
            values = {name: packed[name] for name in packed.files}
    else:
        model = make_model()
        values = inputs()
    reference = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(
        None, values
    )[0]
    converted = from_onnx(model)
    converted_text = str(converted)
    legalized = relax.transform.LegalizeOps()(converted)
    legalized_text = str(legalized)
    compiled = run_vm(tvm, relax, converted, values)

    # This is the local IR intervention.  It does not claim that the full
    # compiler context is invariant: rebuilding the repaired IR can change
    # generated code.  The report therefore remains intervention-dependent.
    repaired = build_relax_repair(relax, values)
    repaired_text = str(repaired)
    repaired_out = run_vm(tvm, relax, repaired, values)

    def comparison(output: np.ndarray) -> dict[str, Any]:
        delta = np.asarray(output, dtype=np.float64) - reference.astype(np.float64)
        return {
            "output": _array_record(output),
            "max_abs_vs_reference": float(np.max(np.abs(delta))),
            "exact_vs_reference": bool(np.array_equal(output, reference)),
            "allclose_vs_reference": bool(np.allclose(output, reference, rtol=0.0, atol=0.0)),
        }

    # The frontend boundary itself is the candidate production site: a
    # specification-preserving Relax op should retain reduction="add".
    has_expected_attr = 'reduction="add"' in converted_text
    has_observed_update = 'reduction="update"' in converted_text
    report = {
        "schema": "tvm_scatter_reduction_case_v0_1",
        "case_id": args.case_id,
        "blind_status": "PATCH_EXCLUDED_REPLAY",
        "claim_limit": "IR_BOUNDARY_PRODUCTION_PLUS_INTERVENTION_DEPENDENT_ATTRIBUTION",
        "tvm_root": str(pathlib.Path(args.tvm_root).resolve()),
        "tvm_version": getattr(tvm, "__version__", "unknown"),
        "python": platform.python_version(),
        "target": "llvm",
        "semantic_spec": "ONNX Runtime ScatterElements opset 18, reduction=add",
        "input": {name: _array_record(value) for name, value in values.items()},
        "reference": _array_record(reference),
        "compiled": comparison(compiled),
        "repaired": comparison(repaired_out),
        "production": {
            "boundary": "ONNX frontend -> Relax",
            "expected_relax_reduction": "add",
            "observed_relax_reduction": "update" if has_observed_update else "unknown",
            "spec_attribute_present": has_expected_attr,
            "local_semantic_discrepancy": bool(has_observed_update and not has_expected_attr),
            "evidence": "converted Relax IR text; not same-input numeric replay",
        },
        "provenance": {
            "onnx_node": "ScatterElements(axis=0,reduction=add)",
            "relax_op": "relax.scatter_elements",
            "compiled_region": "Relax IR -> LegalizeOps/TIR -> build(llvm)",
            "kernel_identity": "not captured in this CPU slice",
            "relax_to_tir_observed": "@T.prim_func" in legalized_text and "scatter_elements" in legalized_text,
            "complete": False,
        },
        "intervention": {
            "type": "IR_repair_rebuild",
            "changed_field": "Relax scatter_elements reduction attribute",
            "non_target_context_invariant": False,
            "context_limitations": ["repaired IR is rebuilt", "kernel identity not captured"],
        },
        "ir": {
            "converted_sha256": _sha(converted_text.encode()),
            "legalized_sha256": _sha(legalized_text.encode()),
            "repaired_sha256": _sha(repaired_text.encode()),
            "converted_text": converted_text,
            "legalized_text": legalized_text,
            "repaired_text": repaired_text,
        },
        "tir_mechanism_signal": {
            "buggy_store_uses_update_directly": "] = updates[" in legalized_text,
            "fixed_store_adds_existing_output": "+ updates[" in legalized_text,
            "interpretation": "structural signal only; not a compiler source-line proof",
        },
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(out),
        "tvm_version": report["tvm_version"],
        "compiled_max_abs": report["compiled"]["max_abs_vs_reference"],
        "repaired_max_abs": report["repaired"]["max_abs_vs_reference"],
        "local_semantic_discrepancy": report["production"]["local_semantic_discrepancy"],
        "claim_limit": report["claim_limit"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
