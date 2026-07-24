"""Patch-excluded replay slice for ONNX Gather negative-index semantics.

The locator consumes only the ONNX model, inputs, and ONNX Runtime output.  It
records the Relax/TIR artifacts produced by the selected TVM checkout; it does
not import or inspect the external fix.  This case is intentionally an
observation/provenance slice before any repair is attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def make_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
    i = helper.make_tensor_value_info("I", TensorProto.INT64, [2])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    node = helper.make_node("Gather", ["X", "I"], ["Y"], axis=0)
    graph = helper.make_graph([node], "gather_negative_witness", [x, i], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def inputs() -> dict[str, np.ndarray]:
    return {
        "X": np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32),
        "I": np.asarray([0, -1], dtype=np.int64),
    }


def run(tvm, relax, mod, values):
    exe = relax.build(relax.transform.LegalizeOps()(mod), target="llvm")
    vm = relax.VirtualMachine(exe, device=tvm.cpu())
    return vm["main"](*(tvm.runtime.tensor(values[k], device=tvm.cpu()) for k in ("X", "I"))).numpy()


def rec(a: np.ndarray) -> dict[str, Any]:
    return {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": digest(a.tobytes()), "min": float(a.min()), "max": float(a.max())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tvm-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--case-dir", default=None)
    p.add_argument("--case-id", default="tvm_gather_negative_index_v0_1")
    args = p.parse_args()

    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    if args.case_dir:
        case = pathlib.Path(args.case_dir)
        model = onnx.load(case / "model.onnx")
        with np.load(case / "input.npz") as packed:
            values = {key: packed[key] for key in packed.files}
        with np.load(case / "positive_control.npz") as packed:
            positive_values = {key: packed[key] for key in packed.files}
    else:
        model, values = make_model(), inputs()
        positive_values = {"X": values["X"], "I": np.asarray([0, 2], dtype=np.int64)}
    reference = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, values)[0]
    converted = from_onnx(model)
    converted_text = str(converted)
    legalized = relax.transform.LegalizeOps()(converted)
    legalized_text = str(legalized)
    output = run(tvm, relax, converted, values)
    positive_reference = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, positive_values)[0]
    positive_output = run(tvm, relax, converted, positive_values)
    delta = output.astype(np.float64) - reference.astype(np.float64)
    report = {
        "schema": "tvm_gather_negative_case_v0_1",
        "case_id": args.case_id,
        "blind_status": "FROZEN_BLIND_LOCATOR_INPUT",
        "tvm_root": str(pathlib.Path(args.tvm_root).resolve()),
        "tvm_version": getattr(tvm, "__version__", "unknown"),
        "target": "llvm",
        "semantic_spec": "ONNX Runtime Gather opset 13, negative indices count from the end",
        "input": {key: rec(value) for key, value in values.items()},
        "reference": rec(reference),
        "compiled": {
            "output": rec(output),
            "exact_vs_reference": bool(np.array_equal(output, reference)),
            "max_abs_vs_reference": float(np.max(np.abs(delta))),
        },
        "positive_control": {
            "input": {key: rec(value) for key, value in positive_values.items()},
            "compiled": rec(positive_output),
            "reference": rec(positive_reference),
            "exact_vs_reference": bool(np.array_equal(positive_output, positive_reference)),
        },
        "provenance": {
            "onnx_node": "Gather(axis=0)",
            "relax_text_sha256": digest(converted_text.encode()),
            "legalized_text_sha256": digest(legalized_text.encode()),
            "relax_text": converted_text,
            "legalized_text": legalized_text,
            "kernel_identity": "not captured in this CPU slice",
            "complete": False,
        },
        "controls": {"negative_index_witness": True, "positive_index_control": "not in this single-input run"},
        "claim_limit": "OBSERVATION_WITH_RELAX_TIR_PROVENANCE_ONLY",
        "limitations": [
            "same-input numeric local replay is not instantiated for frontend conversion",
            "no repair or mediation is claimed",
            "kernel identity and non-target context are not captured",
            "external fix is excluded from locator input",
        ],
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "exact": report["compiled"]["exact_vs_reference"], "max_abs": report["compiled"]["max_abs_vs_reference"]}, sort_keys=True))


if __name__ == "__main__":
    main()
