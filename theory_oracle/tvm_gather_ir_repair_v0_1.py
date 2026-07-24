"""Post-reveal IR repair for the Gather negative-index certificate.

This is intentionally separate from the pre-reveal locator.  It demonstrates
the repair/endpoint part of the evidence contract but does not claim that a
rebuild preserves non-target compiler context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def fp(a: np.ndarray) -> dict:
    return {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tvm-root", required=True)
    p.add_argument("--case-dir", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    import tvm
    from tvm import relax

    case = Path(args.case_dir)
    with np.load(case / "input.npz") as packed:
        values = {key: packed[key] for key in packed.files}
    reference = np.load(args.reference)
    x = relax.Var("X", relax.TensorStructInfo((3, 4), "float32"))
    idx = relax.Var("I", relax.TensorStructInfo((2,), "int64"))
    # Candidate/raw region: exactly the old Relax take, compiled in the same
    # checkout and fed the same boundary tensors.
    raw_bb = relax.BlockBuilder()
    with raw_bb.function("main", [x, idx]):
        with raw_bb.dataflow():
            raw_out = raw_bb.emit(relax.op.take(x, idx, axis=0, mode="fast"))
            raw_bb.emit_output(raw_out)
        raw_bb.emit_func_output(raw_out)
    raw_mod = raw_bb.get()
    raw_exe = relax.build(relax.transform.LegalizeOps()(raw_mod), target="llvm")
    raw_vm = relax.VirtualMachine(raw_exe, device=tvm.cpu())
    raw_output = raw_vm["main"](*(tvm.runtime.tensor(values[k], device=tvm.cpu()) for k in ("X", "I"))).numpy()

    # Specification-preserving repair region.
    bb = relax.BlockBuilder()
    # Variables are bound to a separate builder, so recreate them.
    x = relax.Var("X", relax.TensorStructInfo((3, 4), "float32"))
    idx = relax.Var("I", relax.TensorStructInfo((2,), "int64"))
    with bb.function("main", [x, idx]):
        with bb.dataflow():
            shape = bb.emit(relax.op.shape_to_tensor(relax.op.shape_of(x)))
            dim0 = bb.emit(relax.op.take(shape, relax.const(0, "int64"), axis=0, mode="wrap"))
            negative = bb.emit(relax.op.less(idx, relax.const(0, "int64")))
            shifted = bb.emit(relax.op.add(idx, dim0))
            normalized = bb.emit(relax.op.where(negative, shifted, idx))
            out = bb.emit(relax.op.take(x, normalized, axis=0, mode="fast"))
            bb.emit_output(out)
        bb.emit_func_output(out)
    mod = bb.get()
    legalized = relax.transform.LegalizeOps()(mod)
    exe = relax.build(legalized, target="llvm")
    vm = relax.VirtualMachine(exe, device=tvm.cpu())
    output = vm["main"](*(tvm.runtime.tensor(values[k], device=tvm.cpu()) for k in ("X", "I"))).numpy()
    report = {
        "schema": "tvm_gather_ir_repair_v0_1",
        "case_id": "case_004",
        "intervention": "insert_negative_index_normalization_at_relax_boundary",
        "tvm_root": str(Path(args.tvm_root).resolve()),
        "output": fp(output),
        "raw_output": fp(raw_output),
        "reference": fp(reference),
        "exact_vs_reference": bool(np.array_equal(output, reference)),
        "same_input_local_replay": {
            "boundary_inputs_identical": True,
            "raw_vs_repaired_different": bool(not np.array_equal(raw_output, output)),
            "raw_max_abs_vs_repaired": float(np.max(np.abs(raw_output.astype(np.float64) - output.astype(np.float64)))),
            "interpretation": "IR-level local discrepancy production; not a compiler root-cause proof",
        },
        "ir": {"repaired_relax": str(mod), "repaired_tir": str(legalized)},
        "non_target_context_invariant": False,
        "claim_limit": "INTERVENTION_DEPENDENT_ATTRIBUTION",
        "limitations": ["repair rebuilds Relax/TIR", "kernel identity not captured", "not a root-cause proof"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "exact": report["exact_vs_reference"], "claim_limit": report["claim_limit"]}, sort_keys=True))


if __name__ == "__main__":
    main()
