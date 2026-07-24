"""Probe every automatically mappable TIR region in one TVM checkout.

This process deliberately does not know the operator under test.  It maps
buffer signatures to case inputs, requires exactly one otherwise-unmapped
output buffer, and fails closed for ambiguity.  The paired buggy/fixed
comparison is performed by ``compare_generic_tvm_region_probe_v0_1.py`` in
separate processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from forkcert.relational_oracle import compute_repeatability


def _shape(buffer: Any) -> tuple[int, ...] | None:
    try:
        return tuple(int(dim) for dim in buffer.shape)
    except (TypeError, ValueError):
        return None


def _load_case(case_dir: Path):
    import onnx

    model = onnx.load(case_dir / "model.onnx")
    with np.load(case_dir / "input.npz") as packed:
        values = {name: packed[name] for name in packed.files}
    return model, values


def _fingerprint(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value)
    finite = bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else True
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        "finite": finite,
    }


def _input_matches(buffer: Any, values: dict[str, np.ndarray]) -> list[str]:
    shape = _shape(buffer)
    if shape is None:
        return []
    return [name for name, value in values.items() if tuple(value.shape) == shape and str(value.dtype) == str(buffer.dtype)]


def _probe(tvm, relax, name: str, gv: Any, prim_func: Any, values: dict[str, np.ndarray]):
    buffers = list(prim_func.buffer_map.values())
    assignments: dict[str, str] = {}
    for buffer in buffers:
        matches = _input_matches(buffer, values)
        if len(matches) == 1:
            assignments[str(buffer.data.name)] = matches[0]
        elif len(matches) > 1:
            raise ValueError(f"ambiguous input mapping for {buffer.data}: {matches}")
    outputs = [buffer for buffer in buffers if str(buffer.data.name) not in assignments]
    if len(outputs) != 1 or len(assignments) != len(buffers) - 1:
        raise ValueError("requires one uniquely identified output and all other buffers mapped")
    output_shape = _shape(outputs[0])
    if output_shape is None:
        raise ValueError("symbolic output shape")
    input_names = list(assignments.values())
    params = [relax.Var(name, relax.TensorStructInfo(tuple(values[name].shape), str(values[name].dtype))) for name in input_names]
    bb = relax.BlockBuilder()
    with bb.function("probe", params):
        with bb.dataflow():
            out = bb.emit(relax.call_tir(gv, tuple(params), out_sinfo=relax.TensorStructInfo(output_shape, str(outputs[0].dtype))))
            bb.emit_output(out)
        bb.emit_func_output(out)
    mod = bb.get()
    mod[gv] = prim_func
    exe = relax.build(mod, target="llvm")
    vm = relax.VirtualMachine(exe, device=tvm.cpu())
    args = [tvm.runtime.tensor(values[name], device=tvm.cpu()) for name in input_names]
    outputs_np = [vm["probe"](*args).numpy() for _ in range(2)]
    return outputs_np, {
        "input_names": input_names,
        "input_hashes": {name: _fingerprint(values[name])["sha256"] for name in input_names},
        "output_shape": list(output_shape),
        "output_dtype": str(outputs[0].dtype),
        "buffer_count": len(buffers),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tvm-root", required=True)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    case_dir = Path(args.case_dir).resolve()
    model, values = _load_case(case_dir)
    legalized = relax.transform.LegalizeOps()(from_onnx(model))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for global_var, func in legalized.functions.items():
        if not hasattr(func, "buffer_map"):
            continue
        name = str(global_var.name_hint)
        try:
            outputs, contract = _probe(tvm, relax, name, global_var, func, values)
            stem = re.sub(r"[^A-Za-z0-9_.-]", "_", out.stem + "." + name)
            artifact = out.parent / (stem + ".npy")
            np.save(artifact, outputs[0])
            rows.append({
                "region_id": f"tir::{name}",
                "status": "REPLAYED",
                "contract": contract,
                "outputs": [_fingerprint(value) for value in outputs],
                "repeat_exact": bool(np.array_equal(outputs[0], outputs[1])),
                "repeatability": compute_repeatability(outputs),
                "artifact": str(artifact),
            })
        except (TypeError, ValueError, RuntimeError) as error:
            rows.append({"region_id": f"tir::{name}", "status": "UNINSTANTIATED", "reason": str(error)})
    report = {
        "schema_version": "forkcert.generic-tvm-region-probe.v0.1",
        "case_id": json.loads((case_dir / "case_manifest.json").read_text())["case_id"],
        "tvm_root": str(Path(args.tvm_root).resolve()),
        "input_artifact_sha256": hashlib.sha256((case_dir / "input.npz").read_bytes()).hexdigest(),
        "bug_specific_region_input": False,
        "bug_specific_repair_input": False,
        "regions": rows,
        "claim": "REGION_REPLAY_OR_UNINSTANTIATED",
        "limitations": [
            "isolated TIR replay is not a fixed-suffix mediation experiment",
            "shape/dtype-only mapping fails closed when signatures are ambiguous",
            "a replayed region is not automatically a source or root cause",
        ],
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "replayed": [row["region_id"] for row in rows if row["status"] == "REPLAYED"]}, sort_keys=True))


if __name__ == "__main__":
    main()
