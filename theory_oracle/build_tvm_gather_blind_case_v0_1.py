"""Build the opaque input/reference package for the Gather case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from tvm_gather_negative_case_v0_1 import inputs, make_model


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model, values = make_model(), inputs()
    model_path, input_path, control_path, ref_path = out / "model.onnx", out / "input.npz", out / "positive_control.npz", out / "reference.npy"
    model_path.write_bytes(model.SerializeToString())
    np.savez(input_path, **values)
    np.savez(control_path, X=values["X"], I=np.asarray([0, 2], dtype=np.int64))
    reference = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, values)[0]
    np.save(ref_path, reference)
    manifest = {
        "schema_version": "forkcert.blind_case_package.v0.1",
        "case_id": "case_004",
        "visibility": "patch_free_opaque_case",
        "contract": {"reference_role": "declared_external_onnx_runtime_semantics", "endpoint": "exact output tensor relation", "compiled_output_must_match_reference": True},
        "model": {"path": str(model_path), "sha256": sha(model_path)},
        "input": {"path": str(input_path), "sha256": sha(input_path)},
        "positive_control": {"path": str(control_path), "sha256": sha(control_path)},
        "reference": {"path": str(ref_path), "sha256": sha(ref_path), "shape": list(reference.shape), "dtype": str(reference.dtype)},
        "locator_exclusions": ["issue identifier", "fixed revision", "patch", "pull-request discussion", "root-cause notes"],
    }
    (out / "case_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
