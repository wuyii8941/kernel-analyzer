"""Create the patch-free input package for the TVM ScatterElements slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tvm_scatter_reduction_case_v0_1 import inputs, make_model


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    import onnxruntime as ort

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.onnx"
    input_path = out / "input.npz"
    reference_path = out / "reference.npy"
    model = make_model()
    model_path.write_bytes(model.SerializeToString())
    values = inputs()
    np.savez(input_path, **values)
    reference = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, values)[0]
    np.save(reference_path, reference)
    manifest = {
        "schema_version": "forkcert.blind_case_package.v0.1",
        "case_id": "case_003",
        "visibility": "patch_excluded_opaque_case",
        "contract": {
            "reference_role": "declared_external_semantic_reference",
            "endpoint": "exact output tensor relation",
            "compiled_output_must_match_reference": True,
        },
        "input": {"path": str(input_path), "sha256": sha(input_path)},
        "model": {"path": str(model_path), "sha256": sha(model_path)},
        "reference": {"path": str(reference_path), "sha256": sha(reference_path), "shape": list(reference.shape), "dtype": str(reference.dtype)},
        "locator_exclusions": ["issue identifier", "fixed revision", "patch", "pull-request discussion", "root-cause notes"],
        "status_note": "The external fix was read during candidate screening; this package is not a blind benchmark.",
    }
    (out / "case_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
