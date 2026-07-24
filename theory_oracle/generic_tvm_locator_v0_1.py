"""Bug-agnostic TVM locator for opaque ONNX case packages.

The locator accepts no operator name, region name, trigger condition, or repair
recipe.  It performs only generic work: reference/candidate endpoint
measurement, repeatability, IR-stage snapshots, and reachable TIR-region
inventory.  Any semantic interpretation belongs to a separate post-reveal
evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from typing import Any

import numpy as np
import onnx

from forkcert.relational_oracle import compute_endpoint_oracle, compute_repeatability


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _load_case(case_dir: pathlib.Path):
    model = onnx.load(case_dir / "model.onnx")
    with np.load(case_dir / "input.npz") as packed:
        values = {key: packed[key] for key in packed.files}
    reference = np.load(case_dir / "reference.npy")
    manifest = json.loads((case_dir / "case_manifest.json").read_text())
    return model, values, reference, manifest


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return value.numpy()
    if isinstance(value, (tuple, list)) and len(value) == 1:
        return _as_numpy(value[0])
    return np.asarray(value)


def _run(tvm, relax, mod, values: dict[str, np.ndarray], input_names: list[str]) -> np.ndarray:
    executable = relax.build(relax.transform.LegalizeOps()(mod), target="llvm")
    vm = relax.VirtualMachine(executable, device=tvm.cpu())
    args = [tvm.runtime.tensor(values[name], device=tvm.cpu()) for name in input_names]
    return _as_numpy(vm["main"](*args))


def _inventory(frontend_text: str, legalized_text: str) -> dict[str, Any]:
    """Extract generic textual IR relations; no operator names are assumed."""

    relax_functions = sorted(set(re.findall(r"@R\.function\s+def\s+([A-Za-z_]\w*)", frontend_text)))
    tir_functions = sorted(set(re.findall(r"@T\.prim_func(?:\([^\n]*\))?\s+def\s+([A-Za-z_]\w*)", legalized_text)))
    call_tir_targets = sorted(set(re.findall(r"R\.call_tir\(cls\.([A-Za-z_]\w*)", legalized_text)))
    reachable = [
        {
            "region_id": f"tir::{name}",
            "kind": "TIR_prim_func",
            "symbol": name,
            "reachable_from_main": name in call_tir_targets,
            "source_relation": "Relax call_tir target",
        }
        for name in tir_functions
    ]
    return {
        "frontend_relax_functions": relax_functions,
        "legalized_tir_functions": tir_functions,
        "call_tir_targets": call_tir_targets,
        "candidate_regions": reachable,
        "inventory_method": "generic IR symbol/call_tir extraction",
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

    case_dir = pathlib.Path(args.case_dir).resolve()
    model, values, reference, manifest = _load_case(case_dir)
    input_names = [item.name for item in model.graph.input if item.name in values]
    converted = from_onnx(model)
    frontend_text = str(converted)
    legalized_text = str(relax.transform.LegalizeOps()(converted))
    outputs = [_run(tvm, relax, converted, values, input_names) for _ in range(2)]
    endpoint = compute_endpoint_oracle(reference, outputs[0])
    repeat_endpoints = [compute_endpoint_oracle(reference, value).as_dict() for value in outputs]
    report = {
        "schema_version": "forkcert.generic-tvm-locator.v0.1",
        "case_identity": {
            "case_id": manifest["case_id"],
            "case_package_sha256": _sha((case_dir / "case_manifest.json").read_text()),
            "tvm_root": str(pathlib.Path(args.tvm_root).resolve()),
            "tvm_version": getattr(tvm, "__version__", "unknown"),
            "target": "llvm",
        },
        "input_contract": {
            "model_sha256": hashlib.sha256((case_dir / "model.onnx").read_bytes()).hexdigest(),
            "input_names": input_names,
            "input_artifact_sha256": hashlib.sha256((case_dir / "input.npz").read_bytes()).hexdigest(),
            "reference_artifact_sha256": hashlib.sha256((case_dir / "reference.npy").read_bytes()).hexdigest(),
        },
        "oracle": {
            "endpoint": endpoint.as_dict(),
            "repeat_endpoints": repeat_endpoints,
            "runtime_repeatability": compute_repeatability(outputs),
            "reference_role": manifest.get("contract", {}).get("reference_role", "declared reference artifact"),
        },
        "ir_stages": {
            "frontend_relax": {"sha256": _sha(frontend_text), "text": frontend_text},
            "legalized_tir": {"sha256": _sha(legalized_text), "text": legalized_text},
        },
        "region_inventory": _inventory(frontend_text, legalized_text),
        "automation_contract": {
            "bug_specific_region_input": False,
            "bug_specific_repair_input": False,
            "bug_specific_semantic_rule": False,
            "candidate_selection": "all reachable TIR symbols; no ranking by delta",
            "repair": "not executed by locator",
        },
        "allowed_pre_reveal_claim": "OBSERVATION_PLUS_GENERIC_IR_INVENTORY",
        "limitations": [
            "same-input local replay is not inferred for arbitrary IR symbols",
            "fixed-suffix mediation and repair are evaluator actions, not locator actions",
            "TIR symbol provenance is not source-line or kernel provenance",
            "the locator does not infer correctness from a numeric delta alone",
        ],
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(out),
        "case_id": manifest["case_id"],
        "exact": endpoint.exact_match,
        "candidate_regions": [row["region_id"] for row in report["region_inventory"]["candidate_regions"]],
        "bug_specific_inputs": report["automation_contract"]["bug_specific_region_input"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
