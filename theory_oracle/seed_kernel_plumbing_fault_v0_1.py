#!/usr/bin/env python
"""Introduce and audit a hidden deterministic mutation for plumbing calibration.

This is deliberately a *seeded-fault calibration harness*.  The mutation rule
is not exposed through the opaque case manifest and must never be counted as a
real compiler bug or an external localization result.  Its purpose is narrower:
verify that a captured generated realization can be replayed, that a no-op
control preserves it, and that a one-expression intervention is reported with
its context limits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


TARGET = "tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)"
FAULT = "tmp4 = (tl.sum(tmp3, 1)[:, None] + 1.0).to(tl.float32)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(torch: Any, module: ModuleType, value: Any) -> Any:
    result = module.call([value.clone()])[0]
    torch.cuda.synchronize()
    return result


def delta(left: Any, right: Any) -> dict[str, Any]:
    diff = (left.detach().float() - right.detach().float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "nonzero": int((diff != 0).sum().item()),
    }


def fingerprint(value: Any) -> dict[str, Any]:
    raw = value.detach().contiguous().cpu().numpy().tobytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
    }


def non_target_signature(text: str) -> str:
    return hashlib.sha256(text.replace(TARGET, "<TARGET>").replace(FAULT, "<TARGET>").encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    import torch

    case_dir = args.case_dir.resolve()
    manifest = json.loads((case_dir / "opaque_case_manifest.json").read_text())
    wrappers = [Path(row["path"]) for row in manifest["provenance_inventory"]]
    source = next((path for path in wrappers if path.name == "output_code.py"), None)
    if source is None:
        raise RuntimeError("no debug output_code.py recorded in case manifest")
    original_text = source.read_text(encoding="utf-8")
    if original_text.count(TARGET) != 1:
        raise RuntimeError(f"expected one calibration mutation slot, saw {original_text.count(TARGET)}")
    faulty_text = original_text.replace(TARGET, FAULT)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    original_path = out / "generated_original.py"
    noop_path = out / "generated_noop.py"
    faulty_path = out / "generated_faulty.py"
    original_path.write_text(original_text, encoding="utf-8")
    noop_path.write_text(original_text, encoding="utf-8")
    faulty_path.write_text(faulty_text, encoding="utf-8")

    value = torch.load(case_dir / "input.pt", map_location="cuda", weights_only=True).to("cuda")
    reference = torch.load(case_dir / "reference.pt", map_location="cuda", weights_only=True).to("cuda")
    original = run(torch, load_module(original_path, "forkcert_kernel_cal_original"), value)
    noop = run(torch, load_module(noop_path, "forkcert_kernel_cal_noop"), value)
    faulty = run(torch, load_module(faulty_path, "forkcert_kernel_cal_faulty"), value)
    repaired = run(torch, load_module(original_path, "forkcert_kernel_cal_repaired"), value)

    context_invariant = non_target_signature(original_text) == non_target_signature(faulty_text)
    report = {
        "schema_version": "forkcert.seeded-kernel-plumbing-calibration.v0.1",
        "case_id": manifest["case_id"],
        "role": "seeded_calibration_only",
        "input": fingerprint(value),
        "provenance": {
            "generated_wrapper": str(source),
            "source_sha256": sha256(source),
            "kernel_inventory": manifest["provenance_inventory"],
        },
        "controls": {
            "baseline_matches_reference": bool(torch.equal(original, reference)),
            "noop_matches_baseline": bool(torch.equal(noop, original)),
            "repaired_matches_reference": bool(torch.equal(repaired, reference)),
        },
        "production": {
            "same_input": True,
            "reference_output": fingerprint(original),
            "faulty_output": fingerprint(faulty),
            "delta": delta(original, faulty),
            "observed": not torch.equal(original, faulty),
        },
        "intervention": {
            "type": "direct_generated_expression_repair",
            "context_invariant_except_declared_target": context_invariant,
            "faulty_vs_reference": delta(faulty, reference),
            "repaired_vs_reference": delta(repaired, reference),
            "source_sha256": sha256(original_path),
            "faulty_sha256": sha256(faulty_path),
            "non_target_signature": non_target_signature(original_text),
        },
        "allowed_claim": "SEEDED_CALIBRATION_PLUMBING_PASSED"
        if bool(torch.equal(original, reference))
        and bool(torch.equal(noop, original))
        and not bool(torch.equal(original, faulty))
        and bool(torch.equal(repaired, reference))
        and context_invariant
        else "SEEDED_CALIBRATION_FAILED",
        "limitations": [
            "the seed recipe is calibration-only and is not visible to the opaque case consumer",
            "this proves generated-code replay/intervention plumbing, not localization accuracy",
            "a single generated wrapper has no nontrivial graph-reduction search space",
        ],
    }
    (out / "seeded_fault_record.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"allowed_claim": report["allowed_claim"], "production": report["production"], "controls": report["controls"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

