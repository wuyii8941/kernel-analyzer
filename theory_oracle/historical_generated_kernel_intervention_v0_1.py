#!/usr/bin/env python
"""Run a narrow, auditable intervention on one generated reduction kernel.

The replacement is a hypothesis derived from the captured generated code, not
the hidden historical patch.  The script reports exact textual/context gates
and deliberately caps the claim at intervention-dependent attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any


OLD = "r0_2 + 7724*x0 + 100416*x1"
NEW = "r0_2 + 7724*x0 + 100401*x1"
TARGET_KERNEL = "triton_red_fused_sum_view_1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load generated module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_hash(value: Any) -> dict[str, Any]:
    raw = value.detach().contiguous().cpu().numpy().tobytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
    }


def delta(left: Any, right: Any) -> dict[str, Any]:
    value = (left.detach().float() - right.detach().float()).abs()
    return {
        "max_abs": float(value.max().item()),
        "mean_abs": float(value.mean().item()),
        "nonzero": int((value != 0).sum().item()),
    }


def run_call(torch: Any, module: ModuleType, value: Any) -> Any:
    output = module.call([value.clone()])[0]
    torch.cuda.synchronize()
    return output


def non_target_signature(text: str) -> str:
    """Hash all generated wrapper text except the declared target expression."""

    canonical = text.replace(OLD, "<TARGET_EXPRESSION>").replace(NEW, "<TARGET_EXPRESSION>")
    return hashlib.sha256(canonical.encode()).hexdigest()


def context_summary(text: str) -> dict[str, Any]:
    return {
        "graph_provenance_lines": re.findall(r"# Topologically Sorted Source Nodes: (.+)", text),
        "original_aten_lines": re.findall(r"Original ATen: (.+)", text),
        "kernel_names": sorted(set(re.findall(r"triton_[a-zA-Z0-9_]+", text))),
        "call_sites": re.findall(r"\.run\([^\n]+", text),
        "shape_guards": re.findall(r"assert_size_stride\([^\n]+", text),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-code", required=True)
    parser.add_argument("--input-artifact", required=True)
    parser.add_argument("--reference-artifact", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch

    output_code = Path(args.output_code).resolve()
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    original_copy = workspace / "original_output_code.py"
    intervention_copy = workspace / "intervention_output_code.py"
    original_text = output_code.read_text(encoding="utf-8")
    if original_text.count(OLD) != 1:
        raise RuntimeError(f"expected exactly one target expression, found {original_text.count(OLD)}")
    intervention_text = original_text.replace(OLD, NEW)
    original_copy.write_text(original_text, encoding="utf-8")
    intervention_copy.write_text(intervention_text, encoding="utf-8")

    value = torch.load(Path(args.input_artifact), map_location="cuda").to(device="cuda")
    reference = torch.load(Path(args.reference_artifact), map_location="cuda").to(device="cuda")
    original_module = load_module(original_copy, "forkcert_hist_original_generated")
    original = run_call(torch, original_module, value)
    intervention_module = load_module(intervention_copy, "forkcert_hist_intervention_generated")
    repaired = run_call(torch, intervention_module, value)

    original_context = context_summary(original_text)
    intervention_context = context_summary(intervention_text)
    context_invariant = (
        non_target_signature(original_text) == non_target_signature(intervention_text)
        and original_context["graph_provenance_lines"] == intervention_context["graph_provenance_lines"]
        and original_context["original_aten_lines"] == intervention_context["original_aten_lines"]
        and original_context["shape_guards"] == intervention_context["shape_guards"]
        and len(original_context["call_sites"]) == len(intervention_context["call_sites"])
    )
    report = {
        "schema_version": "forkcert.historical_generated_kernel_intervention.v0.1",
        "case_id": "case_001",
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "intervention": {
            "type": "direct_generated_kernel_code_hypothesis",
            "target_kernel": TARGET_KERNEL,
            "old_expression": OLD,
            "new_expression": NEW,
            "original_output_code_sha256": sha256_file(original_copy),
            "intervention_output_code_sha256": sha256_file(intervention_copy),
        },
        "outputs": {
            "reference": tensor_hash(reference),
            "original": tensor_hash(original),
            "intervention": tensor_hash(repaired),
            "reference_vs_original": delta(reference, original),
            "reference_vs_intervention": delta(reference, repaired),
            "original_vs_intervention": delta(original, repaired),
            "intervention_exact_reference": bool(torch.equal(reference, repaired)),
            "residual_tolerance": 4e-5,
            "intervention_within_control_residual_tolerance": delta(reference, repaired)["max_abs"] <= 4e-5,
        },
        "context": {
            "non_target_text_signature_equal": non_target_signature(original_text)
            == non_target_signature(intervention_text),
            "summary_equal_except_target_text": context_invariant,
            "original": original_context,
            "intervention": intervention_context,
        },
        "claim": {
            "allowed_claim_level": "INTERVENTION_DEPENDENT_ATTRIBUTION" if context_invariant else "OBSERVATION",
            "interpretation": (
                "the declared generated reduction kernel intervention changes the endpoint while the captured "
                "non-target wrapper context remains invariant; this is not a historical-patch or root-cause claim"
                if context_invariant
                else "the intervention changed unverified context; report only observation"
            ),
        },
        "limitations": [
            "the expression replacement is a hypothesis, not the hidden historical patch",
            "no compiler-stage bisection was performed",
            "same-process module replay does not prove all runtime/autotuning state invariant",
            "operator uniqueness and root cause are not claimed",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
