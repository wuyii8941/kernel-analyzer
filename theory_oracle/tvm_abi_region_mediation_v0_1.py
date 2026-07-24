"""Post-reveal TVM region-swap mediation using ABI-aligned IR functions.

The script does not know the bug's operator, trigger, or semantic repair.  It
uses the generic probe comparison to obtain a unique buggy/fixed region pair,
parses the fixed TIR into the buggy TVM process, and swaps only that PrimFunc
in the original buggy Relax module.  The original Relax ``main`` (the suffix
and its call structure) is retained.  Any parse, ABI, or context ambiguity
fails closed.

This is an adapter-level experiment, not a correctness proof: generated-kernel
identity and runtime context are reported separately, and the claim cannot
exceed intervention-dependent attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from forkcert.relational_oracle import compute_endpoint_oracle, compute_repeatability


def _load_case(case_dir: Path):
    import onnx

    model = onnx.load(case_dir / "model.onnx")
    with np.load(case_dir / "input.npz") as packed:
        values = {name: packed[name] for name in packed.files}
    reference = np.load(case_dir / "reference.npy")
    return model, values, reference


def _run(relax, tvm, module, values, input_names):
    executable = relax.build(module, target="llvm")
    vm = relax.VirtualMachine(executable, device=tvm.cpu())
    args = [tvm.runtime.tensor(values[name], device=tvm.cpu()) for name in input_names]
    return [vm["main"](*args).numpy() for _ in range(2)]


def _functions(module):
    return {str(gv.name_hint): func for gv, func in module.functions.items()}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buggy-root", required=True)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--buggy-report", required=True)
    parser.add_argument("--fixed-report", required=True)
    parser.add_argument("--region-comparison", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    from tvm.script import from_source

    case_dir = Path(args.case_dir).resolve()
    buggy_report = json.loads(Path(args.buggy_report).read_text())
    fixed_report = json.loads(Path(args.fixed_report).read_text())
    comparison = json.loads(Path(args.region_comparison).read_text())
    model, values, reference = _load_case(case_dir)
    input_names = [item.name for item in model.graph.input if item.name in values]
    pairs = [row for row in comparison.get("regions", []) if row.get("status") == "COMPARED"]
    if len(pairs) != 1:
        raise SystemExit("mediation requires exactly one unique ABI-aligned region pair")
    pair = pairs[0]
    buggy_name = pair["buggy_region_id"].split("::", 1)[-1]
    fixed_name = pair["fixed_region_id"].split("::", 1)[-1]

    base = relax.transform.LegalizeOps()(from_onnx(model))
    base_functions = _functions(base)
    if buggy_name not in base_functions:
        raise SystemExit(f"buggy region {buggy_name!r} is not in the original Relax module")
    fixed_ir = from_source(fixed_report["ir_stages"]["legalized_tir"]["text"])
    fixed_functions = _functions(fixed_ir)
    if fixed_name not in fixed_functions or not hasattr(fixed_functions[fixed_name], "buffer_map"):
        raise SystemExit(f"fixed region {fixed_name!r} is not a TIR PrimFunc")

    baseline_text = str(base)
    baseline_outputs = _run(relax, tvm, base, values, input_names)
    intervention = relax.transform.LegalizeOps()(from_onnx(model))
    target_gv = next(gv for gv in intervention.get_global_vars() if str(gv.name_hint) == buggy_name)
    intervention[target_gv] = fixed_functions[fixed_name]
    intervention_text = str(intervention)
    intervention_outputs = _run(relax, tvm, intervention, values, input_names)

    base_funcs_after = _functions(base)
    intervention_funcs = _functions(intervention)
    non_target_ids = sorted((set(base_funcs_after) | set(intervention_funcs)) - {buggy_name})
    non_target_equal = all(
        name in base_funcs_after and name in intervention_funcs
        and str(base_funcs_after[name]) == str(intervention_funcs[name])
        for name in non_target_ids
    )
    baseline_oracle = compute_endpoint_oracle(reference, baseline_outputs[0])
    intervention_oracle = compute_endpoint_oracle(reference, intervention_outputs[0])
    mediation_oracle = compute_endpoint_oracle(baseline_outputs[0], intervention_outputs[0])
    report = {
        "schema_version": "forkcert.tvm-abi-region-mediation.v0.1",
        "case_id": buggy_report["case_identity"]["case_id"],
        "adapter": {
            "id": "tvm-relax-onnx-llvm",
            "buggy_root": str(Path(args.buggy_root).resolve()),
            "method": "replace one ABI-aligned TIR PrimFunc in original buggy Relax module",
        },
        "automation_contract": {
            "bug_specific_region_input": False,
            "bug_specific_repair_input": False,
            "bug_specific_semantic_rule": False,
            "pairing": "unique ABI contract from generic region probe",
        },
        "region_pair": pair,
        "same_input_contract": pair.get("same_input_contract", False),
        "intervention": {
            "type": "cross_checkout_primfunc_swap",
            "buggy_region": f"tir::{buggy_name}",
            "fixed_region": f"tir::{fixed_name}",
            "original_relax_main_sha256": _sha(str(base_functions["main"])),
            "intervention_relax_main_sha256": _sha(str(intervention_funcs["main"])),
            "relax_main_unchanged": str(base_functions["main"]) == str(intervention_funcs["main"]),
            "non_target_function_context_invariant": non_target_equal,
            "compiler_kernel_context_invariant": False,
        },
        "oracle": {
            "baseline": baseline_oracle.as_dict(),
            "intervention": intervention_oracle.as_dict(),
            "baseline_vs_intervention": mediation_oracle.as_dict(),
            "baseline_repeatability": compute_repeatability(baseline_outputs),
            "intervention_repeatability": compute_repeatability(intervention_outputs),
        },
        "interpretation": {
            "region_swap_changed_endpoint": not mediation_oracle.exact_match,
            "region_swap_removed_reference_discrepancy": baseline_oracle.exact_match is False and intervention_oracle.exact_match is True,
            "claim": "INTERVENTION_DEPENDENT_ATTRIBUTION",
            "meaning": "only the effect of this post-reveal PrimFunc swap in the retained Relax context",
        },
        "limitations": [
            "the fixed PrimFunc is parsed after reveal and is not an independently supplied semantic repair",
            "generated kernel identity, layout, and autotuning context are not proven invariant",
            "an unchanged endpoint would not prove the region is irrelevant under another suffix",
            "this adapter does not establish a unique compiler source line or mathematical truth",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "claim": report["interpretation"]["claim"], "endpoint_changed": report["interpretation"]["region_swap_changed_endpoint"], "removed_discrepancy": report["interpretation"]["region_swap_removed_reference_discrepancy"]}, sort_keys=True))


if __name__ == "__main__":
    main()
