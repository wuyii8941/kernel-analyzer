#!/usr/bin/env python3
"""Measure reviewed grouped causal softmax outputs with the shared analyzer."""

import argparse
import sys
from pathlib import Path

from scripts.run_numerical_coverage import ROOT, read, save, sha


def select(plan, cases):
    if plan.get("schema") != "grouped-causal-softmax-forward-bound-plan-v1" or not cases:
        raise ValueError("Nonempty grouped causal softmax plan required")
    originals = {case["task_id"]: case for case in plan["cases"]}
    if len(originals) != len(plan["cases"]) or len({c["task_id"] for c in cases}) != len(cases):
        raise ValueError("Unique selected tasks required")
    selected = {}
    for case in cases:
        if case != originals.get(case["task_id"]):
            raise ValueError("Selected grouped softmax case differs from frozen plan")
        symbol = case["expected_symbol"]
        contract = plan["contracts"][symbol]
        if (case["reference_method"] != "GROUPED_CAUSAL_SOFTMAX_FORWARD_COMMON_INPUT"
                or case["reference_output_pointer"] not in contract["output_pointers"]):
            raise ValueError("Grouped softmax output or reference method differs")
        selected[symbol] = contract
    return selected


def verify_dependencies(record, dependencies, visited):
    for name, expected in record.get("source_sha256", {}).items():
        path = Path(name)
        if sha(path) != expected:
            raise ValueError("Frozen grouped softmax dependency changed: " + name)
        dependencies.append(path)
        if path.suffix == ".json" and name not in visited:
            visited.add(name)
            nested = read(path)
            if isinstance(nested, dict):
                verify_dependencies(nested, dependencies, visited)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("family-plan", "case-plan", "output-dir", "spool-dir",
                 "training-bias-profile-v2-output-dir", "input-bank", "release-dir", "model"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--state-bank", type=Path)
    args, remaining = parser.parse_known_args()
    output = args.training_bias_profile_v2_output_dir.resolve()
    for path in (output, args.output_dir, args.spool_dir):
        if path.exists() or not path.resolve().is_relative_to(Path("/data1/tzh")):
            raise ValueError("New output paths under /data1/tzh required")
    plan = read(args.family_plan)
    dependencies = []
    verify_dependencies(plan, dependencies, set())
    cases = read(args.case_plan)["cases"]
    contracts = select(plan, cases)
    translated = output.parent / "grouped_softmax_capture_plan.json"
    save(translated, {"cases": [dict(
        case, reference_method="PARTIAL_REDUCTION_FROM_BOUND_INPUT",
        declared_reference_method=case["reference_method"]
    ) for case in cases]})
    for name, value in (
        ("case-plan", translated), ("output-dir", args.output_dir),
        ("spool-dir", args.spool_dir),
        ("training-bias-profile-v2-output-dir", output), ("input-bank", args.input_bank),
        ("release-dir", args.release_dir), ("model", args.model), ("device", args.device),
    ):
        remaining.extend(["--" + name, str(value)])
    if args.state_bank:
        remaining.extend(["--state-bank", str(args.state_bank)])

    from scripts import capture_bound_endpoint_bias_formation_v21 as capture
    from scripts.same_dtype_semantic_observer import runtime_signature
    from kernel_analyzer.grouped_causal_softmax_observer import observer_class
    from kernel_analyzer.grouped_causal_softmax_reference import select_output

    capture.SameDtypeSemanticCandidateObserver = observer_class(
        capture.SameDtypeSemanticCandidateObserver, contracts, runtime_signature
    )
    selected_outputs = {
        (case["expected_symbol"], case["reference_output_pointer"]) for case in cases
    }

    def reference(metadata, candidate, **unused):
        key = (metadata.get("symbol"), metadata.get("formal_pointer"))
        if key not in selected_outputs:
            raise ValueError("Undeclared grouped softmax output")
        contract = contracts[key[0]]
        return select_output(
            metadata["runtime_pointers"], candidate, formal_pointer=key[1],
            rows=contract["rows"], width=contract["width"], scale=contract["scale"]
        )

    capture.partial_reduction_reference = reference
    old_scope = capture.reference_scope
    capture.reference_scope = lambda method: ({
        "comparison": "SINGLE_GROUPED_CAUSAL_SOFTMAX_OUTPUT_REPLACEMENT",
        "same_local_operands": True,
        "includes_possible_upstream_differences": False,
        "complete_multi_output_implementation_replacement": False,
        "reference_variant": "DECLARED_FP32_EXPRESSION_WITH_ORIGINAL_WRITES",
        "reference_is_absolute_truth": False,
    } if method == "PARTIAL_REDUCTION_FROM_BOUND_INPUT" else old_scope(method))

    dependencies += [Path(__file__), args.family_plan, args.case_plan, translated,
                     args.input_bank, args.model / "config.json"]
    if args.state_bank:
        dependencies.append(args.state_bank)
    dependencies += [ROOT / "src/kernel_analyzer" / name for name in (
        "grouped_causal_softmax_source.py", "grouped_causal_softmax_reference.py",
        "grouped_causal_softmax_observer.py", "parallel_measurement.py",
        "training_numerical_analysis.py", "training_equivalence.py",
        "training_bias_profile.py", "update_write.py", "capture_cost.py",
    )]
    dependencies += [ROOT / "scripts" / name for name in (
        "capture_bound_endpoint_bias_formation_v21.py", "same_dtype_semantic_observer.py",
        "run_parallel_bound_capture.py", "run_training_bias_profile_v2_empirical.py",
    )]
    dependencies += [args.release_dir / name for name in (
        "same_dtype_tasks.json.gz", "campaign.json.gz", "inventory.json.gz",
    )]
    save(output / "family_execution_protocol.json", {
        "schema": "grouped-causal-softmax-forward-capture-v1",
        "contracts": contracts,
        "capture_arguments": remaining,
        "statistical_method_changed": False,
        "claim_scope": "FIXED_SUITE_UPDATE",
        "primary_stage": "PARAMETER_WRITE",
        "contrast_id": "SINGLE_GROUPED_CAUSAL_SOFTMAX_OUTPUT_REPLACEMENT",
        "data_use": "NEW_FAMILY_RUNTIME_INTEGRATION_NOT_BLIND_BIAS_CONFIRMATION",
        "source_sha256": {str(path.resolve()): sha(path) for path in dependencies},
    })
    from scripts.run_parallel_bound_capture import main as run
    from kernel_analyzer.capture_cost import measured_capture
    sys.argv = [sys.argv[0], *remaining]
    measured_capture(run, device=args.device,
                     emit=lambda value: save(output / "capture_cost.json", value))


if __name__ == "__main__":
    main()
