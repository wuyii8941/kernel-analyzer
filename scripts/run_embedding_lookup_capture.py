#!/usr/bin/env python3
"""Measure reviewed pure Triton embedding lookups with the shared analyzer."""

import argparse
import sys
from pathlib import Path

from scripts.run_numerical_coverage import ROOT, read, save, sha


def select(plan, cases):
    if plan.get("schema") != "embedding-lookup-forward-bound-plan-v1" or not cases:
        raise ValueError("Nonempty embedding lookup plan required")
    originals = {case["task_id"]: case for case in plan["cases"]}
    if len(originals) != len(plan["cases"]) or len({case["task_id"] for case in cases}) != len(cases):
        raise ValueError("Unique selected embedding tasks required")
    selected = {}
    for case in cases:
        if case != originals.get(case["task_id"]):
            raise ValueError("Selected embedding case differs from frozen plan")
        symbol = case["expected_symbol"]
        contract = plan["contracts"][symbol]
        if (case["reference_method"] != "EMBEDDING_LOOKUP_COMMON_INPUT"
                or case["reference_output_pointer"] != "out_ptr0"
                or contract["output_pointers"] != ["out_ptr0"]):
            raise ValueError("Embedding output or reference method differs")
        selected[symbol] = contract
    return selected


def verify_dependencies(record, dependencies, visited):
    for name, expected in record.get("source_sha256", {}).items():
        path = Path(name)
        if sha(path) != expected:
            raise ValueError("Frozen embedding dependency changed: " + name)
        dependencies.append(path)
        if path.suffix == ".json" and name not in visited:
            visited.add(name)
            nested = read(path)
            if isinstance(nested, dict):
                verify_dependencies(nested, dependencies, visited)


def unique_argument(arguments, name, default=None):
    values = []
    for index, value in enumerate(arguments):
        if value == name:
            if index + 1 == len(arguments) or arguments[index + 1].startswith("--"):
                raise ValueError("Missing argument: " + name)
            values.append(arguments[index + 1])
        elif value.startswith(name + "="):
            values.append(value.split("=", 1)[1])
    if not values and default is not None:
        return default
    if len(values) != 1:
        raise ValueError("Unique explicit argument required: " + name)
    return values[0]


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
    if args.state_bank is not None:
        raise ValueError("Embedding state-bank input requires a separately audited preflight")
    from scripts.preflight_capture_input import check as check_input
    input_preflight = check_input(
        args.input_bank, args.release_dir / "capture.json",
        int(unique_argument(remaining, "--states")),
        int(unique_argument(remaining, "--warmup-steps", "0")),
    )
    translated = output.parent / "embedding_lookup_capture_plan.json"
    save(translated, {"cases": [dict(
        case, reference_method="PARTIAL_REDUCTION_FROM_BOUND_INPUT",
        declared_reference_method=case["reference_method"],
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
    from kernel_analyzer.embedding_lookup_observer import observer_class
    from kernel_analyzer.embedding_lookup_reference import select_output

    capture.SameDtypeSemanticCandidateObserver = observer_class(
        capture.SameDtypeSemanticCandidateObserver, contracts, runtime_signature
    )
    selected_outputs = {(case["expected_symbol"], case["reference_output_pointer"])
                        for case in cases}

    def reference(metadata, candidate, **unused):
        key = (metadata.get("symbol"), metadata.get("formal_pointer"))
        if key not in selected_outputs:
            raise ValueError("Undeclared embedding output")
        contract = contracts[key[0]]
        preserved = {name: metadata["runtime_pointers"][name]
                     for name in ("in_ptr0", "in_ptr1")}
        return select_output(
            preserved, candidate, tokens=contract["tokens"], width=contract["width"],
            vocabulary_size=contract["vocabulary_size"],
        )

    capture.partial_reduction_reference = reference
    old_scope = capture.reference_scope
    capture.reference_scope = lambda method: ({
        "comparison": "PURE_EMBEDDING_LOOKUP_OUTPUT_REPLACEMENT",
        "same_local_operands": True,
        "includes_possible_upstream_differences": False,
        "complete_multi_output_implementation_replacement": True,
        "reference_variant": "DECLARED_SAME_DTYPE_INDEX_SELECT",
        "reference_is_absolute_truth": False,
    } if method == "PARTIAL_REDUCTION_FROM_BOUND_INPUT" else old_scope(method))

    dependencies += [Path(__file__), args.family_plan, args.case_plan, translated,
                     args.input_bank, args.model / "config.json"]
    if args.state_bank:
        dependencies.append(args.state_bank)
    dependencies += [ROOT / "src/kernel_analyzer" / name for name in (
        "embedding_lookup_source.py", "embedding_lookup_reference.py",
        "embedding_lookup_observer.py", "parallel_measurement.py",
        "training_numerical_analysis.py", "training_equivalence.py",
        "training_bias_profile.py", "update_write.py", "capture_cost.py",
    )]
    dependencies += [ROOT / "scripts" / name for name in (
        "capture_bound_endpoint_bias_formation_v21.py", "same_dtype_semantic_observer.py",
        "run_parallel_bound_capture.py", "run_training_bias_profile_v2_empirical.py",
        "preflight_capture_input.py",
    )]
    dependencies += [args.release_dir / name for name in (
        "same_dtype_tasks.json.gz", "campaign.json.gz", "inventory.json.gz",
    )]
    save(output / "family_execution_protocol.json", {
        "schema": "embedding-lookup-forward-capture-v1",
        "contracts": contracts,
        "capture_arguments": remaining,
        "statistical_method_changed": False,
        "claim_scope": "FIXED_SUITE_UPDATE",
        "primary_stage": "PARAMETER_WRITE",
        "contrast_id": "PURE_EMBEDDING_LOOKUP_OUTPUT_REPLACEMENT",
        "data_use": "NEW_FAMILY_RUNTIME_INTEGRATION_NOT_BLIND_BIAS_CONFIRMATION",
        "input_preflight": input_preflight,
        "source_sha256": {str(path.resolve()): sha(path) for path in dependencies},
    })
    from scripts.run_parallel_bound_capture import main as run
    from kernel_analyzer.capture_cost import measured_capture
    sys.argv = [sys.argv[0], *remaining]
    measured_capture(run, device=args.device,
                     emit=lambda value: save(output / "capture_cost.json", value))


if __name__ == "__main__":
    main()
