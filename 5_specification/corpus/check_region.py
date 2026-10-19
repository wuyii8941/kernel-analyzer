#!/usr/bin/env python3
"""Recheck a captured consecutive region against its generated wrapper.

This checks capture/source prerequisites, task boundaries and wrapper returns.
An explicitly selected softmax task can be evaluated after whole-formula matching.
Device effects and semantic comparison still require separate checks.
"""
import argparse
import json
from fractions import Fraction
from pathlib import Path

import dynamic_three_way as D
from region_source import check_launch_region, map_return_values, check_local_temporary
from region_outputs import bind_outputs
from region_inputs import bind_inputs
from region_task import build_task_region
from kernel_spec.num import Invalid, NotEstablished, Undefined, Unsupported
from region_storage import check_sequence


def select_launches(manifest, first, last):
    if type(first) is not int or type(last) is not int or first < 0 or last < first:
        raise ValueError("launch range must be nonnegative and ordered")
    saved = manifest.get("saved")
    if not isinstance(saved, list):
        raise ValueError("capture manifest has no saved launch list")
    selected = []
    for index in range(first, last + 1):
        matches = [row for row in saved if type(row.get("launch_index")) is int
                   and row["launch_index"] == index]
        if len(matches) != 1:
            raise ValueError(f"launch {index} is missing or ambiguous")
        row = matches[0]
        if (not isinstance(row.get("dir"), str) or Path(row["dir"]).name != row["dir"]
                or row["dir"] in ("", ".", "..") or not isinstance(row.get("kernel"), str)):
            raise ValueError("invalid capture directory or kernel declaration")
        selected.append(row)
    return selected


def check_call_locations(launches, source_path, region):
    calls = region.get("calls", [])
    if len(launches) != len(calls):
        raise ValueError("captured launches and source call sites differ in count")
    for launch, call in zip(launches, calls):
        if launch.kernel_name != call["kernel"]:
            raise ValueError("captured kernel and selected call site disagree")
        if not any(Path(frame.get("file", "")).resolve() == source_path.resolve()
                   and type(frame.get("line")) is int and frame["line"] == call["line"]
                   for frame in (launch.call_stack or [])):
            raise ValueError(f"launch {launch.index} has no matching recorded wrapper call site")


def inspect(capture_dir, output_code, first, last, *, softmax_inputs=None, max_unique=200_000,
            compare_g=False, max_programs=2048, check_device=False, evaluate_task=False,
            max_output_elements=200_000, max_evaluations=2_000_000):
    report = {"status": "not_established", "binding_status": "not_established",
              "reference_status": "not_evaluated", "semantic_verdict": "not_established",
              "stages": {}, "launch_range": [first, last]}
    stage = "capture_manifest"
    try:
        capture_dir, output_code = Path(capture_dir).resolve(), Path(output_code).resolve()
        if compare_g and softmax_inputs is None:
            raise ValueError("composed G comparison requires an explicitly selected task reference")
        if evaluate_task and softmax_inputs is not None:
            raise ValueError("select either graph task evaluation or the independent softmax task reference")
        report.update(capture_dir=str(capture_dir), output_code=str(output_code))
        manifest = json.loads((capture_dir / "manifest.json").read_text())
        selected = select_launches(manifest, first, last)
        if (manifest.get("capture_single_recorder_run") is not True
                or manifest.get("capture_storages_retained") is not True):
            raise ValueError("manifest lacks common recorder run or retained storage lifetime evidence")
        report["stages"][stage] = "checked"
        stage = "captured_sequence"
        from kernel_analyzer.reference_eval.capture import load_launch
        launches = [load_launch(capture_dir / row["dir"]) for row in selected]
        for row, launch in zip(selected, launches):
            if launch.index != row["launch_index"] or launch.kernel_name != row["kernel"]:
                raise ValueError("manifest and raw capture identity disagree")
        sequence = check_sequence(launches, same_recorder_run=True, storages_retained=True)
        report["stages"][stage] = sequence
        if sequence["status"] != "captured storage sequence checked":
            raise ValueError(sequence["reason"])
        stage = "wrapper_calls"
        region = check_launch_region(output_code.read_text(), [row["kernel"] for row in selected])
        if region["status"] != "wrapper launch region checked":
            raise ValueError(region["reason"])
        check_call_locations(launches, output_code, region)
        report["stages"][stage] = region
        stage = "compiler_provenance"
        oc = D.OutputCode(output_code)
        prov_file = Path(manifest["provenance_file"]).resolve()
        if prov_file != prov_file.parent.parent / capture_dir.parent.name / (capture_dir.name + ".jsonl"):
            raise ValueError("manifest provenance path does not match the selected capture model/config")
        task_paths, mapped = set(), []
        for row, launch, call in zip(selected, launches, region["calls"]):
            unit = "/".join([capture_dir.parent.name, capture_dir.name, row["dir"]])
            rec = {"unit": unit, "kernel": row["kernel"], "output_code": str(output_code)}
            records, why = D.provenance(rec, capture_dir.parent.parent, prov_file.parent.parent)
            if records is None:
                raise ValueError(why)
            args, provenance = D.select_call(oc, row["kernel"], records)
            if args is None:
                raise ValueError(provenance)
            if args != call["args"]:
                raise ValueError("parsed launcher arguments disagree")
            scalars, why = D.bind_call_scalars(oc.params[row["kernel"]], args, launch,
                                              oc.launcher_meta.get(row["kernel"]))
            if why:
                raise ValueError(why)
            if not provenance.get("task_graph"):
                raise ValueError("complete task graph provenance missing")
            task_paths.add(str(Path(provenance["task_graph"]).resolve()))
            mapped.append({"launch": launch.index, "kernel": row["kernel"], "scalars": scalars,
                           "parameters": provenance["params"]})
        if len(task_paths) != 1:
            raise ValueError("selected launches belong to different task graphs")
        graph_path = Path(next(iter(task_paths)))
        graph = json.loads(graph_path.read_text())
        if (graph.get("error") or not graph.get("output_code")
                or Path(graph["output_code"]).resolve() != output_code):
            raise ValueError("task graph ownership and selected wrapper disagree")
        report["stages"][stage] = {"status": "checked", "task_graph": str(graph_path), "launches": mapped}
        stage = "region_outputs"
        observations, input_observations, written_storages = [], [], set()
        for launch, call, mapping in zip(launches, region["calls"], mapped):
            writes = D.write_parameters(launch)
            if writes is None:
                raise ValueError("complete IR write parameter set not established")
            byname = {arg.name: arg for arg in launch.args}
            write_only = D.write_only_parameters(launch)
            for parameter, buffer in zip(oc.params[call["kernel"]], call["args"]):
                arg = byname[parameter]
                if arg.kind != "tensor":
                    continue
                observations.append((launch.index, buffer, arg))
                input_observations.append({"launch": launch.index, "buffer": buffer, "arg": arg,
                                           "provenance": mapping["parameters"].get(parameter, {}),
                                           "write_only": parameter in write_only})
                if parameter in writes:
                    written_storages.add(arg.storage_id)
        returned = map_return_values(output_code.read_text(), graph, region)
        bound = bind_outputs(graph, returned, observations, written_storages)
        report["stages"][stage] = {
            "outputs": [{"return": item["return"], "last_observed_launch": item["last_observed_launch"],
                         "storage_id": item["arg"].storage_id, "shape": list(item["arg"].shape),
                         "stride": list(item["arg"].stride), "dtype": item["arg"].dtype}
                        for item in bound["outputs"]],
            "passthrough_returns": bound["passthrough_returns"],
            "constant_returns": bound["constant_returns"],
            "unobserved_returns": bound["unobserved_returns"],
            "unreturned_written_storages": bound["unreturned_written_storages"]}
        stage = "local_temporaries"
        temporaries = []
        report["stages"][stage] = temporaries
        for identity in bound["unreturned_written_storages"]:
            names = sorted({buffer for _, buffer, arg in observations if arg.storage_id == identity})
            temporary = check_local_temporary(output_code.read_text(), region, names)
            temporaries.append(dict(temporary, storage_id=identity))
            if temporary["status"] != "wrapper local temporary checked":
                raise ValueError(temporary.get("reason", "unreturned write is not a proven wrapper temporary"))
        if not bound["outputs"]:
            raise ValueError("no computed returned output was bound")
        stage = "task_inputs"
        inputs, descriptions = bind_inputs(input_observations, written_storages)
        report["stages"][stage] = descriptions
        stage = "task_graph"
        outputs = list(dict.fromkeys(item["return"]["node"] for item in bound["outputs"]))
        task = build_task_region(graph, inputs, outputs)
        from kernel_spec.graph import functional_value_graph
        value_task, local_updates = functional_value_graph(task, input_names=inputs)
        if local_updates:
            report["stages"]["local_update_conditions"] = {"updates": local_updates, "value_graph": value_task}
        report["stages"][stage] = task
        report["status"] = "region task structure checked"
        report["remaining"] = ["complete task input/output/state binding", "IR and device memory effects",
                               "mathematical reference evaluation", "semantic comparison"]
        if evaluate_task:
            stage = "mathematical_reference"
            from region_reference import compare_task
            comparison = compare_task(task,
                {name: {"values": D.logical(arg, "before"), "dtype": arg.dtype} for name, arg in inputs.items()},
                [{"position": item["return"]["position"], "node": item["return"]["node"],
                  "values": D.logical(item["arg"], "after"), "dtype": item["arg"].dtype}
                 for item in bound["outputs"]], max_elements=max_output_elements, max_evaluations=max_evaluations)
            report["stages"][stage] = comparison
            report["reference_status"] = ("captured_graph_complete" if comparison["complete"]
                                          else comparison["outcome"])
            if not comparison["complete"]:
                failure = {"invalid_input": Invalid, "unsupported": Unsupported,
                           "domain_undefined": Undefined}.get(comparison["outcome"], NotEstablished)
                raise failure("complete graph task comparison was not established; see per-output results")
            report["status"] = "captured graph reference evaluated"
            report["remaining"] = ["IR and device memory effects", "independent task specification and contract binding",
                                   "G versus independent task comparison", "base-contract implementation certification"]
        if check_device:
            stage = "device_correspondence"
            from region_device import check
            device = check(launches, [capture_dir / row["dir"] for row in selected],
                           max_programs=max_programs)
            report["stages"][stage] = device
            if device["status"] != "checked":
                raise NotEstablished("device replay or fixed-rule emulation did not agree completely")
        if softmax_inputs is not None:
            stage = "mathematical_reference"
            from softmax_reference import evaluate_region
            if len(softmax_inputs) != 2:
                raise Invalid("softmax reference requires explicit logits and labels node names")
            reference = evaluate_region(task, {name: D.logical(arg, "before") for name, arg in inputs.items()},
                                        logits_input=softmax_inputs[0], labels_input=softmax_inputs[1],
                                        max_unique=max_unique)
            report["stages"][stage] = reference
            report["reference_status"] = "selected task evaluated with rigorous enclosures"
            report["status"] = "region task reference evaluated"
            report["remaining"] = ["IR and device memory effects", "complete kernel/task binding certification",
                                   "G versus task reference comparison"]
            if compare_g:
                stage = "composed_reference_comparison"
                from region_comparison import evaluate_and_compare
                report["stages"][stage] = evaluate_and_compare(launches, bound["outputs"], reference, max_programs)
                report["status"] = "conditional G-f comparison completed"
                report["remaining"] = ["IR and device memory effects", "complete kernel/task binding certification"]
    except (Invalid, NotEstablished, Undefined, Unsupported) as error:
        report["status"] = "not_established"
        report["failed_stage"] = stage
        outcome = {Invalid: "invalid_input", NotEstablished: "not_established",
                   Undefined: "domain_undefined", Unsupported: "unsupported"}[type(error)]
        report["query_outcome"] = outcome
        if stage == "mathematical_reference":
            report["reference_status"] = outcome
        report["reason"] = str(error)
    except (ValueError, KeyError, OSError) as error:
        report["status"] = "not_established"
        report["failed_stage"] = stage
        report["reason"] = str(error)
    except Exception as error:
        report["status"] = "implementation_error"
        report["failed_stage"] = stage
        report["reason"] = f"{type(error).__name__}: {error}"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output-code", required=True, type=Path)
    parser.add_argument("--first", required=True, type=int)
    parser.add_argument("--last", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--softmax-inputs", nargs=2, metavar=("LOGITS_NODE", "LABELS_NODE"))
    parser.add_argument("--max-unique", type=int, default=200_000,
                        help="explicit distinct-exponential budget for the selected softmax reference")
    parser.add_argument("--compare-g", action="store_true", help="evaluate the full composed G and compare all outputs")
    parser.add_argument("--max-programs", type=int, default=2048)
    parser.add_argument("--evaluate-task", action="store_true",
                        help="compare all mapped outputs with the captured compiler graph; this is not an independent f")
    parser.add_argument("--max-output-elements", type=int, default=200_000)
    parser.add_argument("--max-evaluations", type=int, default=2_000_000)
    parser.add_argument("--check-device", action="store_true",
                        help="replay saved binaries twice and check fixed-rule emulation on the full grids")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("--out must be a new file; existing records are never overwritten")
    report = inspect(args.capture_dir, args.output_code, args.first, args.last,
                     softmax_inputs=args.softmax_inputs, max_unique=args.max_unique,
                     compare_g=args.compare_g, max_programs=args.max_programs, check_device=args.check_device,
                     evaluate_task=args.evaluate_task, max_output_elements=args.max_output_elements,
                     max_evaluations=args.max_evaluations)
    def serialize(value):
        if isinstance(value, Fraction):
            return str(value)
        raise TypeError(f"unsupported report value {type(value).__name__}")
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, default=serialize)
    print(report["status"])
    return (0 if report["status"] in ("region task structure checked", "captured graph reference evaluated",
                                     "region task reference evaluated",
                                     "conditional G-f comparison completed")
            else 1 if report["status"] == "implementation_error" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
