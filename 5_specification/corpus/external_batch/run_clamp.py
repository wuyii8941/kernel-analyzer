"""Real public scalar-bound clamp calls with the complete input VJP."""
import argparse
import contextlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import traceback

import torch

import run as batch
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
from kernel_spec import registry
from kernel_spec.numeric_io import encode_finite


PROFILE = "torch210_clamp_scalar_autograd"


def functional(x, minimum, maximum):
    return torch.clamp(x, min=minimum, max=maximum)


def logical_indices(shape):
    return itertools.product(*(range(length) for length in shape))


def inputs(case, seed, device):
    generator = torch.Generator().manual_seed(seed)
    logical_shape = tuple(case["shape"])
    noncontiguous = case.get("noncontiguous", False)
    stored_shape = (*logical_shape[:-2], logical_shape[-1], logical_shape[-2]) \
        if noncontiguous else logical_shape
    x = torch.randint(-24, 25, stored_shape, generator=generator).float() / 8
    x = x.transpose(-1, -2) if noncontiguous else x
    minimum, maximum = case["min"], case["max"]
    boundary_values = []
    if minimum is not None:
        boundary_values.extend([minimum, minimum - 0.5, minimum + 0.5])
    if maximum is not None:
        boundary_values.extend([maximum, maximum + 0.5, maximum - 0.5])
    if minimum is not None and maximum is not None:
        boundary_values.append((minimum + maximum) / 2)
    indices = list(logical_indices(logical_shape))
    if case.get("boundary_heavy"):
        pattern = boundary_values or [0.0]
        for offset, index in enumerate(indices):
            x[index] = pattern[offset % len(pattern)]
    else:
        for index, value in zip(indices, boundary_values):
            x[index] = value
    upstream = (torch.randint(-8, 9, logical_shape, generator=generator).float() / 8)
    for offset, index in enumerate(indices[:len(boundary_values)]):
        upstream[index] = 1.0 if offset % 2 == 0 else -1.0
    return x.to(device), upstream.to(device)


def execute(fn, x, minimum, maximum, upstream, capture=False, allow_missing_grad=False):
    x = x.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
    upstream = upstream.detach().clone(memory_format=torch.preserve_format)
    watched = [x, upstream]
    before = [item.detach().contiguous().view(torch.uint8).clone() for item in watched]
    recorder = TritonLaunchRecorder(copy_tensors=False) if capture else None
    with recorder if recorder else contextlib.nullcontext():
        out = fn(x, minimum, maximum)
        forward_count = len(recorder.launches) if recorder else 0
        try:
            (dx,) = torch.autograd.grad(out, x, upstream, allow_unused=allow_missing_grad)
        except RuntimeError:
            if not allow_missing_grad:
                raise
            dx = None
    if x.is_cuda:
        torch.cuda.synchronize()
    if any(not torch.equal(item.detach().contiguous().view(torch.uint8), old)
           for item, old in zip(watched, before)):
        raise ValueError("input or upstream values changed")
    launches = [] if recorder is None else [
        {"phase": "forward" if index < forward_count else "backward",
         "kernel": launch.kernel_name, "grid": list(launch.grid), "ttir": launch.asm.get("ttir")}
        for index, launch in enumerate(recorder.launches)
    ]
    result = {"out": out.detach()}
    if dx is not None:
        result["dx"] = dx.detach()
    layout = {"input": list(x.stride()), "upstream": list(upstream.stride()),
              "out": list(out.stride()), "out_contiguous": out.is_contiguous(),
              "dx": None if dx is None else list(dx.stride())}
    return result, launches, layout


def mathematical_reference(x, minimum, maximum, upstream):
    query = {"op": "torch.clamp", "input": batch.record(x), "min": minimum,
             "max": maximum, "upstream": batch.record(upstream)}
    result = registry.query("ARITH", PROFILE, query)
    info = {"query": query, "outcome": result.outcome, "note": result.note,
            "labels": result.labels, "scope": "complete public forward and input VJP",
            "mapping": "torch.clamp(Tensor,Scalar?,Scalar?) -> aten.clamp.default",
            "output_mapping": {"out": "out", "grad": "dx"}}
    if result.outcome != "value":
        return info, None
    if set(result.fields) != {"out", "grad"}:
        raise ValueError("clamp reference omitted or added task outputs")
    expected = {"out": result.fields["out"], "dx": result.fields["grad"]}
    info["fields"] = {name: batch.map_values(field, encode_finite)
                      for name, field in expected.items()}
    return info, expected


def exact_math(observed, expected):
    if set(observed) - set(expected):
        raise ValueError("the candidate returned an undeclared task output")
    return {name: batch.reference_comparison(observed[name], expected[name]) for name in observed}


def math_exact(comparison):
    return all(item["outside_mathematical_enclosure"] == 0
               and item["inside_nonpoint_unresolved"] == 0
               and item["nonfinite"] == 0
               and item["exact_equal"] == item["coordinates"]
               for item in comparison.values())


def exact_standard(actual, standard):
    standard_subset = {name: standard[name] for name in actual}
    details = batch.standard_comparison(actual, standard_subset)
    equal = {name: bool(torch.equal(actual[name], standard[name])) for name in actual}
    return {"standard": details, "exact_equal": equal, "compatible": all(equal.values())}


def recorded(result):
    return {name: batch.record(value) for name, value in result.items()}


def coverage(result):
    expected = ["out", "dx"]
    observed = sorted(result)
    return {"expected": expected, "observed": observed,
            "missing": [name for name in expected if name not in result]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "acceptance"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection", type=Path,
                        default=Path(__file__).with_name("selection_clamp_20261019_v1.json"))
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    split = selection[args.split]
    args.out.mkdir(parents=True, exist_ok=False)
    if torch.__version__ != "2.10.0+cu128":
        raise ValueError("standard version mismatch")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")

    spec_root = Path(__file__).resolve().parents[2]
    report = {
        "selection": selection, "split": args.split, "torch": torch.__version__,
        "device": args.device, "samples": [], "formal_binding": "not_established",
        "bias_verdict": "not_established",
        "reference_protocol": ("PyTorch 2.10 public scalar-bound clamp forward and input VJP "
                               "cross-checked against an independent exact coordinate formula."),
        "sources": {"runner": Path(__file__).read_text(),
                    "comparison": Path(batch.__file__).read_text(),
                    "torch_clamp_documentation": torch.clamp.__doc__,
                    "aten_schema": str(torch.ops.aten.clamp.default._schema)},
        "specification_sources": {
            name: (spec_root / name).read_text()
            for name in ("contracts/ARITH.yaml", "adapters/ARITH.py")},
        "retained_official_sources": ["sources/T_CLAMP.txt.gz",
                                      "sources/TS_NATIVE_FUNCTIONS.yaml.gz",
                                      "sources/TS_AUTOGRAD_DERIVATIVES.yaml.gz"],
        "capture_setup": ("For CUDA candidates the recorder is enabled before warmup with no warmup launches "
                          "selected; static CUDA host launch is disabled and numerical precision is unchanged."),
    }

    candidate = context = None
    if args.device == "cuda" and split["implementation"] == "flag_gems":
        import flag_gems
        distribution = importlib.metadata.distribution("flag_gems")
        if distribution.version != "5.0.2":
            raise ValueError("candidate version mismatch")
        report["candidate_sources"] = {
            path: Path(distribution.locate_file("flag_gems/" + path)).read_text()
            for path in ("__init__.py", "ops/clamp.py")}
        report["candidate_metadata"] = distribution.read_text("METADATA")
        report["candidate_license"] = distribution.read_text("LICENSE")
        report["candidate_function_selection"] = ["clamp"]
        report["candidate_scope"] = ("FlagGems replaces aten.clamp forward. Any absent public input VJP "
                                     "is retained as an incomplete training task.")
        candidate = functional
        context = lambda: flag_gems.use_gems(include=["clamp"])
    elif args.device == "cuda" and split["implementation"] == "inductor":
        report["candidate_scope"] = "TorchInductor compiles the public forward and input VJP."
        candidate, context = torch.compile(functional, fullgraph=True, dynamic=True), contextlib.nullcontext

    device = torch.device(args.device)
    for case in split["cases"]:
        if args.device == "cuda" and split["implementation"] == "inductor":
            torch._dynamo.reset()
            candidate = torch.compile(functional, fullgraph=True, dynamic=True)
        for seed in split["seeds"]:
            row = {"case": case["id"], "seed": seed, "min": case["min"], "max": case["max"]}
            report["samples"].append(row)
            print("START", case["id"], seed, flush=True)
            try:
                x, upstream = inputs(case, seed, device)
                row.update(input=batch.record(x), upstream=batch.record(upstream),
                           original_input_stride=list(x.stride()))
                row["mathematical_reference"], expected = mathematical_reference(
                    x, case["min"], case["max"], upstream)
                if expected is None:
                    raise ValueError("selected mathematical reference is unavailable")
                standard, _, standard_layout = execute(
                    functional, x, case["min"], case["max"], upstream)
                row.update(standard=recorded(standard), standard_layout=standard_layout,
                           standard_math=exact_math(standard, expected))
                if not math_exact(row["standard_math"]):
                    raise ValueError("versioned standard differs from the selected exact reference")
                if args.device == "cpu":
                    row["status"] = "reference_crosschecked"
                else:
                    allow_missing_grad = split["implementation"] == "flag_gems"
                    with context():
                        with TritonLaunchRecorder(select=lambda name, index: False, copy_tensors=False):
                            execute(candidate, x, case["min"], case["max"], upstream,
                                    allow_missing_grad=allow_missing_grad)
                        actual, launches, layout = execute(
                            candidate, x, case["min"], case["max"], upstream, capture=True,
                            allow_missing_grad=allow_missing_grad)
                    task_coverage = coverage(actual)
                    row.update(candidate=recorded(actual), launches=launches,
                               observed_layout=layout, input_values_unchanged=True,
                               task_output_coverage=task_coverage,
                               comparison=exact_standard(actual, standard),
                               candidate_math=exact_math(actual, expected))
                    captured = sorted({launch["phase"] for launch in launches})
                    required = {"forward"} if split["implementation"] == "flag_gems" \
                        else {"forward", "backward"}
                    missing_phases = sorted(required - set(captured))
                    row["triton_phase_coverage"] = {
                        "expected": sorted(required), "captured": captured, "missing": missing_phases}
                    if not row["comparison"]["compatible"] or not math_exact(row["candidate_math"]):
                        row["status"] = "compared_mismatch"
                    elif task_coverage["missing"]:
                        row["status"] = "compared_partial_task"
                    else:
                        row["status"] = "compared" if not missing_phases else "compared_partial_triton"
            except Exception as error:
                row.update(status="execution_or_reference_error",
                           error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print("END", case["id"], seed, row["status"], row.get("error", ""), flush=True)

    expected_status = "reference_crosschecked" if args.device == "cpu" else "compared"
    raise SystemExit(int(any(row["status"] != expected_status for row in report["samples"])))


if __name__ == "__main__":
    main()
