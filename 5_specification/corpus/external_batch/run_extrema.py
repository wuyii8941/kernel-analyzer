"""Real public Tensor/Tensor maximum and minimum calls with both complete VJPs."""
import argparse
import contextlib
import importlib.metadata
import json
from pathlib import Path
import traceback

import torch

import run as batch
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
from kernel_spec import registry
from kernel_spec.numeric_io import encode_finite


PROFILE = "torch210_extrema_tensor_autograd"


def maximum(x, other):
    return torch.maximum(x, other)


def minimum(x, other):
    return torch.minimum(x, other)


FUNCTIONS = {"maximum": maximum, "minimum": minimum}


def data_tensor(shape, generator, device, noncontiguous=False):
    logical_shape = tuple(shape)
    stored_shape = (*logical_shape[:-2], logical_shape[-1], logical_shape[-2]) \
        if noncontiguous else logical_shape
    value = torch.randint(-24, 25, stored_shape, generator=generator).float() / 8
    value = value.transpose(-1, -2) if noncontiguous else value
    return value.to(device)


def broadcast_index(output_index, input_shape, output_shape):
    leading = len(output_shape) - len(input_shape)
    return tuple(0 if length == 1 else output_index[leading + axis]
                 for axis, length in enumerate(input_shape))


def inputs(case, seed, device):
    generator = torch.Generator().manual_seed(seed)
    x = data_tensor(case["input_shape"], generator, device,
                    case.get("input_noncontiguous", False))
    other = data_tensor(case["other_shape"], generator, device,
                        case.get("other_noncontiguous", False))
    output_shape = torch.broadcast_shapes(x.shape, other.shape)
    if case.get("all_equal"):
        other.copy_(x)
        tie_indices = list(torch.cartesian_prod(*[torch.arange(length) for length in output_shape]))
        tie_indices = [tuple(int(value) for value in index) for index in tie_indices]
    else:
        zero = tuple(0 for _ in output_shape)
        last = tuple(length - 1 for length in output_shape)
        middle = tuple(length // 2 for length in output_shape)
        tie_indices = [zero, middle, last]
        for index, value in zip(tie_indices, (-0.5, 0.0, 0.5)):
            x[broadcast_index(index, x.shape, output_shape)] = value
            other[broadcast_index(index, other.shape, output_shape)] = value
    upstream = (torch.randint(-8, 9, output_shape, generator=generator).float() / 8).to(device)
    for offset, index in enumerate(tie_indices):
        upstream[index] = 2.0 if offset % 2 == 0 else -2.0
    return x, other, upstream


def execute(fn, x, other, upstream, capture=False, allow_missing_grad=False):
    x = x.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
    other = other.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
    upstream = upstream.detach().clone(memory_format=torch.preserve_format)
    watched = [x, other, upstream]
    before = [item.detach().contiguous().view(torch.uint8).clone() for item in watched]
    recorder = TritonLaunchRecorder(copy_tensors=False) if capture else None
    with recorder if recorder else contextlib.nullcontext():
        out = fn(x, other)
        forward_count = len(recorder.launches) if recorder else 0
        try:
            dx, dother = torch.autograd.grad(out, (x, other), upstream,
                                             allow_unused=allow_missing_grad)
        except RuntimeError:
            if not allow_missing_grad:
                raise
            dx = dother = None
    if x.is_cuda:
        torch.cuda.synchronize()
    if any(not torch.equal(item.detach().contiguous().view(torch.uint8), old)
           for item, old in zip(watched, before)):
        raise ValueError("input, other or upstream values changed")
    launches = [] if recorder is None else [
        {"phase": "forward" if index < forward_count else "backward",
         "kernel": launch.kernel_name, "grid": list(launch.grid), "ttir": launch.asm.get("ttir")}
        for index, launch in enumerate(recorder.launches)
    ]
    result = {"out": out.detach()}
    if dx is not None:
        result["dx"] = dx.detach()
    if dother is not None:
        result["dother"] = dother.detach()
    layout = {"input": list(x.stride()), "other": list(other.stride()),
              "upstream": list(upstream.stride()), "out": list(out.stride()),
              "out_contiguous": out.is_contiguous(),
              "dx": None if dx is None else list(dx.stride()),
              "dother": None if dother is None else list(dother.stride())}
    return result, launches, layout


def mathematical_reference(op, x, other, upstream):
    query = {"op": "torch." + op, "input": batch.record(x), "other": batch.record(other),
             "upstream": batch.record(upstream)}
    result = registry.query("ARITH", PROFILE, query)
    info = {"query": query, "outcome": result.outcome, "note": result.note,
            "labels": result.labels, "scope": "complete public forward and both operand VJPs",
            "mapping": f"torch.{op}(Tensor,Tensor) -> aten.{op}.default",
            "output_mapping": {"out": "out", "grad_input": "dx", "grad_other": "dother"}}
    if result.outcome != "value":
        return info, None
    if set(result.fields) != {"out", "grad_input", "grad_other"}:
        raise ValueError("maximum/minimum reference omitted or added task outputs")
    expected = {"out": result.fields["out"], "dx": result.fields["grad_input"],
                "dother": result.fields["grad_other"]}
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
    expected = ["out", "dx", "dother"]
    observed = sorted(result)
    return {"expected": expected, "observed": observed,
            "missing": [name for name in expected if name not in result]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "acceptance"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection", type=Path,
                        default=Path(__file__).with_name("selection_extrema_20261019_v1.json"))
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
        "reference_protocol": ("PyTorch 2.10 public Tensor/Tensor maximum and minimum forward and both VJPs "
                               "cross-checked against an independent exact coordinate formula."),
        "sources": {"runner": Path(__file__).read_text(),
                    "comparison": Path(batch.__file__).read_text(),
                    "torch_maximum_documentation": torch.maximum.__doc__,
                    "torch_minimum_documentation": torch.minimum.__doc__,
                    "aten_maximum_schema": str(torch.ops.aten.maximum.default._schema),
                    "aten_minimum_schema": str(torch.ops.aten.minimum.default._schema)},
        "specification_sources": {
            name: (spec_root / name).read_text()
            for name in ("contracts/ARITH.yaml", "adapters/ARITH.py")},
        "retained_official_sources": ["sources/T_MAXIMUM.txt.gz", "sources/T_MINIMUM.txt.gz",
                                      "sources/TS_NATIVE_FUNCTIONS.yaml.gz",
                                      "sources/TS_AUTOGRAD_DERIVATIVES.yaml.gz"],
        "capture_setup": ("For CUDA candidates the recorder is enabled before warmup with no warmup launches "
                          "selected; static CUDA host launch is disabled and numerical precision is unchanged."),
    }

    flag_gems = distribution = None
    if args.device == "cuda" and split["implementation"] == "flag_gems":
        import flag_gems
        distribution = importlib.metadata.distribution("flag_gems")
        if distribution.version != "5.0.2":
            raise ValueError("candidate version mismatch")
        report["candidate_sources"] = {
            path: Path(distribution.locate_file("flag_gems/" + path)).read_text()
            for path in ("__init__.py", "ops/maximum.py", "ops/minimum.py")}
        report["candidate_metadata"] = distribution.read_text("METADATA")
        report["candidate_license"] = distribution.read_text("LICENSE")
        report["candidate_function_selection"] = ["maximum", "minimum"]
        report["candidate_scope"] = ("FlagGems replaces each selected forward. Any absent public data VJP "
                                     "is retained as an incomplete training task.")
    elif args.device == "cuda" and split["implementation"] == "inductor":
        report["candidate_scope"] = "TorchInductor compiles each public forward and both VJPs."

    device = torch.device(args.device)
    for case in split["cases"]:
        fn = FUNCTIONS[case["op"]]
        if args.device == "cuda" and split["implementation"] == "flag_gems":
            candidate = fn
            context = lambda op=case["op"]: flag_gems.use_gems(include=[op])
        elif args.device == "cuda" and split["implementation"] == "inductor":
            torch._dynamo.reset()
            candidate = torch.compile(fn, fullgraph=True, dynamic=True)
            context = contextlib.nullcontext
        for seed in split["seeds"]:
            row = {"case": case["id"], "seed": seed, "op": case["op"]}
            report["samples"].append(row)
            print("START", case["id"], seed, flush=True)
            try:
                x, other, upstream = inputs(case, seed, device)
                row.update(input=batch.record(x), other=batch.record(other),
                           upstream=batch.record(upstream),
                           original_strides={"input": list(x.stride()), "other": list(other.stride())})
                row["mathematical_reference"], expected = mathematical_reference(
                    case["op"], x, other, upstream)
                if expected is None:
                    raise ValueError("selected mathematical reference is unavailable")
                standard, _, standard_layout = execute(fn, x, other, upstream)
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
                            execute(candidate, x, other, upstream,
                                    allow_missing_grad=allow_missing_grad)
                        actual, launches, layout = execute(
                            candidate, x, other, upstream, capture=True,
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
