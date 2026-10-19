"""Real public constant-pad calls with complete forward and input VJP."""
import argparse
import contextlib
from fractions import Fraction
import importlib.metadata
import json
from pathlib import Path
import traceback

import torch
import torch.nn.functional as F

import run as batch
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
from kernel_spec import registry
from kernel_spec.numeric_io import encode_finite


PROFILE = "torch210_functional_constant_pad_autograd"


def functional(x, pad, mode, value):
    return F.pad(x, pad, mode=mode, value=value)


def scalar(value):
    return None if value is None else float(Fraction(value))


def output_shape(input_shape, pad):
    result = list(input_shape)
    for pair in range(len(pad) // 2):
        axis = len(result) - 1 - pair
        result[axis] += pad[2 * pair] + pad[2 * pair + 1]
    return tuple(result)


def inputs(case, seed, device):
    generator = torch.Generator().manual_seed(seed)
    logical_shape = tuple(case["input_shape"])
    stored_shape = (*logical_shape[:-2], logical_shape[-1], logical_shape[-2]) \
        if case.get("noncontiguous") else logical_shape
    x = (torch.randint(-24, 25, stored_shape, generator=generator).float() / 8).to(device)
    if case.get("noncontiguous"):
        x = x.transpose(-1, -2)
    upstream = (torch.randint(-8, 9, output_shape(x.shape, case["pad"]),
                              generator=generator).float() / 8).to(device)
    return x, upstream, {"pad": list(case["pad"]), "mode": "constant",
                         "value": scalar(case["value"])}


def execute(fn, x, upstream, options, capture=False, allow_missing_grad=False):
    x = x.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
    upstream = upstream.detach().clone(memory_format=torch.preserve_format)
    watched = [x, upstream]
    before = [item.detach().contiguous().view(torch.uint8).clone() for item in watched]
    recorder = TritonLaunchRecorder(copy_tensors=False) if capture else None
    with recorder if recorder else contextlib.nullcontext():
        out = fn(x, **options)
        forward_count = len(recorder.launches) if recorder else 0
        dx, = torch.autograd.grad(out, x, upstream, allow_unused=allow_missing_grad)
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
    layout = {"input": list(x.stride()), "out": list(out.stride()),
              "out_contiguous": out.is_contiguous()}
    result = {"out": out.detach()}
    if dx is not None:
        result["dx"] = dx.detach()
    return result, launches, layout


def mathematical_reference(x, upstream, options):
    value = None if options["value"] is None else str(Fraction(options["value"]))
    query = {"op": "torch.nn.functional.pad", "input": batch.record(x),
             "pad": options["pad"], "mode": options["mode"], "value": value,
             "upstream": batch.record(upstream)}
    result = registry.query("LAYOUT", PROFILE, query)
    info = {"query": query, "outcome": result.outcome, "note": result.note,
            "labels": result.labels, "scope": "complete public forward and input VJP",
            "mapping": "torch.nn.functional.pad(mode='constant') -> aten.pad/aten.constant_pad_nd",
            "output_mapping": {"out": "out", "grad": "dx"}}
    if result.outcome != "value":
        return info, None
    if set(result.fields) != {"out", "grad"}:
        raise ValueError("constant-pad reference omitted or added task outputs")
    expected = {"out": result.fields["out"], "dx": result.fields["grad"]}
    info["fields"] = {name: batch.map_values(field, encode_finite)
                      for name, field in expected.items()}
    return info, expected


def exact_math(observed, expected):
    if set(observed) != set(expected):
        raise ValueError("the compared task outputs do not match")
    return {name: batch.reference_comparison(observed[name], expected[name])
            for name in observed}


def math_exact(comparison):
    return all(item["outside_mathematical_enclosure"] == 0
               and item["inside_nonpoint_unresolved"] == 0
               and item["nonfinite"] == 0
               and item["exact_equal"] == item["coordinates"]
               for item in comparison.values())


def exact_standard(actual, standard):
    details = batch.standard_comparison(actual, standard)
    equal = {name: bool(torch.equal(actual[name], standard[name])) for name in actual}
    return {"standard": details, "exact_equal": equal, "compatible": all(equal.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "acceptance"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection", type=Path,
                        default=Path(__file__).with_name("selection_constant_pad_20261019_v1.json"))
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
        "reference_protocol": ("PyTorch 2.10 public constant-pad forward and input VJP cross-checked "
                               "against an independent exact coordinate construction."),
        "sources": {"runner": Path(__file__).read_text(),
                    "comparison": Path(batch.__file__).read_text(),
                    "torch_functional_documentation": F.pad.__doc__,
                    "aten_pad_schema": str(torch.ops.aten.pad.default._schema),
                    "aten_constant_pad_schema": str(torch.ops.aten.constant_pad_nd.default._schema)},
        "specification_sources": {
            name: (spec_root / name).read_text()
            for name in ("contracts/LAYOUT.yaml", "adapters/LAYOUT.py")},
        "retained_official_sources": ["sources/T_PAD.txt.gz", "sources/TS_REFS.py.gz",
                                      "sources/TS_NATIVE_FUNCTIONS.yaml.gz"],
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
            for path in ("__init__.py", "ops/pad.py")}
        report["candidate_metadata"] = distribution.read_text("METADATA")
        report["candidate_license"] = distribution.read_text("LICENSE")
        report["candidate_function_selection"] = ["pad", "constant_pad_nd"]
        report["candidate_scope"] = ("FlagGems replaces the public forward pad path. The public input VJP is "
                                     "autograd context and is not counted as a FlagGems Triton backward.")
        candidate = functional
        context = lambda: flag_gems.use_gems(include=["pad", "constant_pad_nd"])
    elif args.device == "cuda" and split["implementation"] == "inductor":
        report["candidate_scope"] = "TorchInductor compiles the public forward and input VJP."
        candidate, context = torch.compile(functional, fullgraph=True, dynamic=True), contextlib.nullcontext

    device = torch.device(args.device)
    for case in split["cases"]:
        if args.device == "cuda" and split["implementation"] == "inductor":
            torch._dynamo.reset()
            candidate = torch.compile(functional, fullgraph=True, dynamic=True)
        for seed in split["seeds"]:
            row = {"case": case["id"], "seed": seed}
            report["samples"].append(row)
            print("START", case["id"], seed, flush=True)
            try:
                x, upstream, options = inputs(case, seed, device)
                row.update(options={**options, "value": case["value"]}, input=batch.record(x),
                           upstream=batch.record(upstream), original_input_strides=list(x.stride()))
                row["mathematical_reference"], expected = mathematical_reference(x, upstream, options)
                if expected is None:
                    raise ValueError("selected mathematical reference is unavailable")
                standard, _, standard_layout = execute(functional, x, upstream, options)
                row.update(standard={name: batch.record(tensor) for name, tensor in standard.items()},
                           standard_layout=standard_layout, standard_math=exact_math(standard, expected))
                if not math_exact(row["standard_math"]):
                    raise ValueError("versioned standard differs from the selected exact reference")
                if args.device == "cpu":
                    row["status"] = "reference_crosschecked"
                else:
                    with context():
                        allow_missing_grad = split["implementation"] == "flag_gems"
                        with TritonLaunchRecorder(select=lambda name, index: False, copy_tensors=False):
                            execute(candidate, x, upstream, options,
                                    allow_missing_grad=allow_missing_grad)
                        actual, launches, layout = execute(
                            candidate, x, upstream, options, capture=True,
                            allow_missing_grad=allow_missing_grad)
                    observed_names = set(actual)
                    expected_names = set(standard)
                    missing_outputs = sorted(expected_names - observed_names)
                    standard_subset = {name: standard[name] for name in actual}
                    reference_subset = {name: expected[name] for name in actual}
                    row.update(candidate={name: batch.record(tensor) for name, tensor in actual.items()},
                               launches=launches, observed_layout=layout, input_values_unchanged=True,
                               task_output_coverage={"expected": sorted(expected_names),
                                                     "observed": sorted(observed_names),
                                                     "missing": missing_outputs},
                               comparison=exact_standard(actual, standard_subset),
                               candidate_math=exact_math(actual, reference_subset))
                    captured = sorted({launch["phase"] for launch in launches})
                    required = {"forward"} if split["implementation"] == "flag_gems" \
                        else {"forward", "backward"}
                    missing = sorted(required - set(captured))
                    row["triton_phase_coverage"] = {
                        "expected": sorted(required), "captured": captured, "missing": missing}
                    if not row["comparison"]["compatible"] or not math_exact(row["candidate_math"]):
                        row["status"] = "compared_mismatch"
                    elif missing_outputs:
                        row["status"] = "compared_partial_task"
                    else:
                        row["status"] = "compared" if not missing else "compared_partial_triton"
            except Exception as error:
                row.update(status="execution_or_reference_error",
                           error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print("END", case["id"], seed, row["status"], row.get("error", ""), flush=True)

    expected_status = "reference_crosschecked" if args.device == "cpu" else "compared"
    raise SystemExit(int(any(row["status"] != expected_status for row in report["samples"])))


if __name__ == "__main__":
    main()
