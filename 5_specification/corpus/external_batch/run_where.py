"""Real public Tensor/Tensor where calls with complete data VJPs."""
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


PROFILE = "torch210_where_tensor_autograd"


def functional(condition, x, other):
    return torch.where(condition, x, other)


def data_tensor(shape, generator, device, noncontiguous=False):
    logical_shape = tuple(shape)
    stored_shape = (*logical_shape[:-2], logical_shape[-1], logical_shape[-2]) \
        if noncontiguous else logical_shape
    value = (torch.randint(-24, 25, stored_shape, generator=generator).float() / 8).to(device)
    return value.transpose(-1, -2) if noncontiguous else value


def inputs(case, seed, device):
    generator = torch.Generator().manual_seed(seed)
    x = data_tensor(case["input_shape"], generator, device, case.get("input_noncontiguous", False))
    other = data_tensor(case["other_shape"], generator, device, case.get("other_noncontiguous", False))
    condition_shape = tuple(case["condition_shape"])
    if case["condition"] == "all_true":
        condition = torch.ones(condition_shape, dtype=torch.bool, device=device)
    elif case["condition"] == "all_false":
        condition = torch.zeros(condition_shape, dtype=torch.bool, device=device)
    else:
        count = int(torch.tensor(condition_shape).prod())
        condition = ((torch.arange(count).reshape(condition_shape) + seed) % 3 != 1).to(device)
    output_shape = torch.broadcast_shapes(condition.shape, x.shape, other.shape)
    upstream = (torch.randint(-8, 9, output_shape, generator=generator).float() / 8).to(device)
    return condition, x, other, upstream


def execute(fn, condition, x, other, upstream, capture=False, allow_missing_grad=False):
    condition = condition.detach().clone(memory_format=torch.preserve_format)
    x = x.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
    other = other.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
    upstream = upstream.detach().clone(memory_format=torch.preserve_format)
    watched = [condition, x, other, upstream]
    before = [item.detach().contiguous().view(torch.uint8).clone() for item in watched]
    recorder = TritonLaunchRecorder(copy_tensors=False) if capture else None
    with recorder if recorder else contextlib.nullcontext():
        out = fn(condition, x, other)
        forward_count = len(recorder.launches) if recorder else 0
        dx, dother = torch.autograd.grad(out, (x, other), upstream,
                                         allow_unused=allow_missing_grad)
    if x.is_cuda:
        torch.cuda.synchronize()
    if any(not torch.equal(item.detach().contiguous().view(torch.uint8), old)
           for item, old in zip(watched, before)):
        raise ValueError("condition, data input or upstream values changed")
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
    layout = {"condition": list(condition.stride()), "input": list(x.stride()),
              "other": list(other.stride()), "out": list(out.stride()),
              "out_contiguous": out.is_contiguous()}
    return result, launches, layout


def mathematical_reference(condition, x, other, upstream):
    query = {"op": "torch.where", "condition": batch.record(condition), "input": batch.record(x),
             "other": batch.record(other), "upstream": batch.record(upstream)}
    result = registry.query("ARITH", PROFILE, query)
    info = {"query": query, "outcome": result.outcome, "note": result.note,
            "labels": result.labels, "scope": "complete public forward and both data VJPs",
            "mapping": "torch.where(Tensor,Tensor,Tensor) -> aten.where.self",
            "output_mapping": {"out": "out", "grad_input": "dx", "grad_other": "dother"}}
    if result.outcome != "value":
        return info, None
    if set(result.fields) != {"out", "grad_input", "grad_other"}:
        raise ValueError("where reference omitted or added task outputs")
    expected = {"out": result.fields["out"], "dx": result.fields["grad_input"],
                "dother": result.fields["grad_other"]}
    info["fields"] = {name: batch.map_values(field, encode_finite)
                      for name, field in expected.items()}
    return info, expected


def exact_math(observed, expected):
    if set(observed) != set(expected):
        raise ValueError("the compared task outputs do not match")
    return {name: batch.reference_comparison(observed[name], expected[name]) for name in observed}


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
                        default=Path(__file__).with_name("selection_where_20261019_v1.json"))
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
        "reference_protocol": ("PyTorch 2.10 public Tensor/Tensor where forward and both data VJPs "
                               "cross-checked against independent exact coordinate routing and reductions."),
        "sources": {"runner": Path(__file__).read_text(),
                    "comparison": Path(batch.__file__).read_text(),
                    "torch_where_documentation": torch.where.__doc__,
                    "aten_schema": str(torch.ops.aten.where.self._schema)},
        "specification_sources": {
            name: (spec_root / name).read_text()
            for name in ("contracts/ARITH.yaml", "adapters/ARITH.py")},
        "retained_official_sources": ["sources/T_WHERE.txt.gz",
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
            for path in ("__init__.py", "ops/where.py")}
        report["candidate_metadata"] = distribution.read_text("METADATA")
        report["candidate_license"] = distribution.read_text("LICENSE")
        report["candidate_function_selection"] = ["where_self"]
        report["candidate_scope"] = ("FlagGems replaces aten.where.self forward. Any absent public data VJP is "
                                     "retained as an incomplete training task.")
        candidate = functional
        context = lambda: flag_gems.use_gems(include=["where_self"])
    elif args.device == "cuda" and split["implementation"] == "inductor":
        report["candidate_scope"] = "TorchInductor compiles the public forward and both data VJPs."
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
                condition, x, other, upstream = inputs(case, seed, device)
                row.update(condition=batch.record(condition), input=batch.record(x),
                           other=batch.record(other), upstream=batch.record(upstream),
                           original_strides={"condition": list(condition.stride()),
                                             "input": list(x.stride()), "other": list(other.stride())})
                row["mathematical_reference"], expected = mathematical_reference(
                    condition, x, other, upstream)
                if expected is None:
                    raise ValueError("selected mathematical reference is unavailable")
                standard, _, standard_layout = execute(functional, condition, x, other, upstream)
                row.update(standard={name: batch.record(tensor) for name, tensor in standard.items()},
                           standard_layout=standard_layout, standard_math=exact_math(standard, expected))
                if not math_exact(row["standard_math"]):
                    raise ValueError("versioned standard differs from the selected exact reference")
                if args.device == "cpu":
                    row["status"] = "reference_crosschecked"
                else:
                    allow_missing_grad = split["implementation"] == "flag_gems"
                    with context():
                        with TritonLaunchRecorder(select=lambda name, index: False, copy_tensors=False):
                            execute(candidate, condition, x, other, upstream,
                                    allow_missing_grad=allow_missing_grad)
                        actual, launches, layout = execute(
                            candidate, condition, x, other, upstream, capture=True,
                            allow_missing_grad=allow_missing_grad)
                    observed_names, expected_names = set(actual), set(standard)
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
