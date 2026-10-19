"""Real public Tensor-list cat calls with complete ordered input VJPs."""
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


PROFILE = "torch210_cat_autograd"


def functional(tensors, dim):
    return torch.cat(tensors, dim=dim)


def data_tensor(shape, generator, device, noncontiguous=False):
    logical_shape = tuple(shape)
    stored_shape = (*logical_shape[:-2], logical_shape[-1], logical_shape[-2]) \
        if noncontiguous else logical_shape
    count = 1
    for length in stored_shape:
        count *= length
    value = (torch.randint(-24, 25, stored_shape, generator=generator).float() / 8).to(device)
    if count == 0:
        value = torch.empty(stored_shape, dtype=torch.float32, device=device)
    return value.transpose(-1, -2) if noncontiguous else value


def inputs(case, seed, device):
    generator = torch.Generator().manual_seed(seed)
    flags = case.get("noncontiguous", [False] * len(case["shapes"]))
    tensors = [data_tensor(shape, generator, device, flag)
               for shape, flag in zip(case["shapes"], flags)]
    out = functional(tensors, case["dim"])
    upstream = (torch.randint(-8, 9, out.shape, generator=generator).float() / 8).to(device)
    return tensors, upstream


def execute(fn, tensors, dim, upstream, capture=False, allow_missing_grad=False):
    tensors = [tensor.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
               for tensor in tensors]
    upstream = upstream.detach().clone(memory_format=torch.preserve_format)
    watched = [*tensors, upstream]
    before = [item.detach().contiguous().view(torch.uint8).clone() for item in watched]
    recorder = TritonLaunchRecorder(copy_tensors=False) if capture else None
    with recorder if recorder else contextlib.nullcontext():
        out = fn(tensors, dim)
        forward_count = len(recorder.launches) if recorder else 0
        try:
            grads = list(torch.autograd.grad(out, tensors, upstream, allow_unused=allow_missing_grad))
        except RuntimeError:
            if not allow_missing_grad:
                raise
            grads = [None] * len(tensors)
    if out.is_cuda:
        torch.cuda.synchronize()
    if any(not torch.equal(item.detach().contiguous().view(torch.uint8), old)
           for item, old in zip(watched, before)):
        raise ValueError("an input tensor or upstream changed")
    launches = [] if recorder is None else [
        {"phase": "forward" if index < forward_count else "backward",
         "kernel": launch.kernel_name, "grid": list(launch.grid), "ttir": launch.asm.get("ttir")}
        for index, launch in enumerate(recorder.launches)
    ]
    layout = {"inputs": [list(tensor.stride()) for tensor in tensors],
              "out": list(out.stride()), "out_contiguous": out.is_contiguous(),
              "grads": [None if grad is None else list(grad.stride()) for grad in grads]}
    return {"out": out.detach(), "grads": [None if grad is None else grad.detach() for grad in grads]}, \
        launches, layout


def mathematical_reference(tensors, dim, upstream):
    query = {"op": "torch.cat", "tensors": [batch.record(tensor) for tensor in tensors],
             "dim": dim, "upstream": batch.record(upstream)}
    result = registry.query("LAYOUT", PROFILE, query)
    info = {"query": query, "outcome": result.outcome, "note": result.note,
            "labels": result.labels, "scope": "complete public forward and every ordered input VJP",
            "mapping": "torch.cat(Tensor[], dim) -> aten.cat.default",
            "output_mapping": {"out": "out", "grads": "one ordered gradient per input"}}
    if result.outcome != "value":
        return info, None
    if set(result.fields) != {"out", "grads"} or len(result.fields["grads"]) != len(tensors):
        raise ValueError("cat reference omitted or added task outputs")
    expected = {"out": result.fields["out"], "grads": result.fields["grads"]}
    info["fields"] = {
        "out": batch.map_values(expected["out"], encode_finite),
        "grads": [batch.map_values(grad, encode_finite) for grad in expected["grads"]],
    }
    return info, expected


def exact_math(observed, expected):
    comparison = {"out": batch.reference_comparison(observed["out"], expected["out"])}
    for index, grad in enumerate(observed["grads"]):
        if grad is not None:
            comparison[f"grad_{index}"] = batch.reference_comparison(grad, expected["grads"][index])
    return comparison


def math_exact(comparison):
    return all(item["outside_mathematical_enclosure"] == 0
               and item["inside_nonpoint_unresolved"] == 0
               and item["nonfinite"] == 0
               and item["exact_equal"] == item["coordinates"]
               for item in comparison.values())


def exact_standard(actual, standard):
    pairs = {"out": (actual["out"], standard["out"])}
    for index, grad in enumerate(actual["grads"]):
        if grad is not None:
            pairs[f"grad_{index}"] = (grad, standard["grads"][index])
    details, equal = {}, {}
    for name, (value, wanted) in pairs.items():
        details[name] = batch.standard_comparison({name: value}, {name: wanted})[name]
        equal[name] = bool(torch.equal(value, wanted))
    return {"standard": details, "exact_equal": equal, "compatible": all(equal.values())}


def recorded(result):
    return {"out": batch.record(result["out"]),
            "grads": [None if grad is None else batch.record(grad) for grad in result["grads"]]}


def coverage(result):
    expected = ["out", *[f"grad_{index}" for index in range(len(result["grads"]))]]
    observed = ["out", *[f"grad_{index}" for index, grad in enumerate(result["grads"])
                          if grad is not None]]
    return {"expected": expected, "observed": observed,
            "missing": [name for name in expected if name not in observed]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "acceptance"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection", type=Path,
                        default=Path(__file__).with_name("selection_cat_20261019_v1.json"))
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
        "reference_protocol": ("PyTorch 2.10 public cat forward and all input VJPs cross-checked "
                               "against an independent exact coordinate copy and upstream partition."),
        "sources": {"runner": Path(__file__).read_text(),
                    "comparison": Path(batch.__file__).read_text(),
                    "torch_cat_documentation": torch.cat.__doc__,
                    "aten_schema": str(torch.ops.aten.cat.default._schema)},
        "specification_sources": {
            name: (spec_root / name).read_text()
            for name in ("contracts/LAYOUT.yaml", "adapters/LAYOUT.py")},
        "retained_official_sources": ["sources/T_CAT.txt.gz",
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
            for path in ("__init__.py", "ops/cat.py")}
        report["candidate_metadata"] = distribution.read_text("METADATA")
        report["candidate_license"] = distribution.read_text("LICENSE")
        report["candidate_function_selection"] = ["cat"]
        report["candidate_scope"] = ("FlagGems replaces aten.cat forward. Every unavailable public input VJP "
                                     "is retained as an incomplete training task.")
        candidate = functional
        context = lambda: flag_gems.use_gems(include=["cat"])
    elif args.device == "cuda" and split["implementation"] == "inductor":
        report["candidate_scope"] = ("TorchInductor compiles the public cat task. Cat VJPs may be graph slices "
                                     "and do not require a separate Triton launch.")
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
                tensors, upstream = inputs(case, seed, device)
                row.update(dim=case["dim"], tensors=[batch.record(tensor) for tensor in tensors],
                           upstream=batch.record(upstream),
                           original_input_strides=[list(tensor.stride()) for tensor in tensors])
                row["mathematical_reference"], expected = mathematical_reference(
                    tensors, case["dim"], upstream)
                if expected is None:
                    raise ValueError("selected mathematical reference is unavailable")
                standard, _, standard_layout = execute(functional, tensors, case["dim"], upstream)
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
                            execute(candidate, tensors, case["dim"], upstream,
                                    allow_missing_grad=allow_missing_grad)
                        actual, launches, layout = execute(
                            candidate, tensors, case["dim"], upstream, capture=True,
                            allow_missing_grad=allow_missing_grad)
                    task_coverage = coverage(actual)
                    row.update(candidate=recorded(actual), launches=launches,
                               observed_layout=layout, input_values_unchanged=True,
                               task_output_coverage=task_coverage,
                               comparison=exact_standard(actual, standard),
                               candidate_math=exact_math(actual, expected))
                    captured = sorted({launch["phase"] for launch in launches})
                    missing_phases = sorted({"forward"} - set(captured))
                    row["triton_phase_coverage"] = {
                        "expected": ["forward"], "captured": captured, "missing": missing_phases,
                        "note": "cat input VJPs are upstream slices and need no arithmetic kernel"}
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
