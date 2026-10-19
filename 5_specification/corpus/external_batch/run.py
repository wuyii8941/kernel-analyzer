"""Run retained external public calls against two explicitly different references.

Results are development evidence, never automatic contract/binding admission.
Run development first; fix the harness before running the selected acceptance set.
Upstream repositories and pre-existing records are not modified.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "5_specification"))
from kernel_spec import registry
from kernel_spec.num import ends
from kernel_spec.numeric_io import encode_finite

SELECTION = Path(__file__).with_name("selection_20261019_v2.json")
LOSSES = {"mse": "MSELoss", "l1": "L1Loss", "smooth_l1": "SmoothL1Loss", "huber": "HuberLoss"}


def record(tensor):
    def finite_json(v):
        return v if math.isfinite(v) else str(v)
    return {"t": str(tensor.dtype).replace("torch.float32", "f32"), "shape": list(tensor.shape),
            "v": map_values(tensor.detach().cpu().tolist(), finite_json)}


def flatten(value):
    if isinstance(value, list):
        return [x for child in value for x in flatten(child)]
    return [value]


def reference_comparison(observed, expected, expected_dtype=torch.float32):
    """Keep exact endpoints; display floats never decide containment or equality."""
    def shape(value):
        if not isinstance(value, list):
            return ()
        child_shapes = [shape(x) for x in value]
        if child_shapes and any(s != child_shapes[0] for s in child_shapes):
            raise ValueError("ragged reference output")
        return (len(value),) + (child_shapes[0] if child_shapes else ())

    if observed.dtype != expected_dtype or tuple(observed.shape) != shape(expected):
        raise ValueError("candidate output shape/dtype differs from the selected task")
    actual = observed.detach().cpu().flatten().tolist()
    wanted = flatten(expected)
    if len(actual) != len(wanted):
        raise ValueError("incomplete reference output")
    outside = exact_equal = unresolved = nonfinite = 0
    residuals = []
    for value, reference in zip(actual, wanted):
        lo, hi = ends(reference)
        if not torch.isfinite(torch.tensor(value, dtype=torch.float64)).item():
            nonfinite += 1
            residuals.append(None)
            continue
        v = Fraction(value)
        outside += int(v < lo or v > hi)
        exact_equal += int(lo == hi == v)
        unresolved += int(lo != hi and lo <= v <= hi)
        residuals.append({"lower": str(v - hi), "upper": str(v - lo)})
    return {"coordinates": len(actual), "exact_equal": exact_equal,
            "outside_mathematical_enclosure": outside, "inside_nonpoint_unresolved": unresolved,
            "nonfinite": nonfinite, "residual_intervals": residuals,
            "meaning": "Pointwise arithmetic differences; not an API bug verdict or statistical bias test."}


def standard_comparison(candidate, standard):
    if set(candidate) != set(standard):
        raise ValueError("missing public output or gradient")
    result = {}
    for name, wanted in standard.items():
        value = candidate[name]
        if value.shape != wanted.shape or value.dtype != wanted.dtype:
            raise ValueError(f"{name}: output shape/dtype mismatch")
        delta = value.double() - wanted.double()
        finite = torch.isfinite(value) & torch.isfinite(wanted)
        result[name] = {
            "coordinates": value.numel(), "nonfinite_pairs": int((~finite).sum()),
            "unequal_coordinates": int((value != wanted).sum()),
            "max_abs_difference": float(delta[finite].abs().max()) if bool(finite.any()) else None,
            "mean_difference": float(delta[finite].mean()) if bool(finite.any()) else None,
            "allclose_rtol_1e-5_atol_1e-6": bool(torch.allclose(value, wanted, rtol=1e-5, atol=1e-6)),
        }
    return result


def make_inputs(op, shape, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-24, 25, shape, generator=g).float() / 8
    target = torch.randint(-24, 25, shape, generator=g).float() / 8 if op in LOSSES else None
    boundary = torch.tensor([0, .5, -.5, 2, -2])
    x.view(-1)[:5] = boundary
    if target is not None:
        target.view(-1)[:5] = 0
    return x.cuda(), target.cuda() if target is not None else None


def execute(fn, x, target, upstream):
    x = x.detach().clone().requires_grad_()
    args = [x]
    if target is not None:
        args.append(target.detach().clone().requires_grad_())
    before = [a.detach().clone() for a in args]
    y = fn(*args)
    g = torch.full_like(y, upstream) if y.ndim == 0 else (
        (torch.arange(y.numel(), device=y.device).reshape(y.shape) % 7 - 3).to(y.dtype) * (upstream / 4))
    grads = torch.autograd.grad(y, args, grad_outputs=g)
    torch.cuda.synchronize()
    if any(not torch.equal(a, b) for a, b in zip(args, before)):
        raise ValueError("public call modified a declared read-only input")
    out = {"out": y.detach(), "dx": grads[0].detach()}
    if target is not None:
        out["dtarget"] = grads[1].detach()
    return out, g


def loss_options(case):
    """One task selection for both public calls and the mathematical query."""
    op = case["op"]
    if "parameters" in case:
        options = case["parameters"]
        allowed = {"reduction"} | ({"beta"} if op == "smooth_l1" else {"delta"} if op == "huber" else set())
        if not isinstance(options, dict) or set(options) - allowed:
            raise ValueError("loss parameters are outside the selected public task")
        return dict(options)
    # Original immutable batch chose these values explicitly in its entry text.
    options = {"reduction": "mean"}
    if op in {"smooth_l1", "huber"}:
        options["beta" if op == "smooth_l1" else "delta"] = .5
    return options


def functions(case):
    library = importlib.import_module(case["library"])
    op = case["op"]
    if op in LOSSES:
        kwargs = loss_options(case)
        return getattr(library, LOSSES[op])(**kwargs), getattr(torch.nn, LOSSES[op])(**kwargs)
    if op == "softmax":
        external = library.Softmax(dim=-1) if case["library"] == "attorch" else lambda x: library.function.softmax(x, 1)
        return external, lambda x: F.softmax(x, dim=1)
    if op == "relu":
        return library.function.relu, torch.relu
    if op in {"sum", "mean"}:
        return lambda x: getattr(library.function, op)(x, 1), lambda x: getattr(torch, op)(x, dim=1)
    raise ValueError(f"no reviewed public task mapping for {op!r}")


def mathematical_queries(case, x, target, g):
    op, cid = case["op"], case["contract"]
    if op in {"smooth_l1", "huber"}:
        # Added after these public calls exposed the gap; retain the original
        # selection's null contract and record the actual reference per query.
        args = [record(x), record(target)]
        profile, operation = "torch210_unweighted_f32", f"aten.{op}_loss"
        kwargs = loss_options(case)
        return [("HUBER", profile, {"op": operation, "args": args, "kwargs": kwargs}, ["out"]),
                ("HUBER", profile, {"op": "vjp." + operation, "args": [record(g)] + args,
                                    "kwargs": kwargs}, ["dx", "dtarget"])], ""
    if cid is None:
        return [], "No admitted mathematical contract; standard implementation comparison remains available."
    if op in {"mse", "l1"}:
        args = [record(x), record(target)]
        kwargs = loss_options(case)
        profile = case.get("profile", "torch210_mean_f32_same_shape" if kwargs.get("reduction", "mean") == "mean"
                           else "torch210_unweighted_f32_same_shape")
        operation = f"aten.{op}_loss"
        return [(cid, profile, {"op": operation, "args": args, "kwargs": kwargs}, ["out"]),
                (cid, profile, {"op": "vjp." + operation, "args": [record(g)] + args,
                                "kwargs": kwargs}, ["dx", "dtarget"])], ""
    if op in {"sum", "mean"}:
        return [(cid, "real", {"op": "aten." + op, "args": [record(x)], "kwargs": {"dim": 1}}, ["out"]),
                (cid, "real", {"op": "vjp.aten." + op, "args": [record(g), record(x)], "kwargs": {"dim": 1}}, ["dx"])], ""
    if op == "relu":
        return [(cid, "torch", {"op": "aten.relu", "args": [record(x)]}, ["out"]),
                (cid, "torch", {"op": "aten.threshold_backward", "args": [record(g), record(x), 0]}, ["dx"])], ""
    if op == "softmax":
        return [(cid, "real", {"op": "aten._softmax", "args": [record(x), 1, False]}, ["out"]),
                (cid, "real", {"op": "vjp.aten.softmax", "args": [record(g), record(x), 1]}, ["dx"])], ""
    return [], f"No reviewed task mapping for operation {op!r}; a contract name alone is insufficient."


def run_case(case, selection):
    result = {**case, "formal_binding": "not_established", "bias_verdict": "not_established", "samples": []}
    try:
        candidate_fn, standard_fn = functions(case)
    except Exception as exc:
        result.update(status="import_error", error=f"{type(exc).__name__}: {exc}")
        return result
    for shape in selection["shapes"]:
        for seed, upstream in zip(selection["seeds"], selection["upstream"]):
            sample = {"shape": shape, "seed": seed, "upstream": upstream}
            result["samples"].append(sample)
            try:
                x, target = make_inputs(case["op"], shape, seed)
                sample["inputs"] = {"x": record(x), **({"target": record(target)} if target is not None else {})}
                standard, g = execute(standard_fn, x, target, upstream)
                sample["standard"] = {k: record(v) for k, v in standard.items()}
                queries, limit = mathematical_queries(case, x, target, g)
                sample["math_limit"] = limit
                references = []
                sample["mathematical"] = references
                evaluated = []
                for cid, profile, query, names in queries:
                    r = registry.query(cid, profile, query)
                    item = {"contract": cid, "profile": profile, "query": query, "outputs": names,
                            "outcome": r.outcome, "note": r.note}
                    references.append(item)
                    if r.outcome == "value":
                        values = r.fields["out"] if len(names) > 1 else [r.fields["out"]]
                        if len(values) != len(names):
                            raise ValueError("reference omitted a public output")
                        item["values"] = {n: map_values(v, encode_finite) for n, v in zip(names, values)}
                        item["standard_differences"] = {n: reference_comparison(standard[n], v) for n, v in zip(names, values)}
                        evaluated.append((item, names, values))
                candidate, _ = execute(candidate_fn, x, target, upstream)
                sample["candidate"] = {k: record(v) for k, v in candidate.items()}
                sample["standard_comparison"] = standard_comparison(candidate, standard)
                for item, names, values in evaluated:
                    item["candidate_differences"] = {n: reference_comparison(candidate[n], v) for n, v in zip(names, values)}
                sample["status"] = "compared"
            except Exception as exc:
                sample.update(status="execution_or_reference_error", error=f"{type(exc).__name__}: {exc}",
                              traceback=traceback.format_exc())
    result["status"] = "compared" if all(s["status"] == "compared" for s in result["samples"]) else "incomplete"
    return result


def map_values(value, fn):
    return [map_values(x, fn) for x in value] if isinstance(value, list) else fn(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "acceptance"], required=True)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    selection = json.loads(args.selection.read_text())
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (args.out / "runner.py").write_text(Path(__file__).read_text())
    import triton
    report = {"torch": torch.__version__, "triton": triton.__version__, "split": args.split,
              "selection": selection, "cases": [], "formal_admission": False}
    if torch.__version__ != "2.10.0+cu128":
        raise RuntimeError("selection fixes PyTorch 2.10.0+cu128")
    for name, repo in selection["repositories"].items():
        path = ROOT / repo["path"]
        commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if commit != repo["commit"] or subprocess.check_output(["git", "-C", str(path), "diff", "HEAD", "--"], text=True):
            raise RuntimeError(f"{name}: source revision differs from selection")
        sys.path.insert(0, str(path))
    for case in selection["cases"]:
        if case["split"] != args.split:
            continue
        print("START", case["id"], flush=True)
        result = run_case(case, selection)
        report["cases"].append(result)
        (args.out / (case["id"] + ".json")).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("END", case["id"], result["status"], result.get("error", ""), flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
