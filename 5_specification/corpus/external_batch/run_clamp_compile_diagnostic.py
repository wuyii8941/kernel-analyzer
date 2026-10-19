"""Minimal uncaptured check for scalar-bound reuse in torch.compile clamp."""
import argparse
import gc
import json
import os
from pathlib import Path

import torch


def functional(x, minimum, maximum):
    return torch.clamp(x, min=minimum, max=maximum)


def make_input(shape, seed, minimum, maximum):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randint(-24, 25, shape, generator=generator).float() / 8
    x.view(-1)[:7] = torch.tensor([
        minimum, minimum - 0.5, minimum + 0.5,
        maximum, maximum + 0.5, maximum - 0.5,
        (minimum + maximum) / 2,
    ])
    upstream = torch.randint(-8, 9, shape, generator=generator).float() / 8
    upstream.view(-1)[:7] = torch.tensor([1., -1., 1., -1., 1., -1., 1.])
    return x.cuda(), upstream.cuda()


def execute(fn, x, upstream, minimum, maximum):
    x = x.detach().clone().requires_grad_(True)
    out = fn(x, minimum, maximum)
    (dx,) = torch.autograd.grad(out, x, upstream)
    torch.cuda.synchronize()
    return out.detach(), dx.detach()


def record(tensor):
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "values": tensor.cpu().tolist()}


def compare(actual, expected):
    difference = actual.double() - expected.double()
    unequal = actual != expected
    return {"equal": bool(torch.equal(actual, expected)),
            "unequal_coordinates": int(unequal.sum()),
            "max_abs_difference": float(difference.abs().max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["fractional_only", "equal_then_fractional"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if torch.__version__ != "2.10.0+cu128" or not torch.cuda.is_available():
        raise RuntimeError("this diagnostic fixes PyTorch 2.10.0+cu128 CUDA")

    report = {"scenario": args.scenario, "torch": torch.__version__, "dynamic": True,
              "capture_or_analysis_tool_loaded": False,
              "target": {"shape": [5, 23], "seed": 20261923, "min": -0.75, "max": 1.25}}
    if args.scenario == "equal_then_fractional":
        first_x, first_upstream = make_input((3, 19), 20261923, 0.25, 0.25)
        torch._dynamo.reset()
        first_compiled = torch.compile(functional, fullgraph=True, dynamic=True)
        first_eager = execute(functional, first_x, first_upstream, 0.25, 0.25)
        first_actual = execute(first_compiled, first_x, first_upstream, 0.25, 0.25)
        report["predecessor"] = {
            "shape": [3, 19], "min": 0.25, "max": 0.25,
            "out": compare(first_actual[0], first_eager[0]),
            "dx": compare(first_actual[1], first_eager[1]),
        }
        del first_compiled
        gc.collect()

    x, upstream = make_input((5, 23), 20261923, -0.75, 1.25)
    eager = execute(functional, x, upstream, -0.75, 1.25)
    torch._dynamo.reset()
    compiled = torch.compile(functional, fullgraph=True, dynamic=True)
    actual = execute(compiled, x, upstream, -0.75, 1.25)
    report["target"].update({
        "input": record(x), "upstream": record(upstream),
        "eager": {"out": record(eager[0]), "dx": record(eager[1])},
        "compiled": {"out": record(actual[0]), "dx": record(actual[1])},
        "comparison": {"out": compare(actual[0], eager[0]),
                       "dx": compare(actual[1], eager[1])},
    })
    cache = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    report["generated_python"] = {
        str(path.relative_to(cache)): path.read_text()
        for path in sorted(cache.rglob("*.py")) if "clamp" in path.read_text()
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    raise SystemExit(int(not all(item["equal"] for item in report["target"]["comparison"].values())))


if __name__ == "__main__":
    main()
