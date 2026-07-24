#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


ISSUES = {
    "layernorm_reciprocal": {
        "issue": "pytorch/pytorch#186577",
        "url": "https://github.com/pytorch/pytorch/issues/186577",
        "contract": "Inductor must preserve F.layer_norm followed by reciprocal; eager and aot_eager are references.",
    },
    "expanded_index_add": {
        "issue": "pytorch/pytorch#183986",
        "url": "https://github.com/pytorch/pytorch/issues/183986",
        "contract": "Inductor must materialize stride-0 expansion semantics before index_add writes.",
    },
}


def tensor_sha256(tensor: Any) -> str:
    values = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def output_summary(torch: Any, tensor: Any) -> dict[str, Any]:
    values = tensor.detach().float().flatten()
    k = min(16, values.numel())
    top_indices = torch.topk(values, k=k).indices.cpu().tolist()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": tensor_sha256(tensor),
        "finite": bool(torch.isfinite(values).all().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "argmax_flat_index": int(values.argmax().item()),
        "top16_flat_indices": [int(value) for value in top_indices],
        "threshold_counts": {
            str(threshold): int((values > threshold).sum().item())
            for threshold in [0.5, 1.0, 2.0, 10.0]
        },
    }


def comparison(torch: Any, reference: Any, output: Any) -> dict[str, Any]:
    delta = (reference.float() - output.float()).abs()
    ref_summary = output_summary(torch, reference)
    out_summary = output_summary(torch, output)
    return {
        "max_abs_delta": float(delta.max().item()),
        "mean_abs_delta": float(delta.mean().item()),
        "argmax_fork": ref_summary["argmax_flat_index"] != out_summary["argmax_flat_index"],
        "top16_candidate_set_fork": set(ref_summary["top16_flat_indices"]) != set(out_summary["top16_flat_indices"]),
        "threshold_count_forks": {
            key: ref_summary["threshold_counts"][key] != out_summary["threshold_counts"][key]
            for key in ref_summary["threshold_counts"]
        },
    }


def layernorm_reciprocal(torch: Any) -> tuple[Any, Callable[[Any], Any]]:
    import torch.nn.functional as F

    x = torch.randn(64, 128, device="cuda")

    def fn(value: Any) -> Any:
        return torch.reciprocal(torch.abs(F.layer_norm(value, [128])) + 1e-6)

    return x, fn


def expanded_index_add(torch: Any) -> tuple[tuple[Any, Any], Callable[..., Any]]:
    source = torch.ones(3, 8, device="cuda")
    indices = torch.tensor([0, 1, 0], device="cuda")

    def fn(src: Any, idx: Any) -> Any:
        expanded = torch.zeros(1, 8, device="cuda").expand(4, -1)
        return expanded.index_add(0, idx, src)

    return (source, indices), fn


def invoke(fn: Callable[..., Any], inputs: Any) -> Any:
    return fn(*inputs) if isinstance(inputs, tuple) else fn(inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="One independent replay of an upstream historical wrong-result issue.")
    parser.add_argument("--case", choices=sorted(ISSUES), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    import torch

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    if args.case == "layernorm_reciprocal":
        inputs, fn = layernorm_reciprocal(torch)
    else:
        inputs, fn = expanded_index_add(torch)
    eager = invoke(fn, inputs)
    torch._dynamo.reset()
    aot = invoke(torch.compile(fn, backend="aot_eager"), inputs)
    torch._dynamo.reset()
    compiled = None
    inductor_exception = None
    try:
        compiled = invoke(torch.compile(fn, backend="inductor"), inputs)
    except Exception as error:  # The fixed expanded-write case intentionally fails closed.
        inductor_exception = {
            "type": type(error).__name__,
            "message": str(error),
        }
    torch.cuda.synchronize()
    eager_aot = comparison(torch, eager, aot)
    eager_inductor = comparison(torch, eager, compiled) if compiled is not None else None
    issue = ISSUES[args.case]
    payload = {
        "schema_version": "forkcert.historical_bug_replay_once.v1",
        "case": args.case,
        **issue,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "outputs": {
            "eager": output_summary(torch, eager),
            "aot_eager": output_summary(torch, aot),
            **({"inductor": output_summary(torch, compiled)} if compiled is not None else {}),
        },
        "comparisons": {
            "eager_vs_aot_eager": eager_aot,
            "eager_vs_inductor": eager_inductor,
        },
        "inductor_exception": inductor_exception,
        "upstream_wrong_result_reproduced": (
            eager_aot["max_abs_delta"] <= 1e-6
            and eager_inductor is not None
            and eager_inductor["max_abs_delta"] > 1e-2
        ),
        "upstream_bug_fail_closed": eager_aot["max_abs_delta"] <= 1e-6 and inductor_exception is not None,
        "claim_scope": (
            "Direct execution of the upstream issue reproducer. A successful replay validates a known historical/"
            "upstream bug on this environment; it is not a newly discovered bug or yet a training-semantic fork."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "outputs"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
