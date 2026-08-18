#!/usr/bin/env python3
"""Exact kernel-versus-output-rounding decomposition for Qwen seq128 v_proj."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.streaming import StreamingGramAccumulator  # noqa: E402
from kernel_analyzer.precision import decompose_low_precision_output  # noqa: E402
from scripts.generated_contrast_observer import _source_identity  # noqa: E402
from scripts.generated_nontriton_fp32_observer import fp32_external_reference  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import (  # noqa: E402
    file_digest, gradient_digest, load_model, tensor_digest,
)
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402


CID = "qwen_seq128_forward_8_output"
TARGET_SHA = "1847d6184bdf781a1b57531571a298c898337332aa694e47a96de633d30ed2af"


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class PrecisionDecomposition:
    """Observe one exact generated call without changing its delivered tensor."""

    def __init__(self, modules: list[Any]) -> None:
        self.modules = modules
        self.restores: list[tuple[Any, Any]] = []
        self.calls = 0
        self.vectors: dict[str, torch.Tensor] = {}
        self.summary: dict[str, Any] | None = None

    def __enter__(self) -> "PrecisionDecomposition":
        seen: set[int] = set()
        for module in self.modules:
            namespace = getattr(module, "extern_kernels", None)
            if namespace is None or id(namespace) in seen:
                continue
            seen.add(id(namespace))
            original = namespace.mm

            def wrapped(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
                _, _, digest = _source_identity()
                result = _original(*args, **kwargs)
                if digest != TARGET_SHA:
                    return result
                actual = kwargs.get("out", result)
                if not isinstance(actual, torch.Tensor):
                    raise RuntimeError("target MM has no tensor output")
                high = fp32_external_reference("mm", args, kwargs)
                vectors = decompose_low_precision_output(actual, high)
                kernel = vectors["kernel"]
                output_rounding = vectors["output_rounding"]
                total = vectors["total"]
                closure = kernel + output_rounding - total
                self.vectors = {
                    "kernel": kernel.cpu(),
                    "output_rounding": output_rounding.cpu(),
                    "total": total.cpu(),
                }
                self.summary = {
                    "coordinates": total.numel(),
                    "kernel_nonzero": int(torch.count_nonzero(kernel).item()),
                    "output_rounding_nonzero": int(torch.count_nonzero(output_rounding).item()),
                    "total_nonzero": int(torch.count_nonzero(total).item()),
                    "kernel_l2": float(torch.linalg.vector_norm(kernel).item()),
                    "output_rounding_l2": float(torch.linalg.vector_norm(output_rounding).item()),
                    "total_l2": float(torch.linalg.vector_norm(total).item()),
                    "kernel_max_abs": float(kernel.abs().max().item()),
                    "output_rounding_max_abs": float(output_rounding.abs().max().item()),
                    "total_max_abs": float(total.abs().max().item()),
                    "closure_max_abs": float(closure.abs().max().item()),
                }
                self.calls += 1
                return result

            namespace.mm = wrapped
            self.restores.append((namespace, original))
        return self

    def __exit__(self, *unused: Any) -> None:
        del unused
        for namespace, original in self.restores:
            namespace.mm = original
        if self.calls != 1 or self.summary is None:
            raise RuntimeError(f"target decomposition executed {self.calls} times")


def endpoint(model: torch.nn.Module, loss: torch.Tensor) -> dict[str, str]:
    return {"loss": tensor_digest(loss), "gradients": gradient_digest(model)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/coverage/cases/qwen128_vproj_precision_decomposition.json",
    )
    args = parser.parse_args()
    if args.states < 2:
        raise ValueError("direction requires at least two independent states")
    bank_path = ROOT / "results/coverage/qwen_seq128_input_bank.json"
    bank = json.loads(bank_path.read_text())
    states = bank.get("states", bank.get("records"))[:args.states]
    release = ROOT / "results/coverage/runtime_releases/qwen_seq128_r1"
    capture = json.loads((release / "capture.json").read_text())
    if file_digest(bank_path) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("input bank does not match frozen release")

    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model("qwen", Path("/data1/tzh/models/Qwen/Qwen3-1.7B"), device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm_tokens = states[0].get("token_ids", states[0].get("input_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    validate_release(wrapper_modules(modules), capture)

    spool = Path("/data1/tzh/cache/kernel_analyzer_contrasts/qwen128_vproj_decomposition")
    grams = {
        name: StreamingGramAccumulator(spool, CID + "_" + name)
        for name in ("kernel", "output_rounding", "total")
    }
    rows: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        state_id = str(state.get("sequence_id", state.get("state_id", index)))
        tokens = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        seed = 24000 + index

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        baseline_loss = candidate(values)
        baseline_loss.backward()
        torch.cuda.synchronize(device)
        baseline = endpoint(model, baseline_loss)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        observer = PrecisionDecomposition(modules)
        with observer:
            observed_loss = candidate(values)
            observed_loss.backward()
        torch.cuda.synchronize(device)
        observed = endpoint(model, observed_loss)
        if observed != baseline:
            raise RuntimeError("read-only decomposition changed endpoints")
        assert observer.summary is not None
        if observer.summary["closure_max_abs"] > 2.0 ** -20:
            raise RuntimeError("precision decomposition failed numerical closure")
        vector_rows = {
            name: grams[name].add_array(state_id, value.numpy())
            for name, value in observer.vectors.items()
        }
        rows.append({
            "state_id": state_id,
            "endpoint_identity": baseline,
            "summary": observer.summary,
            "vectors": vector_rows,
        })
        write(args.output, {
            "schema": "kernel-analyzer-precision-mechanism-decomposition-v1",
            "status": "RUNNING", "candidate_id": CID, "states": rows,
        })
        del values, observer
        torch.cuda.empty_cache()
        print(json.dumps({"event": "STATE_COMPLETE", "state": state_id}), flush=True)

    certificates = {
        name: accumulator.finalize(bootstrap_draws=4000, seed=15000 + offset, cleanup=True)
        for offset, (name, accumulator) in enumerate(grams.items())
    }
    total_matches_live = (
        abs(certificates["total"]["cross_state_inner_product_u"] - 6.681331077855302e-05)
        <= 1e-12
    )
    coherent_sources = [
        name for name in ("kernel", "output_rounding")
        if certificates[name]["status"] == "PASS"
    ]
    payload = {
        "schema": "kernel-analyzer-precision-mechanism-decomposition-v1",
        "status": "COMPLETE_EXACT_PRECISION_MECHANISM_DECOMPOSITION",
        "candidate_id": CID,
        "identity": (
            "actual_low - fp32_same_operands = "
            "(actual_low - bf16(fp32_same_operands)) + "
            "(bf16(fp32_same_operands) - fp32_same_operands)"
        ),
        "terms": {
            "kernel": "local generated-MM difference at fixed operands and BF16 ABI",
            "output_rounding": "deterministic FP32-to-BF16 output rounding",
            "total": "the exact precision contrast used by T1",
        },
        "states": rows,
        "direction": certificates,
        "gates": {
            "read_only_observer_exact": True,
            "algebraic_closure_every_state": True,
            "total_reproduces_frozen_live_t1_u": total_matches_live,
        },
        "coherent_sources": coherent_sources,
        "release_capture_sha256": capture["result_sha256"],
        "input_bank_sha256": file_digest(bank_path),
        "claim_boundary": (
            "Exact source decomposition of local same-operands precision error. It does not "
            "test inherited upstream operand error or live multi-step weight accumulation."
        ),
    }
    payload["result_sha256"] = canonical(payload)
    write(args.output, payload)
    print(json.dumps({
        "event": "DECOMPOSITION_COMPLETE", "coherent_sources": coherent_sources,
        "total_matches_live": total_matches_live,
    }), flush=True)


if __name__ == "__main__":
    main()
