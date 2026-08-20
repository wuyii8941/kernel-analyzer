#!/usr/bin/env python3
"""Exact precision-source decomposition for one compiler-bound external MM."""

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

from kernel_analyzer.precision import decompose_low_precision_output  # noqa: E402
from kernel_analyzer.streaming import StreamingGramAccumulator  # noqa: E402
from scripts.generated_contrast_observer import _source_identity  # noqa: E402
from scripts.generated_nontriton_fp32_observer import fp32_external_reference  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import (  # noqa: E402
    file_digest, gradient_digest, load_model, tensor_digest,
)


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class BoundMMDecomposition:
    def __init__(self, modules: list[Any], target_sha: str) -> None:
        self.modules = modules
        self.target_sha = target_sha
        self.restores: list[tuple[Any, Any]] = []
        self.calls = 0
        self.vectors: dict[str, torch.Tensor] = {}
        self.summary: dict[str, Any] | None = None

    def __enter__(self) -> "BoundMMDecomposition":
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
                if digest != self.target_sha:
                    return result
                actual = kwargs.get("out", result)
                if not isinstance(actual, torch.Tensor):
                    raise RuntimeError("target MM has no tensor output")
                high = fp32_external_reference("mm", args, kwargs)
                vectors = decompose_low_precision_output(actual, high)
                closure = vectors["kernel"] + vectors["output_rounding"] - vectors["total"]
                self.vectors = {name: value.cpu() for name, value in vectors.items()}
                self.summary = {
                    "coordinates": vectors["total"].numel(),
                    "closure_max_abs": float(closure.abs().max().item()),
                }
                for name, value in vectors.items():
                    self.summary[name + "_nonzero"] = int(torch.count_nonzero(value).item())
                    self.summary[name + "_l2"] = float(torch.linalg.vector_norm(value).item())
                    self.summary[name + "_max_abs"] = float(value.abs().max().item())
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
    parser.add_argument("--architecture", choices=("qwen", "mamba", "phi", "deepseek8"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--live-result", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--temp-root", type=Path,
        default=Path("/data1/tzh/cache/kernel_analyzer_contrasts/precision_decomposition"),
    )
    args = parser.parse_args()
    if args.states < 2:
        raise ValueError("direction requires at least two independent states")

    queue = json.loads((ROOT / "results/coverage/bias_candidate_queue.json").read_text())
    bound = next(row for row in queue["candidates"] if row["candidate_id"] == args.candidate_id)
    exact_call = bound["exact_generated_call"]
    if exact_call["function"] != "extern_kernels.mm" or exact_call["source_line_sha256"] != args.target_sha:
        raise RuntimeError("candidate ID does not bind the declared MM source identity")
    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))[:args.states]
    if len(states) != args.states:
        raise RuntimeError("input bank is shorter than the requested population")
    capture = json.loads((args.release_dir / "capture.json").read_text())
    if file_digest(args.input_bank) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("input bank does not match frozen release")
    live = json.loads(args.live_result.read_text())
    precision = next(row for row in live["results"]
                     if row["candidate_id"] == args.candidate_id
                     and row["contrast_axis"] == "PRECISION")
    if int(live["state_count"]) != args.states:
        raise RuntimeError("formal decomposition must use the frozen live population")

    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend="inductor",
        fullgraph=not bool(capture.get("allow_graph_breaks", False)), dynamic=False,
    )
    warm_tokens = states[0].get("token_ids", states[0].get("input_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    # The observer below must execute the exact source-line digest exactly
    # once per state.  Do not reject a scientifically identical target merely
    # because unrelated generated-wrapper bytes or cache metadata changed.

    spool = args.temp_root / hashlib.sha256(args.candidate_id.encode()).hexdigest()[:20]
    grams = {
        name: StreamingGramAccumulator(spool, args.candidate_id + "_" + name)
        for name in ("kernel", "output_rounding", "total")
    }
    rows: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        state_id = str(state.get("sequence_id", state.get("state_id", index)))
        tokens = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        seed = 24000 + index
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        baseline_loss = candidate(values); baseline_loss.backward(); torch.cuda.synchronize(device)
        baseline = endpoint(model, baseline_loss)

        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        observer = BoundMMDecomposition(modules, args.target_sha)
        with observer:
            observed_loss = candidate(values); observed_loss.backward()
        torch.cuda.synchronize(device)
        if endpoint(model, observed_loss) != baseline:
            raise RuntimeError("read-only decomposition changed endpoints")
        assert observer.summary is not None
        if observer.summary["closure_max_abs"] != 0.0:
            raise RuntimeError("precision decomposition failed exact FP32 closure")
        vector_rows = {
            name: grams[name].add_array(state_id, value.numpy())
            for name, value in observer.vectors.items()
        }
        rows.append({"state_id": state_id, "endpoint_identity": baseline,
                     "summary": observer.summary, "vectors": vector_rows})
        write(args.output, {"schema": "kernel-analyzer-precision-mechanism-decomposition-v1",
                            "status": "RUNNING", "candidate_id": args.candidate_id,
                            "states": rows})
        del values, observer
        torch.cuda.empty_cache()
        print(json.dumps({"event": "STATE_COMPLETE", "state": state_id}), flush=True)

    certificates = {
        name: accumulator.finalize(bootstrap_draws=4000, seed=16000 + offset, cleanup=True)
        for offset, (name, accumulator) in enumerate(grams.items())
    }
    expected_u = precision["direction"]["cross_state_inner_product_u"]
    matches_live = abs(certificates["total"]["cross_state_inner_product_u"] - expected_u) <= 1e-12
    coherent_sources = [name for name in ("kernel", "output_rounding")
                        if certificates[name]["status"] == "PASS"]
    payload = {
        "schema": "kernel-analyzer-precision-mechanism-decomposition-v1",
        "status": "COMPLETE_EXACT_PRECISION_MECHANISM_DECOMPOSITION",
        "candidate_id": args.candidate_id,
        "exact_generated_call": exact_call,
        "identity": (
            "actual_low - fp32_same_operands = "
            "(actual_low - bf16(fp32_same_operands)) + "
            "(bf16(fp32_same_operands) - fp32_same_operands)"
        ),
        "terms": {
            "kernel": "local generated-MM difference at fixed operands and low-precision ABI",
            "output_rounding": "deterministic FP32-to-declared-low-dtype output rounding",
            "total": "the exact precision contrast used by T1",
        },
        "states": rows, "direction": certificates,
        "gates": {"candidate_source_identity_exact": True,
                  "read_only_observer_exact": True,
                  "algebraic_closure_every_state": True,
                  "total_reproduces_frozen_live_t1_u": matches_live},
        "coherent_sources": coherent_sources,
        "release_capture_sha256": capture["result_sha256"],
        "live_result_sha256": live["result_sha256"],
        "input_bank_sha256": file_digest(args.input_bank),
        "claim_boundary": (
            "Exact same-operands local precision-source decomposition. Concrete AOT F+B "
            "proof and downstream trajectory are independent gates."
        ),
    }
    if not all(payload["gates"].values()):
        raise RuntimeError("formal decomposition failed a frozen binding gate")
    payload["result_sha256"] = canonical(payload)
    write(args.output, payload)
    print(json.dumps({"event": "DECOMPOSITION_COMPLETE",
                      "coherent_sources": coherent_sources}), flush=True)


if __name__ == "__main__":
    main()
