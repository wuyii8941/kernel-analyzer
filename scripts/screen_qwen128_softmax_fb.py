#!/usr/bin/env python3
"""Typed-reference directional screen for one exact attention-softmax F+B region."""

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
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import (  # noqa: E402
    file_digest, gradient_digest, load_model, tensor_digest,
)
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402


CID = "qwen_seq128_layer27_attention_softmax_fb"
SYMBOL = "triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8"
TARGET_SHA = "536265df506aecc767de55c88596a5fd8dff95629bed91d4b97aabd3457fc0e8"
FORWARD_SYMBOL = (
    "triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_"
    "lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_7"
)
FORWARD_SHA = "2967373ee293b3dfbc79aa01411dd71afe1615f4fc3267c1d4cf8f1a9d286179"
ALPHA = 0.08838834764831845


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu()
    return hashlib.sha256(array.view(torch.uint8).numpy().tobytes()).hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class SoftmaxFBObserver:
    def __init__(self, modules: list[Any]) -> None:
        kernels = []
        forward_kernels = []
        for module in modules:
            kernel = getattr(module, SYMBOL, None)
            if kernel is not None and all(id(kernel) != id(other) for other in kernels):
                kernels.append(kernel)
            forward_kernel = getattr(module, FORWARD_SYMBOL, None)
            if forward_kernel is not None and all(
                id(forward_kernel) != id(other) for other in forward_kernels
            ):
                forward_kernels.append(forward_kernel)
        if len(kernels) != 1:
            raise RuntimeError(f"expected one warmed softmax kernel object, got {len(kernels)}")
        if len(forward_kernels) != 1:
            raise RuntimeError(
                f"expected one warmed forward-softmax kernel object, got {len(forward_kernels)}"
            )
        self.kernel = kernels[0]
        self.forward_kernel = forward_kernels[0]
        self.had_run = "run" in vars(self.kernel)
        self.previous = vars(self.kernel).get("run")
        self.original = self.kernel.run
        self.forward_had_run = "run" in vars(self.forward_kernel)
        self.forward_previous = vars(self.forward_kernel).get("run")
        self.forward_original = self.forward_kernel.run
        self.calls = 0
        self.forward_calls = 0
        self.forward_probability: torch.Tensor | None = None
        self.forward_probability_low: torch.Tensor | None = None
        self.forward_summary: dict[str, Any] | None = None
        self.vectors: dict[str, torch.Tensor] = {}
        self.summary: dict[str, Any] | None = None

    def __enter__(self) -> "SoftmaxFBObserver":
        def forward_wrapped(*args: Any, **kwargs: Any) -> Any:
            _, _, digest = _source_identity()
            if digest != FORWARD_SHA:
                return self.forward_original(*args, **kwargs)
            if len(args) < 5:
                raise RuntimeError("forward softmax region lacks declared pointer operands")
            scores, token_ids, row_max, row_sum, probability_low = args[:5]
            expected = ((1, 16, 128, 128), (1, 128),
                        (1, 16, 128, 1), (1, 16, 128, 1),
                        (1, 16, 128, 128))
            shapes = tuple(tuple(value.shape) for value in
                           (scores, token_ids, row_max, row_sum, probability_low))
            if shapes != expected:
                raise RuntimeError(f"forward softmax operand drift: {shapes}")
            raw_scores = scores.detach().clone().float()
            ids = token_ids.detach().reshape(-1)
            positions = torch.arange(128, device=scores.device)
            valid = ((positions[None, :] <= positions[:, None])
                     & (ids[None, :] == ids[:, None]))
            mask = torch.where(
                valid,
                torch.zeros((), device=scores.device, dtype=torch.float32),
                torch.full((), -3.3895313892515355e38,
                           device=scores.device, dtype=torch.float32),
            )
            logits = raw_scores * ALPHA + mask.reshape(1, 1, 128, 128)
            semantic_probability = torch.softmax(logits, dim=-1)
            result = self.forward_original(*args, **kwargs)
            actual_probability = probability_low.detach().float()
            rounded_probability = semantic_probability.to(probability_low.dtype).float()
            self.forward_probability = semantic_probability
            self.forward_probability_low = actual_probability
            self.forward_summary = {
                "coordinates": semantic_probability.numel(),
                "generated_probability_vs_rounded_semantic_max_abs": float(
                    (actual_probability - rounded_probability).abs().max().item()
                ),
                "semantic_probability_row_sum_max_error": float(
                    (semantic_probability.sum(dim=-1) - 1.0).abs().max().item()
                ),
            }
            self.forward_calls += 1
            return result

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            _, _, digest = _source_identity()
            if digest != TARGET_SHA:
                return self.original(*args, **kwargs)
            if len(args) < 4:
                raise RuntimeError("softmax region lacks declared pointer operands")
            destination, upstream, row_max, row_sum = args[:4]
            expected = ((1, 16, 128, 128), (16, 128, 128),
                        (1, 16, 128, 1), (1, 16, 128, 1))
            shapes = tuple(tuple(value.shape) for value in
                           (destination, upstream, row_max, row_sum))
            if shapes != expected:
                raise RuntimeError(f"softmax region operand drift: {shapes}")
            logits = destination.detach().clone().float()
            gradient = upstream.detach().float().reshape(1, 16, 128, 128)
            maximum = row_max.detach().float()
            denominator = row_sum.detach().float()
            result = self.original(*args, **kwargs)
            if self.forward_probability is None or self.forward_probability_low is None:
                raise RuntimeError("backward softmax ran without its bound forward observation")
            reconstructed_probability = torch.exp(logits - maximum) / denominator
            inner = (gradient * reconstructed_probability).sum(dim=-1, keepdim=True)
            reconstructed_reference = reconstructed_probability * (gradient - inner) * ALPHA
            semantic_probability = self.forward_probability
            semantic_inner = (gradient * semantic_probability).sum(dim=-1, keepdim=True)
            semantic_reference = semantic_probability * (gradient - semantic_inner) * ALPHA
            rounded_probability = semantic_probability.to(torch.bfloat16).float()
            rounded_inner = (gradient * rounded_probability).sum(dim=-1, keepdim=True)
            rounded_forward_reference = (
                rounded_probability * (gradient - rounded_inner) * ALPHA
            )
            actual_forward_probability = self.forward_probability_low
            actual_forward_inner = (
                gradient * actual_forward_probability
            ).sum(dim=-1, keepdim=True)
            actual_forward_reference = (
                actual_forward_probability * (gradient - actual_forward_inner) * ALPHA
            )
            aten_vjp = torch.ops.aten._softmax_backward_data.default(
                gradient, semantic_probability, -1, torch.float32
            ) * ALPHA
            formula_gap = (semantic_reference - aten_vjp).abs().max()
            if (not torch.isfinite(semantic_reference).all()
                    or float(formula_gap.item()) > 2.0 ** -20):
                raise RuntimeError("analytic softmax VJP does not match typed ATen VJP")
            local = decompose_low_precision_output(destination, reconstructed_reference)
            vectors = {
                "kernel": local["kernel"],
                "output_rounding": local["output_rounding"],
                "saved_state_reconstruction": (
                    reconstructed_reference - actual_forward_reference
                ),
                "forward_probability_kernel": (
                    actual_forward_reference - rounded_forward_reference
                ),
                "forward_probability_rounding": (
                    rounded_forward_reference - semantic_reference
                ),
                "semantic_total": destination.detach().float() - semantic_reference,
            }
            closure = (vectors["kernel"] + vectors["output_rounding"]
                       + vectors["saved_state_reconstruction"]
                       + vectors["forward_probability_kernel"]
                       + vectors["forward_probability_rounding"]
                       - vectors["semantic_total"])
            self.vectors = {name: value.cpu() for name, value in vectors.items()}
            self.summary = {
                "coordinates": semantic_reference.numel(),
                "formula_vs_typed_aten_max_abs": float(formula_gap.item()),
                "algebraic_closure_max_abs": float(closure.abs().max().item()),
                "reconstructed_probability_row_sum_max_error": float(
                    (reconstructed_probability.sum(dim=-1) - 1.0).abs().max().item()
                ),
                "reconstructed_vs_semantic_probability_max_abs": float(
                    (reconstructed_probability - semantic_probability).abs().max().item()
                ),
                "forward": self.forward_summary,
            }
            for name, value in vectors.items():
                self.summary[name + "_nonzero"] = int(torch.count_nonzero(value).item())
                self.summary[name + "_l2"] = float(torch.linalg.vector_norm(value).item())
                self.summary[name + "_max_abs"] = float(value.abs().max().item())
            self.calls += 1
            return result

        self.forward_kernel.run = forward_wrapped
        self.kernel.run = wrapped
        return self

    def __exit__(self, *unused: Any) -> None:
        del unused
        if self.had_run:
            self.kernel.run = self.previous
        else:
            delattr(self.kernel, "run")
        if self.forward_had_run:
            self.forward_kernel.run = self.forward_previous
        else:
            delattr(self.forward_kernel, "run")
        if self.calls != 1 or self.forward_calls != 1 or self.summary is None:
            raise RuntimeError(
                f"target softmax F+B region executed F={self.forward_calls}, B={self.calls} times"
            )


def endpoint(model: torch.nn.Module, loss: torch.Tensor) -> dict[str, str]:
    return {"loss": tensor_digest(loss), "gradients": gradient_digest(model)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/coverage/cases/qwen128_softmax_fb_pilot.json",
    )
    args = parser.parse_args()
    if args.states < 2 or args.repeats != 2:
        raise ValueError("protocol requires at least two states and exactly two repeats")
    bank_path = ROOT / "results/coverage/qwen_seq128_input_bank.json"
    states = json.loads(bank_path.read_text()).get("states")[:args.states]
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
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    validate_release(wrapper_modules(modules), capture)

    spool = Path("/data1/tzh/cache/kernel_analyzer_contrasts/qwen128_softmax_fb")
    vector_names = (
        "kernel", "output_rounding", "saved_state_reconstruction",
        "forward_probability_kernel", "forward_probability_rounding", "semantic_total",
    )
    grams = {name: StreamingGramAccumulator(spool, CID + "_" + name)
             for name in vector_names}
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
        repeat_rows = []
        retained: dict[str, torch.Tensor] | None = None
        for repeat in range(args.repeats):
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model.zero_grad(set_to_none=True)
            observer = SoftmaxFBObserver(modules)
            with observer:
                loss = candidate(values); loss.backward()
            torch.cuda.synchronize(device)
            observed = endpoint(model, loss)
            if observed != baseline:
                raise RuntimeError("read-only softmax observer changed full-step endpoints")
            assert observer.summary is not None
            hashes = {name: tensor_sha(value) for name, value in observer.vectors.items()}
            repeat_rows.append({"repeat": repeat, "vector_sha256": hashes,
                                "summary": observer.summary})
            if retained is None:
                retained = observer.vectors
            elif hashes != repeat_rows[0]["vector_sha256"]:
                raise RuntimeError("softmax precision vectors are not repeat exact")
        assert retained is not None
        vector_rows = {name: grams[name].add_array(state_id, value.numpy())
                       for name, value in retained.items()}
        rows.append({"state_id": state_id, "endpoint_identity": baseline,
                     "repeats": repeat_rows, "vectors": vector_rows})
        write(args.output, {"schema": "kernel-analyzer-softmax-fb-typed-screen-v1",
                            "status": "RUNNING", "candidate_id": CID, "states": rows})
        del values, retained
        torch.cuda.empty_cache()
        print(json.dumps({"event": "STATE_COMPLETE", "state": state_id}), flush=True)

    certificates = {name: accumulator.finalize(bootstrap_draws=4000,
                                                seed=17000 + offset, cleanup=True)
                    for offset, (name, accumulator) in enumerate(grams.items())}
    payload = {
        "schema": "kernel-analyzer-softmax-fb-typed-screen-v1",
        "status": "COMPLETE_TYPED_SOFTMAX_FB_PILOT" if args.states < 32
                  else "COMPLETE_TYPED_SOFTMAX_FB_FORMAL",
        "candidate_id": CID,
        "semantic_region": {
            "forward": "A=alpha*(QK^T)+M; P=softmax(A)",
            "backward": "dA=P*(G-sum(G*P)); d(QK^T)=alpha*dA",
            "alpha": ALPHA,
            "actual_endpoint": "generated BF16 d(QK^T) before Q/K BMM VJPs",
            "reference_boundary": (
                "typed AOT softmax VJP using the semantic forward probability computed "
                "from the exact generated forward kernel inputs"
            ),
            "generated_symbol": SYMBOL,
            "generated_source_line_sha256": TARGET_SHA,
        },
        "decomposition": {
            "kernel": "actual_B - round_bf16(VJP(P_reconstructed))",
            "output_rounding": (
                "round_bf16(VJP(P_reconstructed)) - VJP(P_reconstructed)"
            ),
            "saved_state_reconstruction": (
                "VJP(P_reconstructed_from_logits_max_sum) - VJP(P_forward_actual_bf16)"
            ),
            "forward_probability_kernel": (
                "VJP(P_forward_actual_bf16) - VJP(round_bf16(P_semantic))"
            ),
            "forward_probability_rounding": (
                "VJP(round_bf16(P_semantic)) - VJP(P_semantic)"
            ),
            "semantic_total": "actual_B - VJP(P_semantic)",
        },
        "states": rows, "direction": certificates,
        "coherent_sources": [name for name in vector_names[:-1]
                             if certificates[name]["status"] == "PASS"],
        "gates": {
            "typed_independent_reference": True,
            "two_repeats_exact": True,
            "read_only_observer_exact": True,
            "analytic_vjp_matches_typed_aten": True,
            "algebraic_closure_every_state": True,
            "formal_32_state_population": args.states == 32,
        },
        "release_capture_sha256": capture["result_sha256"],
        "input_bank_sha256": file_digest(bank_path),
        "claim_boundary": (
            "Numerical typed-reference screen for one exact generated softmax+scale VJP region. "
            "The exact decomposition includes backward arithmetic, BF16 output rounding, "
            "and the compiler's F-to-B saved-state reconstruction. Concrete AOT and "
            "generated-program bindings remain a separate proof gate."
        ),
    }
    payload["result_sha256"] = canonical(payload)
    write(args.output, payload)
    print(json.dumps({"event": "SCREEN_COMPLETE", "status": payload["status"],
                      "coherent_sources": payload["coherent_sources"]}), flush=True)


if __name__ == "__main__":
    main()
