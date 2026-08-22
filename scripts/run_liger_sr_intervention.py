#!/usr/bin/env python3
"""Run a real Liger fused-CE RN-versus-SR source intervention.

The Triton cross-entropy kernel and the F+B boundary are unchanged.  Only the
BF16 ``dW`` accumulator cast is replaced by stochastic rounding.  Full dW
vectors are reduced online and are never written to the repository.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "archive/round1_code/src")]

MODEL = Path("/data1/tzh/models/Qwen/Qwen3-1.7B")
DESIGN = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/supplementary_state_design_v1.json"
PROTOCOL = ROOT / "results/trajectory/liger_protocol.json"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sr_cast(value: torch.Tensor) -> torch.Tensor:
    """Stochastically round float32 values to BF16 using adjacent BF16 values."""
    if value.dtype != torch.float32:
        value = value.float()
    nearest = value.to(torch.bfloat16)
    nearest_f = nearest.float()
    target_hi = torch.full_like(nearest, float("inf"))
    target_lo = torch.full_like(nearest, float("-inf"))
    up = torch.nextafter(nearest, target_hi).float()
    down = torch.nextafter(nearest, target_lo).float()
    above = value > nearest_f
    lower = torch.where(above, nearest_f, down)
    upper = torch.where(above, up, nearest_f)
    span = upper - lower
    probability = torch.where(span > 0, (value - lower) / span, torch.zeros_like(value))
    draw = torch.rand_like(value)
    sampled = torch.where(draw < probability, upper, lower)
    return torch.where(value == nearest_f, nearest_f, sampled).to(torch.bfloat16)


def sr_accumulate(base: torch.Tensor, contribution: torch.Tensor) -> torch.Tensor:
    if base.dtype != torch.bfloat16:
        return base + contribution.to(base.dtype)
    return sr_cast(base.float() + contribution.float())


def patch_liger_forward() -> dict[str, str]:
    """Patch only the accumulator line and return source provenance."""
    import liger_kernel.ops.fused_linear_cross_entropy as fused

    original = fused.fused_linear_cross_entropy_forward
    source = inspect.getsource(original)
    old = "            grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk).float()"
    new = "            grad_weight = _sr_accumulate(grad_weight, torch.mm(grad_logits_chunk.t(), _input_chunk).float())"
    if source.count(old) != 1:
        raise RuntimeError("Liger accumulator source line was not uniquely found")
    namespace = dict(fused.__dict__)
    namespace["_sr_accumulate"] = sr_accumulate
    transformed = source.replace(old, new, 1)
    exec(compile(transformed, "<liger-sr-forward>", "exec"), namespace)
    fused.fused_linear_cross_entropy_forward = namespace["fused_linear_cross_entropy_forward"]
    return {
        "original_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "transformed_source_sha256": hashlib.sha256(transformed.encode()).hexdigest(),
        "replacement": old + " -> " + new,
    }


def run_region(module: Any, hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = hidden.detach().clone().reshape(-1, hidden.shape[-1]).requires_grad_(True)
    loss = module(weight, h, labels)
    _, grad_weight = torch.autograd.grad(loss, (h, weight), retain_graph=False)
    return loss.detach(), grad_weight.detach().float()


def vector_metrics(result_sum: torch.Tensor, energy: float, path_l2: float, steps: int) -> dict[str, float]:
    if steps == 0:
        return {"steps": 0, "path_l2": 0.0, "resultant_l2": 0.0, "coherence_amplification": 0.0}
    resultant_l2 = float(torch.linalg.vector_norm(result_sum).item())
    return {
        "steps": steps,
        "path_l2": path_l2,
        "energy_sqrt": energy ** 0.5,
        "resultant_l2": resultant_l2,
        "coherence_amplification": resultant_l2 / max(energy ** 0.5, 1e-30),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--sr-repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the host GPU, not the sandbox")
    if args.steps < 2 or args.steps > 24 or args.sr_repeats < 1:
        raise ValueError("steps must be in [2,24] and sr-repeats must be positive")

    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from transformers import AutoModelForCausalLM

    design = json.loads(DESIGN.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    records = {row["sequence_id"]: row for row in design["records"]}
    state_ids = list(protocol["trajectory"]["state_order"])[16 : 16 + args.steps]
    device = torch.device(args.device)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
    ).to(device)
    model.config.use_cache = False
    model.eval()
    weight = model.lm_head.weight
    native = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean", accum_dtype=None).to(device)
    fp32 = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean", accum_dtype=torch.float32).to(device)

    native_sum = torch.zeros_like(weight, dtype=torch.float32)
    native_energy = 0.0
    native_path = 0.0
    sr_sums = [torch.zeros_like(weight, dtype=torch.float32) for _ in range(args.sr_repeats)]
    sr_energies = [0.0 for _ in range(args.sr_repeats)]
    sr_paths = [0.0 for _ in range(args.sr_repeats)]
    records_out: list[dict[str, Any]] = []
    patch_metadata: dict[str, str] | None = None
    for index, state_id in enumerate(state_ids):
        record = records[state_id]
        input_ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            hidden = model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state.detach()
        labels = torch.nn.functional.pad(input_ids, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
        loss_native, grad_native = run_region(native, hidden, weight, labels)
        loss_fp32, grad_fp32 = run_region(fp32, hidden, weight, labels)
        natural = grad_native - grad_fp32
        natural_norm = float(torch.linalg.vector_norm(natural).item())
        native_sum.add_(natural)
        native_energy += natural_norm * natural_norm
        native_path += natural_norm
        row: dict[str, Any] = {
            "step": index + 1,
            "state_id": state_id,
            "forward_native_fp32_equal": bool(torch.equal(loss_native, loss_fp32)),
                "native_residual_l2": natural_norm,
            "sr": [],
        }
        del loss_native, loss_fp32, grad_native
        for repeat in range(args.sr_repeats):
            torch.manual_seed(91000 + repeat * 1000 + index)
            torch.cuda.manual_seed_all(91000 + repeat * 1000 + index)
            if patch_metadata is None:
                patch_metadata = patch_liger_forward()
            sr_module = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean", accum_dtype=None).to(device)
            _, grad_sr = run_region(sr_module, hidden, weight, labels)
            residual_sr = grad_sr - grad_fp32
            residual_norm = float(torch.linalg.vector_norm(residual_sr).item())
            sr_sums[repeat].add_(residual_sr)
            sr_energies[repeat] += residual_norm * residual_norm
            sr_paths[repeat] += residual_norm
            row["sr"].append({
                "repeat": repeat,
                "residual_l2": residual_norm,
                "sr_vs_native_l2": float(torch.linalg.vector_norm(residual_sr - natural).item()),
            })
            del sr_module, grad_sr, residual_sr
        records_out.append(row)
        print(json.dumps({"event": "LIGER_SR_STATE", "step": index + 1, "state_id": state_id}, sort_keys=True), flush=True)
        del grad_fp32, natural, input_ids, hidden, labels
        gc.collect()
        torch.cuda.empty_cache()

    natural_metric = vector_metrics(native_sum, native_energy, native_path, len(state_ids))
    sr_metrics = [vector_metrics(sr_sums[i], sr_energies[i], sr_paths[i], len(state_ids)) for i in range(args.sr_repeats)]
    payload = {
        "schema": "kernel-analyzer-liger-sr-intervention-v1",
        "status": "COMPLETE",
        "case_id": "liger_fused_ce_t128",
        "state_count": len(state_ids),
        "sr_repeats": args.sr_repeats,
        "intervention": "real Liger fused CE BF16 dW accumulator RN replaced by stochastic BF16 rounding; Triton CE and F+B boundary unchanged",
        "source_provenance": patch_metadata,
        "natural_rn": natural_metric,
        "sr": sr_metrics,
        "sr_to_rn_amplification": [float(metric["coherence_amplification"] / max(natural_metric["coherence_amplification"], 1e-30)) for metric in sr_metrics],
        "records": records_out,
        "claim_boundary": "source intervention only; effective-update residual energy is reported, but exact norm matching and multi-step optimizer propagation are not claimed",
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": payload["status"], "natural": payload["natural_rn"]["coherence_amplification"], "sr": [x["coherence_amplification"] for x in payload["sr"]]}, sort_keys=True))


if __name__ == "__main__":
    main()
