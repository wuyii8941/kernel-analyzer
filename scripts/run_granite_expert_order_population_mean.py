#!/usr/bin/env python3
"""Independent mean-direction confirmation for the Granite expert-order case.

This runner is intentionally small and writes no content-addressed metadata.
It samples complete input states with replacement from the declared empirical
bank.  The first half freezes a direction; the second half is untouched until
the direction is fixed.  Both the signed projection mean and the cross-state
U statistic are reported.  The result is conditional on this empirical bank,
the chosen carrier, and the eager Granite implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
import types
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeMoE


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/data1/tzh/models/ibm-granite/granite-3.1-1b-a400m-base")
BANK = ROOT / "results/moe/input_bank.json"
OUT = ROOT / "results/property/granite_expert_order_population_v1"
TARGET = "model.layers.0.block_sparse_moe"
CARRIER = "model.layers.0.input_layernorm.weight"
CALIBRATION = 32
CONFIRMATION = 64
DRAW_SEED = 20260916
LEARNING_RATE = 1e-3


def expert_forward(
    self: GraniteMoeMoE,
    hidden_states: torch.Tensor,
    *,
    order: list[int],
) -> torch.Tensor:
    """Mirror the installed eager MoE and reverse only index-add order."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_dim)
    _, batch_index, batch_gates, expert_size, router_logits = self.router(flat)
    expert_inputs = flat[batch_index]
    hidden = self.input_linear(expert_inputs, expert_size)
    first, second = hidden.chunk(2, dim=-1)
    hidden = self.activation(first) * second
    expert_outputs = self.output_linear(hidden, expert_size)
    expert_outputs = expert_outputs * batch_gates[:, None]
    if order[0] == 0:
        index = batch_index
        values = expert_outputs
    else:
        # batch_index is grouped by expert by the router.  Reversing the
        # contribution stream preserves every value/index pair while changing
        # only the floating-point accumulation order.
        index = batch_index.flip(0)
        values = expert_outputs.flip(0)
    zeros = torch.zeros(
        (batch_size * sequence_length, self.input_size),
        dtype=expert_outputs.dtype,
        device=expert_outputs.device,
    )
    output = zeros.index_add(0, index, values)
    return output.reshape(batch_size, sequence_length, self.input_size), router_logits


def set_order(target: GraniteMoeMoE, order: list[int]) -> None:
    target.forward = types.MethodType(
        lambda self, hidden_states: expert_forward(self, hidden_states, order=order), target
    )


def adamw_zero_moment_write(parameter: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    # Common cold-start AdamW response; no in-place mutation of parameters.
    return -LEARNING_RATE * gradient / (gradient.abs() + 1e-8)


def draw_indices(bank_size: int) -> list[int]:
    rng = random.Random(DRAW_SEED)
    return [rng.randrange(bank_size) for _ in range(CALIBRATION + CONFIRMATION)]


def summarize(vectors: list[torch.Tensor], references: list[torch.Tensor], split: int) -> dict[str, Any]:
    matrix = torch.stack([v.double() for v in vectors])
    ref_matrix = torch.stack([v.double() for v in references])
    n = matrix.shape[0]
    cal = matrix[:split]
    conf = matrix[split:]
    direction = cal.mean(dim=0)
    direction = direction / direction.norm().clamp_min(1e-30)
    projection = conf @ direction
    conf_cross = conf @ conf.T
    off = conf_cross.sum() - torch.diagonal(conf_cross).sum()
    diagonal = torch.diagonal(conf_cross).sum()
    mean = float(projection.mean())
    sd = float(projection.std(unbiased=True)) if len(projection) > 1 else 0.0
    # Descriptive t interval; the primary population conclusion should use
    # the declared iid-with-replacement scope and this interval's assumptions.
    from scipy.stats import t
    half = float(t.ppf(0.975, len(projection) - 1)) * sd / math.sqrt(len(projection))
    one_sided = float(t.ppf(0.95, len(projection) - 1)) * sd / math.sqrt(len(projection))
    return {
        "state_count": int(n),
        "calibration_count": int(split),
        "confirmation_count": int(n - split),
        "calibration_direction_norm": float(cal.mean(dim=0).norm()),
        "confirmation_projection_mean": mean,
        "confirmation_projection_sd": sd,
        "confirmation_projection_two_sided_95": [mean - half, mean + half],
        "confirmation_projection_one_sided_95_lower": mean - one_sided,
        "confirmation_positive_count": int((projection > 0).sum()),
        "confirmation_negative_count": int((projection < 0).sum()),
        "confirmation_zero_count": int((projection == 0).sum()),
        "confirmation_cross_state_u_statistic": float(off / (len(projection) * (len(projection) - 1))),
        "confirmation_diagonal_energy_mean": float(diagonal / len(projection)),
        "reference_energy_rms": float(ref_matrix.norm(dim=1).pow(2).mean().sqrt()),
        "scope": "DECLARED_IID_WITH_REPLACEMENT_FROM_EMPIRICAL_GRANITE_BANK",
        "population_vector_mean_claim": "SUPPORTED_ONLY_IF_THE_SIGNED_PROJECTION_MEAN_TEST_ASSUMPTIONS_HOLD",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=0, help="debug limit; 0 means all units")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("output must stay inside the repository")
    if output.exists() and (output / "result.json").exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output / 'result.json'}")
    bank = json.loads(BANK.read_text())
    states = bank["states"]
    indices = draw_indices(len(states))
    if args.limit:
        indices = indices[: args.limit]
    if len(indices) <= CALIBRATION:
        raise ValueError("limit must leave at least one confirmation unit")

    device = torch.device(args.device)
    torch.manual_seed(20260916)
    torch.cuda.manual_seed_all(20260916)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager", local_files_only=True,
    ).to(device).eval()
    model.config.use_cache = False
    parameter = dict(model.named_parameters())[CARRIER]
    parameter.requires_grad_(True)
    target = dict(model.named_modules())[TARGET]
    if not isinstance(target, GraniteMoeMoE):
        raise RuntimeError("declared Granite expert module is not active")
    original_order = list(range(target.router.num_experts))
    reverse_order = list(reversed(original_order))
    original_forward = target.forward
    effects_gradient: list[torch.Tensor] = []
    effects_write: list[torch.Tensor] = []
    repairs_gradient: list[torch.Tensor] = []
    repairs_write: list[torch.Tensor] = []
    rows = []
    for unit, bank_index in enumerate(indices):
        state = states[bank_index]
        ids = torch.tensor([state["token_ids"][:-1]], dtype=torch.long, device=device)
        labels = ids.clone()
        results: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for name, order in (("candidate", original_order), ("reference", reverse_order)):
            target.forward = original_forward if name == "candidate" else types.MethodType(
                lambda self, hidden_states, _order=order: expert_forward(self, hidden_states, order=_order), target
            )
            model.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=labels, use_cache=False).loss
            loss.backward()
            gradient = parameter.grad.detach().float().cpu().clone()
            write = adamw_zero_moment_write(parameter.detach().cpu(), gradient)
            results[name] = (loss.detach().cpu(), gradient, write)
        target.forward = original_forward
        c_loss, c_gradient, c_write = results["candidate"]
        r_loss, r_gradient, r_write = results["reference"]
        effects_gradient.append(c_gradient - r_gradient)
        effects_write.append(c_write - r_write)
        repairs_gradient.append(r_gradient)
        repairs_write.append(r_write)
        rows.append({
            "unit": unit,
            "bank_index": bank_index,
            "state_id": state.get("state_id", bank_index),
            "loss_exact": bool(torch.equal(c_loss, r_loss)),
            "loss_delta": float(c_loss - r_loss),
            "gradient_effect_l2": float((c_gradient - r_gradient).norm()),
            "write_effect_l2": float((c_write - r_write).norm()),
        })
        print(json.dumps({"event": "GRANITE_EXPERT_MEAN_UNIT", "unit": unit + 1}), flush=True)
        del ids, labels, results
        torch.cuda.empty_cache()

    split = min(CALIBRATION, len(effects_gradient) - 1)
    payload = {
        "schema": "kernel-analyzer-granite-expert-order-population-mean-v1",
        "status": "COMPLETE",
        "case_id": "granite_fp32_expert_order_layer0",
        "candidate": "eager Granite expert contributions in ascending order",
        "reference": "same eager implementation in descending order",
        "dtype": "FP32_BOTH_ARMS",
        "target_parameter": CARRIER,
        "target_module": TARGET,
        "population": "with-replacement draws from results/moe/input_bank.json",
        "draw_seed": DRAW_SEED,
        "draw_indices": indices,
        "calibration_count": split,
        "confirmation_count": len(indices) - split,
        "gradient": summarize(effects_gradient, repairs_gradient, split),
        "parameter_write": summarize(effects_write, repairs_write, split),
        "rows": rows,
        "claim_boundary": (
            "The signed projection and cross-state statistics are conditional on the declared empirical bank, "
            "carrier, zero-moment AdamW response, and iid-with-replacement sampling. They do not establish a "
            "universal Granite or natural-pretraining population claim without those assumptions."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(json.dumps({
        "schema": "kernel-analyzer-granite-expert-order-population-protocol-v1",
        "status": "FROZEN_BEFORE_MEASUREMENT",
        "case_id": payload["case_id"],
        "candidate": payload["candidate"],
        "reference": payload["reference"],
        "population": payload["population"],
        "draw_seed": DRAW_SEED,
        "draw_indices": indices,
        "calibration_count": split,
        "confirmation_count": len(indices) - split,
        "target_parameter": CARRIER,
        "target_module": TARGET,
        "primary_endpoints": [
            "signed confirmation projection mean",
            "confirmation cross-state U statistic",
        ],
    }, indent=2) + "\n")
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(output),
                      "write_mean": payload["parameter_write"]["confirmation_projection_mean"],
                      "write_lower": payload["parameter_write"]["confirmation_projection_one_sided_95_lower"]}))


if __name__ == "__main__":
    main()
