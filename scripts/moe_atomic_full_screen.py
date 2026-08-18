#!/usr/bin/env python3
"""Full-LM F+B screen of one Granite MoE atomic combine intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
import types
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from moe_atomic_replay import canonical_hash, deterministic_moe


ROOT = Path(__file__).resolve().parents[1]


def run_arm(model, moe, routers, input_ids, deterministic: bool, seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    original_forward = moe.forward
    if deterministic:
        moe.forward = types.MethodType(lambda self, x: deterministic_moe(self, x), moe)
    captured = {}
    def capture_hook(_module, _inputs, output):
        captured["output"] = output[0].detach().float().cpu().clone()
    handle = moe.register_forward_hook(capture_hook)
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    loss.backward()
    handle.remove()
    moe.forward = original_forward
    gradients = {
        name: module.layer.weight.grad.detach().float().cpu().clone()
        for name, module in routers.items()
    }
    return float(loss.detach().cpu()), captured["output"], gradients


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/ibm-granite/granite-3.1-1b-a400m-base"))
    parser.add_argument("--input-bank", type=Path, default=ROOT / "results/moe/input_bank.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/moe/atomic_full_screen.json")
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--calibration-states", type=int, default=2)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    bank = json.loads(args.input_bank.read_text())
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager"
    ).to(args.device).train()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    routers = {
        name: module for name, module in model.named_modules()
        if type(module).__name__ == "GraniteMoeTopKGating"
    }
    for module in routers.values():
        module.layer.weight.requires_grad_(True)
    moe = model.model.layers[args.layer].block_sparse_moe

    records = {}
    state_rows = []
    reference_repeat = None
    for state_id in range(args.states):
        bank_row = bank["states"][state_id]
        ids_cpu = torch.tensor(bank_row["token_ids"], dtype=torch.long)
        if hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest() != bank_row["token_sha256"]:
            raise RuntimeError(f"token digest mismatch: {state_id}")
        ids = ids_cpu.unsqueeze(0).to(args.device)
        seed = 19000 + state_id
        ref_loss, ref_output, ref_gradients = run_arm(model, moe, routers, ids, True, seed)
        cand_loss_1, cand_output_1, cand_gradients_1 = run_arm(model, moe, routers, ids, False, seed)
        cand_loss_2, cand_output_2, cand_gradients_2 = run_arm(model, moe, routers, ids, False, seed)
        if state_id == 0:
            rr_loss, rr_output, rr_gradients = run_arm(model, moe, routers, ids, True, seed)
            reference_repeat = {
                "loss_exact": rr_loss == ref_loss,
                "intervention_output_bitwise_exact": torch.equal(rr_output, ref_output),
                "router_gradients_bitwise_exact": all(
                    torch.equal(rr_gradients[name], ref_gradients[name]) for name in routers
                ),
            }
        records[state_id] = {
            "output_delta": (cand_output_1.double() + cand_output_2.double()) / 2 - ref_output.double(),
            "output_runtime": cand_output_2.double() - cand_output_1.double(),
            "gradient_deltas": {
                name: (cand_gradients_1[name].double() + cand_gradients_2[name].double()) / 2
                - ref_gradients[name].double()
                for name in routers
            },
            "gradient_runtime": {
                name: cand_gradients_2[name].double() - cand_gradients_1[name].double()
                for name in routers
            },
        }
        output_signal = torch.linalg.vector_norm(records[state_id]["output_delta"])
        output_runtime = torch.linalg.vector_norm(records[state_id]["output_runtime"])
        gradient_signal = torch.sqrt(sum(
            torch.sum(value ** 2) for value in records[state_id]["gradient_deltas"].values()
        ))
        gradient_runtime = torch.sqrt(sum(
            torch.sum(value ** 2) for value in records[state_id]["gradient_runtime"].values()
        ))
        state_rows.append({
            "state_id": state_id,
            "split": "CALIBRATION" if state_id < args.calibration_states else "HELDOUT",
            "token_sha256": bank_row["token_sha256"],
            "reference_loss": ref_loss,
            "candidate_loss_repeats": [cand_loss_1, cand_loss_2],
            "candidate_mean_loss_delta": (cand_loss_1 + cand_loss_2) / 2 - ref_loss,
            "candidate_runtime_loss_delta": cand_loss_2 - cand_loss_1,
            "local_output_delta_l2": float(output_signal),
            "local_output_runtime_l2": float(output_runtime),
            "router_gradient_delta_l2": float(gradient_signal),
            "router_gradient_runtime_l2": float(gradient_runtime),
        })
        torch.cuda.empty_cache()

    endpoint_rows = []
    endpoint_names = ["layer_output"] + sorted(routers)
    for endpoint in endpoint_names:
        values = {
            state_id: records[state_id]["output_delta"] if endpoint == "layer_output"
            else records[state_id]["gradient_deltas"][endpoint]
            for state_id in records
        }
        runtime_values = {
            state_id: records[state_id]["output_runtime"] if endpoint == "layer_output"
            else records[state_id]["gradient_runtime"][endpoint]
            for state_id in records
        }
        raw = sum(values[i] for i in range(args.calibration_states))
        norm = torch.linalg.vector_norm(raw)
        if norm == 0:
            continue
        direction = raw / norm
        projections = [float(torch.sum(values[i] * direction)) for i in range(args.states)]
        runtime = [float(torch.sum(runtime_values[i] * direction)) for i in range(args.states)]
        heldout = projections[args.calibration_states:]
        mean = sum(heldout) / len(heldout)
        max_runtime = max(abs(value) for value in runtime)
        endpoint_rows.append({
            "endpoint": endpoint if endpoint == "layer_output" else f"{endpoint}.layer.weight_grad",
            "numel": direction.numel(),
            "calibration_projections": projections[:args.calibration_states],
            "heldout_projections": heldout,
            "heldout_positive": sum(value > 0 for value in heldout),
            "heldout_negative": sum(value < 0 for value in heldout),
            "heldout_mean": mean,
            "max_abs_runtime_projection": max_runtime,
            "runtime_projection_over_abs_heldout_mean": max_runtime / abs(mean) if mean else None,
            "persistent_positive": all(value > 0 for value in heldout),
            "persistent_negative": all(value < 0 for value in heldout),
        })
    endpoint_rows.sort(key=lambda row: abs(row["heldout_mean"]), reverse=True)
    output = {
        "schema": "kernel-analyzer-granite-moe-atomic-single-layer-full-fb-screen-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "dtype": "bfloat16",
        "seq_len": bank["seq_len"],
        "layer": args.layer,
        "reference": "layer-only inverse-permutation plus fixed top-k reduction",
        "candidate": "layer-only Transformers zeros.index_add CUDA combine",
        "unchanged": "same model weights, top-k experts, expert arithmetic, inputs, dropout RNG, loss and backward",
        "mathematical_equivalence": "both sum the same 8 gated expert outputs per token",
        "calibration_states": list(range(args.calibration_states)),
        "heldout_states": list(range(args.calibration_states, args.states)),
        "reference_repeat": reference_repeat,
        "state_rows": state_rows,
        "endpoint_rows": endpoint_rows,
        "persistent_endpoint_count": sum(
            row["persistent_positive"] or row["persistent_negative"] for row in endpoint_rows
        ),
        "tensor_values_saved": False,
        "claim_boundary": "SCREEN_ONLY_REQUIRES_INDEPENDENT_STATES_AND_LIVE_WEIGHT_BEFORE_NEW_CASE",
    }
    output["result_sha256"] = canonical_hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "persistent": output["persistent_endpoint_count"],
        "top": endpoint_rows[:6],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
