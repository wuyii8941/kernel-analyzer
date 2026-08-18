#!/usr/bin/env python3
"""Same-input/cotangent F+B replay of Granite MoE atomic combine."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def deterministic_moe(moe, layer_input: torch.Tensor):
    bsz, length, emb_size = layer_input.shape
    flat = layer_input.reshape(-1, emb_size)
    index_sorted, batch_index, batch_gates, expert_size, router_logits = moe.router(flat)
    expert_inputs = flat[batch_index]
    hidden = moe.input_linear(expert_inputs, expert_size)
    left, right = hidden.chunk(2, dim=-1)
    hidden = moe.activation(left) * right
    expert_outputs = moe.output_linear(hidden, expert_size) * batch_gates[:, None]
    # index_sorted maps expert-grouped positions to original [token, top-k] positions.
    inverse = torch.argsort(index_sorted)
    ordered = expert_outputs[inverse].view(bsz * length, moe.router.top_k, emb_size)
    output = ordered.sum(dim=1).view(bsz, length, emb_size)
    return output, router_logits


def local_arm(moe, x: torch.Tensor, cotangent: torch.Tensor, deterministic: bool):
    moe.zero_grad(set_to_none=True)
    leaf = x.detach().clone().requires_grad_(True)
    output, _ = deterministic_moe(moe, leaf) if deterministic else moe(leaf)
    torch.sum(output.float() * cotangent.float()).backward()
    gradients = {
        "input": leaf.grad.detach().float().cpu().clone(),
        "router": moe.router.layer.weight.grad.detach().float().cpu().clone(),
        "input_linear": moe.input_linear.weight.grad.detach().float().cpu().clone(),
        "output_linear": moe.output_linear.weight.grad.detach().float().cpu().clone(),
    }
    return output.detach().float().cpu().clone(), gradients


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/ibm-granite/granite-3.1-1b-a400m-base"))
    parser.add_argument("--input-bank", type=Path, default=ROOT / "results/moe/input_bank.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/moe/atomic_replay.json")
    parser.add_argument("--state-id", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    bank = json.loads(args.input_bank.read_text())
    bank_row = bank["states"][args.state_id]
    ids_cpu = torch.tensor(bank_row["token_ids"], dtype=torch.long)
    if hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest() != bank_row["token_sha256"]:
        raise RuntimeError("token digest mismatch")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager"
    ).to(args.device).train()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    moe = model.model.layers[args.layer].block_sparse_moe
    for parameter in moe.parameters():
        parameter.requires_grad_(True)

    captured = {}
    def pre_hook(_module, inputs):
        captured["input"] = inputs[0].detach().clone()
    def forward_hook(_module, _inputs, output):
        output[0].register_hook(lambda grad: captured.__setitem__("cotangent", grad.detach().clone()))
    pre_handle = moe.register_forward_pre_hook(pre_hook)
    forward_handle = moe.register_forward_hook(forward_hook)
    torch.manual_seed(17000 + args.state_id)
    torch.cuda.manual_seed_all(17000 + args.state_id)
    loss = model(
        input_ids=ids_cpu.unsqueeze(0).to(args.device),
        labels=ids_cpu.unsqueeze(0).to(args.device),
        use_cache=False,
    ).loss
    loss.backward()
    pre_handle.remove()
    forward_handle.remove()
    if set(captured) != {"input", "cotangent"}:
        raise RuntimeError("actual full-model boundary capture failed")

    reference_output, reference_gradients = local_arm(
        moe, captured["input"], captured["cotangent"], deterministic=True
    )
    reference_output_2, reference_gradients_2 = local_arm(
        moe, captured["input"], captured["cotangent"], deterministic=True
    )
    candidate_outputs = []
    candidate_gradients = []
    for _ in range(args.repeats):
        output, gradients = local_arm(
            moe, captured["input"], captured["cotangent"], deterministic=False
        )
        candidate_outputs.append(output)
        candidate_gradients.append(gradients)

    endpoints = {"output": reference_output, **reference_gradients}
    rows = []
    for endpoint, reference in endpoints.items():
        values = candidate_outputs if endpoint == "output" else [g[endpoint] for g in candidate_gradients]
        deltas = [value.double() - reference.double() for value in values]
        calibration_count = min(8, args.repeats // 2)
        raw = sum(deltas[:calibration_count])
        norm = torch.linalg.vector_norm(raw)
        if norm == 0:
            projections = [0.0 for _ in deltas]
        else:
            direction = raw / norm
            projections = [float(torch.sum(delta * direction)) for delta in deltas]
        heldout = projections[calibration_count:]
        rows.append({
            "endpoint": endpoint,
            "numel": reference.numel(),
            "candidate_changed_repeats": sum(not torch.equal(value, reference) for value in values),
            "candidate_unique_sha256": len({hashlib.sha256(value.numpy().tobytes()).hexdigest() for value in values}),
            "delta_l2_min": min(float(torch.linalg.vector_norm(delta)) for delta in deltas),
            "delta_l2_max": max(float(torch.linalg.vector_norm(delta)) for delta in deltas),
            "direction_calibration_repeats": calibration_count,
            "heldout_positive": sum(value > 0 for value in heldout),
            "heldout_negative": sum(value < 0 for value in heldout),
            "heldout_zero": sum(value == 0 for value in heldout),
            "heldout_projection_mean": sum(heldout) / len(heldout),
            "heldout_projections": heldout,
        })

    output = {
        "schema": "kernel-analyzer-granite-moe-atomic-combine-same-boundary-replay-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "dtype": "bfloat16",
        "state_id": args.state_id,
        "token_sha256": bank_row["token_sha256"],
        "layer": args.layer,
        "full_model_loss": float(loss.detach().cpu()),
        "boundary": "same actual MoE input and same actual full-LM output cotangent",
        "reference": "same expert arithmetic; inverse-permutation then fixed top-k reduction",
        "candidate": "Transformers GraniteMoeMoE zeros.index_add CUDA combine",
        "mathematical_equivalence": "both sum gate[t,e] * expert_e(x[t]) over the same frozen top-k experts",
        "reference_repeat": {
            "output_bitwise_exact": torch.equal(reference_output, reference_output_2),
            "gradient_bitwise_exact": {
                name: torch.equal(reference_gradients[name], reference_gradients_2[name])
                for name in reference_gradients
            },
        },
        "candidate_repeats": args.repeats,
        "rows": rows,
        "tensor_values_saved": False,
        "claim_boundary": "ONE_STATE_LOCAL_CAUSAL_REPLAY; DIRECTIONAL_BIAS_REQUIRES_INDEPENDENT_NATURAL_STATES",
    }
    output["result_sha256"] = canonical_hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "reference_repeat": output["reference_repeat"], "rows": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
