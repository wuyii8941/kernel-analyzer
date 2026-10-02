#!/usr/bin/env python3
"""Probe the Granite MoE gate-times-up product materialization boundary."""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/ibm-granite/granite-3.1-1b-a400m-base")
DEFAULT_TEXT = ROOT / "docs/root_cause_closure_current.md"


def windows(tokenizer: Any, text_path: Path, seq_len: int, count: int) -> list[torch.Tensor]:
    tokens = tokenizer(text_path.read_text(encoding="utf-8"), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    stride = seq_len * 2
    need = count * stride + seq_len
    if tokens.numel() < need:
        raise RuntimeError(f"text source has {tokens.numel()} tokens, need {need}")
    return [tokens[i * stride : i * stride + seq_len].clone() for i in range(count)]


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if x.numel() < 2:
        m = float(x.mean().item()); return [m, m]
    m = float(x.mean().item())
    h = 1.96 * float(x.std(unbiased=True).item()) / (x.numel() ** 0.5)
    return [m - h, m + h]


def write(gradient: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    g = gradient.float(); return -lr * g / (g.abs() + eps)


def relative(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect).item()) / max(float(torch.linalg.vector_norm(reference).item()), 1e-30)


def install_reference(block: torch.nn.Module, boundary: str) -> None:
    def forward(self: torch.nn.Module, layer_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, length, emb_size = layer_input.size()
        flat = layer_input.reshape(-1, emb_size)
        _, batch_index, batch_gates, expert_size, router_logits = self.router(flat)
        expert_inputs = flat[batch_index]
        hidden_states = self.input_linear(expert_inputs, expert_size)
        gate, up = hidden_states.chunk(2, dim=-1)
        activated = self.activation(gate)
        # Only the declared product is evaluated in FP32.  All other MoE
        # operations remain on the native path.
        if boundary == "gate_product":
            hidden_states = (activated.float() * up.float()).to(activated.dtype)
        else:
            hidden_states = activated * up
        expert_outputs = self.output_linear(hidden_states, expert_size)
        if boundary == "output_gate":
            expert_outputs = (expert_outputs.float() * batch_gates[:, None].float()).to(expert_outputs.dtype)
        else:
            expert_outputs = expert_outputs * batch_gates[:, None]
        zeros = torch.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
        layer_output = zeros.index_add(0, batch_index, expert_outputs)
        return layer_output.view(bsz, length, self.input_size), router_logits

    block.forward = types.MethodType(forward, block)


def run_once(model: torch.nn.Module, target: torch.nn.Parameter, ids: torch.Tensor, layer: int, reference: bool, boundary: str) -> tuple[float, torch.Tensor]:
    block = model.model.layers[layer].block_sparse_moe
    original = block.forward
    if reference:
        install_reference(block, boundary)
    model.zero_grad(set_to_none=True)
    try:
        loss = model(input_ids=ids, labels=ids, use_cache=False, return_dict=True).loss
        loss.backward()
    finally:
        block.forward = original
    if target.grad is None:
        raise RuntimeError("target gradient is missing")
    return float(loss.detach().cpu().item()), target.grad.detach().float().cpu().clone()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bank = windows(tokenizer, args.text_source, args.sequence_length, args.states)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16).to(args.device).eval()
    target_name = args.parameter or f"model.layers.{args.layer}.block_sparse_moe.input_linear.weight"
    named = dict(model.named_parameters())
    if target_name not in named:
        raise KeyError(target_name)
    target = named[target_name]
    effects: list[torch.Tensor] = []; writes: list[torch.Tensor] = []; refs: list[torch.Tensor] = []; rows: list[dict[str, Any]] = []
    for state_id, tokens in enumerate(bank):
        ids = tokens.unsqueeze(0).to(args.device)
        native_loss, native_grad = run_once(model, target, ids, args.layer, False, args.boundary)
        fp32_loss, fp32_grad = run_once(model, target, ids, args.layer, True, args.boundary)
        native_write = write(native_grad, args.learning_rate, args.eps); fp32_write = write(fp32_grad, args.learning_rate, args.eps)
        grad_effect = fp32_grad - native_grad; write_effect = fp32_write - native_write
        effects.append(grad_effect.double().reshape(-1)); writes.append(write_effect.double().reshape(-1)); refs.append(native_write.double().reshape(-1))
        rows.append({"state_id": state_id, "native_loss": native_loss, "fp32_product_loss": fp32_loss, "loss_difference_fp32_minus_native": fp32_loss-native_loss, "gradient_effect_rms_over_native": relative(grad_effect,native_grad), "write_effect_rms_over_native": relative(write_effect,native_write)})
        del ids; torch.cuda.empty_cache()
    split=args.states//2; direction=torch.stack(effects[:split]).mean(0); direction_norm=float(torch.linalg.vector_norm(direction).item()); projections=[]
    if direction_norm:
        direction/=direction_norm; projections=[float(torch.dot(x,direction).item()) for x in effects[split:]]
    aligned=[float(torch.dot(x,r).item())/max(float(torch.dot(r,r).item()),1e-30) for x,r in zip(writes[split:],refs[split:])]
    label = "gate-times-up" if args.boundary == "gate_product" else "expert-output-times-routing-gate"
    return {"schema":"kernel-analyzer-granite-moe-product-materialization-natural-probe-v1","status":"COMPLETE","model":str(args.model),"operator":f"Granite MoE {label} product materialization","layer":args.layer,"parameter":target_name,"candidate":f"native Granite {label} product in model dtype","reference":f"same Granite MoE block with only {label} product in FP32 then cast back","input_source":str(args.text_source),"comparison_scope":{"same_model_weights":True,"same_input_ids":True,"same_routing_and_expert_projections":True,"single_changed_boundary":f"{label} product materialization"},"rows":rows,"summary":{"state_count":len(rows),"calibration_count":split,"confirmation_count":len(rows)-split,"gradient_effect_rms_mean":sum(r["gradient_effect_rms_over_native"] for r in rows)/len(rows),"write_effect_rms_mean":sum(r["write_effect_rms_over_native"] for r in rows)/len(rows),"confirmation_projection_interval_normal_95":interval(projections) if projections else None,"confirmation_projection_positive":sum(v>0 for v in projections),"confirmation_projection_negative":sum(v<0 for v in projections),"aligned_write_mean":sum(aligned)/len(aligned) if aligned else None,"aligned_write_interval_normal_95":interval(aligned) if aligned else None,"direction_norm":direction_norm},"claim_boundary":f"One Granite MoE checkpoint, one {label} product boundary and the declared text bank; no population or long-run loss claim."}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--model",type=Path,default=DEFAULT_MODEL); p.add_argument("--text-source",type=Path,default=DEFAULT_TEXT); p.add_argument("--parameter",default=None); p.add_argument("--layer",type=int,default=0); p.add_argument("--boundary",choices=("gate_product","output_gate"),default="gate_product"); p.add_argument("--sequence-length",type=int,default=64); p.add_argument("--states",type=int,default=16); p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--eps",type=float,default=1e-8); p.add_argument("--device",default="cuda:0"); p.add_argument("--output",type=Path,required=True); args=p.parse_args(); result=run(args); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); print(json.dumps({"output":str(args.output),**result["summary"]},ensure_ascii=False,sort_keys=True))


if __name__ == "__main__": main()
