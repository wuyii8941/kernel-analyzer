#!/usr/bin/env python3
"""Probe the Mamba selective-scan z-gate product on real text states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, MambaForCausalLM
from transformers.models.mamba import modeling_mamba


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/state-spaces/mamba-130m-hf")
DEFAULT_TEXT = ROOT / "docs/root_cause_closure_current.md"


def windows(tokenizer: Any, source: Path, length: int, count: int) -> list[torch.Tensor]:
    tokens = tokenizer(source.read_text(encoding="utf-8"), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    stride = length * 2
    if tokens.numel() < count * stride + length:
        raise RuntimeError("text source is too short")
    return [tokens[i * stride : i * stride + length].clone() for i in range(count)]


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if x.numel() < 2:
        m = float(x.mean()); return [m, m]
    m = float(x.mean()); h = 1.96 * float(x.std(unbiased=True)) / (x.numel() ** 0.5)
    return [m - h, m + h]


def write(g: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    g = g.float(); return -lr * g / (g.abs() + eps)


def rel(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect)) / max(float(torch.linalg.vector_norm(reference)), 1e-30)


def fp32_z_scan(hidden_states, dt, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False, **kwargs):
    batch_size, intermediate_size, seq_len = hidden_states.shape
    input_dtype = hidden_states.dtype
    if delta_bias is not None:
        dt = dt + delta_bias.to(dt.dtype)[..., None]
    if delta_softplus:
        dt = modeling_mamba.F.softplus(dt)
    B = B.transpose(1, 2); C = C.transpose(1, 2)
    discrete_A = torch.exp(A[None, :, None, :] * dt[:, :, :, None])
    discrete_B = dt[:, :, :, None] * B[:, None, :, :].float()
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
    state = torch.zeros(batch_size, intermediate_size, A.shape[-1], dtype=input_dtype, device=hidden_states.device)
    outputs = []
    for index in range(seq_len):
        state = discrete_A[:, :, index] * state + deltaB_u[:, :, index]
        output = torch.matmul(state.to(input_dtype), C[:, index, :].unsqueeze(-1)).squeeze(-1)
        outputs.append(output)
    scan_output = torch.stack(outputs, dim=-1)
    if D is not None:
        scan_output = scan_output + hidden_states * D[None, :, None]
    if z is not None:
        # Only the final scan-output times SiLU(z) gate is evaluated in FP32.
        scan_output = (scan_output.float() * modeling_mamba.F.silu(z).float()).to(input_dtype)
    if return_last_state:
        return scan_output, state
    return scan_output


def run_once(model, target, ids, reference):
    model.zero_grad(set_to_none=True)
    original = modeling_mamba.mamba_selective_scan
    if reference:
        modeling_mamba.mamba_selective_scan = fp32_z_scan
    try:
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
    finally:
        modeling_mamba.mamba_selective_scan = original
    if target.grad is None:
        raise RuntimeError("target gradient missing")
    return float(loss.detach().cpu()), target.grad.detach().float().cpu().clone()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.input_bank is not None:
        bank_record = json.loads(args.input_bank.read_text(encoding="utf-8"))
        bank = [torch.tensor(row["token_ids"], dtype=torch.long) for row in bank_record["states"][:args.states]]
        if any(int(row.numel()) != args.sequence_length for row in bank):
            raise RuntimeError("input bank sequence length does not match --sequence-length")
        input_source = str(args.input_bank)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        bank = windows(tokenizer, args.text_source, args.sequence_length, args.states)
        input_source = str(args.text_source)
    model = MambaForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True).to(args.device).eval()
    model.config.use_cache = False
    target_name = args.parameter or f"backbone.layers.{args.layer}.mixer.out_proj.weight"
    target = dict(model.named_parameters())[target_name]
    effects=[]; writes=[]; refs=[]; rows=[]
    for state_id,tokens in enumerate(bank):
        ids=tokens.unsqueeze(0).to(args.device)
        lc,gc=run_once(model,target,ids,False); lr,gr=run_once(model,target,ids,True)
        wc=write(gc,args.learning_rate,args.eps); wr=write(gr,args.learning_rate,args.eps); e=wr-wc
        effects.append((gr-gc).double().reshape(-1)); writes.append(e.double().reshape(-1)); refs.append(wc.double().reshape(-1))
        rows.append({"state_id":state_id,"native_loss":lc,"fp32_z_gate_loss":lr,"loss_difference_fp32_minus_native":lr-lc,"gradient_effect_rms_over_native":rel(gr-gc,gc),"write_effect_rms_over_native":rel(e,wc)})
        del ids; torch.cuda.empty_cache()
    split=args.states//2; direction=torch.stack(effects[:split]).mean(0); norm=float(torch.linalg.vector_norm(direction)); projections=[]
    if norm: direction/=norm; projections=[float(torch.dot(x,direction)) for x in effects[split:]]
    aligned=[float(torch.dot(e,r))/max(float(torch.dot(r,r)),1e-30) for e,r in zip(writes[split:],refs[split:])]
    return {"schema":"kernel-analyzer-mamba-z-gate-materialization-natural-probe-v1","status":"COMPLETE","model":str(args.model),"operator":"Mamba selective-scan z-gate product materialization","layer":args.layer,"parameter":target_name,"candidate":"native sequential Mamba selective scan","reference":"same sequential scan with only scan_output*SiLU(z) evaluated in FP32 then cast back","input_source":input_source,"comparison_scope":{"same_model_weights":True,"same_input_ids":True,"same_recurrence":True,"single_changed_boundary":"z gate product materialization"},"rows":rows,"summary":{"state_count":len(rows),"calibration_count":split,"confirmation_count":len(rows)-split,"gradient_effect_rms_mean":sum(r["gradient_effect_rms_over_native"] for r in rows)/len(rows),"write_effect_rms_mean":sum(r["write_effect_rms_over_native"] for r in rows)/len(rows),"confirmation_projection_interval_normal_95":interval(projections) if projections else None,"confirmation_projection_positive":sum(v>0 for v in projections),"confirmation_projection_negative":sum(v<0 for v in projections),"aligned_write_mean":sum(aligned)/len(aligned),"aligned_write_interval_normal_95":interval(aligned),"direction_norm":norm},"claim_boundary":"One real Mamba checkpoint, one z-gate product boundary and the declared text bank; no population or long-run loss claim."}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",type=Path,default=DEFAULT_MODEL); p.add_argument("--text-source",type=Path,default=DEFAULT_TEXT); p.add_argument("--input-bank",type=Path,default=None); p.add_argument("--parameter",default=None); p.add_argument("--layer",type=int,default=0); p.add_argument("--sequence-length",type=int,default=64); p.add_argument("--states",type=int,default=16); p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--eps",type=float,default=1e-8); p.add_argument("--device",default="cuda"); p.add_argument("--output",type=Path,required=True); args=p.parse_args(); result=run(args); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n"); print(json.dumps({"output":str(args.output),**result["summary"]},ensure_ascii=False,sort_keys=True))


if __name__ == "__main__": main()
