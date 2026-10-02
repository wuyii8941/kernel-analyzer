#!/usr/bin/env python3
"""Natural Qwen3 attention output layout/materialisation probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3 import modeling_qwen3


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if x.numel() < 2:
        m = float(x.mean()); return [m, m]
    m = float(x.mean()); h = 1.96 * float(x.std(unbiased=True)) / (x.numel() ** 0.5)
    return [m-h, m+h]


def ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a)) / max(float(torch.linalg.vector_norm(b)), 1e-30)


def write(g: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    g = g.float(); return -lr*g/(g.abs()+eps)


def layout_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = modeling_qwen3.repeat_kv(key, module.num_key_value_groups)
    value_states = modeling_qwen3.repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None: scores = scores + attention_mask
    weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value_states)
    # Same permutation, but use reshape's view/copy decision instead of an
    # explicit transpose followed by contiguous materialisation.
    output = output.permute(0, 2, 1, 3).reshape(*query.shape[:2], -1)
    return output, weights


def run(args: argparse.Namespace) -> dict[str, Any]:
    bank = json.loads(args.input_bank.read_text(encoding="utf-8")); states = bank["states"][:args.states]
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager").to(device).train(); model.config.use_cache=False
    layer = model.model.layers[args.layer]
    target = {
        "q_proj": layer.self_attn.q_proj.weight,
        "k_proj": layer.self_attn.k_proj.weight,
        "v_proj": layer.self_attn.v_proj.weight,
    }[args.parameter]
    original = modeling_qwen3.eager_attention_forward; rows=[]; effects=[]
    try:
        for state in states:
            vals=state.get("token_ids",state.get("input_ids")); ids=torch.tensor([vals],dtype=torch.long,device=device); labels=ids.clone()
            model.zero_grad(set_to_none=True); modeling_qwen3.eager_attention_forward=original; lc=model(input_ids=ids,labels=labels,use_cache=False).loss; lc.backward(); gc=target.grad.detach().float().cpu().clone(); wc=write(gc,args.learning_rate,args.eps)
            model.zero_grad(set_to_none=True); modeling_qwen3.eager_attention_forward=layout_eager; lr=model(input_ids=ids,labels=labels,use_cache=False).loss; lr.backward(); gr=target.grad.detach().float().cpu().clone(); wr=write(gr,args.learning_rate,args.eps)
            e=wc-wr; effects.append(e); rows.append({"state_id":state.get("state_id",state.get("sequence_id")),"loss_difference":float((lc-lr).detach().cpu()),"gradient_effect_rms_over_reference":ratio(gc-gr,gr),"write_effect_rms_over_reference":ratio(e,wr),"write_aligned":float(torch.sum(e*wr))/max(float(torch.sum(wr*wr)),1e-30)})
            del ids,labels,lc,lr; torch.cuda.empty_cache()
    finally: modeling_qwen3.eager_attention_forward=original
    split=len(rows)//2; direction=torch.stack([x.double() for x in effects[:split]]).mean(0); norm=float(torch.linalg.vector_norm(direction)); proj=[]
    if norm: direction/=norm; proj=[float(torch.sum(x.double()*direction)) for x in effects[split:]]
    return {"schema":"kernel-analyzer-qwen3-attention-output-layout-natural-probe-v1","status":"COMPLETE","model":str(args.model),"layer":args.layer,"parameter":args.parameter,"operator":"Qwen3 attention output layout materialization","candidate":"native eager transpose(1,2).contiguous()","reference":"same eager attention with equivalent permute(...).reshape output layout","input_source":"declared real Qwen3 text input bank","claim_boundary":"One checkpoint, one layer/parameter, and declared bank; no population or loss-quality claim.","comparison_scope":{"same_model_weights":True,"same_input_ids":True,"same_attention_path":True,"same_attention_values":True,"single_changed_boundary":"attention output layout materialization"},"rows":rows,"summary":{"state_count":len(rows),"calibration_count":split,"confirmation_count":len(rows)-split,"gradient_effect_rms_mean":sum(r["gradient_effect_rms_over_reference"] for r in rows)/len(rows),"write_effect_rms_mean":sum(r["write_effect_rms_over_reference"] for r in rows)/len(rows),"write_aligned_interval_normal_95":interval([r["write_aligned"] for r in rows]),"loss_difference_interval_normal_95":interval([r["loss_difference"] for r in rows]),"heldout_write_projection_interval_normal_95":interval(proj) if proj else None,"heldout_write_projection_positive":sum(x>0 for x in proj),"heldout_write_projection_negative":sum(x<0 for x in proj),"calibration_direction_norm":norm,"exact_all_states":all(r["gradient_effect_rms_over_reference"]==0.0 and r["write_effect_rms_over_reference"]==0.0 and r["loss_difference"]==0.0 for r in rows)}}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--model",type=Path,default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B")); p.add_argument("--input-bank",type=Path,default=Path("results/coverage/qwen_seq128_input_bank.json")); p.add_argument("--layer",type=int,default=13); p.add_argument("--parameter",choices=("q_proj","k_proj","v_proj"),default="q_proj"); p.add_argument("--states",type=int,default=16); p.add_argument("--learning-rate",type=float,default=1e-4); p.add_argument("--eps",type=float,default=1e-8); p.add_argument("--device",default="cuda:0"); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    out=run(a); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); print(json.dumps(out["summary"],sort_keys=True))


if __name__=="__main__": main()
