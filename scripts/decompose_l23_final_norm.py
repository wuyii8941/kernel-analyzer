#!/usr/bin/env python3
"""Test final-RMSNorm materialization as the source of the layer-23 carrier."""

from __future__ import annotations

import argparse, hashlib, json, os, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD_SRC = REPO / "archive" / "round1_code" / "src"
for path in (OLD_SRC, REPO):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
from scripts.long_horizon_trigger import atomic_json, build_model, load_eval_states, load_milestone, under_root

PARAMETER = "model.layers.23.self_attn.q_proj.weight"
ROWS = slice(1152, 1280); COLUMNS = slice(1664, 1792)

def projection(value, direction, torch):
    return float(torch.dot(value.reshape(-1).float(), direction.reshape(-1).float()) / direction.float().norm())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("results/final/long_horizon_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--direction", type=Path, default=Path("results/final/l23_qproj_tile_direction.pt"))
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--state-index", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    bank_path=under_root(args.bank,"bank"); model_path=under_root(args.model,"model")
    direction_path=under_root(args.direction,"direction"); output_path=under_root(args.output,"output")
    states_idx=args.state_index or list(range(8,40))
    if min(states_idx)<8 or len(set(states_idx))!=len(states_idx): raise ValueError("held-out states required")
    os.environ.setdefault("HF_HOME","/data1/tzh/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE","/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE","/data1/tzh/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE","/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("HF_DATASETS_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR","/data1/tzh/cache/kernel_analyzer/tile_causal_compile")

    import torch
    from torch._dynamo.backends.registry import lookup_backend
    from torch._inductor.codecache import PyCodeCache
    from transformers import AutoTokenizer
    device=torch.device(args.device); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.use_deterministic_algorithms(True,warn_only=True)
    bank=json.loads(bank_path.read_text()); milestone=next(r for r in bank["milestones"] if int(r["step"])==args.step)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=True)
    all_states,evaluation=load_eval_states(tokenizer,1024,max(states_idx)+1,device)
    model=build_model(model_path,device); load_milestone(model,milestone,model_path)
    parameter=dict(model.named_parameters())[PARAMETER]
    direction=torch.load(direction_path,map_location="cpu",weights_only=False)["direction"].float().to(device)

    class LossStep(torch.nn.Module):
        def __init__(self,subject): super().__init__(); self.subject=subject
        def forward(self,input_ids,labels): return self.subject(input_ids=input_ids,labels=labels,use_cache=False,return_dict=False)[0]
    start=len(PyCodeCache.modules); candidate=torch.compile(LossStep(model),backend=lookup_backend("inductor"),fullgraph=True,dynamic=False)
    model.zero_grad(set_to_none=True); loss=candidate(*all_states[states_idx[0]]); loss.backward(); torch.cuda.synchronize(device)
    modules=list(PyCodeCache.modules[start:])
    forward_matches=[]; backward_matches=[]
    for module in modules:
        path=Path(module.__file__); source=path.read_text()
        if "bmm_76]" in source and "mm_198]" in source: backward_matches.append((module,path,source))
        if "hidden_states_394, logits" in source and "prepare_softmax_online" in source: forward_matches.append((module,path,source))
    if len(forward_matches)!=1 or len(backward_matches)!=1:
        raise RuntimeError(f"compile binding failed: forward={len(forward_matches)}, backward={len(backward_matches)}")
    fmod,fpath,fsrc=forward_matches[0]; bmod,bpath,bsrc=backward_matches[0]
    fstart=fsrc.index("def call("); terminal_logits_pos=fsrc.index("hidden_states_394, logits",fstart)
    fcalls=list(re.finditer(r"([A-Za-z0-9_]*add_mean_mul_pow_rsqrt_14)\.run\(",fsrc[fstart:terminal_logits_pos]))
    if not fcalls: raise RuntimeError("final RMSNorm forward kernel not found")
    fcall=fcalls[-1]; fsymbol=fcall.group(1); fabs=fstart+fcall.start(); ford=fsrc[fstart:fabs].count(f"{fsymbol}.run(")
    fkernel=getattr(fmod,fsymbol)
    bstart=bsrc.index("def call("); mmpos=bsrc.index("mm_198]",bstart); mm199=bsrc.index("mm_199]",mmpos)
    bcalls=list(re.finditer(r"([A-Za-z0-9_]*add_div_expand_mul_pow_sum_view_3)\.run\(",bsrc[mmpos:mm199]))
    if len(bcalls)!=1: raise RuntimeError("final RMSNorm backward kernel not found")
    bcall=bcalls[0]; bsymbol=bcall.group(1); babs=mmpos+bcall.start(); bord=bsrc[bstart:babs].count(f"{bsymbol}.run(")
    bkernel=getattr(bmod,bsymbol)

    def eager_forward(h,w):
        x=h.float(); r=torch.rsqrt(x.square().mean(dim=-1,keepdim=True)+1e-6)
        n=w * (x*r).to(h.dtype)
        return r,n
    def eager_vjp(h,w,dn,r):
        x=h.float(); u=(dn*w).float()
        dot=(u*x).mean(dim=-1,keepdim=True)
        return (u*r - x*(r*r*r)*dot).to(h.dtype)

    eager={}
    def capture_eager(inputs):
        eager.clear()
        def hook(_m,vals,out):
            eager["H"]=vals[0].detach().clone(); eager["N"]=out.detach().clone()
            vals[0].register_hook(lambda g: save_grad("T",g)); out.register_hook(lambda g: save_grad("Dn",g))
        def save_grad(name,g): eager[name]=g.detach().clone(); return g
        handle=model.model.norm.register_forward_hook(hook); model.zero_grad(set_to_none=True)
        try:
            loss=model(input_ids=inputs[0],labels=inputs[1],use_cache=False,return_dict=False)[0]; loss.backward(); torch.cuda.synchronize(device)
        finally: handle.remove()
        if set(eager)!={"H","N","Dn","T"}: raise RuntimeError(f"eager capture incomplete: {sorted(eager)}")
        return float(loss.detach().float().cpu()),parameter.grad.detach()[ROWS,COLUMNS].clone()

    original_f=fkernel.run; original_b=bkernel.run
    def run_candidate(inputs,arm):
        if arm not in {"C","F_REF","FB_REF","SHAM"}: raise ValueError(arm)
        fc={"v":0}; bc={"v":0}; obs={}; model.zero_grad(set_to_none=True)
        def wf(*vals,**kw):
            o=fc["v"];fc["v"]+=1
            if o!=ford:return original_f(*vals,**kw)
            result=original_f(*vals,**kw); obs["f"]=True
            h=vals[0]; r=vals[1]; w=vals[3]; n=vals[4]
            obs["H"]=h.detach().clone();obs["R_candidate"]=r.detach().clone();obs["N_candidate"]=n.detach().clone()
            rr,nn=eager_forward(h,w)
            obs["R_reference"]=rr.detach().clone();obs["N_reference"]=nn.detach().clone()
            if arm in {"F_REF","FB_REF"}: r.copy_(rr); n.copy_(nn)
            elif arm=="SHAM": r.copy_(r.clone()); n.copy_(n.clone())
            return result
        def wb(*vals,**kw):
            o=bc["v"];bc["v"]+=1
            if o!=bord:return original_b(*vals,**kw)
            obs["b"]=True; dn=vals[0].detach().clone();h=vals[2];r=vals[3]
            if arm=="FB_REF": vals[0].copy_(eager_vjp(h,vals[1],dn,r)); result=None
            else: result=original_b(*vals,**kw)
            obs["T"]=vals[0].detach().clone(); return result
        fkernel.run=wf;bkernel.run=wb
        try:
            loss=candidate(*inputs);loss.backward();torch.cuda.synchronize(device)
        finally:fkernel.run=original_f;bkernel.run=original_b
        if not obs.get("f") or not obs.get("b"):raise RuntimeError("final norm F+B not observed")
        return float(loss.detach().float().cpu()),parameter.grad.detach()[ROWS,COLUMNS].clone(),obs

    rows=[]
    for state in states_idx:
        inputs=all_states[state];eloss,etile=capture_eager(inputs)
        er,en=eager_forward(eager["H"],model.model.norm.weight.detach()); et=eager_vjp(eager["H"],model.model.norm.weight.detach(),eager["Dn"],er)
        closs,tc,oc=run_candidate(inputs,"C");_,tf,of=run_candidate(inputs,"F_REF");_,tfb,ofb=run_candidate(inputs,"FB_REF");_,tsh,osh=run_candidate(inputs,"SHAM")
        fremove=tc.float()-tf.float(); bremove=tf.float()-tfb.float(); total=tc.float()-tfb.float()
        row={
            "state_index":state,"offset":evaluation["offsets"][state],"token_sha256":evaluation["token_sha256"][state],"eager_loss":eloss,"candidate_loss":closs,
            "eager_forward_replay_matches":bool(torch.equal(en,eager["N"])),"eager_forward_replay_max_abs":float((en.float()-eager["N"].float()).abs().max()),
            "eager_vjp_replay_matches":bool(torch.equal(et,eager["T"])),"eager_vjp_replay_max_abs":float((et.float()-eager["T"].float()).abs().max()),
            "candidate_minus_eager_projection":projection(tc.float()-etile.float(),direction,torch),
            "forward_materialization_removal_projection":projection(fremove,direction,torch),"backward_materialization_additional_removal_projection":projection(bremove,direction,torch),
            "joint_final_norm_removal_projection":projection(total,direction,torch),"joint_residual_projection":projection(tfb.float()-etile.float(),direction,torch),
            "stage_closure_max_abs":float((fremove+bremove-total).abs().max()),"candidate_restoration_sham_max_abs":float((tsh.float()-tc.float()).abs().max()),
            "candidate_r_vs_materialized_r_max_abs":float((oc["R_candidate"]-oc["R_reference"]).abs().max()),"candidate_n_vs_materialized_n_max_abs":float((oc["N_candidate"].float()-oc["N_reference"].float()).abs().max()),
        };rows.append(row);print(json.dumps({"state":state,"total":row["candidate_minus_eager_projection"],"F":row["forward_materialization_removal_projection"],"B":row["backward_materialization_additional_removal_projection"]},sort_keys=True),flush=True)
    result={"schema":"kernel-analyzer-l23-final-norm-materialization-v1","status":"COMPLETE","checkpoint_step":args.step,"state_indices":states_idx,"parameter":PARAMETER,"tile":{"rows":[ROWS.start,ROWS.stop],"columns":[COLUMNS.start,COLUMNS.stop]},"equations":["H=BF16(residual+MLP)","r=rsqrt(mean(FP32(H)^2)+eps)","N=w*BF16(FP32(H)*r)","dH=r*u-H*r^3*mean(u*H), u=FP32(Dn*w)"],"mechanism":"Inductor fuses residual add into RMSNorm and delays the BF16 materialization used by eager before the RMS reduction; it also delays the normalized-value BF16 cast before weight multiplication","binding":{"forward_kernel":fsymbol,"forward_ordinal":ford,"backward_kernel":bsymbol,"backward_ordinal":bord,"forward_source_sha256":hashlib.sha256(fpath.read_bytes()).hexdigest(),"backward_source_sha256":hashlib.sha256(bpath.read_bytes()).hexdigest(),"candidate_values_used_to_select_boundary":False},"rows":rows,"tensor_values_saved":False}
    result["result_sha256"]=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(",",":")).encode()).hexdigest();atomic_json(output_path,result)
    print(json.dumps({"output":str(output_path),"rows":len(rows)},sort_keys=True))

if __name__=="__main__":main()
