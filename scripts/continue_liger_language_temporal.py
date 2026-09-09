#!/usr/bin/env python3
"""Continue all existing paired runs with sparse, predeclared direct-effect windows.

The training state advances normally. Counterfactual computation does not write
weights or moments. This is a post-confirmation extension, not new unseen data.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
import numpy as np
import torch
from kernel_analyzer.update_write import adamw_parameter_write
from scripts import run_liger_language_pilot as pilot
from scripts.probe_liger_language_checkpoints import Moments
from scripts.run_liger_language_confirmation import OUT as TRAINING, N
from scripts.run_liger_single_boundary_collapse import file_sha256
from scripts.run_training_numerical_v2 import BASE, ROOT, save_new

OUT = BASE / "language_temporal_extension"
WINDOWS = [(1025,1056),(2017,2048),(3041,3072),(4065,4096)]
SOURCES = ("scripts/continue_liger_language_temporal.py", "scripts/probe_liger_language_checkpoints.py",
           "scripts/run_liger_language_pilot.py", "scripts/run_liger_single_boundary_collapse.py", "src/kernel_analyzer/update_write.py")


def hashes(): return {p:file_sha256(ROOT/p) for p in SOURCES}


def freeze():
    checkpoints = {}
    for pair in range(N):
        for c in ("candidate","reference"):
            d=json.loads((TRAINING/f"pair{pair}"/c/"status.json").read_text())
            checkpoints[f"{pair}/{c}"]=d["final_checkpoint_sha256"]
    save_new(OUT/"plan.json", {"schema":"liger-language-temporal-extension-v1", "pairs":N,
        "data_use":"POST_CONFIRMATION_EXTENSION_ALL_EXISTING_PAIRS_NO_SELECTION",
        "start_step":1024,"end_step":4096,"direct_windows":WINDOWS,
        "evaluation_steps":[2048,3072,4096],"checkpoint_sha256":checkpoints,"sources":hashes(),
        "state_policy":"Continue actual master weights and AdamW moments without reset; same-state counterfactuals never written",
        "direct_scope":"tied embedding/head parameter only; all model parameters train normally",
        "direction_check":"Full-coordinate within-window split means and across-window mean Gram; first window direction never refit to later windows",
        "claim_boundary":"Only sampled windows have direct evidence. No claim that every intervening update is directional, no iid assumption on training steps, no result-dependent stopping."})


def train(pair, condition, device):
    plan=json.loads((OUT/"plan.json").read_text())
    if hashes()!=plan["sources"]: raise RuntimeError("Frozen extension source changed")
    source=TRAINING/f"pair{pair}"; protocol=json.loads((source/"protocol.json").read_text())
    old=source/condition/"final.pt"
    if file_sha256(old)!=plan["checkpoint_sha256"][f"{pair}/{condition}"]: raise RuntimeError("Checkpoint changed")
    for name,digest in protocol["liger_sources"].items():
        if file_sha256(pilot.PACKAGE/name)!=digest: raise RuntimeError("Liger source changed")
    for split in ("train","validation"):
        if file_sha256(source/f"{split}.npy")!=protocol["data"][split]["encoded_sha256"]: raise RuntimeError("Tokens changed")
    output=OUT/f"pair{pair}"/condition; output.mkdir(parents=True,exist_ok=False)
    save_new(output/"execution.json",{"plan_sha256":file_sha256(OUT/"plan.json"),"condition":condition,"pair":pair,"status":"STARTED"})
    spec=importlib.util.spec_from_file_location("liger_kernel",pilot.PACKAGE/"__init__.py",submodule_search_locations=[str(pilot.PACKAGE)])
    module=importlib.util.module_from_spec(spec);sys.modules["liger_kernel"]=module;spec.loader.exec_module(module)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(old,map_location="cpu",weights_only=True)
    if state["steps"]!=1024:raise RuntimeError("Wrong checkpoint")
    model,target=pilot.build_model(protocol,device)
    if list(dict(model.named_parameters()))!=list(state["master"]):raise RuntimeError("Parameter order mismatch")
    master={n:torch.nn.Parameter(p.to(device)) for n,p in state["master"].items()}
    optimizer=torch.optim.AdamW(list(master.values()),**protocol["optimizer"])
    optimizer.load_state_dict(state["optimizer"])
    opt=protocol["optimizer"]
    for key in ("lr","eps","weight_decay","foreach","fused"):
        if optimizer.param_groups[0][key]!=opt[key]:raise RuntimeError("Optimizer mismatch")
    losses=pilot.make_loss_modules(protocol,device)
    natural="CANDIDATE" if condition=="candidate" else "REPAIR"
    opposite="REPAIR" if condition=="candidate" else "CANDIDATE"
    tokens=np.load(source/"train.npy"); validation=np.load(source/"validation.npy")
    summaries=[];means=[];evaluations=[];stats=None
    with (output/"steps.jsonl").open("x") as log,(output/"direct.jsonl").open("x") as direct:
        for step in range(1025,4097):
            pilot.materialize(model,master)
            x,y,offsets=pilot.batch_from_stream(tokens,batch_size=2,sequence_length=128,stream=0,step=step-1,seed=20260906,repeat_within_batch=False,device=device)
            loss,_=pilot.backward_pass(model,losses[natural],x,y,target_name=target)
            if not np.isfinite(loss):raise RuntimeError("Nonfinite training loss")
            for name,p in model.named_parameters():master[name].grad=p.grad.detach().float() if p.grad is not None else None
            window=next(((a,b) for a,b in WINDOWS if a<=step<=b),None)
            if window:
                if step==window[0]:stats=Moments()
                before=master[target].detach().clone()
                natural_gradient=master[target].grad.detach().clone()
                _,counter_gradient=pilot.backward_pass(model,losses[opposite],x,y,target_name=target)
                moments=optimizer.state[master[target]]
                counter=adamw_parameter_write(before,counter_gradient,first=moments["exp_avg"],second=moments["exp_avg_sq"],prior_step=int(moments["step"]),learning_rate=opt["lr"],beta1=opt["betas"][0],beta2=opt["betas"][1],epsilon=opt["eps"],weight_decay=opt["weight_decay"])
                if not torch.equal(master[target].grad,natural_gradient):raise RuntimeError("Counterfactual changed training gradient")
                if step==window[0]:
                    expected=adamw_parameter_write(before,natural_gradient,first=moments["exp_avg"],second=moments["exp_avg_sq"],prior_step=int(moments["step"]),learning_rate=opt["lr"],beta1=opt["betas"][0],beta2=opt["betas"][1],epsilon=opt["eps"],weight_decay=opt["weight_decay"])
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            if window:
                actual=master[target].detach()-before
                if step==window[0] and not torch.equal(actual,expected):raise RuntimeError("Actual training write differs from cloned optimizer replay")
                effect=actual-counter if condition=="candidate" else counter-actual
                reference=counter if condition=="candidate" else actual
                record=stats.add(effect,reference)
                direct.write(json.dumps({"step":step,"offsets":offsets,**record},allow_nan=False)+"\n");direct.flush()
                if step==window[1]:
                    result={"start":window[0],"end":window[1],**stats.result()}
                    result["scope"]="32 successive evolving training states; descriptive temporal window, not iid samples"
                    means.append(stats.total/stats.n); summaries.append(result)
                    save_new(output/f"window_{step}.json",result)
                    stats=None
            log.write(json.dumps({"step":step,"loss":loss,"offsets":offsets},allow_nan=False)+"\n");log.flush()
            if step in plan["evaluation_steps"]:
                pilot.materialize(model,master);values=[]
                with torch.no_grad():
                    for i in range(protocol["evaluation_batches"]):
                        vx,vy,_=pilot.batch_from_stream(validation,batch_size=2,sequence_length=128,stream=0,step=i,seed=20260907,repeat_within_batch=False,device=device)
                        h=model.transformer(input_ids=vx,use_cache=False).last_hidden_state
                        logits=torch.nn.functional.linear(h.float(),model.lm_head.weight.float())
                        values.append(float(torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),vy.reshape(-1))))
                if not all(np.isfinite(values)):raise RuntimeError("Nonfinite evaluation")
                evaluations.append({"step":step,"shared_evaluation_loss":float(np.mean(values))})
                torch.save({"master":{n:p.detach().cpu() for n,p in master.items()},"optimizer":optimizer.state_dict(),"steps":step},output/f"checkpoint_{step}.pt")
                save_new(output/f"evaluation_{step}.json",evaluations[-1])
                print(json.dumps({"pair":pair,"condition":condition,"step":step}),flush=True)
    gram=[[float((a*b).sum()) for b in means] for a in means]
    save_new(output/"summary.json",{"status":"COMPLETE_TEMPORAL_EXTENSION","pair":pair,"condition":condition,"windows":summaries,"window_mean_gram":gram,"evaluations":evaluations,"plan_sha256":file_sha256(OUT/"plan.json"),"checkpoint_sha256":file_sha256(output/"checkpoint_4096.pt"),"scope":plan["claim_boundary"]})


def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=("freeze","train"));p.add_argument("--pairs",nargs="+",type=int,choices=range(N));p.add_argument("--device",default="cuda:0");args=p.parse_args()
    if args.command=="freeze":freeze();return
    if not args.pairs:p.error("train requires --pairs")
    for pair in args.pairs:
        for condition in ("candidate","reference"):train(pair,condition,torch.device(args.device))


if __name__=="__main__":main()
