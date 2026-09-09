#!/usr/bin/env python3
"""A new MoE combination implementation, using the frozen numerical analyzer.

Only the order of independent expert contributions changes; all arithmetic is
FP32. Candidate is the installed eager implementation, not an invented bug.
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
import types
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeExperts
from kernel_analyzer.training_numerical_analysis import analyze_artifact
from kernel_analyzer.update_write import adamw_parameter_write, WRITE_PROTOCOL
from scripts.run_training_bias_profile_v2_empirical import _new_stage_store, _new_original_statistics, _append_contrast, _finish_stages
from scripts.run_liger_single_boundary_collapse import file_sha256
from scripts.run_liger_language_pilot import DATA
from scripts.run_training_numerical_v2 import BASE, ROOT, hashes, save_new

OUT=BASE/"granite_expert_order_confirmation_v2"
MODEL=Path("/data1/tzh/models/ibm-granite/granite-3.1-1b-a400m-base")
TARGET="model.layers.0.block_sparse_moe.experts"
CARRIER="model.layers.0.input_layernorm.weight"


def reversed_forward():
    source=textwrap.dedent(inspect.getsource(GraniteMoeExperts.forward))
    tree=ast.parse(source)
    loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.iter,ast.Name) and n.iter.id=="expert_hit"]
    if len(loops)!=1:raise RuntimeError("No unique expert accumulation loop")
    loops[0].iter=ast.Call(func=ast.Attribute(value=ast.Name(id="expert_hit",ctx=ast.Load()),attr="flip",ctx=ast.Load()),args=[ast.Constant(0)],keywords=[])
    ast.fix_missing_locations(tree)
    namespace=dict(inspect.unwrap(GraniteMoeExperts.forward).__globals__)
    exec(compile(tree,"<frozen-reverse-expert-order>","exec"),namespace)
    return namespace["forward"],ast.unparse(tree)


def freeze():
    import pyarrow as pa
    _,variant=reversed_forward()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    source=DATA/"wikitext-validation.arrow"
    with pa.memory_map(str(source),'r') as f:
        table=pa.ipc.open_stream(f).read_all().slice(0,2000)
        tokens=tokenizer.encode('\n'.join(table.column('text').to_pylist()),add_special_tokens=False)
    if len(tokens)<32*33:raise RuntimeError("Insufficient input tokens")
    protocol=json.loads((BASE/"protocol.json").read_text())
    protocol.update(cases={"granite_expert_order":"granite_fp32_expert_order_layer0"},
        data_use="NEW_MODEL_AND_MOE_COMBINATION_IMPLEMENTATION_FIXED_SUITE_CONFIRMATION",
        selected_before_measurement=True,selection="Locally available Granite MoE; first layer; expert-sum order chosen by semantics, not error size",
        contrast_scope="Natural forward/backward comparison: original ascending expert order versus reverse expert order only at the first MoE; both FP32",
        source_sha256=hashes(),adapter_sha256=file_sha256(Path(__file__)),
        model_sources={p.name:file_sha256(p) for p in sorted(MODEL.iterdir()) if p.suffix in {'.json','.safetensors'}},
        model_implementation_sha256=file_sha256(Path(inspect.getfile(GraniteMoeExperts))),
        variant_source=variant,parameter_scope=CARRIER,target_module=TARGET,
        dtype="float32",tf32=False,experts_implementation="eager",optimizer_state="cold AdamW at each state",
        supersedes_unexecuted_adapter="granite_expert_order_confirmation: unit test found wrapper globals rather than original function globals; no model measurement was run",
        reference_not_ground_truth=True,negative_and_abstention_retained=True,
        family="MOE_EXPERT_CONTRIBUTION_ACCUMULATION",
        semantic_argument="Each expert computes a pure function of the same hidden states and fixed routing. Permuting their sum preserves the real-valued output and derivative; floating-point addition order may differ.",
        limitations="New relative comparison, not a claim the reverse order is better or that no earlier study analyzed MoE summation")
    save_new(OUT/"protocol.json",protocol)
    save_new(OUT/"inputs.json",{"source_sha256":file_sha256(source),"state_ids":[f"granite-order-{i:02d}" for i in range(32)],"token_ids":[tokens[i*33:(i+1)*33] for i in range(32)]})


def measure(device):
    protocol=json.loads((OUT/"protocol.json").read_text());inputs=json.loads((OUT/"inputs.json").read_text())
    if hashes()!=protocol["source_sha256"] or file_sha256(Path(__file__))!=protocol["adapter_sha256"]:raise RuntimeError("Frozen method changed")
    for name,digest in protocol["model_sources"].items():
        if file_sha256(MODEL/name)!=digest:raise RuntimeError("Model file changed")
    if file_sha256(Path(inspect.getfile(GraniteMoeExperts)))!=protocol["model_implementation_sha256"]:raise RuntimeError("Model code changed")
    reverse,code=reversed_forward()
    if code!=protocol["variant_source"]:raise RuntimeError("Variant changed")
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.float32,attn_implementation="eager",experts_implementation="eager").to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    parameter=dict(model.named_parameters())[CARRIER];parameter.requires_grad_(True)
    target=dict(model.named_modules())[TARGET]
    if type(target) is not GraniteMoeExperts:raise RuntimeError("Different expert runtime")
    if target.config._experts_implementation!="eager":raise RuntimeError("Declared eager expert implementation is not active")
    original=target.forward
    store=_new_stage_store(include_parameter_write=True);stats=_new_original_statistics();identities=[]
    for index,ids in enumerate(inputs["token_ids"]):
        tokens=torch.tensor([ids[:-1]],device=device);labels=tokens.clone()
        measured={}
        for condition in ("candidate","repeat","reference"):
            target.forward=types.MethodType(reverse,target) if condition=="reference" else original
            local=[]
            hook=target.register_forward_hook(lambda _m,_i,o:local.append(o.detach().clone()))
            try:
                model.zero_grad(set_to_none=True)
                loss=model(input_ids=tokens,labels=labels,use_cache=False).loss
                loss.backward()
            finally:hook.remove()
            if len(local)!=1:raise RuntimeError("Target invocation count differs")
            gradient=parameter.grad.detach().clone()
            write=adamw_parameter_write(parameter,gradient)
            measured[condition]=(local[0],gradient,write,float(loss))
        target.forward=original
        if not all(torch.equal(a,b) for a,b in zip(measured['candidate'][:3],measured['repeat'][:3])) or measured['candidate'][3]!=measured['repeat'][3]:raise RuntimeError("No-op repeat differs")
        for j,stage in enumerate(("LOCAL","PARAMETER_GRADIENT","PARAMETER_WRITE")):
            c,r=measured['candidate'][j],measured['reference'][j]
            _append_contrast(store,stage,c-r,r,stats)
        identities.append({"state_id":inputs['state_ids'][index],"exact_repeat":True,"implementation":"GraniteMoeExperts eager CUDA/PyTorch","target":TARGET})
        save_new(OUT/"states"/f"{index:02d}.json",{"state_id":inputs['state_ids'][index],"original_coordinate_statistics":{k:v[-1] for k,v in stats.items() if v}})
        print(json.dumps({"case":"granite_expert_order","state":index+1}),flush=True)
    raw={"schema":"kernel-analyzer-training-bias-profile-v2-raw-case","status":"COMPLETE","case_id":"granite_fp32_expert_order_layer0","contrast_id":"NATURAL_IMPLEMENTATION_COMPARISON_EXPERT_ORDER_ONLY","state_ids":inputs['state_ids'],"calibration_state_ids":inputs['state_ids'][:16],"confirmation_state_ids":inputs['state_ids'][16:],"carrier":CARRIER,"runtime_boundary":{"target":TARGET,"identities":identities,"implementation_source_sha256":protocol['model_implementation_sha256']},"determinism":{"all_exact":True},"parameter_write_protocol":WRITE_PROTOCOL,"original_coordinate_statistics":stats,"stages":_finish_stages(store),"protocol_sha256":file_sha256(OUT/'protocol.json'),"input_sha256":file_sha256(OUT/'inputs.json')}
    save_new(OUT/'raw.json',raw)
    report=analyze_artifact(raw,protocol);report['provenance'].update(data_use=protocol['data_use'],raw_sha256=file_sha256(OUT/'raw.json'))
    save_new(OUT/'recomputed.json',report)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','measure'));p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.command=='freeze':freeze();return
    try:measure(torch.device(a.device))
    except Exception as error:
        save_new(OUT/'failure.json',{'status':'ABSTAIN_EXECUTION_OR_REFERENCE_FAILURE','error':str(error),'replacement_used':False})
        raise


if __name__=='__main__':main()
