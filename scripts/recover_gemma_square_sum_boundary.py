#!/usr/bin/env python3
"""Bind the original internal Gemma output to its actual sum-of-squares formula.

No AOT endpoint is fabricated. This adapter uses the existing internal-reduction
capture path, with a separately checked formula and exact runtime source guard.
The numerical analysis code and old capture files are unchanged.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import torch
from scripts.run_training_numerical_v2 import BASE, ROOT, hashes, save_new
from scripts.run_liger_single_boundary_collapse import file_sha256

SYMBOL="triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_10"
OLD=ROOT/"results/property/three_mechanism_profiles_v1/runs/gemma4_text128_scan_0037/runtime_release"
OUT=BASE/"recovery/gemma_bound_square_sum"
EXPECTED='''
xnumel = 128
r0_numel = 1536
rnumel = r0_numel
RBLOCK: tl.constexpr = R0_BLOCK
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
xmask = xindex < xnumel
r0_base = tl.arange(0, R0_BLOCK)[None, :]
rbase = r0_base
x0 = xindex
_tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
    r0_index = r0_offset + r0_base
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = _tmp4 + tmp3
    _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
tmp4 = tl.sum(_tmp4, 1)[:, None]
tl.store(out_ptr0 + (x0), tmp4, xmask)
'''


def kernel(text):
    tree=ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==SYMBOL for t in node.targets):
            code=node.value.args[1].value
            return next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name==SYMBOL)
    raise ValueError("Declared kernel missing")


def check(text):
    fn=kernel(text); prefix=ast.parse(EXPECTED).body
    if [ast.dump(x) for x in fn.body[:len(prefix)]] != [ast.dump(x) for x in prefix]:
        raise ValueError("Square-sum source formula differs")
    # The original output must not be overwritten later in the same kernel.
    for node in fn.body[len(prefix):]:
        if any(isinstance(x,ast.Name) and x.id=="out_ptr0" for x in ast.walk(node)):
            raise ValueError("Output reused after the proved reduction")
    return hashlib.sha256(ast.dump(fn).encode()).hexdigest()


def reference(metadata,candidate,**unused):
    source=metadata.get("runtime_pointers",{}).get("in_ptr0")
    if not isinstance(source,torch.Tensor) or source.dtype!=torch.bfloat16 or source.numel()!=128*1536 or not source.is_contiguous():
        raise RuntimeError("Bound source storage differs")
    if candidate.dtype!=torch.float32 or candidate.numel()!=128 or not candidate.is_contiguous():
        raise RuntimeError("Bound output storage differs")
    return source.reshape(128,1536).float().square().sum(-1).reshape(candidate.shape)


def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=("prepare","capture"));p.add_argument("--device",default="cuda:0");args=p.parse_args()
    old_source=OLD/"trace/model__0_forward_segment0_executed/output_code.py"
    signature=check(old_source.read_text())
    if args.command=="prepare":
        save_new(OUT/"protocol.json",{"case_id":"gemma4_text128_scan_0037","target":"forward:17:out_ptr0",
            "reference_contract":"sum_j float32(in_ptr0[row,j])**2, with FP32 evaluation",
            "proof":"Masked disjoint contiguous reduction partitions; each j in [0,1536) included once; masked padding zero; no later output write",
            "mathematical_scope":"Real sum-of-squares identity, not exact floating-point equality or zero mean rounding error",
            "reference_method_compatibility_slot":"PARTIAL_REDUCTION_FROM_BOUND_INPUT dispatched to this explicit square-sum adapter",
            "aot_endpoint_claimed":False,"larger_boundary_substituted":False,
            "source_ast_sha256":signature,"source_file_sha256":file_sha256(old_source),"adapter_sha256":file_sha256(Path(__file__)),
            "capture_sources":hashes(),"data_use":"EXISTING_CASE_NEW_EXPLICIT_REFERENCE_BINDING_NOT_UNSEEN_CONFIRMATION"})
        release=OUT/"runtime_release";release.mkdir()
        for name in ("capture.json","campaign.json.gz","inventory.json.gz","trace"):
            (release/name).symlink_to((OLD/name).resolve(),target_is_directory=(OLD/name).is_dir())
        tasks=json.loads(__import__('gzip').open(BASE/"recovery/gemma_math_completed/same_dtype_tasks.json.gz","rt").read())
        # Preserve the honest no-AOT status; include_unresolved_tasks already
        # supports explicit internal-reference computations in the old runner.
        import gzip
        with gzip.open(release/"same_dtype_tasks.json.gz","wt") as handle:json.dump(tasks,handle)
        save_new(OUT/"case_plan.json",{"cases":[{"case_id":"gemma4_text128_scan_0037","task_id":"forward:17:out_ptr0","carrier":"model.language_model.per_layer_model_projection.weight","reference_method":"PARTIAL_REDUCTION_FROM_BOUND_INPUT","explicit_reference_contract":"BOUND_ROW_SQUARE_SUM_V1"}]})
        return
    protocol=json.loads((OUT/"protocol.json").read_text())
    if hashes()!=protocol["capture_sources"] or file_sha256(Path(__file__))!=protocol["adapter_sha256"]:raise RuntimeError("Frozen adapter changed")
    from scripts import capture_bound_endpoint_bias_formation_v21 as capture
    observer=capture.SameDtypeSemanticCandidateObserver
    class Guarded(observer):
        def __init__(self,**kwargs):
            observed=[]
            for module in kwargs["modules"]:
                if hasattr(module,SYMBOL):observed.append(check(Path(module.__file__).read_text()))
            if observed!=[protocol["source_ast_sha256"]]:raise RuntimeError("Actual executed kernel differs from proved source")
            super().__init__(**kwargs)
    capture.SameDtypeSemanticCandidateObserver=Guarded
    capture.partial_reduction_reference=reference
    sys.argv=["capture_bound_endpoint_bias_formation_v21.py","--architecture","gemma4","--model","/data1/tzh/models/google/gemma-4-E2B","--input-bank","results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128.json","--state-bank","results/property/three_mechanism_profiles_v1/input_banks/gemma4_text128_scan_0037.json","--release-dir",str(OUT/"runtime_release"),"--case-plan",str(OUT/"case_plan.json"),"--output-dir",str(OUT/"legacy"),"--spool-dir","/data1/tzh/cache/kernel-analyzer/gemma_bound_square_sum","--device",args.device,"--states","32","--allow-graph-breaks","--training-bias-profile-v2-output-dir",str(OUT/"raw")]
    capture.main()


if __name__=="__main__":main()
