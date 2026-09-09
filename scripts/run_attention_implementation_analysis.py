#!/usr/bin/env python3
"""Measure one forced Flash-SDPA versus math-attention LLM boundary."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
MODEL=Path('/data1/tzh/models/Qwen/Qwen3-1.7B')
BANK=ROOT/'results/coverage/qwen_seq64_input_bank.json'
TARGET='model.layers.0.self_attn.q_proj.weight'
STAGES=('LOCAL','PARAMETER_GRADIENT','PARAMETER_WRITE')


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def load(path): return json.loads(path.read_text())


def save_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False); handle.write('\n')


def sources():
    import transformers.integrations.sdpa_attention as integration
    import transformers.models.qwen3.modeling_qwen3 as modeling
    paths=[Path(__file__).resolve(),
           ROOT/'src/kernel_analyzer/fixed_suite_implementation_capture.py',
           ROOT/'src/kernel_analyzer/optimizer_implementation_capture.py',
           ROOT/'src/kernel_analyzer/training_numerical_analysis.py',
           ROOT/'src/kernel_analyzer/training_bias_profile.py',
           ROOT/'src/kernel_analyzer/training_equivalence.py',
           ROOT/'src/kernel_analyzer/update_write.py',BANK,MODEL/'config.json',
           MODEL/'model.safetensors.index.json',
           Path(inspect.getfile(modeling)).resolve(),Path(inspect.getfile(integration)).resolve()]
    paths.extend(sorted(MODEL.glob('model-*.safetensors')))
    return {str(path):sha(path) for path in paths}


def freeze(output):
    if output.exists(): raise ValueError('freeze requires a new output directory')
    states=load(BANK)['states'][:32]
    if len(states)!=32: raise ValueError('exactly 32 input states required')
    state_ids=[str(row['sequence_id']) for row in states]
    protocol={
        'schema':'attention-implementation-analysis-v1',
        'status':'FROZEN_BEFORE_FULL_SUITE',
        'case_id':'qwen_layer0_flash_sdpa_vs_math_attention',
        'operator_family':'FUSED_CAUSAL_ATTENTION',
        'candidate':{'implementation':'torch.nn.functional.scaled_dot_product_attention',
                     'forced_backend':'SDPBackend.FLASH_ATTENTION'},
        'reference':{'implementation':'torch.nn.functional.scaled_dot_product_attention',
                     'forced_backend':'SDPBackend.MATH','reference_is_absolute_truth':False},
        'boundary':{'layer':0,'selected_sdpa_call_ordinal':0,
                    'all_later_attention_calls':'SDPBackend.MATH',
                    'causal_representation':'attn_mask=None,is_causal=True',
                    'input_has_padding':False},
        'model':str(MODEL),'input_bank':str(BANK),'target_parameter':TARGET,
        'state_ids':state_ids,'calibration_state_ids':state_ids[:16],
        'confirmation_state_ids':state_ids[16:],
        'primary_stage':'PARAMETER_WRITE','claim_scope':'FIXED_SUITE_UPDATE',
        'fixed_suite_margins':{'full_update_rms':0.01},
        'contrast_id':'LOCAL_ATTENTION_IMPLEMENTATION_SUBSTITUTION',
        'optimizer':{'implementation':'torch.optim.AdamW','lr':1e-4,
                     'betas':[.9,.999],'eps':1e-8,'weight_decay':0.0,
                     'moments':'ZERO_FOR_EACH_STATE','parameter_representation':'FP32_MASTER'},
        'data_use':'DEVELOPMENT_NEW_FAMILY_CONFIRMATION_NOT_BLIND',
        'prior_information':('Single-state feasibility showed nonzero local, gradient and loss '
                             'differences before this suite was frozen.'),
        'automation_boundary':{'human_review':['causal attention semantics','backend pair',
                                              'layer, parameter, margin and states'],
                               'automatic_after_freeze':['matched model replay','backend fail-closed',
                                                         'determinism','AdamW write','statistics','report']},
        'source_sha256':sources(),
    }
    save_new(output/'protocol.json',protocol)


def verify(output):
    protocol=load(output/'protocol.json')
    if protocol.get('schema')!='attention-implementation-analysis-v1':
        raise ValueError('unexpected protocol')
    for name,expected in protocol['source_sha256'].items():
        if not Path(name).is_file() or sha(Path(name))!=expected:
            raise ValueError('frozen dependency changed: '+name)
    return protocol


class AttentionBackendReplacement:
    def __init__(self,selected):
        self.selected=selected; self.calls=[]; self.original=None
    def __enter__(self):
        import torch.nn.functional as functional
        from torch.nn.attention import SDPBackend,sdpa_kernel
        self.original=functional.scaled_dot_product_attention; original=self.original
        calls=self.calls; selected=self.selected
        def wrapped(q,k,v,attn_mask=None,dropout_p=0.0,is_causal=False,scale=None,enable_gqa=False):
            ordinal=len(calls)
            backend=(SDPBackend.FLASH_ATTENTION
                     if selected=='FLASH' and ordinal==0 else SDPBackend.MATH)
            calls.append({'ordinal':ordinal,'backend':str(backend),'q_shape':list(q.shape),
                          'input_mask_was_none':attn_mask is None,
                          'input_is_causal':bool(is_causal)})
            with sdpa_kernel(backend):
                return original(q,k,v,None,dropout_p,True,scale=scale,enable_gqa=enable_gqa)
        functional.scaled_dot_product_attention=wrapped
        return self
    def __exit__(self,*unused):
        import torch.nn.functional as functional
        functional.scaled_dot_product_attention=self.original


def capture(output,device):
    import torch
    from transformers import AutoModelForCausalLM
    from kernel_analyzer.fixed_suite_implementation_capture import FixedSuiteImplementationCapture
    from kernel_analyzer.training_numerical_analysis import analyze_artifact
    from kernel_analyzer.update_write import adamw_parameter_write

    protocol=verify(output)
    if (output/'raw.json').exists(): raise ValueError('capture refuses to overwrite results')
    torch.manual_seed(314159); torch.cuda.manual_seed_all(314159)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,
                                               local_files_only=True).to(device).eval()
    model.config.use_cache=False; model.config._attn_implementation='sdpa'
    parameters=dict(model.named_parameters())
    if TARGET not in parameters: raise ValueError('target parameter absent')
    target=parameters[TARGET]
    for name,parameter in parameters.items(): parameter.requires_grad_(name==TARGET)
    base=target.detach().clone()

    def branch(row,selected):
        local=[]
        hook=model.model.layers[0].self_attn.register_forward_hook(
            lambda module,inputs,value:local.append(value[0].detach().float().cpu().clone()))
        try:
            with torch.no_grad(): target.copy_(base)
            model.zero_grad(set_to_none=True)
            tokens=torch.tensor([row['input_ids']],dtype=torch.long,device=device)
            with AttentionBackendReplacement(selected) as replacement:
                loss=model(input_ids=tokens,labels=tokens).loss
                loss.backward()
            torch.cuda.synchronize()
            if len(local)!=1 or target.grad is None: raise RuntimeError('boundary capture failed')
            if len(replacement.calls)!=model.config.num_hidden_layers:
                raise RuntimeError('unexpected attention call count')
            expected='SDPBackend.FLASH_ATTENTION' if selected=='FLASH' else 'SDPBackend.MATH'
            if expected not in replacement.calls[0]['backend']:
                raise RuntimeError('selected backend was not forced at the target call')
            if any('SDPBackend.MATH' not in row['backend'] for row in replacement.calls[1:]):
                raise RuntimeError('a later attention layer did not use the common backend')
            return {'loss':loss.detach().float().cpu(),'local':local[0],
                    'gradient':target.grad.detach().float().cpu().clone(),
                    'runtime_calls':replacement.calls}
        finally: hook.remove()

    collector=FixedSuiteImplementationCapture(protocol['state_ids'],STAGES)
    repetitions=[]; runtime_identity=None
    for index,row in enumerate(load(BANK)['states'][:32]):
        candidate=branch(row,'FLASH'); repeat=branch(row,'FLASH'); reference=branch(row,'MATH')
        checks={name:torch.equal(candidate[name],repeat[name])
                for name in ('loss','local','gradient')}
        if not all(checks.values()): raise RuntimeError(f'candidate repeat failed at state {index}')
        if runtime_identity is None: runtime_identity={
            'candidate_calls':candidate['runtime_calls'],'reference_calls':reference['runtime_calls']}
        write_candidate=adamw_parameter_write(base,candidate['gradient'],learning_rate=1e-4,
                                               beta1=.9,beta2=.999,weight_decay=0.0)
        write_reference=adamw_parameter_write(base,reference['gradient'],learning_rate=1e-4,
                                               beta1=.9,beta2=.999,weight_decay=0.0)
        collector.append({'LOCAL':(candidate['local'],reference['local']),
                          'PARAMETER_GRADIENT':(candidate['gradient'],reference['gradient']),
                          'PARAMETER_WRITE':(write_candidate,write_reference)})
        repetitions.append({'state_id':protocol['state_ids'][index],'exact':checks,
                            'candidate_loss':float(candidate['loss']),
                            'reference_loss':float(reference['loss'])})
        print(json.dumps({'event':'ATTENTION_IMPLEMENTATION_STATE','state':index+1}),flush=True)
    statistics,stages=collector.finish()
    raw={'schema':'kernel-analyzer-attention-implementation-raw-v1','status':'COMPLETE',
         'case_id':protocol['case_id'],'contrast_id':protocol['contrast_id'],
         'state_ids':protocol['state_ids'],'calibration_state_ids':protocol['calibration_state_ids'],
         'confirmation_state_ids':protocol['confirmation_state_ids'],
         'runtime_boundary':{'kind':'FORCED_SDPA_BACKEND_ONE_LAYER','identity':runtime_identity,
                             'fail_closed_backend_context':True},
         'carrier':TARGET,'determinism':{'all_exact':True,'rows':repetitions},
         'parameter_write_protocol':{'version':'adamw-readback-v2',
             'measurement':'parameter_after_step_minus_parameter_before_step',
             'representation':'FP32_MASTER_INITIALIZED_FROM_STORED_PARAMETER'},
         'original_coordinate_statistics':statistics,'stages':stages,
         'protocol_sha256':sha(output/'protocol.json'),'primary_update_endpoint':'PARAMETER_WRITE',
         'claim_boundary':('32 frozen Qwen text states, one forced layer-0 causal attention backend; '
                           'fixed suite, cold AdamW, no population or training-quality claim')}
    save_new(output/'raw.json',raw)
    save_new(output/'analysis.json',analyze_artifact(raw,protocol))


def report(output):
    from kernel_analyzer.training_numerical_analysis import analyze_artifact
    protocol=load(output/'protocol.json'); raw=load(output/'raw.json')
    save_new(output/'recomputed_analysis.json',analyze_artifact(raw,protocol))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','capture','report'))
    parser.add_argument('--output-root',type=Path,required=True); parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args(); output=args.output_root.resolve()
    if not output.is_relative_to(Path('/data1/tzh')): raise ValueError('output must be under /data1/tzh')
    if args.command=='freeze': freeze(output)
    elif args.command=='capture': capture(output,args.device)
    else: report(output)


if __name__=='__main__': main()
