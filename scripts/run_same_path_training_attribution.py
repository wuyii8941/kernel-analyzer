"""Frozen same-data attribution replay, not a new external confirmation set.

Reuses the original training implementation, tokens, seeds, evaluations and
settings. Only residual feedback is toggled; both modes retain residual storage.
Final checkpoints are saved and loaded again for independent endpoint evaluation.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from unittest.mock import patch

from scripts import run_adamw8bit_error_compensation_training as original

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / 'results/property/numerical_coverage_v1/adamw8bit_error_compensation_training_v1'


def freeze(output: Path):
    from kernel_analyzer.compensation_control import CompensationControl
    from torchao.optim import AdamW8bit
    from torchao.optim.subclass_8bit import OptimState8bit
    from transformers import AutoModelForCausalLM
    from transformers.models.mamba.modeling_mamba import MambaForCausalLM
    if output.exists():
        raise ValueError('Use a new experiment directory')
    old = original.verify_protocol(HISTORICAL)
    paths = {Path(__file__), Path(inspect.getfile(CompensationControl)),
             Path(inspect.getfile(AdamW8bit)), Path(inspect.getfile(OptimState8bit)),
             Path(inspect.getfile(MambaForCausalLM)),
             ROOT/'src/kernel_analyzer/adamw8bit_error_compensation.py',
             ROOT/'scripts/verify_adamw8bit_error_compensation_training.py',
             HISTORICAL/'protocol.json', HISTORICAL/'summary.json'}
    paths.update(Path(p) for p in old['source_sha256'])
    paths.update(HISTORICAL/'streams'/f'stream_{i:02d}.json' for i in range(8))
    protocol = dict(schema='same-path-training-attribution-v1',
        data_use='FROZEN_SAME_DATA_CAUSAL_REPLAY_NOT_UNSEEN_GENERALIZATION',
        steps=old['steps'], stream_count=8, train_banks=old['train_banks'],
        evaluation_steps=old['evaluation_steps'], evaluation_states=old['evaluation_states'],
        optimizer=old['optimizer'], conditions=['OFF','ON'],
        primary='FINAL_EVALUATION_LOSS_OFF_MINUS_ON',
        material_margin=old['material_improvement_margin'],
        inference_scope='PAIRED_T_INTERVAL_CONDITIONAL_ON_STREAM_SAMPLING_ASSUMPTIONS; NO_UNIVERSAL_FINITE_SAMPLE_CLAIM',
        multiplicity='ONE_PRIMARY_CONTRAST; HISTORICAL_BRIDGE_AND_INTERMEDIATE_STEPS_DESCRIPTIVE',
        comparison='identical legacy scalar arithmetic and residual allocation; only residual read multiplier 0/1 differs',
        reference_bridge='Historical compiled default and compensated loss reused; ON must first reproduce historical compensated',
        pilot_overlap='historical stream0 early data previously observed; report exclusion sensitivity',
        no_early_stopping=True, checkpoint_re_evaluation_required=True,
        source_sha256={str(p.resolve()):original.sha(p) for p in sorted(paths)})
    original.save_new(output/'protocol.json', protocol)
    original.save_new(output/'source_snapshot.json',
        {str(p.resolve()):p.read_text() for p in sorted(paths) if p.suffix=='.py'})
    print(json.dumps({'status':'FROZEN','streams':8,'conditions':['OFF','ON'],'steps':old['steps']}),flush=True)


def checked(output):
    protocol=original.load(output/'protocol.json')
    for p,h in protocol['source_sha256'].items():
        if original.sha(Path(p))!=h:raise ValueError('Frozen dependency changed: '+p)
    return protocol


def run(output: Path, device: str, worker: int, workers: int):
    import torch
    import torchao
    import transformers
    from transformers import AutoModelForCausalLM
    from kernel_analyzer.compensation_control import CompensationControl
    protocol=checked(output)
    load_model=AutoModelForCausalLM.from_pretrained
    for stream in range(worker,protocol['stream_count'],workers):
        order=['OFF','ON'] if stream%2==0 else ['ON','OFF']
        for mode in order:
            destination=output/'runs'/f'stream_{stream:02d}_{mode}.json'
            if destination.exists():
                record=original.load(destination)
                if record.get('status')!='COMPLETE' or record.get('protocol_sha256')!=original.sha(output/'protocol.json'):
                    raise ValueError('Invalid existing result')
                if original.sha(Path(record['checkpoint']))!=record['checkpoint_sha256']:
                    raise ValueError('Existing checkpoint differs')
                continue
            retained={}
            def loader(*args,**kwargs):
                model=load_model(*args,**kwargs);retained['model']=model;return model
            def factory(condition,parameters,settings):
                if condition!=mode:raise ValueError('Wrong comparison mode')
                optimizer=CompensationControl(parameters,compensation_enabled=mode=='ON',block_size=256,**settings)
                retained['optimizer']=optimizer
                return optimizer
            print(json.dumps({'event':'START','stream':stream,'mode':mode,'device':device}),flush=True)
            with patch.object(AutoModelForCausalLM,'from_pretrained',side_effect=loader), patch.object(original,'make_optimizer',side_effect=factory):
                result=original.run_condition(protocol,stream,mode,device)
            model=retained.pop('model');retained.clear()
            checkpoint=output/'checkpoints'/f'stream_{stream:02d}_{mode}.pt'
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            if checkpoint.exists():raise ValueError('Unfinished checkpoint exists; preserve and inspect it')
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint)
            del model
            torch.cuda.empty_cache()
            model=load_model(original.MODEL,dtype=torch.float32,local_files_only=True).to(device)
            model.config.use_cache=False
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
            values=original.evaluate(model,original.load(original.EVAL_BANK)['states'][:protocol['evaluation_states']],device)
            reported=result['evaluation_loss_by_step'][str(protocol['steps'])]
            equal=values==reported
            result.update(status='COMPLETE' if equal else 'CHECKPOINT_REEVALUATION_FAILED',
                stream_index=stream, protocol_sha256=original.sha(output/'protocol.json'),
                checkpoint=str(checkpoint), checkpoint_sha256=original.sha(checkpoint),
                reevaluated_loss=values, exact_endpoint_reproduction=equal,
                runtime=dict(torch=torch.__version__,torchao=getattr(torchao,'__version__','unknown'),
                             transformers=transformers.__version__,gpu=torch.cuda.get_device_name(device)))
            original.save_new(destination,result)
            print(json.dumps({'event':'COMPLETE','stream':stream,'mode':mode,'checkpoint_verified':equal,
                              'final_loss':sum(values)/len(values)}),flush=True)
            del model
            torch.cuda.empty_cache()
            if not equal:raise RuntimeError('Saved model does not exactly reproduce endpoint')


def summarize(output: Path):
    from scripts.verify_adamw8bit_error_compensation_training import interval
    protocol=checked(output);rows=[]
    for stream in range(8):
        pair={mode:original.load(output/'runs'/f'stream_{stream:02d}_{mode}.json') for mode in ['OFF','ON']}
        for r in pair.values():
            if r['status']!='COMPLETE' or not r['exact_endpoint_reproduction'] or original.sha(Path(r['checkpoint']))!=r['checkpoint_sha256']:
                raise ValueError('Invalid checkpoint evidence')
        old={r['condition']:r for r in original.load(HISTORICAL/'streams'/f'stream_{stream:02d}.json')['records']}
        end=str(protocol['steps'])
        mean=lambda r:sum(r['evaluation_loss_by_step'][end])/len(r['evaluation_loss_by_step'][end])
        reference_on=old['ADAMW8BIT_COMPENSATED_BLOCK256']
        rows.append(dict(stream=stream,off_minus_on=mean(pair['OFF'])-mean(pair['ON']),
            default_minus_off=mean(old['ADAMW8BIT_BLOCK256'])-mean(pair['OFF']),
            historical_on_minus_on=mean(reference_on)-mean(pair['ON']),
            historical_on_exact_parameter_match=reference_on['final_parameter_sha256']==pair['ON']['final_parameter_sha256']))
    gains=[r['off_minus_on'] for r in rows];ci=interval(gains)
    result=dict(schema='same-path-training-attribution-v1',rows=rows,
        primary=dict(mean=sum(gains)/len(gains),interval_95=ci,
            decision='MATERIAL_IMPROVEMENT' if ci[0]>protocol['material_margin'] else 'DETECTABLE_IMPROVEMENT' if ci[0]>0 else 'NOT_CONFIRMED'),
        excluding_pilot_stream=dict(mean=sum(gains[1:])/7,interval_95=interval(gains[1:]),data_use='SENSITIVITY'),
        status='MEASURED_WITH_CHECKPOINT_REEVALUATION',
        all_historical_on_parameters_reproduced=all(r['historical_on_exact_parameter_match'] for r in rows),
        scope=protocol['data_use'])
    original.save_new(output/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['freeze','run','summarize'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:2')
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();a.output=a.output.resolve()
    if not a.output.is_relative_to(ROOT):p.error('All new files must stay inside repository')
    if not 0<=a.worker<a.workers:p.error('Invalid worker index')
    if a.action=='freeze':freeze(a.output)
    elif a.action=='run':run(a.output,a.device,a.worker,a.workers)
    else:summarize(a.output)
