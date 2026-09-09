"""Real-text, full F+B Top-k comparison; fixed weights and cold AdamW replay.

Engineering integration probe, not a population test or long-run consequence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from kernel_analyzer.router_selection import selected_router_topk,forward_sha256
from kernel_analyzer.update_write import adamw_parameter_write,WRITE_PROTOCOL
from scripts.run_training_bias_profile_v2_empirical import _original_coordinate_row
from scripts.run_numerical_coverage import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','input-bank','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',required=True)
    p.add_argument('--states',type=int,default=2)
    p.add_argument('--layer',type=int,default=0)
    p.add_argument('--deterministic',action='store_true')
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        raise ValueError('New output under /data1/tzh required')
    bank=json.loads(a.input_bank.read_text());states=bank['states'][:a.states]
    if a.states<1 or len(states)!=a.states:raise ValueError('Insufficient states')
    if a.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='eager').to(a.device).train()
    model.config.use_cache=False
    router=model.model.layers[a.layer].block_sparse_moe.router
    for parameter in model.parameters():parameter.requires_grad_(False)
    router.layer.weight.requires_grad_(True)
    parameter_name=f'model.layers.{a.layer}.block_sparse_moe.router.layer.weight'
    source=forward_sha256(router)
    base=router.layer.weight.detach().cpu().clone()
    protocol=dict(schema='granite-selection-probe-v1',input_sha256=sha(a.input_bank),
        model_config_sha256=sha(a.model/'config.json'),router_source_sha256=source,
        parameter_scope=[parameter_name],states=a.states,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        comparison='ORIGINAL_TOPK_VS_STABLE_SORT_TOPK',backend='TRANSFORMERS_EAGER_ATEN',
        state_scope='FIXED_CHECKPOINT_TEXT_SUITE_COLD_OPTIMIZER',
        write_protocol=WRITE_PROTOCOL,equivalence_decision='NOT_ASSESSED',
        source_sha256={str(path.resolve()):sha(path) for path in [Path(__file__),
            Path('src/kernel_analyzer/router_selection.py'),Path('src/kernel_analyzer/selection_reference.py'),
            Path('src/kernel_analyzer/update_write.py'),Path('scripts/run_training_bias_profile_v2_empirical.py')]})
    save(a.output/'protocol.json',protocol)
    summaries=[]
    for index,state in enumerate(states):
        tokens=torch.tensor(state.get('token_ids',state.get('input_ids')),dtype=torch.long)
        if state.get('token_sha256') and hashlib.sha256(tokens.numpy().tobytes()).hexdigest()!=state['token_sha256']:
            raise ValueError('Token hash differs')
        ids=tokens.unsqueeze(0).to(a.device)
        outcomes=[]
        for replace in (False,True,False,True):
            torch.manual_seed(27000+index);torch.cuda.manual_seed_all(27000+index)
            model.zero_grad(set_to_none=True)
            records=[];captured=[]
            with selected_router_topk(router,expected_source_sha256=source,replace=replace,
                                      records=records,captured=captured):
                loss=model(input_ids=ids,labels=ids,use_cache=False).loss
                loss.backward()
            if len(captured)!=1:raise ValueError('Unexpected selected-router call count')
            grad=router.layer.weight.grad
            if grad is None:raise ValueError('Router parameter gradient absent')
            gradient=grad.detach().float().cpu().clone()
            write=adamw_parameter_write(base,gradient)
            outcomes.append(dict(**captured[0],gradient=gradient,write=write,
                                 loss=loss.detach().float().cpu(),records=records))
        if not torch.equal(base,router.layer.weight.detach().cpu()):raise ValueError('Model weights changed')
        same_scores=all(torch.equal(outcomes[0]['scores'],o['scores']) for o in outcomes[1:])
        deterministic=all(torch.equal(outcomes[j][key],outcomes[j+2][key])
            for j in (0,1) for key in ('values','indices','gradient','write','loss'))
        candidate,reference=outcomes[:2]
        statistics={stage:_original_coordinate_row(candidate[key]-reference[key],reference[key])
            for stage,key in [('LOCAL','values'),('PARAMETER_GRADIENT','gradient'),('PARAMETER_WRITE','write')]}
        path=a.output/f'state_{index:03d}.pt';torch.save(outcomes,path)
        record=dict(state_id=state.get('state_id',str(index)),same_scores=same_scores,
            repeat_determinism=deterministic,original_coordinate_statistics=statistics,
            changed_index_coordinates=int((candidate['indices']!=reference['indices']).sum()),
            candidate_minus_reference_loss=float(candidate['loss']-reference['loss']),
            raw_path=str(path),raw_sha256=sha(path),diagnostics=[o['records'] for o in outcomes],
            status='MEASURED_ENGINEERING_PROBE' if same_scores and deterministic else 'INVALID_COMPARISON')
        save(a.output/f'state_{index:03d}.json',record);summaries.append(record)
        print(json.dumps(dict(event='SELECTION_STATE_COMPLETE',state=index,status=record['status'])),flush=True)
    save(a.output/'summary.json',dict(protocol=protocol,records=summaries,
        full_three_stage_protocol_complete=False,training_quality_claim=False))


if __name__=='__main__':main()
