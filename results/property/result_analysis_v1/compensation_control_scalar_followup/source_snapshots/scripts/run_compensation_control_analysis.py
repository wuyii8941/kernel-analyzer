"""Small controlled-gradient check before any new training attribution claim."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:2')
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error('Use a new output inside kernel-analyzer')
    import torch
    from torchao.optim import AdamW8bit
    from torchao.optim.subclass_8bit import OptimState8bit
    from kernel_analyzer.compensation_control import CompensationControl, TensorScalarCompensationControl
    from kernel_analyzer.adamw8bit_error_compensation import AdamW8bitErrorCompensated
    sources = [Path(__file__), Path(inspect.getfile(AdamW8bit)), Path(inspect.getfile(OptimState8bit)),
               Path(inspect.getfile(CompensationControl)), Path(inspect.getfile(AdamW8bitErrorCompensated))]
    settings = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    protocol = dict(schema='compensation-same-path-development-v1',
                    data_use='CONTROLLED_SYNTHETIC_DEVELOPMENT_NOT_TRAINING_CONFIRMATION',
                    sizes=[4096, 32768], seeds=[11, 29], steps=8, settings=settings,
                    torch_version=torch.__version__, device=a.device,
                    source_sha256={str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources},
                    question='Does compensation-off reproduce default; does on reproduce frozen compensated code?',
                    no_population_or_training_claim=True)
    out.mkdir(parents=True)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows=[]
    def value(x):
        return x.dequantize() if isinstance(x, OptimState8bit) else x
    def difference(x,y):
        u=(x-y).double(); denom=float(y.double().square().sum())
        return dict(exact_equal=bool(torch.equal(x,y)), max_abs=float(u.abs().max()),
                    relative_rms=(float(u.square().sum())/denom)**0.5 if denom else None)
    for size in protocol['sizes']:
        for seed in protocol['seeds']:
            gen=torch.Generator(device=a.device).manual_seed(seed)
            base=torch.randn(size, generator=gen, device=a.device)
            names=['default_compiled','default_eager','off','on','frozen_compensated','fp32','tensor_off','tensor_on']
            params={name:torch.nn.Parameter(base.clone()) for name in names}
            opts={name:AdamW8bit([params[name]], block_size=256, **settings) for name in names[:2]}
            opts.update({name:CompensationControl([params[name]], compensation_enabled=name=='on', **settings) for name in ['off','on']})
            opts['frozen_compensated']=AdamW8bitErrorCompensated([params['frozen_compensated']], **settings)
            opts['fp32']=torch.optim.AdamW([params['fp32']], foreach=False, fused=False, **settings)
            opts.update({name:TensorScalarCompensationControl([params[name]], compensation_enabled=name=='tensor_on', **settings) for name in ['tensor_off','tensor_on']})
            for step in range(1,protocol['steps']+1):
                grad=torch.randn(size,generator=gen,device=a.device)*torch.logspace(-4,0,size,device=a.device)
                before={k:v.detach().clone() for k,v in params.items()}
                for name in names:
                    params[name].grad=grad.clone()
                    if name=='default_eager':
                        with patch('torch.compile', side_effect=lambda f, **kwargs:f): opts[name].step()
                    else: opts[name].step()
                writes={k:v.detach()-before[k] for k,v in params.items()}
                comparisons={}
                for left,right in [('on','frozen_compensated'),('off','default_compiled'),('off','default_eager'),('default_eager','default_compiled'),('on','fp32'),('off','fp32'),('tensor_off','default_eager'),('tensor_on','fp32'),('tensor_off','fp32')]:
                    state_left=opts[left].state[params[left]];state_right=opts[right].state[params[right]]
                    comparisons[left+'__'+right]=dict(write=difference(writes[left],writes[right]),
                        **{key:difference(value(state_left[key]),value(state_right[key])) for key in ['exp_avg','exp_avg_sq']})
                rows.append(dict(size=size,seed=seed,step=step,comparisons=comparisons))
    torch.cuda.synchronize(a.device)
    result=dict(schema=protocol['schema'],status='MEASURED_NOT_TRAINING_CONFIRMED',rows=rows,
                on_matches_frozen_all_steps=all(r['comparisons']['on__frozen_compensated']['write']['exact_equal'] for r in rows),
                off_matches_default_all_steps=all(r['comparisons']['off__default_compiled']['write']['exact_equal'] for r in rows),
                off_matches_eager_default_all_steps=all(r['comparisons']['off__default_eager']['write']['exact_equal'] for r in rows),
                tensor_off_matches_eager_default_all_steps=all(r['comparisons']['tensor_off__default_eager']['write']['exact_equal'] for r in rows))
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__': main()
