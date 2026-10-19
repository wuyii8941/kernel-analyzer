"""Observe the original three reduction paths before reviewing their binding."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

SPEC = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SPEC), str(SPEC/'corpus')]
import call_binding
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder


def record(t):
    return {'t':'f32','shape':list(t.shape),'v':t.detach().cpu().tolist()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    path=SPEC/'corpus/calls/flaggems_mse.py'
    spec=importlib.util.spec_from_file_location('flaggems_mse_call',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    plan_path=SPEC/'corpus/call_bindings/flaggems_mse.json'
    plan=json.loads(plan_path.read_text())
    generator=torch.Generator().manual_seed(20261117)
    x=torch.randn((7,41),generator=generator).cuda()
    target=torch.randn((7,41),generator=generator).cuda()
    report={'plan':plan,'cases':[]}
    for reduction in ['none','sum','mean']:
        before={'x':record(x),'target':record(target)}
        with TritonLaunchRecorder() as recorder:
            outputs=module.call({'x':x,'target':target,'reduction':reduction})
        torch.cuda.synchronize()
        obs={'subject':plan['subject'],'launch_scope_complete':len(recorder.launches)==recorder.launch_count,
             'launches':[launch.kernel_name for launch in recorder.launches],
             'tensor_arguments':{name:{'before':before[name],'after':record(t)} for name,t in [('x',x),('target',target)]},
             'scalars':{'reduction':reduction},'returns':{'loss':record(outputs['loss'])},'public_outputs':['/returns/loss']}
        binding=call_binding.inspect(plan,obs,plan_path=plan_path)
        standard=F.mse_loss(x,target,reduction=reduction)
        case={'reduction':reduction,'observation':obs,'binding':binding,
              'standard':record(standard),'max_standard_difference':float((outputs['loss']-standard).abs().max()),
              'launches':[{'name':launch.kernel_name,'grid':launch.grid,'ttir':launch.asm['ttir'],
                           'parameters':[{'name':a.name,'kind':a.kind,'shape':a.shape,'dtype':a.dtype,
                                          **({'value':a.value} if a.kind!='tensor' else {})} for a in launch.args]}
                          for launch in recorder.launches]}
        report['cases'].append(case)
        print(reduction,obs['launches'],binding['binding_status'],binding.get('reference_status'),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(report,f,indent=2,default=str);f.write('\n')
    if any(c['binding'].get('binding_status') not in ('development_checked','bound') or
           c['binding'].get('reference_status')!='value' for c in report['cases']):
        raise SystemExit(1)


if __name__=='__main__':main()
