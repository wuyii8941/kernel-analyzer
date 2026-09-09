#!/usr/bin/env python3
"""Run one original saved kernel before attempting full training integration.

Synthetic operands validate execution/addressing only, not training bias.
"""
import argparse
import ast
import json
import os
from pathlib import Path

from scripts.launch_detached_experiment import DATA_CACHE_ENV
from scripts.finalize_numerical_family import read, sha
from kernel_analyzer.selected_silu_product_reference import check_source, reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose a new output under /data1/tzh')
    os.environ.update(DATA_CACHE_ENV)
    manifest=read(a.manifest)
    adapter=Path(__file__).resolve().parents[1]/'src/kernel_analyzer/selected_silu_product_reference.py'
    if manifest['adapter_sha256']!=sha(adapter): raise ValueError('Adapter changed')
    selected=sorted((s['source'],r['symbol'],r['contract']) for s in manifest['sources']
                    for r in s['rows'] if r['status']=='SOURCE_CHECKED')
    if not selected: raise ValueError('No source-checked kernel')
    source,symbol,contract=selected[0]
    text=Path(source).read_text()
    if check_source(text,symbol)!=contract: raise ValueError('Source contract changed')
    assignment=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id==symbol for t in n.targets))
    kernel_source=assignment.value.args[1].value
    a.output.mkdir(parents=True)
    def save(name,value):
        with (a.output/name).open('x') as f: json.dump(value,f,indent=2,allow_nan=False)
    save('protocol.json',dict(symbol=symbol,source=source,source_sha256=sha(Path(source)),
        adapter_sha256=sha(adapter),script_sha256=sha(Path(__file__)),
        manifest_sha256=sha(a.manifest),selection='LEXICOGRAPHIC_SOURCE_THEN_SYMBOL',
        fixtures=['normal','wide_activation','zero_gradient'],seed=20260907,
        activation_layout='BATCH_CHANNEL_TIME_DENSE_TRANSPOSE_LIKE_SAVED_CALLER',
        scope='Synthetic local execution only; no training bias or allclose policy claim'))
    import torch
    from torch._inductor.async_compile import AsyncCompile
    torch.cuda.set_device(0)
    compiler=AsyncCompile()
    scope={symbol:compiler.triton(symbol,kernel_source)}
    compiler.wait(scope)
    kernel=scope[symbol]
    channels,length=contract['elements'],contract['sequence_length']
    torch.manual_seed(20260907)
    records=[]
    for fixture in ('normal','wide_activation','zero_gradient'):
        gradient=torch.randn(channels,length,device='cuda',dtype=torch.bfloat16)
        activation=torch.randn(length,channels,device='cuda',dtype=torch.float32).T.unsqueeze(0)
        if fixture=='wide_activation': activation.mul_(40)
        if fixture=='zero_gradient': gradient.zero_()
        output=torch.empty(channels,device='cuda',dtype=torch.bfloat16)
        gradient_before=gradient.clone(); activation_before=activation.clone()
        # This saved Inductor version embeds Grid1D in the kernel metadata;
        # use the same invocation signature as its original generated caller.
        kernel.run(gradient,activation,output,channels,
                   stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        expected=reference(dict(input_output_storage_aliases=[],runtime_pointers=dict(
            in_ptr0=gradient_before,in_ptr1=activation_before)),output,contract)
        if not torch.equal(gradient,gradient_before) or not torch.equal(activation,activation_before):
            raise RuntimeError('Read-only operands changed')
        if not torch.isfinite(output).all() or not torch.isfinite(expected).all():
            raise RuntimeError('Nonfinite preflight output')
        if fixture=='zero_gradient' and torch.count_nonzero(output):
            raise RuntimeError('Zero-gradient expression failed')
        records.append(dict(fixture=fixture,elements=channels,
            differing_elements=int(torch.count_nonzero(output!=expected)),
            max_abs_difference=float((output.double()-expected.double()).abs().max()),
            inputs_unchanged=True))
    save('result.json',dict(status='LOCAL_EXECUTION_COMPLETE',records=records,
        training_measurement_complete=False,torch_version=torch.__version__,
        device=torch.cuda.get_device_name(0)))
    print(json.dumps(records))


if __name__=='__main__': main()
