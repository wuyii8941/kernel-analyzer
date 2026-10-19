"""Inspect a real public pooling call before admitting its task correspondence."""
import argparse
import json
from pathlib import Path
import sys

import torch

SPEC = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SPEC), str(SPEC / 'corpus')]
import call_binding
from calls.inductor_avg_pool2d import call
from kernel_analyzer import check
from kernel_analyzer.provenance import _Trace
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder


def record(value):
    return {'t': 'f32', 'shape': list(value.shape), 'v': value.detach().cpu().tolist()}


def inputs(seed):
    generator = torch.Generator().manual_seed(seed)
    return {name: (torch.rand(shape, generator=generator) * 2 - 1).cuda()
            for name, shape in [('x', (1, 1, 4, 4)), ('upstream', (1, 1, 2, 2))]}


class Case(check.Case):
    name = 'inductor_public_avg_pool2d_program_boundary'
    implementation = 'PyTorch 2.10 Inductor public avg_pool2d and complete input VJP'
    def inputs(self, seed):
        return inputs(seed)
    def launch(self, values):
        return call(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    path = SPEC / 'corpus/call_bindings/inductor_avg_pool2d.json'
    plan = json.loads(path.read_text())
    values = inputs(20261225)
    before = {k: record(v) for k, v in values.items()}
    with TritonLaunchRecorder(select=lambda name, index: False, copy_tensors=False):
        call(values)
    with TritonLaunchRecorder(copy_tensors=False, keep_storages=True) as recorder:
        actual = call(values)
    torch.cuda.synchronize()
    observation = {'subject': plan['subject'], 'launch_scope_complete': recorder.launch_count == len(recorder.launches),
        'launches': [launch.kernel_name for launch in recorder.launches],
        'tensor_arguments': {k: {'before': before[k], 'after': record(v)} for k, v in values.items()},
        'scalars': {}, 'returns': {k: record(v) for k, v in actual.items()},
        'public_outputs': ['/returns/' + k for k in actual]}
    report = {'probe_source': Path(__file__).read_text(),
        'call_source': (SPEC / 'corpus/calls/inductor_avg_pool2d.py').read_text(), 'plan': plan,
        'observation': observation, 'binding': call_binding.inspect(plan, observation, plan_path=path),
        'return_arguments': {name: [(i, a.name) for i, launch in enumerate(recorder.launches) for a in launch.args
            if a.kind == 'tensor' and a.storage_id == value.untyped_storage()._cdata] for name, value in actual.items()},
        'input_arguments': {name: [(i, a.name) for i, launch in enumerate(recorder.launches) for a in launch.args
            if a.kind == 'tensor' and a.storage_id == value.untyped_storage()._cdata] for name, value in values.items()},
        'launches': [{'name': launch.kernel_name, 'grid': list(launch.grid), 'ttir': launch.asm.get('ttir'),
            'ptx': launch.asm.get('ptx'), 'arguments': [{'name': a.name, 'kind': a.kind, 'shape': a.shape,
                'dtype': a.dtype, **({'value': a.value} if a.kind != 'tensor' else {})} for a in launch.args]}
            for launch in recorder.launches]}
    # Dispatch tracing can change how torch.compile executes. Keep its launch
    # sequence separate; it must never replace the primary recorded invocation.
    with TritonLaunchRecorder(copy_tensors=False) as traced_recorder:
        trace = _Trace(traced_recorder)
        with trace.mode:
            traced_outputs = call(values)
    report['separate_dispatch_trace'] = {
        'launches': [launch.kernel_name for launch in traced_recorder.launches],
        'outputs': {k: record(v) for k, v in traced_outputs.items()},
        'operators': [str(event[2]) for event in trace.events],
        'untraced_dispatch_interiors': trace.enclosed_launches,
        'meaning': 'Diagnostic only; a different launch sequence supplies no aligned producer evidence'}
    print('binding', report['binding']['binding_status'], report['binding']['reference_status'], flush=True)
    print('return_arguments', report['return_arguments'], flush=True)
    report['mode_a_diagnostic'] = check.run(Case(), dev=[20261225], conf=[20261226], repeats=1)
    with args.out.open('x') as file:
        json.dump(report, file, indent=2, default=str)
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
