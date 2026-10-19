"""Inspect full public RMSNorm task correspondence and the actual program-reference boundary."""
import argparse
import json
from pathlib import Path
import sys

import torch

SPEC = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SPEC), str(SPEC / 'corpus')]
import call_binding
from calls.flaggems_rms_norm import call
from kernel_analyzer import check
from kernel_analyzer.provenance import _Trace
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder


def record(t):
    return {'t': 'f32', 'shape': list(t.shape), 'v': t.detach().cpu().tolist()}


def inputs(seed):
    g = torch.Generator().manual_seed(seed)
    return {'x': (torch.randn(2, 7, generator=g) / 1024).cuda(),
            'weight': (torch.rand(7, generator=g) + .5).cuda(),
            'upstream': (torch.rand(2, 7, generator=g) * 2 - 1).cuda()}


def tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in tensors(v)]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in tensors(v)]
    return []


class Case(check.Case):
    name = 'flaggems_public_rms_norm_program_boundary'
    implementation = 'FlagGems 5.0.2, public F.rms_norm eps=None with full VJP'
    def inputs(self, seed):
        return inputs(seed)
    def launch(self, inp):
        return call(inp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    plan_path = SPEC / 'corpus/call_bindings/flaggems_rms_norm.json'
    plan = json.loads(plan_path.read_text())
    values = inputs(20261211)
    before = {n: record(v) for n, v in values.items()}
    with TritonLaunchRecorder(select=lambda name, index: False, copy_tensors=False):
        call(values)
    with TritonLaunchRecorder(copy_tensors=False, keep_storages=True) as rec:
        trace = _Trace(rec)
        with trace.mode:
            outputs = call(values)
    torch.cuda.synchronize()
    obs = {'subject': plan['subject'], 'launch_scope_complete': rec.launch_count == len(rec.launches),
           'launches': [l.kernel_name for l in rec.launches],
           'tensor_arguments': {n: {'before': before[n], 'after': record(v)} for n, v in values.items()},
           'scalars': {}, 'returns': {n: record(v) for n, v in outputs.items()},
           'public_outputs': ['/returns/' + n for n in outputs]}
    report = {'probe_source': Path(__file__).read_text(), 'call_source': (SPEC / 'corpus/calls/flaggems_rms_norm.py').read_text(),
              'plan': plan, 'observation': obs, 'binding': call_binding.inspect(plan, obs, plan_path=plan_path),
              'untraced_dispatch_interiors': trace.enclosed_launches}
    report['launches'] = [{'name': l.kernel_name, 'grid': l.grid, 'ttir': l.asm['ttir'],
                          'arguments': [{'name': a.name, 'kind': a.kind, 'shape': a.shape, 'dtype': a.dtype,
                                         **({'value': a.value} if a.kind != 'tensor' else {})} for a in l.args]}
                         for l in rec.launches]
    report['return_producers'] = {}
    for name, value in outputs.items():
        identity = value.untyped_storage()._cdata
        report['return_producers'][name] = {
            'captured_tensor_arguments': [(i, a.name) for i, l in enumerate(rec.launches) for a in l.args
                                          if a.kind == 'tensor' and a.storage_id == identity],
            'aten_events': [{'after_launches': event[0], 'operator': str(event[2])}
                            for event in trace.events if any(t.untyped_storage()._cdata == identity for t in tensors(event[5]))]}
    print('binding', report['binding']['binding_status'], report['binding']['reference_status'], flush=True)
    print('return producers', report['return_producers'], flush=True)
    # Mode A is deliberately separate from a drafted task binding. These two
    # development inputs diagnose evaluator coverage, not population bias.
    report['mode_a_diagnostic'] = check.run(Case(), dev=[20261211], conf=[20261212], repeats=1)
    with args.out.open('x') as f:
        json.dump(report, f, indent=2, default=str)
    print('diagnostic saved', args.out, flush=True)


if __name__ == '__main__':
    main()
