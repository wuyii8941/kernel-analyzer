"""Versioned public-style attention training calls with complete mathematical VJPs."""
import argparse
import contextlib
import gzip
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import traceback

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import run as batch
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
from kernel_spec import registry
from kernel_spec.numeric_io import encode_finite


PROFILE = 'torch210_sdpa_defined_real_vjp'
TRITON_TUTORIAL = Path(__file__).resolve().parents[3] / '.cache/untrusted/triton_tutorial_v360/06-fused-attention.py'


def standard(q, k, v, mask, **options):
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, **options)


def load_tutorial():
    spec = importlib.util.spec_from_file_location('retained_triton_3_6_attention', TRITON_TUTORIAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tutorial_function(module):
    def call(q, k, v, mask, **options):
        if mask is not None or options.get('enable_gqa', False):
            raise ValueError('the retained tutorial interface does not accept mask or GQA')
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError('the retained tutorial requires equal q/k/v shapes')
        scale = options.get('scale')
        if scale is None:
            scale = 1.0 / (q.shape[-1] ** 0.5)
        return module.attention(q, k, v, options.get('is_causal', False), scale, False)
    return call


def record(value):
    return None if value is None else batch.record(value)


def execute(fn, q, k, v, mask, upstream, options, capture=False):
    values = [item.detach().clone().requires_grad_(True) for item in (q, k, v)]
    actual_mask = None if mask is None else mask.detach().clone()
    watched = [*values, upstream] + ([] if actual_mask is None else [actual_mask])
    before = [item.detach().contiguous().view(torch.uint8).clone() for item in watched]
    recorder = TritonLaunchRecorder(copy_tensors=False, keep_storages=True) if capture else None
    with recorder if recorder is not None else contextlib.nullcontext():
        out = fn(*values, actual_mask, **options)
        forward_count = len(recorder.launches) if recorder else 0
        gradients = torch.autograd.grad(out, values, upstream)
    if out.is_cuda:
        torch.cuda.synchronize()
    if not all(torch.equal(item.detach().contiguous().view(torch.uint8), old)
               for item, old in zip(watched, before)):
        raise ValueError('q, k, v, mask or upstream bytes changed')
    result = dict(out=out.detach(), dq=gradients[0].detach(), dk=gradients[1].detach(), dv=gradients[2].detach())
    launches = [] if recorder is None else [
        {'phase': 'forward' if number < forward_count else 'backward', 'kernel': launch.kernel_name,
         'grid': list(launch.grid), 'ttir': launch.asm.get('ttir')}
        for number, launch in enumerate(recorder.launches)]
    matches = {} if recorder is None else {name: [
        {'launch': number, 'argument': argument.name} for number, launch in enumerate(recorder.launches)
        for argument in launch.args if argument.kind == 'tensor' and
        argument.storage_id == value.untyped_storage()._cdata] for name, value in result.items()}
    return result, {'launches': launches, 'output_storage_arguments': matches,
                    'triton_scope': 'not_established',
                    'scope_note': 'Storage matching and phase presence do not establish complete producer coverage or a formal task binding.'}


def inputs(case, seed, device='cuda'):
    batch_size, hq, hkv, length, source, head_dim = case['shape']
    dtype_name = case.get('dtype', 'f32')
    dtype = {'f16': torch.float16, 'bf16': torch.bfloat16, 'f32': torch.float32}[dtype_name]
    generator = torch.Generator().manual_seed(seed)
    def draw(shape):
        return (torch.randint(-4, 5, shape, generator=generator).float() / 8).to(device=device, dtype=dtype)
    q = draw((batch_size, hq, length, head_dim))
    k = draw((batch_size, hkv, source, head_dim))
    v = draw((batch_size, hkv, source, head_dim))
    mask = None
    if case.get('mask') == 'boolean':
        mask = torch.tensor([[[(i + 2*j) % 3 != 0 for j in range(source)] for i in range(length)]], device=device)
        mask = mask.expand(batch_size, hq, length, source).clone()
    elif case.get('mask') == 'additive':
        mask = torch.tensor([[[(i-j) % 4 / -4 for j in range(source)] for i in range(length)]],
                            dtype=torch.float32, device=device).expand(batch_size, hq, length, source).clone()
    options = {name: case.get(name, default) for name, default in
               (('is_causal', False), ('scale', None), ('enable_gqa', False))}
    # Only the versioned standard determines the public output/upstream shape.
    out = standard(q, k, v, mask, **options)
    upstream = draw(out.shape)
    return q, k, v, mask, upstream, options


def mathematical_reference(q, k, v, mask, upstream, options):
    dtype_name = {torch.float16: 'f16', torch.bfloat16: 'bf16', torch.float32: 'f32'}[q.dtype]
    query = {'q': q.detach().cpu().tolist(), 'k': k.detach().cpu().tolist(), 'v': v.detach().cpu().tolist(),
             'upstream': upstream.detach().cpu().tolist(), 'dtype': dtype_name,
             'attn_mask': None if mask is None else mask.detach().cpu().tolist(),
             'attn_mask_dtype': None if mask is None else ('b8' if mask.dtype == torch.bool else 'f32'),
             'dropout_p': 0, **options}
    result = registry.query('ATTN', PROFILE, query)
    info = {'query': query, 'outcome': result.outcome, 'note': result.note, 'labels': result.labels,
            'scope': 'complete public output and q/k/v VJP under the defined-row exact-real profile'}
    if result.outcome != 'value':
        return info, None
    if set(result.fields) != {'out', 'dq', 'dk', 'dv'}:
        raise ValueError('attention reference omitted or added task outputs')
    info['fields'] = {name: batch.map_values(value, encode_finite) for name, value in result.fields.items()}
    return info, result.fields


def compare(actual, expected, mathematical, declared_atol=None):
    required = {'out', 'dq', 'dk', 'dv'}
    if set(actual) != required or set(expected) != required or set(mathematical) != required:
        raise ValueError('missing or extra attention output/gradient')
    numerical = batch.standard_comparison(actual, expected)
    result = {'standard': numerical,
              'candidate_math': {name: batch.reference_comparison(
                  actual[name], mathematical[name], expected_dtype=actual[name].dtype) for name in required},
              'standard_math': {name: batch.reference_comparison(
                  expected[name], mathematical[name], expected_dtype=expected[name].dtype) for name in required},
              'compatible': all(not value['nonfinite_pairs'] and value['allclose_rtol_1e-5_atol_1e-6']
                                for value in numerical.values())}
    if declared_atol is not None:
        by_output = {name: bool(torch.allclose(actual[name], expected[name], rtol=0, atol=declared_atol))
                     for name in required}
        result['candidate_source_tolerance'] = {
            'atol': declared_atol,
            'rtol': 0,
            'by_output': by_output,
            'all_outputs': all(by_output.values()),
            'meaning': 'Reproduction of the retained candidate source test threshold; not exact semantic equality.',
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['development', 'acceptance'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selection', type=Path, default=Path(__file__).with_name('selection_attention_20261019_v1.json'))
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    split = selection[args.split]
    args.out.mkdir(parents=True, exist_ok=False)
    if torch.__version__ != '2.10.0+cu128':
        raise ValueError('standard version mismatch')
    root = Path(__file__).resolve().parents[2]
    report = {'selection': selection, 'split': args.split, 'torch': torch.__version__,
              'device': torch.cuda.get_device_name(), 'samples': [],
              'formal_binding': 'not_established', 'bias_verdict': 'not_established',
              'precision': {'matmul_precision': torch.get_float32_matmul_precision(),
                            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32},
              'sources': {'runner': Path(__file__).read_text(), 'torch_functional': Path(inspect.getfile(F)).read_text()},
              'specification_sources': {name: (root / name).read_text() for name in
                                        ('contracts/ATTN.yaml', 'adapters/ATTN.py', 'kernel_spec/d1_attention.py')},
              'official_document': gzip.decompress((root / 'sources/T_SDPA.txt.gz').read_bytes()).decode(),
              'capture_setup': 'Recorder active before warmup, selecting none. No precision setting changed.'}
    if split['implementation'] == 'flag_gems':
        import flag_gems
        dist = importlib.metadata.distribution('flag_gems')
        if dist.version != '5.0.2':
            raise ValueError('candidate version mismatch')
        candidate = flag_gems.scaled_dot_product_attention
        source = Path(dist.locate_file('flag_gems/ops/attention.py'))
        report['candidate_sources'] = {'attention': source.read_text(), 'metadata': dist.read_text('METADATA')}
    else:
        if importlib.metadata.version('triton') != '3.6.0':
            raise ValueError('Triton version mismatch')
        module = load_tutorial()
        candidate = tutorial_function(module)
        report['candidate_sources'] = {'tutorial': TRITON_TUTORIAL.read_text()}
    for case in split['cases']:
        for seed in split['seeds']:
            row = {'case': case['id'], 'seed': seed}
            report['samples'].append(row)
            print('START', case['id'], seed, flush=True)
            try:
                q, k, v, mask, upstream, options = inputs(case, seed)
                row.update(inputs={'q': record(q), 'k': record(k), 'v': record(v), 'mask': record(mask)},
                           upstream=record(upstream), options=options)
                row['mathematical_reference'], mathematical = mathematical_reference(q, k, v, mask, upstream, options)
                expected, _ = execute(standard, q, k, v, mask, upstream, options)
                row['standard'] = {name: record(value) for name, value in expected.items()}
                if mathematical is None:
                    raise ValueError('complete mathematical reference unavailable')
                with TritonLaunchRecorder(select=lambda name, index: False, copy_tensors=False):
                    execute(candidate, q, k, v, mask, upstream, options)
                actual, capture = execute(candidate, q, k, v, mask, upstream, options, capture=True)
                declared_atol = 0.01 if split['implementation'] == 'triton_tutorial_3_6' and q.dtype == torch.float16 else None
                row.update(candidate={name: record(value) for name, value in actual.items()}, capture=capture,
                           comparison=compare(actual, expected, mathematical, declared_atol=declared_atol),
                           input_values_unchanged=True)
                row['status'] = 'compared' if {item['phase'] for item in capture['launches']} == {'forward', 'backward'} else 'partial_triton_capture'
            except Exception as exc:
                row.update(status='execution_or_reference_error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
            (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print('END', case['id'], seed, row['status'], row.get('error', ''), flush=True)
    failed = any(
        row['status'] != 'compared' or not row['comparison'].get(
            'candidate_source_tolerance', {'all_outputs': row['comparison']['compatible']})['all_outputs']
        for row in report['samples']
    )
    raise SystemExit(int(failed))


if __name__ == '__main__':
    main()
