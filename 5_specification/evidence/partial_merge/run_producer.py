"""Retained attention producer plus merge, with both empty-partial representations."""
import argparse
import ast
import gzip
import importlib.util
import inspect
import json
import math
import sys
import tempfile
from pathlib import Path

from run import ROOT, retained_function


def compiled_record(kernel):
    return {'ttir': kernel.asm['ttir'], 'signature': {str(k): str(v) for k, v in kernel.src.signature.items()},
            'argument_names': list(kernel.src.fn.arg_names),
            'constexprs': [{'index': list(k), 'value': v} for k, v in kernel.src.constants.items()]}


def call_record(function, *args, **kwargs):
    """Record the exact bound Python arguments before launch, including defaults."""
    import torch
    bound = inspect.signature(function.fn).bind(*args, **kwargs)
    bound.apply_defaults()
    return {name: ({'kind': 'tensor', 'data_ptr': value.data_ptr()} if isinstance(value, torch.Tensor)
                   else {'kind': type(value).__name__, 'value': value})
            for name, value in bound.arguments.items()}


def run(out):
    if out.exists():
        raise RuntimeError('output must be a new file')
    import torch
    import triton
    snippets, functions = [], {}
    for filename in ('VLLM_HELPERS.py.gz', 'VLLM_UA.py.gz'):
        path = ROOT/'5_specification/sources'/filename
        source = gzip.open(path, 'rt').read()
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and any(isinstance(d, ast.Attribute) and d.attr == 'jit' for d in node.decorator_list):
                snippet, lines = retained_function(path, node.name)
                snippets.append(snippet)
                functions[node.name] = {'source': filename, 'lines': lines}
    header = 'import torch, triton\nimport triton.language as tl\nfloat8_info=torch.finfo(torch.float8_e4m3fn)\n'
    with tempfile.TemporaryDirectory(prefix='attention-producer-', dir=ROOT/'.cache/tmp') as folder:
        path = Path(folder)/'retained_attention.py'
        path.write_text(header+'\n'.join(snippets))
        spec = importlib.util.spec_from_file_location('retained_attention', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        length, width, segments, block, block_q = 33, 32, 4, 16, 32
        q = torch.zeros((length, 1, width), dtype=torch.float16, device='cuda')
        k = torch.full((3, block, 1, width), float('nan'), dtype=torch.float16, device='cuda')
        v = torch.full_like(k, float('nan'))
        order = [2, 0, 1]
        for j in range(length):
            physical, slot = order[j//block], j % block
            k[physical, slot, 0] = 0
            v[physical, slot, 0] = torch.arange(width, device='cuda')+2*j-5
        table = torch.tensor([order], dtype=torch.int32, device='cuda')
        lengths = torch.tensor([length], dtype=torch.int32, device='cuda')
        prefix = torch.tensor([0, length], dtype=torch.int32, device='cuda')
        inputs = {'query': q, 'keys': k, 'values': v, 'block_table': table, 'seq_lens': lengths, 'query_offsets': prefix}
        def bits(tensor):
            return tensor.view(torch.int16 if tensor.element_size() == 2 else torch.int32).flatten().cpu().tolist()
        before = {name: bits(tensor) for name, tensor in inputs.items()}
        report = {'scope': 'One finite causal paged-attention task, no full-wrapper or general certification',
                  'semantic_verdict': 'not_established', 'source_functions': functions, 'dependency_setup': header,
                  'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'triton': triton.__version__,
                  'task': {'seq_len': length, 'head_size': width, 'heads': 1, 'block_size': block, 'tile_size': block,
                           'block_q': block_q, 'segments': segments, 'physical_blocks': order,
                           'causal': True, 'scale': 1, 'window': [-1, -1], 'softcap': 0,
                           'query_values': 'zero', 'key_values': 'zero', 'value_formula': '2*key_position+coordinate-5',
                           'expected_output_formula': 'query_position+coordinate-5'},
                  'received_inputs': before, 'runs': []}
        for sentinel in (-333, 777):
            maxima = torch.full((length, 1, segments), sentinel, dtype=torch.float32, device='cuda')
            masses = torch.full_like(maxima, sentinel)
            numerators = torch.full((length, 1, segments, width), sentinel, dtype=torch.float32, device='cuda')
            storage = torch.full((length*36+2,), sentinel, dtype=torch.float32, device='cuda')
            output = torch.as_strided(storage, (length, 1, width), (36, 32, 1), storage_offset=1)
            all_buffers = dict(inputs, output=output, maxima=maxima, masses=masses, numerators=numerators)
            def versions():
                return {name: tensor._version for name, tensor in all_buffers.items()}
            def layouts():
                return {name: {'storage_ptr': tensor.untyped_storage().data_ptr(),
                               'storage_nbytes': tensor.untyped_storage().nbytes(),
                               'data_ptr': tensor.data_ptr(), 'shape': list(tensor.shape),
                               'stride': list(tensor.stride()), 'dtype': str(tensor.dtype),
                               'device': str(tensor.device)} for name, tensor in all_buffers.items()}
            versions_before_producer = versions()
            layouts_before_producer = layouts()
            arguments = dict(output_ptr=output, query_ptr=q, key_cache_ptr=k, value_cache_ptr=v,
                sink_ptr=None, block_tables_ptr=table, seq_lens_ptr=lengths, alibi_slopes_ptr=None, qq_bias_ptr=None,
                scale=1.0, q_scale=1.0, k_scale=1.0, v_scale=1.0, out_scale=1.0, softcap=0.0,
                num_query_heads=1, num_queries_per_kv=1, block_table_stride=3,
                query_stride_0=32, query_stride_1=32, output_stride_0=36, output_stride_1=32, qq_bias_stride_0=0,
                BLOCK_SIZE=16, TILE_SIZE=16, HEAD_SIZE=32, HEAD_SIZE_PADDED=32,
                USE_ALIBI_SLOPES=False, USE_ALIBI_SQRT=False, USE_QQ_BIAS=False, USE_SOFTCAP=False, USE_SINKS=False,
                SLIDING_WINDOW=0, USE_CAUSAL=True, USE_PER_SEQ_CAUSAL=False, per_seq_causal_ptr=None,
                USE_MM_PREFIX=False, MAX_MM_RANGES=0, mm_prefix_range_ptr=None, rswa_prefix_lens_ptr=None,
                R_SWA_WINDOW=0, USE_R_SWA=False,
                stride_k_cache_0=512, stride_k_cache_1=32, stride_k_cache_2=32, stride_k_cache_3=1,
                stride_v_cache_0=512, stride_v_cache_1=32, stride_v_cache_2=32, stride_v_cache_3=1,
                query_start_len_ptr=prefix, BLOCK_Q=32, num_seqs=1, BLOCK_M=32, NUM_SEGMENTS_PER_SEQ=4,
                USE_FP8=False, IS_3D=True, segm_output_ptr=numerators, segm_max_ptr=maxima, segm_expsum_ptr=masses)
            producer_arguments = call_record(module.kernel_unified_attention, **arguments)
            producer = module.kernel_unified_attention[(2, 1, 4)](**arguments, num_warps=4)
            torch.cuda.synchronize()
            partials = {'maxima': bits(maxima), 'masses': bits(masses), 'numerators': bits(numerators)}
            after_producer = {name: bits(tensor) for name, tensor in inputs.items()}
            output_after_producer = bits(storage)
            versions_after_producer = versions()
            layouts_after_producer = layouts()
            versions_before_merge = versions()
            layouts_before_merge = layouts()
            merge_args = (output, numerators, maxima, masses, lengths, 1, 1,
                          1.0, 36, 32, 3, 16, 32, 32, prefix, 32, 4, False)
            merge_kwargs = dict(FP8_MIN=-448.0, FP8_MAX=448.0)
            merge_arguments = call_record(module.reduce_segments, *merge_args, **merge_kwargs)
            consumer = module.reduce_segments[(length, 1)](*merge_args, **merge_kwargs, num_warps=4)
            torch.cuda.synchronize()
            actual = storage.cpu().tolist()
            expected = {1+t*36+c: t+c-5 for t in range(length) for c in range(width)}
            differences = [i for i, value in expected.items() if not math.isfinite(actual[i]) or actual[i] != value]
            guards = [i for i, value in enumerate(actual) if i not in expected and value != sentinel]
            report['runs'].append({'initial_storage': sentinel, 'producer_grid': [2, 1, 4], 'merge_grid': [33, 1],
                'partials': partials, 'inputs_after_producer': after_producer,
                'output_after_producer': output_after_producer, 'final_storage': bits(storage),
                'inputs_after_merge': {name: bits(tensor) for name, tensor in inputs.items()},
                'partials_after_merge': {'maxima': bits(maxima), 'masses': bits(masses), 'numerators': bits(numerators)},
                'output_mismatch_addresses': differences, 'changed_guard_addresses': guards,
                'producer_call_arguments': producer_arguments, 'merge_call_arguments': merge_arguments,
                'producer_compiled': compiled_record(producer), 'merge_compiled': compiled_record(consumer)})
            report['runs'][-1]['storage_versions'] = {
                'before_producer': versions_before_producer, 'after_producer': versions_after_producer,
                'before_merge': versions_before_merge, 'after_merge': versions()}
            report['runs'][-1]['storage_layouts'] = {
                'before_producer': layouts_before_producer, 'after_producer': layouts_after_producer,
                'before_merge': layouts_before_merge, 'after_merge': layouts()}
        with out.open('x') as f:
            json.dump(report, f, indent=2, allow_nan=False)
        print(json.dumps({'device': report['device'], 'coordinates': length*width,
                          'runs': [{'differences': len(r['output_mismatch_addresses']), 'changed_guards': len(r['changed_guard_addresses'])}
                                   for r in report['runs']]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    run(parser.parse_args().out)
