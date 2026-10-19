"""Replay the retained vLLM merge function on a declared finite test task.

Extracts the unchanged function and its two unchanged helpers into a temporary
module. It does not run the full vLLM launcher or attention producer. Equal
segment shifts make every expected coordinate an independently computed rational.
"""
import argparse
import ast
import gzip
import importlib.util
import json
import sys
import tempfile
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def retained_function(path, name):
    source = gzip.open(path, 'rt').read()
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise RuntimeError(f'expected exactly one function {name}')
    node = nodes[0]
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return '\n'.join(source.splitlines()[start-1:node.end_lineno])+'\n', [start, node.end_lineno]


def run(out):
    if out.exists():
        raise RuntimeError('output must be a new file')
    import torch
    import triton
    sys.path.insert(0, str(ROOT/'5_specification'))
    from kernel_spec import registry
    contract = registry.load('K14_COLL')
    snippets, locations = [], {}
    for file, name in (('VLLM_HELPERS.py.gz', 'cdiv_fn'), ('VLLM_HELPERS.py.gz', 'find_seq_idx'),
                       ('VLLM_UA.py.gz', 'reduce_segments')):
        snippet, lines = retained_function(ROOT/'5_specification/sources'/file, name)
        snippets.append(snippet)
        locations[name] = {'source': file, 'lines': lines}
    # Bounds only supply unused Python defaults. USE_FP8=False; explicit bounds
    # are also supplied at launch. No vLLM platform/FP8 behavior is exercised.
    header = ('import torch, triton\nimport triton.language as tl\n'
              'float8_info = torch.finfo(torch.float8_e4m3fn)\n')
    with tempfile.TemporaryDirectory(prefix='partial-merge-', dir=ROOT/'.cache/tmp') as folder:
        module_path = Path(folder)/'retained_merge.py'
        module_path.write_text(header+'\n'.join(snippets))
        spec = importlib.util.spec_from_file_location('retained_merge', module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        report = {'scope': 'Selected merge invocation only; no full producer binding or semantic certification',
                  'source_functions': locations, 'dependency_setup': header,
                  'torch': torch.__version__, 'triton': triton.__version__,
                  'device': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
                  'profile': 'scaled_numerator_out', 'semantic_verdict': 'not_established', 'cases': []}
        for seq_lens, offsets in (([5, 0, 3], [0, 2, 2, 3]), ([1, 9], [0, 1, 3])):
            tokens, heads, segments, width, tile = offsets[-1], 2, 4, 4, 2
            maxima = torch.full((tokens, heads, segments), float('nan'), device='cuda')
            masses = torch.full_like(maxima, float('nan'))
            numerators = torch.full((tokens, heads, segments, width), float('nan'), device='cuda')
            expected, tasks = {}, []
            for seq in range(len(seq_lens)):
                if offsets[seq] == offsets[seq+1]:
                    continue
                length = seq_lens[seq]
                if length <= 0:
                    raise RuntimeError('this experiment excludes queries with empty key sequences')
                # Independent partition construction: each segment holds consecutive
                # groups of ceil(length/(segments*tile)) tiles, ignoring empty slots.
                tiles = (length+segments*tile-1)//(segments*tile)
                starts = list(range(0, length, tiles*tile))
                for token in range(offsets[seq], offsets[seq+1]):
                    for head in range(heads):
                        parts = []
                        for segment, start in enumerate(starts):
                            mass = (1, 1, 2, 4)[segment]
                            values = [token*16+head*8+segment*2+c-5 for c in range(width)]
                            numerator = [mass*v for v in values]
                            maxima[token, head, segment] = token-head
                            masses[token, head, segment] = mass
                            numerators[token, head, segment] = torch.tensor(numerator, device='cuda')
                            parts.append({'shift': token-head, 'mass': mass, 'numerator': numerator})
                        total = sum(p['mass'] for p in parts)
                        hand = [sum(F(p['numerator'][c]) for p in parts)/total for c in range(width)]
                        task = dict(operation='merge', participant_ids=[f'segment-{s}' for s in range(len(parts))],
                                    empty_policy='undefined', parts=parts)
                        result = contract.adapter.reference('scaled_numerator_out', task).fields
                        assert result == {'out': hand}
                        tasks.append({'token': token, 'head': head, 'input': task, 'expected': list(map(str, hand))})
                        for c, value in enumerate(hand):
                            expected[1+token*12+head*4+c] = value
            lengths = torch.tensor(seq_lens, dtype=torch.int32, device='cuda')
            prefix = torch.tensor(offsets, dtype=torch.int32, device='cuda')
            buffers = {'maxima': maxima, 'masses': masses, 'numerators': numerators,
                       'seq_lens': lengths, 'query_offsets': prefix}
            def snapshot_inputs():
                return {name: tensor.view(torch.int32).flatten().cpu().tolist()
                        for name, tensor in buffers.items()}
            received = snapshot_inputs()
            runs = []
            for sentinel in (-12345, 23456):
                storage = torch.full((tokens*12+2,), sentinel, dtype=torch.float32, device='cuda')
                output = torch.as_strided(storage, (tokens, heads, width), (12, 4, 1), storage_offset=1)
                compiled = module.reduce_segments[(tokens, heads)](
                    output, numerators, maxima, masses, lengths, len(seq_lens), heads,
                    1.0, 12, 4, 0, tile, width, width, prefix, 1, segments, False,
                    FP8_MIN=-448.0, FP8_MAX=448.0, num_warps=4)
                torch.cuda.synchronize()
                actual = storage.cpu().tolist()
                mismatches = [i for i, v in expected.items() if F(actual[i]) != v]
                changed_guards = [i for i, v in enumerate(actual) if i not in expected and v != sentinel]
                runs.append({'initial_storage': sentinel, 'complete_storage': actual,
                             'output_mismatch_addresses': mismatches, 'changed_guard_addresses': changed_guards,
                             'inputs_after': snapshot_inputs()})
            compiled_record = {
                'ttir': compiled.asm['ttir'],
                'argument_names': list(compiled.src.fn.arg_names),
                'signature': {str(k): str(v) for k, v in compiled.src.signature.items()},
                'constexprs': [{'index': list(k) if isinstance(k, tuple) else k, 'value': v}
                              for k, v in compiled.src.constants.items()],
            }
            report['cases'].append({'seq_lens': seq_lens, 'query_offsets': offsets, 'grid': [tokens, heads],
                'segments': segments, 'tile_size': tile, 'value_width': width, 'output_strides': [12, 4, 1],
                'output_storage_offset': 1, 'inactive_input_slots': 'NaN',
                'received_inputs': received, 'input_encoding': 'float32 buffers as signed int32 bits; index buffers as int32 values',
                'compiled': compiled_record,
                'tasks': tasks, 'runs': runs})
    with out.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({'device': report['device'], 'cases': len(report['cases']),
                      'coordinates_per_run': sum(len(c['tasks'])*4 for c in report['cases']),
                      'all_equal': all(not r['output_mismatch_addresses'] and not r['changed_guard_addresses']
                                       for c in report['cases'] for r in c['runs'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
