"""Evaluate every program of the recorded merge calls with the existing DSL.

The recorded buffers are relocated into disjoint synthetic address ranges. This
is valid for this kernel's parameter-relative addressing, not a raw-address or
aliasing certificate. No capture hashes or automatic admission are generated.
"""
import argparse
import importlib.util
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'2_tool/src'))
from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch
from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator, NumericMode

_spec = importlib.util.spec_from_file_location('partial_merge_evidence_verify', Path(__file__).with_name('verify.py'))
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)
verify, require = _audit.verify, _audit.require


def launch_from_case(case, run):
    compiled = case['compiled']
    names = compiled['argument_names']
    signature = compiled['signature']
    expected_names = ['output_ptr', 'segm_output_ptr', 'segm_max_ptr', 'segm_expsum_ptr', 'seq_lens_ptr',
                      'num_seqs', 'num_query_heads', 'out_scale_inv', 'output_stride_0', 'output_stride_1',
                      'block_table_stride', 'TILE_SIZE', 'HEAD_SIZE', 'HEAD_SIZE_PADDED', 'query_start_len_ptr',
                      'BLOCK_Q', 'NUM_SEGMENTS_PER_SEQ', 'USE_FP8', 'FP8_MIN', 'FP8_MAX']
    require(names == expected_names and set(signature) == set(names), 'unexpected compiled interface')
    constants = {tuple(r['index']): r['value'] for r in compiled['constexprs']}
    require(len(constants) == len(compiled['constexprs']), 'duplicate compiled constants')
    require(constants == {(6,): 2, (11,): 2, (12,): 4, (13,): 4, (15,): 1,
                          (16,): 4, (17,): False, (18,): -448.0, (19,): 448.0}, 'unexpected compiled constants')
    received = case['received_inputs']
    tensors = {'output_ptr': ('output', [3, 2, 4], [12, 4, 1]),
               'segm_output_ptr': ('numerators', [3, 2, 4, 4], [32, 16, 4, 1]),
               'segm_max_ptr': ('maxima', [3, 2, 4], [8, 4, 1]),
               'segm_expsum_ptr': ('masses', [3, 2, 4], [8, 4, 1]),
               'seq_lens_ptr': ('seq_lens', [len(case['seq_lens'])], [1]),
               'query_start_len_ptr': ('query_offsets', [len(case['query_offsets'])], [1])}
    scalars = {'num_seqs': len(case['seq_lens']), 'out_scale_inv': 1.0,
               'output_stride_0': 12, 'output_stride_1': 4, 'block_table_stride': 0}
    args = []
    for i, name in enumerate(names):
        if name in tensors:
            field, shape, stride = tensors[name]
            is_float = field not in ('seq_lens', 'query_offsets')
            require(signature[name] == ('*fp32' if is_float else '*i32'), 'unexpected buffer type')
            if field == 'output':
                before = np.full(38, run['initial_storage'], dtype=np.float32).view(np.uint8)
                after = np.array(run['complete_storage'], dtype=np.float32).view(np.uint8)
            else:
                before = np.array(received[field], dtype=np.int32).view(np.uint8)
                after = np.array(run['inputs_after'][field], dtype=np.int32).view(np.uint8)
            base = (i+1)*65536
            args.append(CapturedArg(i, name, 'tensor', False, signature[name],
                dtype='float32' if is_float else 'int32', shape=tuple(shape), stride=tuple(stride),
                element_size=4, data_ptr=base+(4 if field == 'output' else 0), storage_ptr=base,
                storage_nbytes=len(before), storage_id=base, before=before, after=after))
        elif (i,) in constants:
            require(signature[name] == 'constexpr', 'constant unexpectedly became a runtime parameter')
            kind = 'bool' if type(constants[i,]) is bool else 'float' if type(constants[i,]) is float else 'int'
            args.append(CapturedArg(i, name, kind, True,
                                    signature[name], value=constants[i,]))
        else:
            require(name in scalars and signature[name] == ('fp32' if name == 'out_scale_inv' else 'i32' if name == 'num_seqs' else 'i64'),
                    'unexpected scalar type')
            args.append(CapturedArg(i, name, 'float' if name == 'out_scale_inv' else 'int', False,
                                    signature[name], value=scalars[name]))
    return CapturedLaunch(0, 'reduce_segments', '', (3, 2, 1), args,
                          {'ttir': compiled['ttir']}, None, {}, None, executed=True)


def evaluate(report):
    verify(report)
    rows = []
    for case_index, case in enumerate(report['cases']):
        expected = {1+t['token']*12+t['head']*4+c: F(value)
                    for t in case['tasks'] for c, value in enumerate(t['expected'])}
        for run_index, run in enumerate(case['runs']):
            launch = launch_from_case(case, run)
            result = KernelReferenceEvaluator.from_text(launch.asm['ttir'], mode=NumericMode.NUMERICAL_DIFFERENCE).evaluate(launch)
            reads = {a.name: result.loaded_elements[a.storage_ptr][0].tolist()
                     for a in launch.args if a.kind == 'tensor' and a.storage_ptr in result.loaded_elements}
            scalar_indices = sorted((t['token']*2+t['head'])*4+s
                                    for t in case['tasks'] for s in range(len(t['input']['parts'])))
            value_indices = [i*4+c for i in scalar_indices for c in range(4)]
            reads_match = (reads.get('segm_max_ptr') == scalar_indices
                           and reads.get('segm_expsum_ptr') == scalar_indices
                           and reads.get('segm_output_ptr') == value_indices
                           and 'output_ptr' not in reads)
            output = launch.args[0].storage_ptr
            buffer = result.buffers[output]
            written = np.flatnonzero(buffer.written).tolist()
            classes = result.element_classes(output)
            coordinates = []
            for index in sorted(expected):
                target = expected[index]
                lo, hi = float(buffer.lo[index]), float(buffer.hi[index])
                relation = 'not_established'
                if classes[index] == 'complete_composed' and math.isfinite(lo) and math.isfinite(hi):
                    lo, hi = F(lo), F(hi)
                    relation = ('exact_equal' if lo == hi == target else 'outside'
                                if target < lo or target > hi else 'enclosed_not_equal_proved')
                coordinates.append({'address': index, 'class': str(classes[index]),
                    'lower': str(lo), 'upper': str(hi), 'task': str(target),
                    'relation': relation})
            rows.append({'case': case_index, 'run': run_index, 'programs': len(result.programs),
                'aborted': {str(k): v for k, v in result.aborted.items()}, 'writes': written,
                'reads': reads, 'reads_match_declared_partials': reads_match,
                'writes_match_task': written == sorted(expected),
                'other_stored_buffers': [a.name for a in launch.args if a.kind == 'tensor'
                                        and a.storage_ptr != output and result.buffers[a.storage_ptr].written.any()],
                'coordinates': coordinates, 'notes': result.notes})
    return {'scope': 'Complete grids for recorded inputs, relocated pointer-relative memory; not universal kernel or producer binding',
            'mode': NumericMode.NUMERICAL_DIFFERENCE, 'semantic_verdict': 'not_established', 'runs': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).with_name('observed_with_ttir.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError('output must be a new file')
    report = evaluate(json.loads(args.input.read_text()))
    with args.out.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps([{'case': r['case'], 'run': r['run'], 'programs': r['programs'],
                      'aborted': r['aborted'], 'writes_match_task': r['writes_match_task'],
                      'classes': sorted(set(c['class'] for c in r['coordinates'])),
                      'relations': sorted(set(c['relation'] for c in r['coordinates']))} for r in report['runs']]))
