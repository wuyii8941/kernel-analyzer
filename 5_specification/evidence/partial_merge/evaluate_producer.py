"""Compose the recorded producer and merge in the existing reference evaluator.

Only the fixed recorded task is accepted. Address relocation preserves the
recorded script's distinct allocations and shared intermediate buffers. The
absence of a version-counter capture is reported, not replaced by a fake one.
"""
import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'2_tool/src'))
from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch
from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence, NumericMode

spec = importlib.util.spec_from_file_location('producer_independent_audit', Path(__file__).with_name('verify_producer.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

FIELDS = ['output', 'query', 'keys', 'values', 'block_table', 'seq_lens', 'query_offsets', 'maxima', 'masses', 'numerators']
IDS = {name: (i+1)*65536 for i, name in enumerate(FIELDS)}
POINTERS = dict(output_ptr='output', query_ptr='query', key_cache_ptr='keys', value_cache_ptr='values',
    block_tables_ptr='block_table', seq_lens_ptr='seq_lens', query_start_len_ptr='query_offsets',
    segm_output_ptr='numerators', segm_max_ptr='maxima', segm_expsum_ptr='masses')
SHAPES = {'output': ((33, 1, 32), (36, 32, 1)), 'query': ((33, 1, 32), (32, 32, 1)),
          'keys': ((3, 16, 1, 32), (512, 32, 32, 1)), 'values': ((3, 16, 1, 32), (512, 32, 32, 1)),
          'block_table': ((1, 3), (3, 1)), 'seq_lens': ((1,), (1,)), 'query_offsets': ((2,), (1,)),
          'maxima': ((33, 1, 4), (4, 4, 1)), 'masses': ((33, 1, 4), (4, 4, 1)),
          'numerators': ((33, 1, 4, 32), (128, 128, 32, 1))}


def config():
    values = dict(scale=1.0, q_scale=1.0, k_scale=1.0, v_scale=1.0, out_scale=1.0, out_scale_inv=1.0, softcap=0.0,
        num_query_heads=1, num_queries_per_kv=1, block_table_stride=3, query_stride_0=32, query_stride_1=32,
        output_stride_0=36, output_stride_1=32, qq_bias_stride_0=0, BLOCK_SIZE=16, TILE_SIZE=16,
        HEAD_SIZE=32, HEAD_SIZE_PADDED=32, SLIDING_WINDOW=0, MAX_MM_RANGES=0, R_SWA_WINDOW=0,
        stride_k_cache_0=512, stride_k_cache_1=32, stride_k_cache_2=32, stride_k_cache_3=1,
        stride_v_cache_0=512, stride_v_cache_1=32, stride_v_cache_2=32, stride_v_cache_3=1,
        BLOCK_Q=32, num_seqs=1, BLOCK_M=32, NUM_SEGMENTS_PER_SEQ=4, KV_QUANT_MODE=0,
        FP8_MIN=-448.0, FP8_MAX=448.0, CHUNK_LOOKBACK=-1, CHUNK_SIZE=-1)
    for name in ('USE_ALIBI_SLOPES USE_ALIBI_SQRT USE_QQ_BIAS USE_SOFTCAP USE_SINKS USE_PER_SEQ_CAUSAL '
                 'USE_MM_PREFIX USE_R_SWA USE_FP8 USE_TD USE_TD_QO Q_IS_FP8 MM_PREFIX_CLAMP_SW').split():
        values[name] = False
    values.update(USE_CAUSAL=True, IS_3D=True)
    for name in ('sink_ptr alibi_slopes_ptr qq_bias_ptr per_seq_causal_ptr mm_prefix_range_ptr rswa_prefix_lens_ptr '
                 'k_scale_cache_ptr v_scale_cache_ptr stride_ks_blk stride_ks_slot stride_ks_head '
                 'stride_vs_blk stride_vs_slot stride_vs_head').split():
        values[name] = None
    return values


def check_storage_layouts(run):
    """Validate captured layouts before using the fixed-task address relocation.

    Older records lack these observations; never manufacture them from SHAPES.
    """
    stages = run.get('storage_layouts')
    if stages is None:
        return False
    audit.require(isinstance(stages, dict) and set(stages) ==
                  {'before_producer', 'after_producer', 'before_merge', 'after_merge'}, 'storage layout stages incomplete')
    first = stages['before_producer']
    audit.require(isinstance(first, dict) and set(first) == set(FIELDS), 'storage layout fields incomplete')
    audit.require(all(stage == first for stage in stages.values()), 'storage layout changed between launches')
    ranges, devices = [], set()
    for name, item in first.items():
        audit.require(isinstance(item, dict) and set(item) ==
                      {'storage_ptr', 'storage_nbytes', 'data_ptr', 'shape', 'stride', 'dtype', 'device'},
                      'storage layout record incomplete')
        audit.require(all(type(item[k]) is int and item[k] > 0 for k in ('storage_ptr', 'storage_nbytes', 'data_ptr')),
                      'storage address or size invalid')
        shape, stride = SHAPES[name]
        audit.require(item['shape'] == list(shape) and item['stride'] == list(stride)
                      and all(type(n) is int for n in item['shape'] + item['stride']), 'captured tensor layout mismatch')
        half = name in ('query', 'keys', 'values')
        integer = name in ('block_table', 'seq_lens', 'query_offsets')
        dtype, size = ('torch.float16', 2) if half else ('torch.int32', 4) if integer else ('torch.float32', 4)
        audit.require(item['dtype'] == dtype, 'captured dtype mismatch')
        count = 1190 if name == 'output' else math.prod(shape)
        audit.require(item['storage_nbytes'] == count*size, 'captured storage extent mismatch')
        audit.require(item['data_ptr'] == item['storage_ptr'] + (4 if name == 'output' else 0), 'captured storage offset mismatch')
        audit.require(isinstance(item['device'], str) and item['device'].startswith('cuda:'), 'captured device invalid')
        devices.add(item['device'])
        ranges.append((item['storage_ptr'], item['storage_ptr']+item['storage_nbytes']))
    audit.require(len(devices) == 1, 'buffers are on different devices')
    ranges.sort()
    audit.require(all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])), 'captured storages overlap')
    return True


def launches(report, run):
    layouts_checked = check_storage_layouts(run)
    has_calls = [stage+'_call_arguments' in run for stage in ('producer', 'merge')]
    audit.require(all(has_calls) or not any(has_calls), 'call argument stages incomplete')
    if any(has_calls):
        audit.require(layouts_checked, 'call pointer binding requires captured storage layouts')
    audit.require(run['partials_after_merge'] == run['partials'], 'intermediate mutation')
    versions = run.get('storage_versions')
    if versions is not None:
        audit.require(isinstance(versions, dict) and set(versions) ==
                      {'before_producer', 'after_producer', 'before_merge', 'after_merge'}, 'version stages incomplete')
        for values in versions.values():
            audit.require(isinstance(values, dict) and set(values) == set(FIELDS)
                          and all(type(v) is int and v >= 0 for v in values.values()), 'version record incomplete or invalid')
    result = []
    for stage in ('producer', 'merge'):
        compiled = run[stage+'_compiled']
        names, signature = compiled['argument_names'], compiled['signature']
        audit.require(len(set(names)) == len(names) and set(names) == set(signature), 'compiled argument list mismatch')
        call = run.get(stage+'_call_arguments')
        if all(has_calls):
            audit.require(isinstance(call, dict) and set(call) == set(names), 'captured call arguments incomplete')
            for name, value in call.items():
                if name in POINTERS:
                    pointer = run['storage_layouts']['before_'+stage][POINTERS[name]]['data_ptr']
                    audit.require(isinstance(value, dict) and set(value) == {'kind', 'data_ptr'}
                                  and value['kind'] == 'tensor' and type(value['data_ptr']) is int
                                  and value['data_ptr'] == pointer, 'captured call pointer mismatch')
                else:
                    expected = config().get(name)
                    audit.require(name in config() and isinstance(value, dict) and set(value) == {'kind', 'value'}
                                  and value['kind'] == type(expected).__name__ and type(value['value']) is type(expected)
                                  and value['value'] == expected, 'captured call scalar mismatch')
        constants = {tuple(c['index']): c['value'] for c in compiled['constexprs']}
        audit.require(len(constants) == len(compiled['constexprs']), 'duplicate compiled constant')
        args = []
        for i, name in enumerate(names):
            sig = signature[name]
            if sig == 'constexpr':
                audit.require((i,) in constants and name in config() and constants[i,] == config()[name], 'compiled task choice mismatch')
                value = constants[i,]
                kind = 'none' if value is None else 'bool' if type(value) is bool else 'float' if type(value) is float else 'int'
                args.append(CapturedArg(i, name, kind, True, sig, value=value))
            elif name in POINTERS:
                field = POINTERS[name]
                half = field in ('query', 'keys', 'values')
                integer = field in ('block_table', 'seq_lens', 'query_offsets')
                dtype, elem = ('float16', 2) if half else ('int32', 4) if integer else ('float32', 4)
                audit.require(sig == ('*fp16' if half else '*i32' if integer else '*fp32'), 'pointer type mismatch')
                if field == 'output':
                    before = (np.full(1190, run['initial_storage'], dtype=np.float32).view(np.uint8) if stage == 'producer'
                              else np.array(run['output_after_producer'], dtype=np.int32).view(np.uint8))
                    after = np.array(run['output_after_producer'] if stage == 'producer' else run['final_storage'], dtype=np.int32).view(np.uint8)
                elif field in ('maxima', 'masses', 'numerators'):
                    count = 4224 if field == 'numerators' else 132
                    before = (np.full(count, run['initial_storage'], dtype=np.float32).view(np.uint8) if stage == 'producer'
                              else np.array(run['partials'][field], dtype=np.int32).view(np.uint8))
                    after = np.array(run['partials'][field] if stage == 'producer' else run['partials_after_merge'][field], dtype=np.int32).view(np.uint8)
                else:
                    before_data = report['received_inputs'] if stage == 'producer' else run['inputs_after_producer']
                    after_data = run['inputs_after_producer'] if stage == 'producer' else run['inputs_after_merge']
                    before = np.array(before_data[field], dtype=np.int16 if half else np.int32).view(np.uint8)
                    after = np.array(after_data[field], dtype=np.int16 if half else np.int32).view(np.uint8)
                shape, stride = SHAPES[field]
                base = IDS[field]
                args.append(CapturedArg(i, name, 'tensor', False, sig, dtype=dtype, shape=shape, stride=stride,
                    element_size=elem, storage_ptr=base, storage_id=base, data_ptr=base+(4 if field == 'output' else 0),
                    storage_nbytes=len(before), before=before, after=after,
                    mutations=None if versions is None else versions['before_'+stage][field]))
            else:
                audit.require(name in config() and (i,) not in constants, 'unknown runtime argument')
                value = config()[name]
                wanted = 'fp32' if type(value) is float else 'i32' if name == 'num_seqs' else 'i64'
                audit.require(sig == wanted, 'runtime scalar type mismatch')
                args.append(CapturedArg(i, name, 'float' if type(value) is float else 'int', False, sig, value=value))
        grid = (2, 1, 4) if stage == 'producer' else (33, 1, 1)
        result.append(CapturedLaunch(len(result), 'kernel_unified_attention' if stage == 'producer' else 'reduce_segments',
            '', grid, args, {'ttir': compiled['ttir']}, None, {}, None, executed=True))
    return result


def evaluate(report):
    audit.verify(report)
    results = []
    for run in report['runs']:
        sequence = evaluate_sequence(launches(report, run), mode=NumericMode.NUMERICAL_DIFFERENCE)
        versions = run.get('storage_versions')
        stage_changes = ([] if versions is None else
            [stage for stage in ('producer', 'merge') if versions['before_'+stage] != versions['after_'+stage]])
        final = sequence.launches[-1]
        buffer = sequence.memory[IDS['output']]
        classes = final.element_classes(IDS['output'])
        coordinates = []
        for t in range(33):
            for c in range(32):
                index, target = 1+t*36+c, F(t+c-5)
                lo, hi = float(buffer.lo[index]), float(buffer.hi[index])
                k = audit.decode(run['final_storage'][index])
                relation, device = 'not_established', 'not_established'
                if classes[index] == 'complete_composed' and math.isfinite(lo) and math.isfinite(hi):
                    lo, hi = F(lo), F(hi)
                    relation = 'exact_equal' if lo == hi == target else 'outside' if target < lo or target > hi else 'enclosed_not_equal_proved'
                    if math.isfinite(k):
                        device = 'outside' if F(k) < lo or F(k) > hi else 'enclosed_not_equal_proved'
                coordinates.append({'address': index, 'class': str(classes[index]), 'lower': str(lo), 'upper': str(hi),
                    'task': str(target), 'device': str(F(k)) if math.isfinite(k) else str(k),
                    'G_vs_f': relation, 'K_vs_G': device})
        results.append({'initial_storage': run['initial_storage'],
            'call_argument_evidence': all(stage+'_call_arguments' in run for stage in ('producer', 'merge')),
            'storage_layout_evidence': run.get('storage_layouts') is not None,
            'address_relocation_basis': ('captured layouts and disjoint storage ranges' if run.get('storage_layouts') is not None
                                         else 'runner declarations only; actual storage layouts not recorded'),
            'version_evidence': versions is not None, 'stage_version_changes': stage_changes,
            'composed_without_external_reloads': versions is not None and not stage_changes and not sequence.external_writes,
            'launches': [{'programs': len(r.programs), 'aborted': {str(k): v for k, v in r.aborted.items()},
                          'stored_buffers': [name for name, ident in IDS.items() if ident in r.stored], 'notes': r.notes}
                         for r in sequence.launches],
            'carry_overs': sequence.carry_overs, 'external_writes': sequence.external_writes,
            'written_output_addresses': np.flatnonzero(buffer.written).tolist(), 'coordinates': coordinates,
            'classes': dict(Counter(c['class'] for c in coordinates)),
            'G_vs_f': dict(Counter(c['G_vs_f'] for c in coordinates)),
            'K_vs_G': dict(Counter(c['K_vs_G'] for c in coordinates))})
    return {'scope': 'Recorded fixed producer→merge task, full grids, composed reference memory; no general binding certification',
            'numeric_encoding': 'finite established bounds, task values and device values are exact rational strings',
            'semantic_verdict': 'not_established', 'mode': NumericMode.NUMERICAL_DIFFERENCE, 'runs': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).with_name('producer_with_arguments.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError('output must be a new file')
    result = evaluate(json.loads(args.input.read_text()))
    with args.out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps([{k: v for k, v in r.items() if k not in ('coordinates', 'written_output_addresses', 'carry_overs')}
                      for r in result['runs']]))
