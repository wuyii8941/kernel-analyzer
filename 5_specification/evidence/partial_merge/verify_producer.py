"""Independent finite-workload partition, state and output audit (stdlib only)."""
import argparse
import json
import math
import struct
from fractions import Fraction as F
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode(value, width=32):
    require(type(value) is int and -(2**(width-1)) <= value < 2**(width-1), 'invalid input bit record')
    return struct.unpack('<f' if width == 32 else '<e', struct.pack('<i' if width == 32 else '<h', value))[0]


def verify(report):
    require(report['semantic_verdict'] == 'not_established', 'no automatic semantic certification')
    task = report['task']
    require(task == {'seq_len': 33, 'head_size': 32, 'heads': 1, 'block_size': 16, 'tile_size': 16,
        'block_q': 32, 'segments': 4, 'physical_blocks': [2, 0, 1], 'causal': True, 'scale': 1,
        'window': [-1, -1], 'softcap': 0, 'query_values': 'zero', 'key_values': 'zero',
        'value_formula': '2*key_position+coordinate-5', 'expected_output_formula': 'query_position+coordinate-5'}, 'changed task')
    received = report['received_inputs']
    require(set(received) == {'query', 'keys', 'values', 'block_table', 'seq_lens', 'query_offsets'}, 'missing input')
    require(len(received['query']) == 1056 and len(received['keys']) == 1536 and len(received['values']) == 1536, 'input buffer length')
    require(received['block_table'] == [2, 0, 1] and received['seq_lens'] == [33]
            and received['query_offsets'] == [0, 33], 'received metadata mismatch')
    require(all(decode(x, 16) == 0 for x in received['query']), 'query differs from task')
    logical = []
    used = set()
    for j in range(33):
        base = (received['block_table'][j//16]*16+j%16)*32
        used.update(range(base, base+32))
        require(all(decode(x, 16) == 0 for x in received['keys'][base:base+32]), 'key differs from task')
        values = [decode(x, 16) for x in received['values'][base:base+32]]
        require(values == [2*j+c-5 for c in range(32)], 'value differs from task')
        logical.append(list(map(F, values)))
    for name in ('keys', 'values'):
        require(all(math.isnan(decode(x, 16)) for i, x in enumerate(received[name]) if i not in used), 'unused cache tail sentinel changed')
    runs = report['runs']
    require(len(runs) == 2 and [r['initial_storage'] for r in runs] == [-333, 777], 'missing repeated initialization')
    summaries = []
    for run in runs:
        require(run['producer_grid'] == [2, 1, 4] and run['merge_grid'] == [33, 1], 'grid mismatch')
        require(run['inputs_after_producer'] == received and run['inputs_after_merge'] == received, 'input modified')
        parts = run['partials']
        require(set(parts) == {'maxima', 'masses', 'numerators'} and len(parts['maxima']) == 132
                and len(parts['masses']) == 132 and len(parts['numerators']) == 4224, 'partial buffer incomplete')
        require(run['partials_after_merge'] == parts, 'partial changed across merge')
        require(len(run['output_after_producer']) == 1190 and all(decode(x) == run['initial_storage']
                for x in run['output_after_producer']), 'producer unexpectedly wrote final output')
        empty_counts = {'no_loop': 0, 'masked_row': 0}
        expected, partition = {}, []
        for t in range(33):
            seen = []
            total_mass = 0
            total_numerator = [F(0)]*32
            row_parts = []
            for segment in range(4):
                i = t*4+segment
                maximum, mass = decode(parts['maxima'][i]), decode(parts['masses'][i])
                numerator = [decode(x) for x in parts['numerators'][i*32:(i+1)*32]]
                if segment == 3:
                    require(all(x == run['initial_storage'] for x in [maximum, mass, *numerator]), 'inactive segment was overwritten')
                    continue
                keys = [j for j in range(segment*16, min((segment+1)*16, 33)) if j <= t]
                seen.extend(keys)
                wanted = [sum((logical[j][c] for j in keys), F(0)) for c in range(32)]
                require(numerator == wanted, 'producer numerator differs from independent key sum')
                if keys:
                    require(maximum == 0 and mass == len(keys), 'nonempty maximum or mass differs')
                    total_mass += F(mass)
                    total_numerator = [a+F(b) for a, b in zip(total_numerator, numerator)]
                    state = 'nonempty'
                else:
                    # Is the segment outside the whole query block's prefix?
                    prefix_end = min((t//32+1)*32, 33)
                    no_loop = segment*16 >= prefix_end
                    require((maximum == -math.inf and mass == 1) if no_loop else (maximum == 0 and mass == 0),
                            'empty producer representation differs from declared control flow')
                    state = 'no_loop' if no_loop else 'masked_row'
                    empty_counts[state] += 1
                row_parts.append({'segment': segment, 'keys': keys, 'state': state})
            require(seen == list(range(t+1)), 'key partition is incomplete or duplicated')
            direct = [sum(logical[j][c] for j in range(t+1))/(t+1) for c in range(32)]
            require([v/total_mass for v in total_numerator] == direct, 'partial union differs from full attention')
            for c, value in enumerate(direct):
                expected[1+t*36+c] = value
            partition.append({'query': t, 'partials': row_parts})
        actual = run['final_storage']
        require(len(actual) == 1190, 'incomplete output storage')
        values = list(map(decode, actual))
        differences = [i for i, target in expected.items() if not math.isfinite(values[i]) or F(values[i]) != target]
        guards = [i for i, value in enumerate(values) if i not in expected and value != run['initial_storage']]
        require(differences == run['output_mismatch_addresses'] and guards == run['changed_guard_addresses'], 'false difference flags')
        errors = [F(values[i])-target for i, target in expected.items() if math.isfinite(values[i])]
        summaries.append({'queries': 33, 'output_coordinates': len(expected), 'empty_states': empty_counts,
            'nonempty_partials': 99-sum(empty_counts.values()), 'partition': partition,
            'device_differences': len(differences), 'changed_guards': len(guards),
            'positive_errors': sum(e > 0 for e in errors), 'negative_errors': sum(e < 0 for e in errors),
            'nonfinite_outputs': len(expected)-len(errors), 'maximum_absolute_error': str(max(map(abs, errors)))})
    require(runs[0]['final_storage'][1:33] == runs[1]['final_storage'][1:33], 'first output row differs across repeats')
    require(all(runs[0]['final_storage'][1+t*36+c] == runs[1]['final_storage'][1+t*36+c]
                for t in range(33) for c in range(32)), 'output differs across repeats')
    return {'scope': 'Fixed finite causal workload; exact producer sums and key partition, observed K-f differences; no general binding or statistical bias claim',
            'semantic_verdict': 'not_established', 'runs': summaries}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).with_name('producer_observed.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError('output must be a new file')
    result = verify(json.loads(args.input.read_text()))
    with args.out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps([{k: v for k, v in r.items() if k != 'partition'} for r in result['runs']]))
