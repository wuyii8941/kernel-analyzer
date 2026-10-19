"""Independent stdlib audit of the fixed device observations; no reference imports."""
import argparse
import json
import math
import struct
from fractions import Fraction as F
from pathlib import Path


def require(condition, note):
    if not condition:
        raise ValueError(note)


def decode(bits):
    require(type(bits) is int and -(2**31) <= bits < 2**31, 'invalid float32 bit record')
    return struct.unpack('<f', struct.pack('<i', bits))[0]


def verify(report):
    require(report['semantic_verdict'] == 'not_established', 'device replay cannot certify semantics')
    require(report['profile'] == 'scaled_numerator_out', 'wrong task profile')
    cases = report['cases']
    workload = (([5, 0, 3], [0, 2, 2, 3]), ([1, 9], [0, 1, 3]))
    require(len(cases) == len(workload), 'missing workload case')
    summaries = []
    for case, (lengths, offsets) in zip(cases, workload):
        require(case['seq_lens'] == lengths and case['query_offsets'] == offsets, 'changed workload')
        require(case['grid'] == [3, 2] and case['segments'] == 4 and case['tile_size'] == 2
                and case['value_width'] == 4 and case['output_strides'] == [12, 4, 1]
                and case['output_storage_offset'] == 1, 'changed launch or storage mapping')
        received = case['received_inputs']
        require(set(received) == {'maxima', 'masses', 'numerators', 'seq_lens', 'query_offsets'}, 'missing input buffer')
        require(received['seq_lens'] == lengths and received['query_offsets'] == offsets, 'received sequence metadata differs')
        require(len(received['maxima']) == 24 and len(received['masses']) == 24
                and len(received['numerators']) == 96, 'incomplete input buffers')
        require(len(case['tasks']) == 6, 'missing token/head task')
        tasks = {(t['token'], t['head']): t for t in case['tasks']}
        require(set(tasks) == {(t, h) for t in range(3) for h in range(2)}, 'duplicate or missing token/head task')
        expected = {}
        for token in range(3):
            seq = next(i for i in range(len(lengths)) if offsets[i] <= token < offsets[i+1])
            # Find the smallest whole number of tiles per segment that covers
            # the sequence using at most four segments. No kernel helper used.
            tiles = 1
            while 4*2*tiles < lengths[seq]:
                tiles += 1
            active = [s for s in range(4) if s*2*tiles < lengths[seq]]
            for head in range(2):
                parts = []
                for segment in range(4):
                    index = (token*2+head)*4+segment
                    m = decode(received['maxima'][index])
                    mass = decode(received['masses'][index])
                    numerator = [decode(v) for v in received['numerators'][index*4:index*4+4]]
                    if segment not in active:
                        require(all(math.isnan(v) for v in [m, mass, *numerator]), 'inactive input sentinel changed')
                        continue
                    intended_mass = (1, 1, 2, 4)[segment]
                    intended = [intended_mass*(token*16+head*8+segment*2+c-5) for c in range(4)]
                    require(m == token-head and mass == intended_mass and numerator == intended, 'received input differs from declared task')
                    parts.append({'shift': F(m), 'mass': F(mass), 'numerator': list(map(F, numerator))})
                # All actual shifts agree: exp(m_i-M)=1 exactly. Divide sums of
                # actual received numerators by actual received masses.
                total = sum(p['mass'] for p in parts)
                answer = [sum(p['numerator'][c] for p in parts)/total for c in range(4)]
                task = tasks[token, head]
                require(task['input'] == dict(operation='merge', participant_ids=[f'segment-{s}' for s in active],
                        empty_policy='undefined', parts=parts), 'task record does not describe received buffers')
                require(list(map(F, task['expected'])) == answer, 'recorded expected result differs from independent sum')
                for c, value in enumerate(answer):
                    expected[1+token*12+head*4+c] = value
        runs = case['runs']
        require(len(runs) == 2 and [r['initial_storage'] for r in runs] == [-12345, 23456], 'missing distinct-initialization replay')
        counts = []
        for run in runs:
            require(run['inputs_after'] == received, 'input buffer changed during kernel invocation')
            actual = run['complete_storage']
            require(len(actual) == 38 and all(type(v) in (int, float) and math.isfinite(v) for v in actual), 'incomplete or nonfinite storage record')
            differences = [i for i, target in expected.items() if F(actual[i]) != target]
            guards = [i for i, value in enumerate(actual) if i not in expected and value != run['initial_storage']]
            require(run['output_mismatch_addresses'] == differences, 'recorded mismatch flags disagree with storage')
            require(run['changed_guard_addresses'] == guards, 'recorded guard flags disagree with storage')
            counts.append({'output_differences': len(differences), 'changed_guards': len(guards)})
        summaries.append({'output_coordinates': len(expected), 'guard_coordinates': 38-len(expected), 'runs': counts})
    return {'scope': 'Fixed merge calls; post-experiment independent arithmetic/storage audit, not blind validation or full binding',
            'cases': summaries, 'semantic_verdict': 'not_established'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('observed', nargs='?', type=Path, default=Path(__file__).with_name('observed.json'))
    args = parser.parse_args()
    print(json.dumps(verify(json.loads(args.observed.read_text())), indent=2))
