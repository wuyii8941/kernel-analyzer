import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'evidence/partial_merge'
sys.path.insert(0, str(ROOT))
from kernel_spec import registry, attention_partials

spec = importlib.util.spec_from_file_location('verify_partial_producer', EVIDENCE/'verify_producer.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ProducerEvidence(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((EVIDENCE/'producer_observed.json').read_text())

    def test_partition_empty_states_and_real_differences_are_preserved(self):
        result = audit.verify(self.report)
        for run in result['runs']:
            self.assertEqual(run['empty_states'], {'no_loop': 32, 'masked_row': 16})
            self.assertEqual(run['nonempty_partials'], 51)
            self.assertEqual(run['device_differences'], 122)
            self.assertEqual((run['positive_errors'], run['negative_errors']), (120, 2))
            self.assertEqual(run['maximum_absolute_error'], '1/262144')
            self.assertEqual(run['changed_guards'], 0)
        self.assertEqual(result['semantic_verdict'], 'not_established')

    def test_no_loop_empty_mass_must_not_be_silently_reinterpreted(self):
        run = self.report['runs'][0]
        run['partials']['masses'][2] = 0
        run['partials_after_merge']['masses'][2] = 0
        with self.assertRaisesRegex(ValueError, 'empty producer representation'):
            audit.verify(self.report)

    def test_all_producer_coordinates_are_checked(self):
        run = self.report['runs'][0]
        run['partials']['numerators'][4191] += 1
        run['partials_after_merge']['numerators'][4191] += 1
        with self.assertRaisesRegex(ValueError, 'producer numerator'):
            audit.verify(self.report)

    def test_missing_state_or_false_success_flags_are_rejected(self):
        self.report['runs'][0]['output_mismatch_addresses'] = []
        with self.assertRaisesRegex(ValueError, 'false difference flags'):
            audit.verify(self.report)
        self.report = json.loads((EVIDENCE/'producer_observed.json').read_text())
        self.report['runs'][0]['partials']['masses'].pop()
        with self.assertRaisesRegex(ValueError, 'partial buffer incomplete'):
            audit.verify(self.report)

    def test_current_contracts_agree_on_the_complete_received_task(self):
        audited = audit.verify(self.report)
        received = self.report['received_inputs']
        query = [[[audit.decode(x, 16) for x in received['query'][t*32:(t+1)*32]]] for t in range(33)]
        def cache(name):
            result = []
            for block in range(3):
                slots = []
                for slot in range(16):
                    start = (block*16+slot)*32
                    values = [audit.decode(x, 16) for x in received[name][start:start+32]]
                    # The independent audit proves these physical tail slots are
                    # outside the logical task, so null does not invent a value.
                    slots.append(None if block == 1 and slot > 0 else [values])
                result.append(slots)
            return result
        task = dict(query=query, cu_seqlens_q=[0, 33], seq_lens=[33], key_cache=cache('keys'),
                    value_cache=cache('values'), block_table=[received['block_table']],
                    scale=1, causal=True, window=[-1, -1], softcap=0, dtype='f16')
        expected = [[[t+c-5 for c in range(32)]] for t in range(33)]
        self.assertEqual(registry.load('PAGEDATTN').adapter.reference('vllm016_causal_2d_real', task).fields,
                         {'out': expected})
        merge = registry.load('K14_COLL').adapter
        raw = self.report['runs'][0]['partials']
        for row in audited['runs'][0]['partition']:
            t = row['query']
            parts = []
            for part in row['partials']:
                index = t*4+part['segment']
                if part['keys']:
                    parts.append(attention_partials.scaled(audit.decode(raw['maxima'][index]),
                        audit.decode(raw['masses'][index]),
                        [audit.decode(v) for v in raw['numerators'][index*32:(index+1)*32]]))
                else:
                    # Empty-key proof and raw sentinel checks came first. L alone
                    # is insufficient: both raw (M=-inf,L=1) and (M=0,L=0) occur.
                    parts.append(attention_partials.empty(32))
            inputs = dict(operation='merge', participant_ids=['segment-0', 'segment-1', 'segment-2'],
                          empty_policy='undefined', parts=parts)
            self.assertEqual(merge.reference('scaled_numerator_out', inputs).fields, {'out': expected[t][0]})


if __name__ == '__main__':
    unittest.main()
