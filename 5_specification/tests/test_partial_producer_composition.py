import copy
import importlib.util
import json
import unittest
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]/'evidence/partial_merge'
spec = importlib.util.spec_from_file_location('evaluate_producer_chain', ROOT/'evaluate_producer.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ProducerComposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observed = json.loads((ROOT/'producer_with_arguments.json').read_text())
        cls.result = audit.evaluate(cls.observed)

    def test_full_composition_preserves_distinct_numerical_and_semantic_questions(self):
        expected_addresses = [1+t*36+c for t in range(33) for c in range(32)]
        for run in self.result['runs']:
            self.assertTrue(run['storage_layout_evidence'])
            self.assertTrue(run['call_argument_evidence'])
            self.assertTrue(run['composed_without_external_reloads'])
            self.assertEqual(run['external_writes'], [])
            self.assertEqual([l['programs'] for l in run['launches']], [8, 33])
            self.assertTrue(all(not l['aborted'] for l in run['launches']))
            self.assertEqual(run['written_output_addresses'], expected_addresses)
            self.assertEqual(run['classes'], {'complete_composed': 1056})
            self.assertEqual(run['G_vs_f'], {'enclosed_not_equal_proved': 1056})
            self.assertEqual(run['K_vs_G'], {'enclosed_not_equal_proved': 934, 'outside': 122})
            self.assertTrue(all(c['evidence'] == 'bytes and version counters' for c in run['carry_overs']))
            for coordinate in run['coordinates']:
                lo, hi, target = map(F, (coordinate['lower'], coordinate['upper'], coordinate['task']))
                self.assertLessEqual(lo, target)
                self.assertGreaterEqual(hi, target)
                if coordinate['K_vs_G'] == 'outside':
                    value = F(coordinate['device'])
                    self.assertTrue(value < lo or value > hi)
        self.assertEqual(self.result['semantic_verdict'], 'not_established')

    def test_storage_layout_mismatch_or_alias_is_rejected_before_evaluation(self):
        for change in ('offset', 'stride', 'extent', 'alias', 'continuity', 'missing'):
            run = copy.deepcopy(self.observed['runs'][0])
            stages = run['storage_layouts']
            if change == 'continuity':
                stages['before_merge']['masses']['storage_ptr'] += 4
            elif change == 'missing':
                del stages['after_merge']
            else:
                for layout in stages.values():
                    if change == 'offset': layout['output']['data_ptr'] += 4
                    elif change == 'stride': layout['output']['stride'][0] = 32
                    elif change == 'extent': layout['output']['storage_nbytes'] -= 4
                    else:
                        layout['masses']['storage_ptr'] = layout['maxima']['storage_ptr']
                        layout['masses']['data_ptr'] = layout['maxima']['data_ptr']
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.launches(self.observed, run)

    def test_older_record_does_not_acquire_layout_evidence(self):
        old = json.loads((ROOT/'producer_with_versions.json').read_text())
        for run in old['runs']:
            self.assertFalse(audit.check_storage_layouts(run))

    def test_actual_call_parameters_must_match_task_and_captured_buffers(self):
        for change in ('scalar', 'type', 'pointer', 'missing_field', 'missing_stage', 'null_stage'):
            run = copy.deepcopy(self.observed['runs'][0])
            args = run['producer_call_arguments']
            if change == 'scalar': args['scale']['value'] = 2.0
            elif change == 'type': args['scale']['value'] = 1
            elif change == 'pointer': args['query_ptr']['data_ptr'] = args['key_cache_ptr']['data_ptr']
            elif change == 'missing_field': del args['scale']
            elif change == 'missing_stage': del run['merge_call_arguments']
            else: run['merge_call_arguments'] = None
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.launches(self.observed, run)

    def test_legacy_call_parameters_remain_explicitly_unrecorded(self):
        report = copy.deepcopy(self.observed)
        for run in report['runs']:
            del run['producer_call_arguments'], run['merge_call_arguments']
        result = audit.evaluate(report)
        self.assertTrue(all(not run['call_argument_evidence'] for run in result['runs']))

    def test_same_bytes_with_intervening_version_change_are_not_silent_carryover(self):
        report = copy.deepcopy(self.observed)
        versions = report['runs'][0]['storage_versions']
        versions['before_merge']['masses'] += 1
        versions['after_merge']['masses'] += 1
        run = audit.evaluate(report)['runs'][0]
        self.assertFalse(run['composed_without_external_reloads'])
        self.assertTrue(any(e['buffer'] == 'segm_expsum_ptr' for e in run['external_writes']))

    def test_missing_versions_are_not_filled_with_fake_zeroes(self):
        report = copy.deepcopy(self.observed)
        del report['runs'][0]['storage_versions']
        run = audit.evaluate(report)['runs'][0]
        self.assertFalse(run['version_evidence'])
        self.assertFalse(run['composed_without_external_reloads'])
        self.assertTrue(all('no version evidence' in c['evidence'] for c in run['carry_overs']))

    def test_unsupported_producer_does_not_yield_a_numerical_difference_claim(self):
        report = copy.deepcopy(self.observed)
        compiled = report['runs'][0]['producer_compiled']
        self.assertIn('tt.dot ', compiled['ttir'])
        compiled['ttir'] = compiled['ttir'].replace('tt.dot ', 'unknown.dot ')
        run = audit.evaluate(report)['runs'][0]
        self.assertTrue(run['launches'][0]['aborted'])
        self.assertNotIn('outside', run['K_vs_G'])
        self.assertNotIn('outside', run['G_vs_f'])


if __name__ == '__main__':
    unittest.main()
