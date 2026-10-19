import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]/'evidence/partial_merge'
spec = importlib.util.spec_from_file_location('verify_partial_merge', ROOT/'verify.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PartialMergeDeviceEvidence(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((ROOT/'observed.json').read_text())

    def test_complete_observations_and_guards(self):
        result = audit.verify(self.report)
        self.assertEqual(result['semantic_verdict'], 'not_established')
        self.assertEqual(sum(c['output_coordinates'] for c in result['cases']), 48)
        for case in result['cases']:
            self.assertEqual(case['guard_coordinates'], 14)
            for run in case['runs']:
                self.assertEqual(run, {'output_differences': 0, 'changed_guards': 0})

    def test_missing_input_output_or_task_is_rejected(self):
        for field in ('cases', 'tasks', 'complete_storage', 'numerators'):
            report = copy.deepcopy(self.report)
            if field == 'cases': report['cases'].pop()
            elif field == 'tasks': report['cases'][0]['tasks'].pop()
            elif field == 'complete_storage': report['cases'][0]['runs'][0][field].pop()
            else: report['cases'][0]['received_inputs'][field].pop()
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.verify(report)

    def test_declared_task_is_checked_against_actual_received_inputs(self):
        self.report['cases'][0]['received_inputs']['maxima'][0] = 1065353216
        with self.assertRaisesRegex(ValueError, 'received input differs'):
            audit.verify(self.report)

    def test_input_mutation_or_fabricated_result_flags_are_rejected(self):
        for kind in ('input', 'output', 'guard'):
            report = copy.deepcopy(self.report)
            run = report['cases'][0]['runs'][0]
            if kind == 'input': run['inputs_after']['seq_lens'][0] += 1
            elif kind == 'output': run['complete_storage'][32] += 1
            else: run['complete_storage'][0] = 0
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit.verify(report)

    def test_honest_counterexample_is_retained_not_reported_as_pass(self):
        run = self.report['cases'][0]['runs'][0]
        run['complete_storage'][32] += 1
        run['output_mismatch_addresses'] = [32]
        result = audit.verify(self.report)
        self.assertEqual(result['cases'][0]['runs'][0]['output_differences'], 1)
        self.assertEqual(result['semantic_verdict'], 'not_established')


if __name__ == '__main__':
    unittest.main()
