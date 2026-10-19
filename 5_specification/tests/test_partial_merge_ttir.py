import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]/'evidence/partial_merge'
spec = importlib.util.spec_from_file_location('evaluate_partial_merge_ttir', ROOT/'evaluate_ttir.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PartialMergeTTIR(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((ROOT/'observed_with_ttir.json').read_text())

    def test_complete_grids_reads_writes_and_honest_enclosure_status(self):
        report = audit.evaluate(self.report)
        self.assertEqual(len(report['runs']), 4)
        for run in report['runs']:
            self.assertEqual(run['programs'], 6)
            self.assertEqual(run['aborted'], {})
            self.assertTrue(run['writes_match_task'])
            self.assertTrue(run['reads_match_declared_partials'])
            self.assertEqual(run['other_stored_buffers'], [])
            self.assertEqual(len(run['coordinates']), 24)
            for coordinate in run['coordinates']:
                self.assertEqual(coordinate['class'], 'complete_composed')
                self.assertEqual(coordinate['relation'], 'enclosed_not_equal_proved')
        self.assertEqual(report['semantic_verdict'], 'not_established')

    def test_unsupported_ir_does_not_manufacture_a_contradiction(self):
        compiled = self.report['cases'][0]['compiled']
        self.assertIn('math.exp ', compiled['ttir'])
        compiled['ttir'] = compiled['ttir'].replace('math.exp ', 'unknown.exp ')
        run = audit.evaluate(self.report)['runs'][0]
        self.assertEqual(len(run['aborted']), 6)
        self.assertFalse(run['writes_match_task'])
        self.assertTrue(all(c['relation'] == 'not_established' for c in run['coordinates']))

    def test_changed_compiled_task_is_not_silently_rebound(self):
        self.report['cases'][0]['compiled']['constexprs'][0]['value'] = 3
        with self.assertRaisesRegex(ValueError, 'unexpected compiled constants'):
            audit.evaluate(self.report)

    def test_new_capture_preserves_prior_device_observations(self):
        previous = json.loads((ROOT/'observed.json').read_text())
        for current, old in zip(self.report['cases'], previous['cases']):
            self.assertEqual({k: v for k, v in current.items() if k != 'compiled'}, old)


if __name__ == '__main__':
    unittest.main()
