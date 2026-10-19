"""Unavailable tasks and broken reference installations are different outcomes."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import NotEstablished, Result


class QueryFailures(unittest.TestCase):
    def test_unknown_contract_is_not_established(self):
        result = registry.query('UNKNOWN_TASK_NO_CONTRACT', 'base', {})
        self.assertEqual(result.outcome, 'not_established')
        self.assertEqual(result.labels['semantic_verdict'], 'not_established')
        self.assertEqual(registry.check_output('UNKNOWN_TASK_NO_CONTRACT', 'base', {}, {})[0], 'not_established')

    def test_corrupt_contract_is_an_implementation_error(self):
        with patch.object(registry, 'load_contract', side_effect=ValueError('corrupt contract')):
            result = registry.query('MOE', 'renorm_aux_top1', {})
        self.assertEqual(result.outcome, 'implementation_error')
        self.assertIn('corrupt contract', result.note)

    def test_missing_adapter_is_not_an_unknown_task(self):
        with patch.object(registry, 'import_adapter', side_effect=FileNotFoundError('missing adapter')):
            result = registry.query('MOE', 'renorm_aux_top1', {})
        self.assertEqual(result.outcome, 'implementation_error')
        self.assertIn('missing adapter', result.note)

    def test_admission_failure_does_not_become_mathematical_undefinedness(self):
        with patch.object(registry, 'activation', side_effect=ValueError('broken source index')):
            result = registry.query('MOE', 'renorm_aux_top1', {})
        self.assertEqual(result.outcome, 'implementation_error')
        self.assertIn('broken source index', result.note)

    def test_known_reference_unavailability_remains_not_established(self):
        with patch.object(registry, 'load', side_effect=NotEstablished('module origin conflict')):
            result = registry.query('MOE', 'renorm_aux_top1', {})
        self.assertEqual(result.outcome, 'not_established')

    def test_comparison_adapter_load_failure_is_reported(self):
        ref = Result('value', {'out': 1}, labels={'semantic_verdict': 'not_established'})
        with patch.object(registry, 'query', return_value=ref), \
             patch.object(registry, 'load', side_effect=FileNotFoundError('adapter disappeared')):
            verdict, note, labels = registry.check_output('MOE', 'renorm_aux_top1', {}, {'out': 1})
        self.assertEqual(verdict, 'not_established')
        self.assertIn('implementation_error', note)
        self.assertIn('adapter disappeared', note)
        self.assertEqual(labels['semantic_verdict'], 'not_established')


if __name__ == '__main__':
    unittest.main()
