"""The average-pooling task includes its full input VJP and selected ceil divisor."""
import copy
import json
from pathlib import Path
import sys
import unittest

SPEC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SPEC), str(SPEC / 'corpus')]
import call_binding as B


class InductorAveragePoolingBinding(unittest.TestCase):
    def fixture(self):
        plan = json.loads((SPEC / 'corpus/call_bindings/inductor_avg_pool2d.json').read_text())
        plan['status'] = 'drafted'
        x = {'t': 'f32', 'shape': [1, 1, 4, 4], 'v': [[[[1]*4 for _ in range(4)]]]}
        upstream = {'t': 'f32', 'shape': [1, 1, 2, 2], 'v': [[[[0, 0], [0, 1]]]]}
        grad = {'t': 'f32', 'shape': [1, 1, 4, 4], 'v': [[[[0]*4, [0]*4, [0, 0, .25, .25], [0, 0, .25, .25]]]]}
        observed = {'subject': plan['subject'], 'launch_scope_complete': True,
            'launches': ['forward_pool', 'backward_pool'],
            'tensor_arguments': {k: {'before': copy.deepcopy(v), 'after': copy.deepcopy(v)}
                                 for k, v in [('x', x), ('upstream', upstream)]},
            'scalars': {}, 'returns': {'y': {**upstream, 'v': [[[[1, 1], [1, 1]]]]}, 'dx': grad},
            'public_outputs': ['/returns/y', '/returns/dx']}
        return plan, observed

    def test_explicit_task_and_complete_reference_do_not_admit_draft(self):
        report = B.inspect(*self.fixture())
        self.assertEqual(report['binding_status'], 'development_checked', report)
        self.assertEqual(report['reference_status'], 'value')
        self.assertFalse(report['reference_usable_for_bias_measurement'])
        task = report['tasks'][0]
        self.assertEqual(task['inputs']['kernel_size'], [3, 3])
        self.assertTrue(task['inputs']['ceil_mode'])
        self.assertTrue(task['labels']['task_choice'])
        self.assertEqual(task['fixed_call_against_f'], 'inside_declared_result')

    def test_wrong_divisor_is_observed_without_refusing_the_task_binding(self):
        plan, observed = self.fixture()
        observed['returns']['dx']['v'][0][0][2][2] = 0.1111111119389534
        report = B.inspect(plan, observed)
        self.assertEqual(report['binding_status'], 'development_checked', report)
        self.assertEqual(report['tasks'][0]['fixed_call_against_f'], 'outside_declared_result')

    def test_every_input_and_output_is_required(self):
        for source in ('/tensor_arguments/x/before', '/tensor_arguments/upstream/before'):
            plan, observed = self.fixture()
            plan['tasks'][0]['input_map'] = [m for m in plan['tasks'][0]['input_map'] if m.get('source') != source]
            with self.subTest(source=source):
                self.assertEqual(B.inspect(plan, observed)['binding_status'], 'rejected')
        plan, observed = self.fixture()
        plan['tasks'][0]['output_map'] = plan['tasks'][0]['output_map'][:1]
        self.assertEqual(B.inspect(plan, observed)['binding_status'], 'rejected')
        plan, observed = self.fixture()
        observed['returns']['dx'] = {'t': 'f32', 'shape': [1, 1, 2, 2], 'v': [[[[.25, .25], [.25, .25]]]]}
        self.assertEqual(B.inspect(plan, observed)['binding_status'], 'rejected')

    def test_changed_state_partial_capture_and_extra_option_are_rejected(self):
        for name in ('x', 'upstream'):
            plan, observed = self.fixture()
            observed['tensor_arguments'][name]['after']['v'][0][0][0][0] = 2
            with self.subTest(name=name):
                self.assertEqual(B.inspect(plan, observed)['binding_status'], 'rejected')
        plan, observed = self.fixture(); observed['launch_scope_complete'] = False
        self.assertEqual(B.inspect(plan, observed)['binding_status'], 'rejected')
        plan, observed = self.fixture(); observed['scalars']['divisor_override'] = 9
        self.assertEqual(B.inspect(plan, observed)['binding_status'], 'rejected')

    def test_missing_explicit_task_statement_is_rejected(self):
        plan, observed = self.fixture()
        del plan['tasks'][0]['task_statement']
        self.assertEqual(B.inspect(plan, observed)['binding_status'], 'rejected')
