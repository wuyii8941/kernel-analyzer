"""Full public RMSNorm correspondence is separate from a Triton-only decomposition."""
import copy
import json
from pathlib import Path
import sys
import unittest

SPEC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SPEC), str(SPEC / 'corpus')]
import call_binding as B


class FlagGemsRMSNormBinding(unittest.TestCase):
    def fixture(self):
        plan = json.loads((SPEC / 'corpus/call_bindings/flaggems_rms_norm.json').read_text())
        plan['status'] = 'drafted'
        matrix = {'t': 'f32', 'shape': [2, 7], 'v': [[0]*7 for _ in range(2)]}
        vector = {'t': 'f32', 'shape': [7], 'v': [1]*7}
        observed = {'subject': plan['subject'], 'launch_scope_complete': True,
                    'launches': ['rms_norm_kernel', 'rms_norm_grad_dx_kernel', 'rms_norm_grad_dw_kernel'],
                    'tensor_arguments': {k: {'before': copy.deepcopy(v), 'after': copy.deepcopy(v)}
                        for k, v in [('x', matrix), ('weight', vector), ('upstream', matrix)]},
                    'scalars': {}, 'returns': {'y': copy.deepcopy(matrix), 'dx': copy.deepcopy(matrix),
                        'dweight': {'t': 'f32', 'shape': [7], 'v': [0]*7}},
                    'public_outputs': ['/returns/y', '/returns/dx', '/returns/dweight']}
        return plan, observed

    def test_full_reference_and_public_default_do_not_admit_a_draft(self):
        report = B.inspect(*self.fixture())
        self.assertEqual(report['binding_status'], 'development_checked', report)
        self.assertEqual(report['reference_status'], 'value', report)
        self.assertEqual(report['tasks'][0]['inputs']['eps'], '1/8388608')
        self.assertFalse(report['reference_usable_for_bias_measurement'])

    def test_every_tensor_input_is_required(self):
        for name in ('x', 'weight', 'upstream'):
            plan, obs = self.fixture()
            plan['tasks'][0]['input_map'] = [m for m in plan['tasks'][0]['input_map']
                if m.get('source') != f'/tensor_arguments/{name}/before']
            with self.subTest(name=name):
                self.assertEqual(B.inspect(plan, obs)['binding_status'], 'rejected')

    def test_nontriton_final_gradient_cannot_be_omitted(self):
        plan, obs = self.fixture()
        plan['tasks'][0]['output_map'] = plan['tasks'][0]['output_map'][:-1]
        self.assertEqual(B.inspect(plan, obs)['binding_status'], 'rejected')
        plan, obs = self.fixture()
        obs['returns']['dweight'] = {'t': 'f32', 'shape': [1, 7], 'v': [[0]*7]}
        self.assertEqual(B.inspect(plan, obs)['binding_status'], 'rejected')

    def test_state_changes_and_partial_launch_scope_are_rejected(self):
        for name in ('x', 'weight', 'upstream'):
            plan, obs = self.fixture()
            obs['tensor_arguments'][name]['after']['v'][0] = 2 if name == 'weight' else [2]*7
            with self.subTest(name=name):
                self.assertEqual(B.inspect(plan, obs)['binding_status'], 'rejected')
        plan, obs = self.fixture()
        obs['launch_scope_complete'] = False
        self.assertEqual(B.inspect(plan, obs)['binding_status'], 'rejected')

    def test_an_unmapped_epsilon_cannot_override_the_fixed_public_task(self):
        plan, obs = self.fixture()
        obs['scalars']['eps'] = 1e-5
        self.assertEqual(B.inspect(plan, obs)['binding_status'], 'rejected')
