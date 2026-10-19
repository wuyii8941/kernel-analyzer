"""A declared interface may omit LSE; an observed result may not choose its own scope."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry, schema
from kernel_spec.num import Result


class ProfileOutputs(unittest.TestCase):
    def contract(self):
        c = registry.load('K14_COLL')
        c.data = copy.deepcopy(c.data)
        c.data['profiles']['scaled_numerator']['outputs'] = ['out']
        c.data['examples'] = [e for e in c.data['examples'] if e['profile'] == 'scaled_numerator']
        for example in c.data['examples']:
            example['expected'].get('fields', {}).pop('lse', None)
        return c

    def test_declared_subset_keeps_original_guarantee(self):
        c = self.contract()
        self.assertEqual(schema.profile_outputs(c.data, 'scaled_numerator'), {'out': c.data['outputs']['out']})
        self.assertEqual(schema.profile_outputs(c.data, 'normalized_lse'),
                         {name: c.data['outputs'][name] for name in ('out', 'lse')})
        self.assertEqual(schema.profile_outputs(c.data, 'all_reduce_sum'),
                         {'rank_outputs': c.data['outputs']['rank_outputs']})

    def test_invalid_scope_is_rejected_by_structure_and_runtime(self):
        for bad in ([], ['unknown'], ['out', 'out'], 'out', None, [True], [{'name': 'out'}]):
            c = self.contract()
            c.data['profiles']['scaled_numerator']['outputs'] = bad
            with self.subTest(outputs=bad):
                with self.assertRaises(ValueError):
                    schema.profile_outputs(c.data, 'scaled_numerator')
                self.assertTrue(any('profile outputs' in e for e in schema.validate(c.data, registry.load_sources())))
                _, reasons, _ = registry.activation(c, 'scaled_numerator')
                self.assertTrue(any('profile outputs' in e for e in reasons))

    def test_query_checks_exactly_declared_fields_and_guarantees(self):
        c = self.contract()
        for fields, outcome in (({'out': [1]}, 'value'), ({}, 'implementation_error'),
                                ({'out': [1], 'lse': 0}, 'implementation_error'),
                                ({'out': [1], 0: 1}, 'implementation_error'),
                                ({'out': [0.1]}, 'implementation_error')):
            c.adapter = SimpleNamespace(reference=lambda profile, inp: Result('value', fields))
            with patch.object(registry, 'load', return_value=c), \
                 patch.object(registry, 'activation', return_value=(True, [], {})):
                self.assertEqual(registry.query('K14_COLL', 'scaled_numerator', {}).outcome, outcome)

    def test_example_cannot_omit_field_from_full_output_profile(self):
        c = registry.load('K14_COLL')
        c.data['examples'][0]['expected']['fields'].pop('lse')
        self.assertTrue(any('output keys' in e for e in schema.validate(c.data, registry.load_sources())))
        c = self.contract()
        c.data['examples'][0]['expected']['fields']['lse'] = 0
        self.assertTrue(any('output keys' in e for e in schema.validate(c.data, registry.load_sources())))

    def test_comparison_entry_enforces_fields_even_if_adapter_ignores_them(self):
        c = self.contract()
        labels = {'semantic_verdict': 'not_established'}
        for profile, fields in (('scaled_numerator', {'out': [1]}),
                                ('normalized_lse', {'out': [1], 'lse': 0})):
            compare = Mock(return_value=True)
            c.adapter = SimpleNamespace(check_output=compare)
            ref = Result('value', fields, labels=labels)
            candidates = [None, [], {}, dict(fields, extra=0)]
            candidates.extend({k: v for k, v in fields.items() if k != missing} for missing in fields)
            with patch.object(registry, 'load', return_value=c), patch.object(registry, 'query', return_value=ref):
                for observed in candidates:
                    with self.subTest(profile=profile, observed=observed):
                        verdict, note, result_labels = registry.check_output('K14_COLL', profile, {}, observed)
                        self.assertEqual(verdict, 'contradiction')
                        self.assertIn('output fields', note)
                        self.assertEqual(result_labels['semantic_verdict'], 'not_established')
                        compare.assert_not_called()
                self.assertEqual(registry.check_output('K14_COLL', profile, {}, fields)[0], 'consistent')
                compare.assert_called_once()

    def test_missing_fields_do_not_bypass_reference_admission(self):
        ref = Result('not_established', note='contract not admitted',
                     labels={'semantic_verdict': 'not_established'})
        with patch.object(registry, 'query', return_value=ref), patch.object(registry, 'load') as load:
            verdict, note, labels = registry.check_output('K14_COLL', 'normalized_lse', {}, {})
        self.assertEqual(verdict, 'not_established')
        self.assertIn('contract not admitted', note)
        load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
