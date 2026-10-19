import copy
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import NotEstablished, ends, evaluate


class K14MigrationTests(unittest.TestCase):
    def setUp(self):
        self.contract = registry.load('K14_COLL')
        self.adapter = self.contract.adapter

    def test_complete_output_and_input_immutability(self):
        inp = copy.deepcopy(self.contract.data['examples'][1]['input'])
        before = copy.deepcopy(inp)
        expected = {'out': [2, 7, -1], 'lse': 0}
        self.assertEqual(self.adapter.reference('normalized_lse', inp).fields, expected)
        self.assertTrue(self.adapter.check_output('normalized_lse', inp, expected))
        for observed in ({'out': [2, 7, -1]}, {'out': [2, 7], 'lse': 0},
                         {'out': [2, 7, -2], 'lse': 0}, {'out': [2, 7, -1], 'lse': 1}):
            self.assertFalse(self.adapter.check_output('normalized_lse', inp, observed))
        self.assertEqual(inp, before)

    def test_out_only_interface_is_a_declared_task_not_a_query_filter(self):
        inp = self.contract.data['examples'][0]['input']
        out = {'out': [F(7, 2), 1]}
        self.assertEqual(self.adapter.reference('scaled_numerator_out', inp).fields, out)
        self.assertTrue(self.adapter.check_output('scaled_numerator_out', inp, out))
        self.assertFalse(self.adapter.check_output('scaled_numerator', inp, out))
        self.assertFalse(self.adapter.check_output('scaled_numerator_out', inp, {'out': [F(7, 2)]}))
        self.assertFalse(self.adapter.check_output('scaled_numerator_out', inp, dict(out, lse=0)))
        altered = dict(inp, outputs=['out'])
        self.assertEqual(evaluate(self.adapter.reference, 'scaled_numerator', altered).outcome, 'unsupported')

    def test_containment_is_not_an_equality_certificate(self):
        inp = self.contract.data['examples'][0]['input']
        result = self.adapter.reference('scaled_numerator', inp).fields
        lo, hi = ends(result['lse'])
        observed = {'out': [F(7, 2), 1], 'lse': (lo+hi)/2}
        with self.assertRaises(NotEstablished):
            self.adapter.check_output('scaled_numerator', inp, observed)
        observed['out'][-1] = 2
        self.assertFalse(self.adapter.check_output('scaled_numerator', inp, observed))

    def test_complete_participants_and_selected_scope(self):
        inp = self.contract.data['examples'][0]['input']
        for ids in ([], ['one'], ['same', 'same'], ['', 'two'], [1, 2], 'ab'):
            result = evaluate(self.adapter.reference, 'scaled_numerator', dict(inp, participant_ids=ids))
            self.assertEqual(result.outcome, 'invalid_input')
        for field in inp:
            bad = {k: v for k, v in inp.items() if k != field}
            self.assertEqual(evaluate(self.adapter.reference, 'scaled_numerator', bad).outcome, 'invalid_input')
        for extra in ({'operation': 'allreduce'}, {'base': '2'}, {'weights': [1, 1]}):
            self.assertEqual(evaluate(self.adapter.reference, 'scaled_numerator', dict(inp, **extra)).outcome, 'unsupported')
        self.assertEqual(evaluate(self.adapter.reference, 'normalized_lse', inp).outcome, 'invalid_input')

    def test_collective_complete_outputs_and_exact_values(self):
        examples = {e['id']: e for e in self.contract.data['examples']}
        for example_id in ('K14.E8', 'K14.E11', 'K14.E12', 'K14.E13'):
            example = examples[example_id]
            inp = copy.deepcopy(example['input'])
            expected = self.adapter.reference(example['profile'], inp).fields
            self.assertTrue(self.adapter.check_output(example['profile'], inp, expected))
            self.assertEqual(inp, example['input'])

            missing_rank = copy.deepcopy(expected)
            missing_rank['rank_outputs'].pop()
            self.assertFalse(self.adapter.check_output(example['profile'], inp, missing_rank))

            missing_coordinate = copy.deepcopy(expected)
            missing_coordinate['rank_outputs'][0].pop()
            self.assertFalse(self.adapter.check_output(example['profile'], inp, missing_coordinate))

            extra_field = dict(copy.deepcopy(expected), work_handle=None)
            self.assertFalse(self.adapter.check_output(example['profile'], inp, extra_field))

            wrong = copy.deepcopy(expected)
            wrong['rank_outputs'][0][0] += F(1, 10**80)
            self.assertFalse(self.adapter.check_output(example['profile'], inp, wrong))

    def test_empty_does_not_hide_bad_width_and_draft_stays_closed(self):
        inp = copy.deepcopy(self.contract.data['examples'][1]['input'])
        inp['parts'][1]['out'].pop()
        self.assertEqual(evaluate(self.adapter.reference, 'normalized_lse', inp).outcome, 'invalid_input')
        verdict, _, labels = registry.check_output('K14_COLL', 'normalized_lse', inp, {},
            allow_draft=True, task_bindings={'normalized_lse': 'Development test only'})
        self.assertEqual(verdict, 'not_established')
        self.assertEqual(labels['implementation_status'], 'pending')
        self.assertEqual(labels['semantic_verdict'], 'not_established')


if __name__ == '__main__':
    unittest.main()
