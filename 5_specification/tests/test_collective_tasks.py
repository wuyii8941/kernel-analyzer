import copy
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import collectives as C
from kernel_spec.num import Invalid, Undefined, Unsupported


class CollectiveTasks(unittest.TestCase):
    def task(self, **kw):
        return dict(operation='all_reduce', participant_ids=['rank_b', 'rank_a'],
                    values=[[1, 2], [3, 4]], **kw)

    def test_distinct_reductions_and_all_participant_outputs(self):
        for profile, extra, expected in (
            ('all_reduce_sum', {}, [4, 6]), ('all_reduce_mean', {}, [2, 3]),
            ('all_reduce_weighted_mean', {'weights': [Q(3, 2), Q(1, 2)]}, [Q(3, 2), Q(5, 2)]),
            ('all_reduce_count_mean', {'counts': [3, 1]}, [Q(3, 2), Q(5, 2)])):
            inp = self.task(**extra)
            before = copy.deepcopy(inp)
            self.assertEqual(C.reference(profile, inp), [expected, expected])
            self.assertEqual(inp, before)

    def test_count_mean_matches_independent_sample_expansion(self):
        samples = [
            [Q(1, 3), Q(7, 2), Q(-5, 4)],
            [Q(11, 3)],
            [],
        ]
        local_means = [
            sum(row, Q(0)) / len(row) if row else Q(987, 11)
            for row in samples
        ]
        inp = dict(operation='all_reduce', participant_ids=['left', 'right', 'empty'],
                   values=[[x] for x in local_means], counts=[len(row) for row in samples])
        all_samples = [x for row in samples for x in row]
        independently_expanded = sum(all_samples, Q(0)) / len(all_samples)
        self.assertEqual(C.reference('all_reduce_count_mean', inp),
                         [[independently_expanded]] * len(samples))

    def test_incomplete_counts_and_weights_never_drop_participants(self):
        for profile, field in (('all_reduce_weighted_mean', 'weights'), ('all_reduce_count_mean', 'counts')):
            for bad in ([1], [1, 2, 3], '31', [True, 1], [-1, 2]):
                with self.subTest(profile=profile, bad=bad), self.assertRaises(Invalid):
                    C.reference(profile, self.task(**{field: bad}))
        with self.assertRaises(Invalid):
            C.reference('all_reduce_count_mean', self.task(counts=[3.9, 1]))
        with self.assertRaises(Undefined):
            C.reference('all_reduce_count_mean', self.task(counts=[0, 0]))

    def test_rank_order_and_all_shards(self):
        inp = self.task()
        inp.update(operation='reduce_scatter', values=[[1, 2, 3, 4], [10, 20, 30, 40]])
        self.assertEqual(C.reference('reduce_scatter_sum', inp), [[11, 22], [33, 44]])
        inp.update(operation='all_gather', values=[[1, 2], [3, 4]])
        self.assertEqual(C.reference('all_gather_equal', inp), [[1, 2, 3, 4], [1, 2, 3, 4]])
        inp['values'] = [[1], [2, 3]]
        with self.assertRaises(Invalid): C.reference('all_gather_equal', inp)

    def test_empty_coordinates_and_invalid_participant_structures(self):
        inp = self.task()
        inp['values'] = [[], []]
        self.assertEqual(C.reference('all_reduce_sum', inp), [[], []])
        for bad in ([], ['one'], ['same', 'same'], ['one', ''], 'ab'):
            with self.assertRaises(Invalid):
                C.reference('all_reduce_sum', dict(inp, participant_ids=bad))
        with self.assertRaises(Invalid):
            C.reference('all_reduce_sum', dict(inp, values=[[1], [2, 3]]))
        with self.assertRaises(Unsupported):
            C.reference('all_reduce_sum', self.task(weights=[3, 1]))
        with self.assertRaises(Unsupported):
            C.reference('all_reduce_sum', dict(self.task(), operation='all_gather'))
        with self.assertRaises(Invalid):
            C.reference('reduce_scatter_sum', dict(inp, operation='reduce_scatter', values=[[1], [2]]))


if __name__ == '__main__': unittest.main()
