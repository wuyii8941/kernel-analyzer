"""Independent token-level reconstruction of the explicitly selected rank-mean task."""
import itertools
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import Result, evaluate


class RankMeanIndependent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("RANKMEAN").adapter

    def test_every_partition_preserves_the_token_mean(self):
        tokens = [Q(1, 3), Q(-7, 5), Q(11, 2), Q(0)]
        # Online mean over individual tokens, without rank sums or rank counts.
        expected = Q(0)
        for number, token in enumerate(tokens, 1):
            expected += (token - expected) / number
        for assignment in itertools.product(range(3), repeat=len(tokens)):
            groups = [[token for token, owner in zip(tokens, assignment) if owner == rank] for rank in range(3)]
            for order in ((0, 1, 2), (2, 0, 1)):
                inp = dict(S=[sum(groups[r], Q(0)) for r in order], n=[len(groups[r]) for r in order],
                           participant_ids=[f"rank-{r}" for r in order])
                result = self.adapter.reference("base", inp)
                self.assertEqual(result.fields, {"loss": expected})
                self.assertTrue(self.adapter.check_output("base", inp, {"loss": expected}))
                self.assertFalse(self.adapter.check_output("base", inp, {"loss": expected + Q(1, 10**80)}))

    def test_missing_inputs_are_invalid_not_reference_failures(self):
        complete = dict(S=[2], n=[1], participant_ids=["rank-0"])
        for key in complete:
            inp = {k: v for k, v in complete.items() if k != key}
            with self.subTest(missing=key):
                self.assertEqual(evaluate(self.adapter.reference, "base", inp).outcome, "invalid_input")
        for inp in (None, [], "task"):
            self.assertEqual(evaluate(self.adapter.reference, "base", inp).outcome, "invalid_input")

    def test_unrepresentable_participant_identifiers_are_not_mathematical_failures(self):
        for ids in ([[0]], [{"rank": 0}]):
            result = evaluate(self.adapter.reference, "base", dict(S=[2], n=[1], participant_ids=ids))
            self.assertEqual(result.outcome, "unsupported")

    def test_task_options_cannot_be_silently_ignored(self):
        base = dict(S=[6, 4], n=[3, 1], participant_ids=["a", "b"])
        for options in ({"weights": [1, 3]}, {"reduction": "sum"}, {"world_size": 3}):
            inp = dict(base, **options)
            with self.subTest(options=options):
                self.assertEqual(evaluate(self.adapter.reference, "base", inp).outcome, "unsupported")
                result = evaluate(lambda: Result("value", {"matches": self.adapter.check_output("base", inp, {"loss": Q(5, 2)})}))
                self.assertEqual(result.outcome, "unsupported")

    def test_empty_participants_and_large_exact_counts(self):
        for inp, expected in ((dict(S=[0, 0], n=[0, 0], participant_ids=["a", "b"]), "domain_undefined"),
                              (dict(S=[1, 0], n=[0, 0], participant_ids=["a", "b"]), "invalid_input"),
                              (dict(S=[1, 2], n=[1, 2], participant_ids=["a", "a"]), "invalid_input")):
            self.assertEqual(evaluate(self.adapter.reference, "base", inp).outcome, expected)
        count = 10**100 + 3
        result = self.adapter.reference("base", dict(S=[count, 1], n=[count, 2], participant_ids=[0, 1]))
        self.assertEqual(result.fields, {"loss": Q(count + 1, count + 2)})
