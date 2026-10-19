"""Complete sorting outputs: enumerate small joint outputs independently of marginal index sets."""
import itertools
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A, sorting as S, registry
from kernel_spec.num import NotEstablished, Result, Unsupported, hull
from kernel_spec.real import R


class SortingMembership(unittest.TestCase):
    @staticmethod
    def tensor(t, values, shape):
        return {"t": t, "v": values, "shape": shape}

    def test_registry_checks_complete_output_after_activation(self):
        inp = {"op": "aten.topk", "args": [{"t": "f32", "v": [4, 2, 1]}, 2]}
        with patch.object(registry, "activation", return_value=(True, [], {"draft_contract": True})):
            good = registry.check_output("SORT", "finite_unique", inp, {"out": [
                self.tensor("f32", [4, 2], [2]), self.tensor("i64", [0, 1], [2])]})
            bad = registry.check_output("SORT", "finite_unique", inp, {"out": [
                self.tensor("f32", [4, 2], [2]), self.tensor("i64", [0, 0], [2])]})
        self.assertEqual(good[0], "consistent")
        self.assertEqual(bad[0], "contradiction")
        self.assertTrue(good[2]["draft_contract"])

    def test_complete_output_requires_dtype_and_explicit_shape(self):
        inp = {"op": "aten.sort", "args": [{"t": "f32", "v": [[2, 1], [4, 3]], "shape": [2, 2]}, 1]}
        good = {"out": [self.tensor("f32", [[1, 2], [3, 4]], [2, 2]),
                         self.tensor("i64", [[1, 0], [1, 0]], [2, 2])]}
        contract = registry.load("SORT")
        self.assertTrue(contract.adapter.check_output("finite_unique", inp, good))
        for bad in (
            {"out": [[[1, 2], [3, 4]], [[1, 0], [1, 0]]]},
            {"out": [self.tensor("f16", [[1, 2], [3, 4]], [2, 2]), good["out"][1]]},
            {"out": [self.tensor("f32", [[1, 2], [3, 4]], [4]), good["out"][1]]},
            {"out": [good["out"][0], self.tensor("f32", [[1, 0], [1, 0]], [2, 2])]},
        ):
            with self.subTest(bad=bad):
                self.assertFalse(contract.adapter.check_output("finite_unique", inp, bad))

    def test_active_profile_refuses_tied_rows(self):
        inp = {"op": "aten.sort", "args": [{"t": "f32", "v": [1, 0, 1], "shape": [3]}]}
        contract = registry.load("SORT")
        observed = {"out": [self.tensor("f32", [0, 1, 1], [3]), self.tensor("i64", [1, 0, 2], [3])]}
        with self.assertRaises(Unsupported):
            contract.adapter.check_output("finite_unique", inp, observed)

    def test_registry_does_not_bypass_reference_refusal(self):
        with patch.object(registry, "query", return_value=Result("not_established", note="not reviewed")), \
             patch.object(registry, "load") as load:
            verdict, note, _ = registry.check_output("SORT", "real", {}, {})
        self.assertEqual(verdict, "not_established")
        self.assertIn("not reviewed", note)
        load.assert_not_called()

    def test_registry_has_no_elementwise_fallback(self):
        with patch.object(registry, "query", return_value=Result("value", {})):
            verdict, _, _ = registry.check_output("GDR", "base", {}, {})
        self.assertEqual(verdict, "not_established")

    def test_duplicate_tied_indices_are_rejected(self):
        a = A.tensor([4, 4, 1], "f32")
        self.assertFalse(S.check_output(a, [0, 0], [4, 4], op="topk", k=2))
        self.assertTrue(S.check_output(a, [1, 0], [4, 4], op="topk", k=2))
        self.assertFalse(S.check_output(a, [2, 0, 0], [1, 4, 4]))

    def test_all_small_joint_outputs(self):
        for xs in itertools.product(range(2), repeat=3):
            a = A.tensor(list(xs), "f32")
            for descending in (False, True):
                for k in range(4):
                    target = sorted(xs, reverse=descending)[:k]
                    allowed = {p for p in itertools.permutations(range(3), k)
                               if [xs[i] for i in p] == target}
                    for candidate in itertools.product(range(3), repeat=k):
                        observed = [xs[i] for i in candidate]
                        self.assertEqual(S.check_output(a, candidate, observed, op="topk", k=k,
                                                        descending=descending), candidate in allowed,
                                         (xs, k, descending, candidate))

    def test_value_binding_and_stability(self):
        a = A.tensor([2, 2, 1], "f32")
        self.assertFalse(S.check_output(a, [2, 0, 1], [1, 2, 9]))
        self.assertTrue(S.check_output(a, [2, 1, 0], op="argsort"))
        self.assertFalse(S.check_output(a, [2, 1, 0], op="argsort", stable=True))
        self.assertTrue(S.check_output(a, [0, 1, 2], [2, 2, 1], descending=True, stable=True))

    def test_indices_are_not_coerced(self):
        a = A.tensor([1], "f32")
        for indices in ([False], [0.0], [1], [-1], []):
            self.assertFalse(S.check_output(a, indices, [1]))
        self.assertFalse(S.check_output(a, [0], [float("nan")]))

    def test_rows_are_checked_independently(self):
        a = A.tensor([[2, 1], [2, 1]], "f32")
        self.assertTrue(S.check_output(a, [[0, 0], [1, 1]], [[2, 1], [2, 1]], dim=0))
        self.assertFalse(S.check_output(a, [[0, 0], [0, 1]], [[2, 1], [2, 1]], dim=0))

    def test_uncertain_value_is_not_certified(self):
        a = A.Ten([R(hull(0, 1))], "f32")
        with self.assertRaises(NotEstablished):
            S.check_output(a, [0], [0])


if __name__ == "__main__":
    unittest.main()
