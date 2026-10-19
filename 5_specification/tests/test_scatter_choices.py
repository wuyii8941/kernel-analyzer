"""Scatter choices retain node identity; index ranges follow the retained torch docs."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A, graph as G, lazy
from kernel_spec.num import Invalid, NotEstablished


class ScatterChoices(unittest.TestCase):
    def fixture(self):
        return dict(inputs={"x": {"t": "f32", "v": [0]}, "idx": {"t": "i64", "v": [0, 0]},
                            "src": {"t": "f32", "v": [1, 3]}},
                    nodes=[dict(name="s", op="aten.scatter", args=[{"ref": "x"}, 0, {"ref": "idx"}, {"ref": "src"}])],
                    outputs=["s", "s"])

    def test_complete_outputs_share_one_scatter_choice(self):
        self.assertEqual(G.evaluate(self.fixture()), frozenset({((1,), (1,)), ((3,), (3,))}))

    def test_reused_choice_cancels_exactly(self):
        g = self.fixture()
        g["nodes"].append(dict(name="z", op="aten.sub", args=[{"ref": "s"}, {"ref": "s"}]))
        g["outputs"] = ["z"]
        self.assertEqual(G.evaluate(g), [[0]])

    def test_budget_preserves_refusal_instead_of_partial_answers(self):
        with self.assertRaises(NotEstablished):
            G.evaluate(self.fixture(), max_choices=1)

    def test_equal_writes_need_only_one_choice(self):
        g = self.fixture()
        g["inputs"]["src"]["v"] = [2, 2]
        self.assertEqual(G.evaluate(g, max_choices=1), [[2], [2]])

    def test_lazy_ambiguity_is_a_typed_refusal_not_an_arithmetic_error(self):
        for values in ([1., 3.], [2., 2.]):
            out = lazy.l_scatter(lazy.from_array(np.array([0.]), "f32"), 0,
                                 lazy.from_array(np.array([0, 0]), "i64"),
                                 lazy.from_array(np.array(values), "f32"))
            if values[0] == values[1]:
                self.assertEqual(out.get((0,)).v, 2)
            else:
                with self.assertRaises(NotEstablished):
                    out.get((0,))

    def test_scatter_indices_are_not_wrapped(self):
        for index in (-1, 2):
            with self.subTest(index=index):
                with self.assertRaises(Invalid):
                    A.scatter(A.tensor([0, 0], "f32"), 0, A.tensor([index], "i64"), A.tensor([5], "f32"))
                with self.assertRaises(Invalid):
                    lazy.l_scatter(lazy.from_array(np.array([0., 0.]), "f32"), 0,
                                   lazy.from_array(np.array([index]), "i64"), lazy.from_array(np.array([5.]), "f32"))


if __name__ == "__main__":
    unittest.main()
