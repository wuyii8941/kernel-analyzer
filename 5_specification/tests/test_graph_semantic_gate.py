import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A, graph as G, lazy as L
from kernel_spec.num import Invalid, Undefined, Unsupported


class GraphSemanticGate(unittest.TestCase):
    def lazy_inputs(self, graph):
        return {name: L.as_lazy(A.tensor(value["v"], value["t"])) for name, value in graph["inputs"].items()}

    def test_mathfn_domain_is_shared_by_complete_and_lazy_paths(self):
        graph = {"inputs": {"x": {"t": "f32", "v": [0]}},
                 "nodes": [{"name": "y", "op": "aten.log", "args": [{"ref": "x"}]}], "outputs": ["y"]}
        with self.assertRaises(Undefined):
            G.evaluate(graph)
        self.assertIsInstance(L.evaluate_lazy(graph, self.lazy_inputs(graph))["y"], Undefined)

    def test_whole_node_undefined_granularity_is_shared(self):
        graph = {"inputs": {"x": {"t": "f32", "v": [-1, 4]}}, "nodes": [
            {"name": "l", "op": "aten.log", "args": [{"ref": "x"}]},
            {"name": "y", "op": "aten.select", "args": [{"ref": "l"}, 0, 1]}], "outputs": ["y"]}
        with self.assertRaises(Undefined):
            G.evaluate(graph)
        env = L.evaluate_lazy(graph, self.lazy_inputs(graph))
        self.assertIsInstance(env["l"], Undefined)
        self.assertIsInstance(env["y"], Undefined)

    def test_only_mathematical_undefined_may_be_unused(self):
        base = {"inputs": {"bad": {"t": "f32", "v": [-1]}, "x": {"t": "f32", "v": [3]}}, "nodes": [
            {"name": "u", "op": "aten.log", "args": [{"ref": "bad"}]},
            {"name": "y", "op": "aten.mul", "args": [{"ref": "x"}, 2]}], "outputs": ["y"]}
        self.assertEqual(G.evaluate(base), [[6]])
        self.assertEqual(L.evaluate_lazy(base, self.lazy_inputs(base))["y"].get((0,)).v, 6)

        above_scope = {"inputs": {"bad": {"t": "f32", "v": [101]}, "x": {"t": "f32", "v": [3]}},
                       "nodes": base["nodes"], "outputs": ["y"]}
        for call in (lambda: G.evaluate(above_scope),
                     lambda: L.evaluate_lazy(above_scope, self.lazy_inputs(above_scope))):
            with self.assertRaises(Unsupported):
                call()

        invalid = {"inputs": {"bad": {"t": "f32", "v": [1, 2]}, "x": {"t": "f32", "v": [3]}}, "nodes": [
            {"name": "u", "op": "aten.view", "args": [{"ref": "bad"}, [3]]}, base["nodes"][1]], "outputs": ["y"]}
        for call in (lambda: G.evaluate(invalid), lambda: L.evaluate_lazy(invalid, self.lazy_inputs(invalid))):
            with self.assertRaises(Invalid):
                call()

    def test_output_slice_is_explicit_and_does_not_admit_the_whole_graph(self):
        graph = {"inputs": {"bad": {"t": "f32", "v": [101]}, "x": {"t": "f32", "v": [3]}}, "nodes": [
            {"name": "u", "op": "aten.exp", "args": [{"ref": "bad"}]},
            {"name": "y", "op": "aten.mul", "args": [{"ref": "x"}, 2]}], "outputs": ["y"]}
        with self.assertRaises(Unsupported):
            L.evaluate_lazy(graph, self.lazy_inputs(graph))
        env = L.evaluate_lazy(graph, self.lazy_inputs(graph), scope="output_slice")
        self.assertEqual(env["y"].get((0,)).v, 6)
        self.assertNotIn("u", env)

    def test_physical_input_is_not_a_logical_rational(self):
        graph = {"inputs": {"x": {"t": "f32", "v": ["1/3"]}},
                 "nodes": [{"name": "y", "op": "aten.mul", "args": [{"ref": "x"}, 3]}], "outputs": ["y"]}
        with self.assertRaises(Invalid):
            G.evaluate(graph)
        inputs = {"x": L.as_lazy(A.tensor(["1/3"], "f32"))}
        with self.assertRaises(Invalid):
            L.evaluate_lazy(graph, inputs)

    def test_helper_availability_is_not_contract_admission(self):
        graph = {"inputs": {"x": {"t": "i64", "v": [1, 2]}},
                 "nodes": [{"name": "y", "op": "aten.copy", "args": [{"ref": "x"}, {"ref": "x"}]}],
                 "outputs": ["y"]}
        inputs = {"x": L.as_lazy(A.tensor([1, 2], "i64"))}
        with self.assertRaises(Unsupported):
            L.evaluate_lazy(graph, inputs)
        env = L.evaluate_lazy(graph, inputs, admission="development_helper")
        self.assertEqual([env["y"].get((i,)) for i in range(2)], [1, 2])

    def test_selected_profile_and_public_argument_aliases_are_enforced(self):
        integer_reduce = {"inputs": {"x": {"t": "i64", "v": [1, 2]}},
                          "nodes": [{"name": "y", "op": "aten.sum", "args": [{"ref": "x"}]}], "outputs": ["y"]}
        for call in (lambda: G.evaluate(integer_reduce),
                     lambda: L.evaluate_lazy(integer_reduce, self.lazy_inputs(integer_reduce))):
            with self.assertRaises(Unsupported):
                call()

        matrix = {"inputs": {"a": {"t": "f32", "v": [[1, 2]]}, "b": {"t": "f32", "v": [[3], [4]]}},
                  "nodes": [{"name": "y", "op": "aten.mm", "args": [],
                             "kwargs": {"self": {"ref": "a"}, "mat2": {"ref": "b"}}}], "outputs": ["y"]}
        self.assertEqual(G.evaluate(matrix), [[[11]]])
        env = L.evaluate_lazy(matrix, self.lazy_inputs(matrix))
        self.assertEqual(env["y"].get((0, 0)).v, 11)

    def test_admitted_operation_without_a_lazy_backend_is_unestablished(self):
        graph = {"inputs": {"x": {"t": "f32", "v": [2, 2]}},
                 "nodes": [{"name": "y", "op": "aten.argsort", "args": [{"ref": "x"}]}], "outputs": ["y"]}
        from kernel_spec.num import NotEstablished
        with self.assertRaises(NotEstablished):
            L.evaluate_lazy(graph, self.lazy_inputs(graph))


if __name__ == "__main__":
    unittest.main()
