"""Independent rational control interpreter for GRAPH composition structure.

It intentionally does not call graph._evaluate, graph_semantics.execute_node or
the ATen arithmetic helpers.  Its small operation set is enough to check graph
order, shared names, branches, tuple projection, conversions and the admitted
local update on exact rational examples.
"""
from fractions import Fraction
import operator
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import graph as G, lazy as L
from kernel_spec.aten_adapter import decode
from kernel_spec.num import Undefined
from kernel_spec.real import R


def rational(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        return Fraction(value)
    return Fraction(value)


def tensor(record):
    values = record["v"]
    array = np.array(values, dtype=object)
    if "shape" in record:
        array = array.reshape(record["shape"])
    return np.vectorize(rational, otypes=[object])(array) if array.size else array


class RationalControl:
    def evaluate(self, graph):
        env = {name: tensor(value) for name, value in graph.get("inputs", {}).items()}

        def resolve(value):
            if isinstance(value, dict) and "ref" in value:
                result = env[value["ref"]]
                return result[value["index"]] if "index" in value else result
            if isinstance(value, dict) and "t" in value:
                return tensor(value)
            if isinstance(value, list):
                return [resolve(item) for item in value]
            if isinstance(value, (int, Fraction)) and not isinstance(value, bool):
                return rational(value)
            if isinstance(value, str):
                try:
                    return rational(value)
                except ValueError:
                    return value
            return value

        for node in graph["nodes"]:
            args = [resolve(value) for value in node.get("args", [])]
            kwargs = {name: resolve(value) for name, value in (node.get("kwargs") or {}).items()}
            op = node["op"]
            if op == "aten.add":
                value = args[0] + args[1] * kwargs.get("alpha", Fraction(1))
            elif op == "aten.sub":
                value = args[0] - args[1] * kwargs.get("alpha", Fraction(1))
            elif op == "aten.mul":
                value = args[0] * args[1]
            elif op in {"prims.convert_element_type", "aten._to_copy"}:
                value = np.array(args[0], dtype=object, copy=True)
            elif op in {"aten.view", "aten.reshape"}:
                value = np.reshape(args[0], tuple(int(item) for item in args[1]))
            elif op == "aten.max":
                axis = int(args[1])
                keep = bool(args[2]) if len(args) > 2 else False
                indices = np.argmax(args[0], axis=axis)
                values = np.take_along_axis(args[0], np.expand_dims(indices, axis), axis=axis)
                if not keep:
                    values = np.squeeze(values, axis=axis)
                else:
                    indices = np.expand_dims(indices, axis)
                value = values, indices.astype(object)
            elif op == "operator.getitem":
                value = operator.getitem(args[0], int(args[1]))
            elif op == "aten.full":
                size = args[0] if args else kwargs["size"]
                fill = args[1] if len(args) > 1 else kwargs["value"]
                value = np.full(tuple(int(item) for item in size), fill, dtype=object)
            elif op in {"aten.index_put", "aten.index_put_"}:
                value = np.array(args[0], dtype=object, copy=True)
                indices, updates = args[1], args[2]
                accumulate = bool(args[3]) if len(args) > 3 else bool(kwargs.get("accumulate", False))
                for source in np.ndindex(*updates.shape):
                    target = tuple(int(index[source[:index.ndim]]) for index in indices)
                    value[target] = value[target] + updates[source] if accumulate else updates[source]
            else:
                raise AssertionError(f"control interpreter does not implement {op}")
            env[node["name"]] = value
        return [self.plain(resolve(output if isinstance(output, dict) else {"ref": output}))
                for output in graph["outputs"]]

    def plain(self, value):
        if isinstance(value, np.ndarray):
            return self.plain(value.tolist())
        if isinstance(value, (list, tuple)):
            return [self.plain(item) for item in value]
        if isinstance(value, np.integer):
            return int(value)
        return value


def plain_lazy(value):
    if isinstance(value, L.Lazy):
        array = np.empty(value.shape, dtype=object)
        for index in np.ndindex(*value.shape):
            item = value.get(index)
            array[index] = item.v if isinstance(item, R) else item
        return array.tolist() if value.shape else array[()].v if isinstance(array[()], R) else array[()]
    if isinstance(value, (list, tuple)):
        return [plain_lazy(item) for item in value]
    return value


class GraphIndependentStructure(unittest.TestCase):
    def setUp(self):
        self.control = RationalControl()

    def check(self, graph):
        expected = self.control.evaluate(graph)
        self.assertEqual(G.evaluate(graph), expected)
        inputs = {name: L.as_lazy(decode(value)) for name, value in graph.get("inputs", {}).items()}
        env = L.evaluate_lazy(graph, inputs)
        actual = [plain_lazy(env[out["ref"]] if isinstance(out, dict) else env[out]) for out in graph["outputs"]]
        self.assertEqual(actual, expected)

    def test_chain_shared_branch_and_conversion(self):
        graphs = [
            {"inputs": {"x": {"t": "f32", "v": [1, 2]}}, "nodes": [
                {"name": "a", "op": "aten.mul", "args": [{"ref": "x"}, 2]},
                {"name": "y", "op": "aten.add", "args": [{"ref": "a"}, 1]}], "outputs": ["y"]},
            {"inputs": {"x": {"t": "f32", "v": [1, 2]}}, "nodes": [
                {"name": "p", "op": "aten.mul", "args": [{"ref": "x"}, {"ref": "x"}]},
                {"name": "y", "op": "aten.add", "args": [{"ref": "p"}, {"ref": "p"}]}], "outputs": ["p", "y"]},
            {"inputs": {"x": {"t": "f32", "v": [2, 3]}}, "nodes": [
                {"name": "a", "op": "aten.add", "args": [{"ref": "x"}, 1]},
                {"name": "b", "op": "aten.sub", "args": [{"ref": "x"}, 1]},
                {"name": "y", "op": "aten.mul", "args": [{"ref": "a"}, {"ref": "b"}]},
                {"name": "c", "op": "prims.convert_element_type", "args": [{"ref": "y"}, "f64"]}],
             "outputs": ["c"]},
        ]
        for graph in graphs:
            with self.subTest(nodes=len(graph["nodes"])):
                self.check(graph)

    def test_multi_result_identity_and_output_order(self):
        graph = {"inputs": {"x": {"t": "f32", "v": [[1, 5, 2]]}}, "nodes": [
            {"name": "pair", "op": "aten.max", "args": [{"ref": "x"}, 1, True]},
            {"name": "value", "op": "operator.getitem", "args": [{"ref": "pair"}, 0]},
            {"name": "index", "op": "operator.getitem", "args": [{"ref": "pair"}, -1]}],
            "outputs": ["index", "value", "value"]}
        self.check(graph)

    def test_private_local_update(self):
        graph = {"inputs": {}, "nodes": [
            {"name": "base", "op": "aten.full", "args": [[3], 2], "kwargs": {"dtype": "f32"}},
            {"name": "out", "op": "aten.index_put_", "args": [
                {"ref": "base"}, [{"t": "i64", "v": [1]}], {"t": "f32", "v": [4]}, True]}],
            "outputs": ["out"]}
        self.check(graph)

    def test_empty_tensor_shape_is_preserved_by_both_structural_interpreters(self):
        graph = {"inputs": {"x": {"t": "f32", "v": [], "shape": [0, 2]}},
                 "nodes": [{"name": "y", "op": "aten.view", "args": [{"ref": "x"}, [0, 1, 2]]}],
                 "outputs": ["y"]}
        self.assertEqual(self.control.evaluate(graph), [[]])
        self.assertEqual(G.evaluate(graph), [[]])
        env = L.evaluate_lazy(graph, {"x": L.as_lazy(decode(graph["inputs"]["x"]))})
        self.assertEqual(env["y"].shape, (0, 1, 2))

    def test_rejected_domain_and_joint_choice_are_independently_fixed(self):
        bad = {"inputs": {"x": {"t": "f32", "v": [0]}},
               "nodes": [{"name": "y", "op": "aten.log", "args": [{"ref": "x"}]}], "outputs": ["y"]}
        with self.assertRaises(Undefined):
            G.evaluate(bad)
        tied = {"inputs": {"x": {"t": "f32", "v": [2, 2]}},
                "nodes": [{"name": "i", "op": "aten.argsort", "args": [{"ref": "x"}]}], "outputs": ["i"]}
        self.assertEqual(G.evaluate(tied), frozenset({((0, 1),), ((1, 0),)}))


if __name__ == "__main__":
    unittest.main()
