"""In-place input views need explicit compiler origins and layout evidence."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
import dynamic_three_way as D


class InputViews(unittest.TestCase):
    def setUp(self):
        self.source = """def call():
    assert_size_stride(buf0, (2, 3), (1, 2))
    buf1 = reinterpret_tensor(buf0, (3, 2), (2, 1), 0)
    kernel.run(buf1)
"""
        self.node = dict(name="pre", meta='Tensor "f32[2, 3][1, 2]cuda:0"')
        raw = torch.arange(6, dtype=torch.float32).view(torch.uint8)
        self.arg = SimpleNamespace(dtype="float32", shape=(3, 2), stride=(2, 1), before=raw, after=raw.clone(),
                                   data_ptr=100, storage_ptr=100, element_size=4, storage_nbytes=24, window=None)
        self.prov = {"params": {"in_out": {"role": "inplace", "other_origin_nodes": {"buf0": "pre", "buf1": "post"}}}}

    def mapping(self, source=None, prov=None, node=None, arg=None):
        views, layouts = D.compiler_view_metadata(self.source if source is None else source)
        return D.input_view(self.node if node is None else node, self.arg if arg is None else arg,
                            "in_out", "buf1", self.prov if prov is None else prov,
                            SimpleNamespace(views=views, layout_assertions=layouts))

    def test_transposed_pre_state_is_read_with_its_own_coordinates(self):
        mapped = self.mapping()
        self.assertIsNotNone(mapped)
        np.testing.assert_array_equal(D.logical(mapped, "before"), [[0, 2, 4], [1, 3, 5]])
        self.assertEqual(self.arg.shape, (3, 2))
        self.assertEqual(self.arg.stride, (2, 1))

    def test_equal_shapes_do_not_hide_different_pre_state_strides(self):
        source = self.source.replace("(2, 3)", "(2, 2)").replace("(3, 2)", "(2, 2)")
        arg = copy.copy(self.arg)
        arg.shape = (2, 2)
        node = dict(self.node, meta='Tensor "f32[2, 2][1, 2]cuda:0"')
        mapped = self.mapping(source=source, node=node, arg=arg)
        self.assertIsNotNone(mapped)
        np.testing.assert_array_equal(D.logical(mapped, "before"), [[0, 2], [1, 3]])

    def test_element_count_is_not_evidence_of_alias_or_layout(self):
        sources = [self.source.replace("    assert_size_stride(buf0, (2, 3), (1, 2))\n", ""),
                   self.source.replace("(1, 2)", "(3, 1)"),
                   self.source.replace("(2, 1), 0)", "(2, 1), 1)"),
                   self.source + "    buf1 = another_tensor\n",
                   self.source + "    buf0 = another_tensor\n    buf0 = yet_another_tensor\n",
                   self.source.replace("    assert_size_stride", "    buf0 = reinterpret_tensor(base, (2, 3), (1, 2), 0)\n    assert_size_stride"),
                   self.source.replace("reinterpret_tensor(buf0", "reinterpret_tensor(unrelated")]
        for source in sources:
            with self.subTest(source=source):
                self.assertIsNone(self.mapping(source=source))
        prov = copy.deepcopy(self.prov)
        prov["params"]["in_out"]["other_origin_nodes"]["buf0"] = "different_task"
        self.assertIsNone(self.mapping(prov=prov))

    def test_gapped_or_conflicting_source_layout_is_not_guessed(self):
        node = dict(self.node, meta='Tensor "f32[2, 3][1, 3]cuda:0"')
        self.assertIsNone(self.mapping(source=self.source.replace("(1, 2)", "(1, 3)"), node=node))
        self.assertIsNone(self.mapping(source=self.source + "    assert_size_stride(buf0, (2, 3), (3, 1))\n"))


if __name__ == "__main__":
    unittest.main()
