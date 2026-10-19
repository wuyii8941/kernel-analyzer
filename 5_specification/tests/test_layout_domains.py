"""Layout domains from the retained view/reshape, expand, permute and t sources."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A
from kernel_spec.num import Invalid


class LayoutDomains(unittest.TestCase):
    def test_view_rejects_ambiguous_empty_shape(self):
        empty = A.tensor([], "i64")
        for shape in ([0, -1], [-1, 0], [-1, -1], [-2], [0.0], [False]):
            with self.subTest(shape=shape), self.assertRaises(Invalid):
                A.view(empty, shape)
        for shape, expected in (([-1, 3], (0, 3)), ([0, 3], (0, 3)), ([-1], (0,))):
            self.assertEqual(A.view(empty, shape).shape, expected)

    def test_permutation_requires_each_existing_axis_once(self):
        x = A.tensor([[1, 2, 3], [4, 5, 6]], "i64")
        for dims in ([2, 1], [-3, 1], [0, 0], [1], [False, 1], [0.0, 1]):
            with self.subTest(dims=dims), self.assertRaises(Invalid):
                A.permute(x, dims)
        self.assertEqual(A.permute(x, [-1, -2]).a.tolist(), [[1, 4], [2, 5], [3, 6]])
        self.assertEqual(A.permute(A.tensor(7, "i64"), []).shape, ())

    def test_expand_cannot_infer_a_new_axis(self):
        x = A.tensor([1, 2], "i64")
        for shape in ([-1, 2], [], [2.0], [True], [-2]):
            with self.subTest(shape=shape), self.assertRaises(Invalid):
                A.expand(x, shape)
        self.assertEqual(A.expand(x, [3, -1]).a.tolist(), [[1, 2]] * 3)

    def test_t_requires_at_most_two_dimensions(self):
        with self.assertRaises(Invalid):
            A.t(A.tensor([[[1, 2]]], "i64"))
        self.assertEqual(A.t(A.tensor(7, "i64")).a.item(), 7)

    def test_unsqueeze_validates_before_normalizing_negative_axes(self):
        x = A.tensor([1, 2], "i64")
        for dim in (-4, -3, 2, True, 0.0):
            with self.subTest(dim=dim), self.assertRaises(Invalid):
                A.unsqueeze(x, dim)
        self.assertEqual(A.unsqueeze(x, -2).a.tolist(), [[1, 2]])
        self.assertEqual(A.unsqueeze(x, -1).a.tolist(), [[1], [2]])

    def test_squeeze_validates_all_axes_even_when_size_is_not_one(self):
        x = A.tensor([[1, 2]], "i64")
        for dims in ([2], [-3], [1, 1], [0, -2], [True], [0.0]):
            with self.subTest(dims=dims), self.assertRaises(Invalid):
                A.squeeze(x, dims)
        self.assertEqual(A.squeeze(x, [-2]).a.tolist(), [1, 2])
        self.assertEqual(A.squeeze(x, [1]).shape, (1, 2))

    def test_scalar_layout_axes(self):
        scalar = A.tensor(7, "i64")
        for dim in (0, -1):
            self.assertEqual(A.squeeze(scalar, dim).a.item(), 7)
            self.assertEqual(A.transpose(scalar, dim, 0).a.item(), 7)
        with self.assertRaises(Invalid):
            A.squeeze(scalar, [0, -1])
        for dim in (1, -2):
            with self.subTest(dim=dim), self.assertRaises(Invalid):
                A.transpose(scalar, dim, 0)

    def test_transpose_invalid_axes_are_input_errors(self):
        x = A.tensor([[1, 2]], "i64")
        for dim in (2, -3, True, 0.0):
            with self.subTest(dim=dim), self.assertRaises(Invalid):
                A.transpose(x, dim, 0)
        self.assertEqual(A.transpose(x, -1, -2).a.tolist(), [[1], [2]])


if __name__ == "__main__":
    unittest.main()
