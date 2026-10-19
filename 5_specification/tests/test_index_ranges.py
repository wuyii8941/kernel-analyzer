"""Nonnegative indexed operators must not inherit Python's negative-index convention."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A, lazy
from kernel_spec.num import Invalid, NotEstablished


class IndexRanges(unittest.TestCase):
    def test_invalid_indices_are_rejected_by_each_operation(self):
        for index in (-1, 2, 200):
            ix = A.tensor([index], "i64")
            x = A.tensor([10, 20], "f32")
            for name, fn in (("gather", lambda: A.gather(x, 0, ix)),
                             ("select", lambda: A.index_select(x, 0, ix)),
                             ("add", lambda: A.index_add(x, 0, ix, A.tensor([1], "f32"))),
                             ("embedding", lambda: A.embedding(A.tensor([[10], [20]], "f32"), ix))):
                with self.subTest(index=index, op=name), self.assertRaises(Invalid):
                    fn()
            for fn in (lambda: lazy.l_gather(lazy.from_array(np.array([10., 20.]), "f32"), 0,
                                            lazy.from_array(np.array([index]), "i64")).get((0,)),
                       lambda: lazy.l_embedding(lazy.from_array(np.array([[10.], [20.]]), "f32"),
                                               lazy.from_array(np.array([index]), "i64")).get((0, 0))):
                with self.assertRaises(Invalid):
                    fn()

    def test_floating_indices_are_not_coerced_even_when_empty_or_integral(self):
        for values in ([0.5], [0.0], []):
            ix = A.tensor(values, "f32")
            x = A.tensor([10, 20], "f32")
            weight = A.tensor([[10], [20]], "f32")
            src = A.tensor([1] * len(values), "f32")
            operations = (
                lambda: A.gather(x, 0, ix),
                lambda: A.index_select(x, 0, ix),
                lambda: A.index_add(x, 0, ix, src),
                lambda: A.embedding(weight, ix),
                lambda: A.index(x, [ix]),
                lambda: A.scatter(x, 0, ix, src=src),
                lambda: lazy.l_gather(x, 0, ix),
                lambda: lazy.l_embedding(weight, ix),
                lambda: lazy.l_index(x, [ix]),
                lambda: lazy.l_scatter(x, 0, ix, src=src),
            )
            for number, operation in enumerate(operations):
                if not values and number in (0, 5, 6, 9):
                    result = operation()
                    self.assertEqual(result.shape, (0,) if number in (0, 6) else (2,))
                    continue
                with self.subTest(values=values, operation=number), self.assertRaises(Invalid):
                    operation()

    def test_gather_validates_shape_and_dimension(self):
        for shape, indices, dim in (((2, 2), [0], 0), ((2, 2), [[0, 0, 0]], 0), ((2,), [0], 1), ((2,), [0], -2)):
            x, ix = A.tensor(np.ones(shape), "i64"), A.tensor(indices, "i64")
            for fn in (A.gather, lazy.l_gather):
                with self.subTest(shape=shape, dim=dim, fn=fn.__name__), self.assertRaises(Invalid):
                    fn(x, dim, ix)

    def test_gather_scalar_conventions(self):
        for values, indices, expected in ((7, 0, 7), (7, [0, 0], [7, 7]), ([7, 9], 1, 9)):
            x, ix = A.tensor(values, "i64"), A.tensor(indices, "i64")
            out, partial = A.gather(x, 0, ix), lazy.l_gather(x, 0, ix)
            self.assertEqual(A.plain(out), expected)
            self.assertEqual(partial.shape, out.shape)
            for coordinate in np.ndindex(out.shape):
                self.assertEqual(partial.get(coordinate), out.a[coordinate])

    def test_empty_index_still_requires_valid_dimension(self):
        x, ix = A.tensor([1, 2], "i64"), A.tensor([], "f32")
        for dimension in (-2, 1):
            for fn in (lambda: A.gather(x, dimension, ix),
                       lambda: lazy.l_gather(x, dimension, ix),
                       lambda: A.scatter(x, dimension, ix, value=3),
                       lambda: lazy.l_scatter(x, dimension, ix, value=3)):
                with self.assertRaises(Invalid):
                    fn()

    def test_scatter_checks_source_and_nonindexed_dimensions(self):
        x = A.tensor([[1, 2], [3, 4]], "i64")
        for indices, values in (([[0, 0, 0]], [[8, 9, 10]]), ([[0, 0]], [[8]])):
            for fn in (A.scatter, lazy.l_scatter):
                with self.assertRaises(Invalid):
                    fn(x, 0, A.tensor(indices, "i64"), A.tensor(values, "i64"))
        for fn in (A.scatter, lazy.l_scatter):
            with self.assertRaises(NotEstablished):
                fn(x, 0, A.tensor([0], "i64"), A.tensor([8], "i64"))

    def test_scatter_extra_source_entries_and_scalar_inputs(self):
        for values, indices, source, expected in (([1, 2], [0], [8, 99], [8, 2]), (1, 0, 8, 8), ([1, 2], 0, 8, [8, 2])):
            x, ix, src = A.tensor(values, "i64"), A.tensor(indices, "i64"), A.tensor(source, "i64")
            out, partial = A.scatter(x, 0, ix, src), lazy.l_scatter(x, 0, ix, src)
            self.assertEqual(A.plain(out), expected)
            self.assertEqual(partial.shape, out.shape)
            for coordinate in np.ndindex(out.shape):
                self.assertEqual(partial.get(coordinate), out.a[coordinate])

    def test_index_add_cannot_ignore_extra_source_rows(self):
        with self.assertRaises(Invalid):
            A.index_add(A.tensor([10, 20], "f32"), 0, A.tensor([0], "i64"), A.tensor([1, 99], "f32"))

    def test_embedding_preserves_arbitrary_index_shape(self):
        out = A.embedding(A.tensor([[10, 11], [20, 21]], "f32"), A.tensor([[[1, 0]]], "i64"))
        self.assertEqual(A.plain(out), [[[[20, 21], [10, 11]]]])

    def test_nonadjacent_advanced_indices_reject_out_of_range(self):
        x = A.tensor(np.arange(24).reshape(2, 3, 4), "i64")
        for axis, bad in ((0, 2), (0, -3), (2, 4), (2, -5)):
            indices = [A.tensor([0], "i64"), None, A.tensor([0], "i64")]
            indices[axis] = A.tensor([bad], "i64")
            with self.subTest(axis=axis, value=bad):
                with self.assertRaises(Invalid):
                    A.index(x, indices)
                with self.assertRaises(Invalid):
                    lazy.l_index(x, indices).get((0, 0))

    def test_nonadjacent_advanced_indices_match_independent_coordinates(self):
        x = A.tensor(np.arange(24).reshape(2, 3, 4), "i64")
        indices = [A.tensor([[-1], [0]], "i64"), None, A.tensor([[0, -1]], "i64")]
        eager, partial = A.index(x, indices), lazy.l_index(x, indices)
        self.assertEqual(eager.shape, (2, 2, 3))
        self.assertEqual(partial.shape, (2, 2, 3))
        for i, j, k in np.ndindex(2, 2, 3):
            expected = (1 - i) * 12 + k * 4 + j * 3
            self.assertEqual(eager.a[i, j, k], expected)
            self.assertEqual(partial.get((i, j, k)), expected)

    def test_advanced_indexing_still_accepts_negative_indices(self):
        self.assertEqual(A.plain(A.index(A.tensor([10, 20], "f32"), [A.tensor([-1], "i64")])), [20])

    def test_valid_repeated_indices_and_embedding_scalar_shape(self):
        x, ix = A.tensor([10, 20], "f32"), A.tensor([1, 0, 1], "i64")
        self.assertEqual(A.plain(A.gather(x, 0, ix)), [20, 10, 20])
        self.assertEqual(A.plain(A.index_select(x, 0, ix)), [20, 10, 20])
        self.assertEqual(A.plain(A.index_add(x, 0, ix, A.tensor([3, 4, 5], "f32"), alpha=2)), [18, 36])
        weight = A.tensor([[10, 11], [20, 21]], "f32")
        self.assertEqual(A.plain(A.embedding(weight, A.tensor(1, "i64"))), [20, 21])
        self.assertEqual(A.embedding(weight, A.tensor([[]], "i64")).shape, (1, 0, 2))

    def test_embedding_backward_cannot_drop_or_wrap_contributions(self):
        for grad, ix in (([[1]], [-1]), ([[1]], [2]), ([[1], [99]], [0])):
            with self.subTest(grad=grad, ix=ix), self.assertRaises(Invalid):
                A.embedding_dense_backward(A.tensor(grad, "f32"), A.tensor(ix, "i64"), 2)
        result = A.embedding_dense_backward(A.tensor([[1], [3], [7]], "f32"), A.tensor([1, 0, 1], "i64"), 2)
        self.assertEqual(A.plain(result), [[3], [8]])


if __name__ == "__main__":
    unittest.main()
