import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A, lazy as L
from kernel_spec.num import Invalid, Undefined

class IndexPutDomains(unittest.TestCase):
    def test_out_of_range_is_not_wrapped(self):
        for k in (-4, 3, 2**64):
            for fn in (A.index_put, L.l_index_put):
                with self.subTest(k=k, fn=fn.__name__), self.assertRaises(Invalid):
                    fn(A.tensor([0,0,0], 'i64'), [A.tensor([k], 'i64')], A.tensor([7], 'i64'))

    def test_source_shape_cannot_discard_elements(self):
        for shape in ((2,3), (1,2,2)):
            for fn in (A.index_put, L.l_index_put):
                with self.subTest(shape=shape, fn=fn.__name__), self.assertRaises(Invalid):
                    fn(A.tensor([[0,0],[0,0]], 'i64'), [A.tensor([0,1], 'i64')], A.tensor(np.ones(shape), 'i64'))

    def test_duplicate_invalidates_whole_result(self):
        for fn in (A.index_put, L.l_index_put):
            with self.subTest(fn=fn.__name__), self.assertRaises(Undefined):
                fn(A.tensor([0,0,0], 'i64'), [A.tensor([0,0], 'i64')], A.tensor([1,2], 'i64'))

    def test_values_dtype_must_match_even_for_empty_indices(self):
        for indices in ([0], []):
            for target, source in [('f32','f64'),('i64','f32'),('b8','i64')]:
                for fn in (A.index_put,L.l_index_put):
                    with self.subTest(indices=indices,target=target,source=source,fn=fn.__name__),self.assertRaises(Invalid):
                        fn(A.tensor([0,0],target),[A.tensor(indices,'i64')],A.tensor([1]*len(indices),source))

    def test_accumulate_must_be_boolean(self):
        for flag in (0,1,None,'false',[],[False]):
            for fn in (A.index_put,L.l_index_put):
                with self.subTest(flag=flag,fn=fn.__name__),self.assertRaises(Invalid):
                    fn(A.tensor([0,0],'i64'),[A.tensor([],'i64')],A.tensor([],'i64'),flag)

    def test_boolean_accumulation_preserves_boolean_values(self):
        for fn in (A.index_put,L.l_index_put):
            out=fn(A.tensor([True,False,False],'b8'),[A.tensor([0,0,1],'i64')],A.tensor([True,True,True],'b8'),True)
            for i,expected in enumerate((True,True,False)):
                value=out.get((i,)) if isinstance(out,L.Lazy) else out.a[i]
                self.assertIs(type(value),bool)
                self.assertEqual(value,expected)

    def test_complete_negative_index_and_accumulation(self):
        for fn in (A.index_put, L.l_index_put):
            out=fn(A.tensor([[1,2],[3,4],[5,6]], 'i64'), [A.tensor([-1,0,-1], 'i64')], A.tensor([[10],[20],[30]], 'i64'), True)
            expected=[[21,22],[3,4],[45,46]]
            self.assertEqual(out.shape,(3,2))
            for i in np.ndindex(out.shape):
                self.assertEqual(out.get(i) if isinstance(out,L.Lazy) else out.a[i],expected[i[0]][i[1]])

if __name__=='__main__': unittest.main()
