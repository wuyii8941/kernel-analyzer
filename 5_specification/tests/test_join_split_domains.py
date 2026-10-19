import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A,lazy as L
from kernel_spec.num import Invalid

class JoinSplitDomains(unittest.TestCase):
    def test_cat_rejects_incompatible_shapes_and_dimensions(self):
        for shapes,dim in [([],0),([()],0),([(2,2),(3,3)],0),([(0,2),(0,3)],0),
                           ([(2,2),(2,)],0),([(2,2)],2),([(2,2)],-3)]:
            ts=[A.tensor(np.zeros(s,dtype=np.int64),'i64') for s in shapes]
            for fn in (A.cat,L.l_cat):
                with self.subTest(fn=fn.__name__,shapes=shapes,dim=dim),self.assertRaises(Invalid):fn(ts,dim)

    def test_cat_empty_inputs_still_contribute_dtype(self):
        for fn in (A.cat,L.l_cat):
            out=fn([A.tensor([1,2],'i64'),A.tensor([],'f32')])
            self.assertEqual(out.dtype,'f32')
            self.assertEqual(out.shape,(2,))
            empty=fn([A.tensor([],'i64'),A.tensor([],'f64')])
            self.assertEqual(empty.dtype,'f64')
            self.assertEqual(empty.shape,(0,))
            for dim in (-3, 2):
                self.assertEqual(fn([A.tensor([], 'i64')], dim).shape, (0,))

    def test_split_rejects_incomplete_negative_and_noninteger_sections(self):
        for sizes in ([1],[3],[-1,3],[1.,1.],[True,1]):
            for fn in (A.split_with_sizes,L.l_split_with_sizes):
                with self.subTest(fn=fn.__name__,sizes=sizes),self.assertRaises(Invalid):fn(A.tensor([1,2],'i64'),sizes)
        for fn in (A.split,L.l_split):
            for size in (-1,0,1.5,True):
                with self.subTest(fn=fn.__name__,size=size),self.assertRaises(Invalid):fn(A.tensor([1,2],'i64'),size)
        for fn in (A.split_with_sizes,L.l_split_with_sizes):
            with self.assertRaises(Invalid):fn(A.tensor([1,2],'i64'),[2],1)

    def test_complete_parts_and_empty_parts(self):
        for fn in (A.split_with_sizes,L.l_split_with_sizes):
            parts=fn(A.tensor([[1,2,3],[4,5,6]],'i64'),[1,0,2],-1)
            self.assertEqual([p.shape for p in parts],[(2,1),(2,0),(2,2)])
            expected=[[[1],[4]],[[],[]],[[2,3],[5,6]]]
            for p,e in zip(parts,expected):
                for idx in np.ndindex(p.shape):self.assertEqual(p.get(idx) if isinstance(p,L.Lazy) else p.a[idx],e[idx[0]][idx[1]])
        for fn in (A.split,L.l_split):
            self.assertEqual([p.shape for p in fn(A.tensor([],'i64'),0)],[(0,)])
            self.assertEqual([p.shape for p in fn(A.tensor([1,2,3],'i64'),2)],[(2,),(1,)])

if __name__=='__main__':unittest.main()
