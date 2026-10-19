import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A,lazy as L
from kernel_spec.num import Invalid

class SliceScatterDomains(unittest.TestCase):
    def test_select_scatter_requires_exact_source_and_valid_index(self):
        for shape,src_shape,dim,index in [((2,3),(1,),0,0),((2,3),(4,),0,0),
                                         ((2,3),(3,),0,2),((2,3),(3,),0,-3),
                                         ((0,3),(0,),1,3),((2,3),(3,),2,0),
                                         ((0,),(0,),0,0),((),(),0,0)]:
            for fn in (A.select_scatter,L.l_select_scatter):
                with self.subTest(fn=fn.__name__,shape=shape,src=src_shape,dim=dim,index=index), self.assertRaises(Invalid):
                    fn(A.tensor(np.zeros(shape,dtype=np.int64),'i64'),A.tensor(np.ones(src_shape,dtype=np.int64),'i64'),dim,index)

    def test_slice_scatter_cannot_ignore_extra_or_empty_source(self):
        for shape,src_shape,dim,start,end,step in [((4,),(3,),0,0,2,1),((4,),(1,),0,0,2,1),
                                                 ((4,),(1,),0,2,2,1),((0,),(1,),0,0,0,1),
                                                 ((2,3),(2,1),2,0,1,1),((2,3),(2,1),-3,0,1,1),
                                                 ((),(),0,0,1,1),((0,),(0,),0,0,0,0)]:
            for fn in (A.slice_scatter,L.l_slice_scatter):
                with self.subTest(fn=fn.__name__,shape=shape,src=src_shape), self.assertRaises(Invalid):
                    fn(A.tensor(np.zeros(shape,dtype=np.int64),'i64'),A.tensor(np.ones(src_shape,dtype=np.int64),'i64'),dim,start,end,step)

    def test_complete_coordinate_values_and_base_preserved(self):
        base=A.tensor([[0,1,2,3],[4,5,6,7]],'i64')
        for fn,args,expected in [(A.select_scatter,(A.tensor([8,9],'i64'),-1,-1),[[0,1,2,8],[4,5,6,9]]),
                                 (L.l_select_scatter,(A.tensor([8,9],'i64'),-1,-1),[[0,1,2,8],[4,5,6,9]]),
                                 (A.slice_scatter,(A.tensor([[8,9],[10,11]],'i64'),-1,-4,9,2),[[8,1,9,3],[10,5,11,7]]),
                                 (L.l_slice_scatter,(A.tensor([[8,9],[10,11]],'i64'),-1,-4,9,2),[[8,1,9,3],[10,5,11,7]])]:
            out=fn(base,*args)
            self.assertEqual(out.shape,(2,4))
            for idx in np.ndindex(out.shape):
                value=out.get(idx) if isinstance(out,L.Lazy) else out.a[idx]
                self.assertEqual(value,expected[idx[0]][idx[1]])
        self.assertEqual(A.plain(base),[[0,1,2,3],[4,5,6,7]])

    def test_empty_source_and_empty_remaining_dimensions_are_valid(self):
        for fn in (A.slice_scatter,L.l_slice_scatter):
            out=fn(A.tensor([1,2],'i64'),A.tensor([],'i64'),0,2,0)
            self.assertEqual(out.shape,(2,))
            self.assertEqual([out.get((i,)) if isinstance(out,L.Lazy) else out.a[i] for i in range(2)],[1,2])
        for fn in (A.select_scatter,L.l_select_scatter):
            out=fn(A.tensor(np.zeros((0,3),dtype=np.int64),'i64'),A.tensor([],'i64'),1,-1)
            self.assertEqual(out.shape,(0,3))

if __name__=='__main__': unittest.main()
