import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A, lazy as L
from kernel_spec.num import Invalid

class SelectSliceDomains(unittest.TestCase):
    def test_select_rejects_invalid_axes_and_indices_before_reading(self):
        for shape, dim, index in [((2,),0,2),((2,),0,-3),((2,),1,0),((2,),-2,0),
                                  ((0,2),1,2),((0,),0,0),((),0,0),
                                  ((2,),0,1.5),((2,),True,0),((2,),0,False)]:
            for fn, x in [(A.select,A.tensor(np.zeros(shape,dtype=np.int64),'i64')),
                          (L.l_select,L.Lazy(shape,'i64',lambda _: self.fail('unexpected read')))]:
                with self.subTest(op=fn.__name__,shape=shape,dim=dim,index=index), self.assertRaises(Invalid):
                    fn(x,dim,index)

    def test_slice_rejects_invalid_metadata_even_for_empty_output(self):
        for shape, args in [((2,),(1,0,0,1)),((2,),(-2,0,0,1)),((),(0,0,0,1)),
                            ((0,),(0,0,0,0)),((0,),(0,0,0,-1)),((0,),(0,0.5,0,1)),
                            ((0,),(0,0,False,1)),((2,),(0,None,None,1.5))]:
            for fn, x in [(A.slice_,A.tensor(np.zeros(shape,dtype=np.int64),'i64')),
                          (L.l_slice,L.Lazy(shape,'i64',lambda _: self.fail('unexpected read')))]:
                with self.subTest(op=fn.__name__,shape=shape,args=args), self.assertRaises(Invalid):
                    fn(x,*args)

    def test_slice_matches_independent_python_coordinates(self):
        data=[[10,11,12,13],[20,21,22,23]]
        for start in (None,-9,-2,0,2,9):
            for end in (None,-9,-1,0,3,9):
                for step in (1,2,5):
                    expected=[row[start:end:step] for row in data]
                    for fn in (A.slice_,L.l_slice):
                        out=fn(A.tensor(data,'i64'),-1,start,end,step)
                        self.assertEqual(out.shape,(2,len(expected[0])))
                        for idx in np.ndindex(out.shape):
                            value=out.get(idx) if isinstance(out,L.Lazy) else out.a[idx]
                            self.assertEqual(value,expected[idx[0]][idx[1]])

    def test_select_negative_index_and_empty_remaining_axis(self):
        for fn in (A.select,L.l_select):
            x=A.tensor([[10,11],[20,21]],'i64'); out=fn(x,-1,-1)
            vals=[out.get((i,)) if isinstance(out,L.Lazy) else out.a[i] for i in range(2)]
            self.assertEqual(vals,[11,21])
            self.assertEqual(fn(A.tensor(np.zeros((0,2),dtype=np.int64),'i64'),1,-1).shape,(0,))

if __name__=='__main__': unittest.main()
