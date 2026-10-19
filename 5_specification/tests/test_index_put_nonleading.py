import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A,lazy as L
from kernel_spec.num import Invalid,Undefined

class NonleadingIndexPut(unittest.TestCase):
    def check(self,base,indices,values,expected,accumulate=False):
        for fn in (A.index_put,L.l_index_put):
            out=fn(A.tensor(base,'i64'),indices,A.tensor(values,'i64'),accumulate)
            self.assertEqual(out.shape,np.shape(base))
            for i in np.ndindex(out.shape):
                self.assertEqual(out.get(i) if isinstance(out,L.Lazy) else out.a[i],expected[i])

    def test_middle_axis_keeps_outer_dimensions(self):
        base=np.arange(24).reshape(2,3,4);expected=base.copy()
        values=np.arange(16).reshape(2,2,4)+100
        for b in range(2):
            for j,col in enumerate((2,0)):
                for k in range(4):expected[b,col,k]=values[b,j,k]
        self.check(base,[None,A.tensor([2,0],'i64')],values,expected)

    def test_separated_axes_move_broadcast_dimensions_to_front(self):
        base=np.arange(24).reshape(2,3,4);expected=base.copy()
        values=np.arange(12).reshape(2,2,3)+100
        for b,row in enumerate((1,0)):
            for j,col in enumerate((3,1)):
                for mid in range(3):expected[row,mid,col]=values[b,j,mid]
        self.check(base,[A.tensor([[1],[0]],'i64'),None,A.tensor([[3,1]],'i64')],values,expected)

    def test_repeated_nonleading_indices_accumulate_or_are_undefined(self):
        base=np.arange(6).reshape(2,3);expected=base.copy();expected[:,1]+=7
        indices=[None,A.tensor([1,-2],'i64')]
        self.check(base,indices,[3,4],expected,True)
        for fn in (A.index_put,L.l_index_put):
            with self.assertRaises(Undefined):fn(A.tensor(base,'i64'),indices,A.tensor([3,4],'i64'))

    def test_empty_unindexed_axis_does_not_hide_out_of_range_index(self):
        for fn in (A.index_put,L.l_index_put):
            with self.assertRaises(Invalid):
                fn(A.tensor(np.zeros((0,3)),'i64'),[None,A.tensor([3],'i64')],A.tensor(7,'i64'))

    def test_empty_selection_preserves_all_values_and_validates_source(self):
        base=np.arange(6).reshape(2,3);indices=[None,A.tensor([],'i64')]
        self.check(base,indices,np.empty((2,0),dtype=int),base)
        for fn in (A.index_put,L.l_index_put):
            with self.assertRaises(Invalid):fn(A.tensor(base,'i64'),indices,A.tensor(np.zeros((3,0)),'i64'))

if __name__=='__main__':unittest.main()
