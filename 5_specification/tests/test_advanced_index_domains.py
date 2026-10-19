import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A,lazy as L
from kernel_spec.num import Invalid,Unsupported

class AdvancedIndexDomains(unittest.TestCase):
    def operations(self,a,indices):
        return [(A.index,(a,indices)),(L.l_index,(a,indices)),
                (A.index_put,(a,indices,A.tensor(7,'i64'))),(L.l_index_put,(a,indices,A.tensor(7,'i64')))]

    def test_masks_not_interpreted_as_integer_positions(self):
        for dtype in ('u8','b8'):
            for fn,args in self.operations(A.tensor([10,20,30],'i64'),[A.tensor([1,0,1],dtype)]):
                with self.subTest(dtype=dtype,fn=fn.__name__),self.assertRaises(Unsupported):fn(*args)

    def test_small_integer_index_types_rejected(self):
        for dtype in ('i8','i16'):
            for values in ([0],[]):
                for fn,args in self.operations(A.tensor([10,20,30],'i64'),[A.tensor(values,dtype)]):
                    with self.subTest(dtype=dtype,values=values,fn=fn.__name__),self.assertRaises(Invalid):fn(*args)

    def test_index_metadata_errors_are_typed(self):
        a=A.tensor([[10,20],[30,40]],'i64')
        for indices in ([A.tensor([0],'i64')]*3,[A.tensor([0,1],'i64'),A.tensor([0,1,0],'i64')]):
            for fn,args in self.operations(a,indices):
                with self.subTest(fn=fn.__name__,indices=len(indices)),self.assertRaises(Invalid):fn(*args)

    def test_large_out_of_range_index_is_not_machine_cast(self):
        with self.assertRaises(Invalid):A.index(A.tensor([10,20],'i64'),[A.tensor([2**64],'i64')])

    def test_scalar_indices_write_one_complete_coordinate(self):
        for fn in (A.index_put,L.l_index_put):
            for accumulate in (False,True):
                out=fn(A.tensor([[10,20],[30,40]],'i64'),[A.tensor(1,'i64'),A.tensor(0,'i64')],A.tensor(7,'i64'),accumulate)
                expected=[[10,20],[37 if accumulate else 7,40]]
                for i in np.ndindex(out.shape):
                    self.assertEqual(out.get(i) if isinstance(out,L.Lazy) else out.a[i],expected[i[0]][i[1]])

    def test_broadcast_multiple_integer_indices_complete_values(self):
        a=A.tensor([[10,20,30],[40,50,60]],'i64')
        for dtype in ('i32','i64'):
            indices=[A.tensor([[0],[1]],dtype),A.tensor([[2,0]],dtype)]
            for fn in (A.index,L.l_index):
                out=fn(a,indices)
                self.assertEqual(out.shape,(2,2))
                for i in np.ndindex(out.shape):
                    self.assertEqual(out.get(i) if isinstance(out,L.Lazy) else out.a[i],[[30,10],[60,40]][i[0]][i[1]])

if __name__=='__main__':unittest.main()
