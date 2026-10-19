import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A,lazy as L
from kernel_spec.num import Invalid

class WhereBroadcastDomains(unittest.TestCase):
    def test_nonboolean_condition_rejected_before_any_coordinate(self):
        for shape in ((2,),(0,)):
            for dtype in ('i64','f32'):
                for fn in (A.where,L.l_where):
                    with self.subTest(shape=shape,dtype=dtype,fn=fn.__name__),self.assertRaises(Invalid):
                        fn(A.tensor(np.zeros(shape),dtype),1,2)

    def test_incompatible_shapes_are_typed_errors(self):
        for sa,sb in [((2,),(3,)),((0,2),(0,3)),((2,1),(3,2))]:
            a=A.tensor(np.zeros(sa),'i64');b=A.tensor(np.zeros(sb),'i64')
            for fn in (A.add,L.l_add,A.where,L.l_where):
                with self.subTest(sa=sa,sb=sb,fn=fn.__name__),self.assertRaises(Invalid):
                    fn(A.tensor(True,'b8'),a,b) if fn in (A.where,L.l_where) else fn(a,b)

    def test_boolean_broadcast_selects_every_coordinate(self):
        c=A.tensor([[True],[False]],'b8');a=A.tensor([[1,2,3]],'i64');b=A.tensor([7,8,9],'i64')
        for fn in (A.where,L.l_where):
            out=fn(c,a,b)
            self.assertEqual(out.shape,(2,3));self.assertEqual(out.dtype,'i64')
            expected=[[1,2,3],[7,8,9]]
            for idx in np.ndindex(out.shape):self.assertEqual(out.get(idx) if isinstance(out,L.Lazy) else out.a[idx],expected[idx[0]][idx[1]])

    def test_scalar_and_zero_extent_broadcasting_remain_valid(self):
        for fn in (A.where,L.l_where):
            scalar=fn(True,1,2)
            self.assertEqual(scalar.shape,())
            self.assertEqual(scalar.get(()) if isinstance(scalar,L.Lazy) else scalar.a.item(),1)
            out=fn(A.tensor(np.zeros((0,1),dtype=bool),'b8'),A.tensor([[1,2,3]],'i64'),0)
            self.assertEqual(out.shape,(0,3))

    def test_lazy_broadcasting_does_not_depend_on_numpy_address_sizes(self):
        x=L.Lazy((2**64,),'i64',lambda _:0)
        out=L.l_add(x,1)
        self.assertEqual(out.shape,(2**64,))
        self.assertEqual(out.get((2**64-1,)),1)

if __name__=='__main__':unittest.main()
