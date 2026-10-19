import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A,lazy as L
from kernel_spec.num import Invalid

class CreationShapes(unittest.TestCase):
    def test_invalid_shapes_are_typed_input_errors_before_value_conversion(self):
        for shape in ([-1],[0,-1],[True],[2.0],[-2],2,'23'):
            for fn in (A.full,L.l_full):
                with self.subTest(shape=shape,fn=fn.__name__),self.assertRaises(Invalid):
                    fn(shape,object(),'i64')

    def test_full_supports_scalar_and_empty_shapes(self):
        for shape in ([],[0],[2,0,3],[2,3],[np.int64(2)]):
            for fn in (A.full,L.l_full):
                out=fn(shape,7,'i64')
                self.assertEqual(out.shape,tuple(shape))
                for idx in np.ndindex(out.shape):self.assertEqual(out.get(idx) if isinstance(out,L.Lazy) else out.a[idx],7)

    def test_creation_wrappers_cannot_ignore_invalid_sizes(self):
        x=A.tensor([1,2],'i64')
        for fn in (lambda:A.zeros([-1]),lambda:A.ones([1.5]),lambda:A.new_zeros(x,[False])):
            with self.assertRaises(Invalid):fn()

if __name__=='__main__':unittest.main()
