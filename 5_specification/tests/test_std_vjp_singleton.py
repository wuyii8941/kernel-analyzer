import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A
from kernel_spec.num import Undefined

class SingletonStdVJP(unittest.TestCase):
    def test_scalar_std_is_constant_zero_when_denominator_positive(self):
        for x in (-7,0,5):
            for correction in (0,-2):
                for dim in (None,0,-1):
                    out=A.vjp_std(A.tensor(3,'f64'),A.tensor(x,'f64'),dim=dim,correction=correction)
                    self.assertEqual(out.shape,());self.assertEqual(out.a.item().v,0)

    def test_singleton_reduced_axes_return_every_zero_gradient(self):
        import torch
        for shape,dims in [((2,1,3),1),((2,1,1),(1,2)),((1,),0),((0,1,3),1)]:
            for keepdim in (False,True):
                x=torch.arange(int(np.prod(shape)),dtype=torch.float64).reshape(shape).requires_grad_()
                y=torch.std(x,dim=dims,correction=0,keepdim=keepdim)
                g=torch.ones_like(y)*3;y.backward(g)
                result=A.vjp_std(A.tensor(g.numpy(),'f64'),A.tensor(x.detach().numpy(),'f64'),dim=dims,correction=0,keepdim=keepdim)
                self.assertEqual(result.shape,shape)
                for i in np.ndindex(shape):
                    self.assertEqual(result.a[i].v,0)
                    self.assertEqual(x.grad[i].item(),0)

    def test_multiple_equal_elements_do_not_gain_an_unselected_subgradient(self):
        # For std([3+h,3], correction=0), the value is |h|/2.
        # Its left and right slopes disagree at h=0.
        with self.assertRaises(Undefined):
            A.vjp_std(A.tensor(1,'f64'),A.tensor([3,3],'f64'),correction=0)

    def test_nonpositive_degrees_of_freedom_and_infinite_input_stay_undefined(self):
        for correction in (1,2):
            with self.assertRaises(Undefined):
                A.vjp_std(A.tensor(1,'f64'),A.tensor(3,'f64'),correction=correction)
        with self.assertRaises(Undefined):
            A.vjp_std(A.tensor(1,'f64'),A.tensor(float('inf'),'f64'),correction=0)

if __name__=='__main__':unittest.main()
