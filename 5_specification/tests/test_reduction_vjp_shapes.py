import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A
from kernel_spec.num import Invalid

class ReductionVJPShapes(unittest.TestCase):
    reductions=(A.vjp_sum,A.vjp_mean,A.vjp_prod,A.vjp_amax,A.vjp_amin,
                A.vjp_max_dim,A.vjp_min_dim,A.vjp_var,A.vjp_std,A.vjp_logsumexp,A.vjp_norm)

    def test_reduction_upstream_must_match_entire_forward_output(self):
        for fn in self.reductions:
            for shape in ((2,3),(0,3)):
                for keepdim in (False,True):
                    x=A.tensor(np.ones(shape),'f64')
                    g=A.tensor(np.ones((1,1) if keepdim else (1,)),'f64')
                    with self.subTest(fn=fn.__name__,shape=shape,keepdim=keepdim),self.assertRaises(Invalid):
                        fn(g,x,dim=1,keepdim=keepdim)

    def test_keepdim_is_boolean_even_for_empty_output(self):
        for fn in self.reductions:
            with self.subTest(fn=fn.__name__),self.assertRaises(Invalid):
                fn(A.tensor(np.ones((0,1)),'f64'),A.tensor(np.ones((0,3)),'f64'),dim=1,keepdim='false')

    def test_cumsum_preserves_input_shape_and_validates_axis(self):
        x=A.tensor(np.zeros((2,3)),'f64')
        for shape in ((1,3),(2,1),(2,3,1)):
            with self.subTest(shape=shape),self.assertRaises(Invalid):
                A.vjp_cumsum(A.tensor(np.ones(shape),'f64'),x,1)
        for dim in (2,-3,1.5,True):
            with self.subTest(dim=dim),self.assertRaises(Invalid):
                A.vjp_cumsum(A.tensor(np.ones((2,3)),'f64'),x,dim)

    def test_scalar_cumsum_gradient_is_identity(self):
        for dim in (0,-1):
            out=A.vjp_cumsum(A.tensor(7,'f64'),A.tensor(3,'f64'),dim)
            self.assertEqual(out.shape,());self.assertEqual(out.a.item().v,7)

    def test_valid_cumsum_complete_gradients_against_autograd(self):
        import torch
        for shape in ((),(4,),(2,3),(0,3)):
            for dim in range(-max(1,len(shape)),max(1,len(shape))):
                with self.subTest(shape=shape,dim=dim):
                    x=torch.ones(shape,dtype=torch.float64,requires_grad=True)
                    g=torch.arange(x.numel(),dtype=torch.float64).reshape(shape)+1
                    torch.cumsum(x,dim).backward(g)
                    result=A.vjp_cumsum(A.tensor(g.numpy(),'f64'),A.tensor(x.detach().numpy(),'f64'),dim)
                    self.assertEqual(result.shape,shape)
                    for i in np.ndindex(shape):self.assertEqual(result.a[i].v,int(x.grad[i].item()))

if __name__=='__main__':unittest.main()
