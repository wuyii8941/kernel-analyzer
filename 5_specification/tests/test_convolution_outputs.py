import sys,unittest,itertools
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A
from kernel_spec.num import Invalid

class ConvolutionOutputs(unittest.TestCase):
    def call(self,mask):
        x=A.tensor([[[1,2,3]]],'f64');w=A.tensor([[[1]]],'f64');g=A.tensor([[[2,3,4]]],'f64')
        return A.convolution_backward(g,x,w,[1],[1],[0],[1],False,[0],1,mask)

    def test_output_mask_cannot_truncate_or_invent_gradient_positions(self):
        for mask in ([],[True],[True,False],[True]*4,'yes',[True,False,1],[True,None,False]):
            with self.subTest(mask=mask),self.assertRaises(Invalid):self.call(mask)

    def test_every_boolean_mask_preserves_three_output_positions(self):
        expected=[[[[2,3,4]]],[[[20]]],[9]]
        for mask in itertools.product((False,True),repeat=3):
            out=self.call(mask)
            self.assertEqual(len(out),3)
            for i,enabled in enumerate(mask):
                if enabled:self.assertEqual(A.plain(out[i]),expected[i])
                else:self.assertIsNone(out[i])

    def test_grouped_dilated_gradients_match_independent_autograd(self):
        import torch
        import torch.nn.functional as F
        x=torch.arange(56,dtype=torch.float64).reshape(2,4,7).remainder(5).sub(2).requires_grad_()
        w=torch.arange(16,dtype=torch.float64).reshape(4,2,2).remainder(3).sub(1).requires_grad_()
        b=torch.zeros(4,dtype=torch.float64,requires_grad=True)
        y=F.conv1d(x,w,b,stride=2,padding=1,dilation=2,groups=2)
        g=torch.arange(y.numel(),dtype=torch.float64).reshape(y.shape).remainder(3).sub(1)
        expected=torch.autograd.grad(y,(x,w,b),g)
        args=(A.tensor(g,'f64'),A.tensor(x,'f64'),A.tensor(w,'f64'),[4],[2],[1],[2],False,[0],2,[True]*3)
        actual=A.convolution_backward(*args)
        for out,e in zip(actual,expected):
            self.assertEqual(out.shape,tuple(e.shape))
            for idx in np.ndindex(out.shape):self.assertEqual(out.a[idx].v,e[idx].item())

if __name__=='__main__':unittest.main()
