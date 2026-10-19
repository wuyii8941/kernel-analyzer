import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A,lazy as L
from kernel_spec.num import Invalid,NotEstablished

class ConvolutionDomains(unittest.TestCase):
    def test_backward_requires_exact_upstream_shape(self):
        x=A.tensor([[[1,2,3]]],'f64');w=A.tensor([[[1]]],'f64')
        for shape in ((2,1,3),(1,2,3),(1,1,2),(1,1,4),(1,1,3,1)):
            with self.subTest(shape=shape),self.assertRaises(Invalid):
                A.convolution_backward(A.tensor(np.ones(shape),'f64'),x,w,[1],[1],[0],[1],False,[0],1,[True]*3)

    def test_forward_validates_all_metadata_before_coordinates(self):
        # Include zero batch so evaluation cannot hide missing checks.
        for shape,wshape,stride,padding,dilation,groups,bias in [
            ((0,2,3),(2,1,1),[1],[0],[1],1,None),
            ((0,1,3),(1,1,1),[0],[0],[1],1,None),
            ((0,1,3),(1,1,1),[1],[-1],[1],1,None),
            ((0,1,3),(1,1,1),[1],[0],[-1],1,None),
            ((0,1,3),(1,1,1),[1,2],[0],[1],1,None),
            ((0,1,3),(1,1,1),[1],[0],[1],0,None),
            ((0,1,3),(1,1,1),[1],[0],[1],1,A.tensor([0,1],'f64')),
            ((1,1,3),(1,1,4),[1],[0],[1],1,None)]:
            x=A.tensor(np.ones(shape),'f64');w=A.tensor(np.ones(wshape),'f64')
            for fn in (A.convolution,L.l_convolution):
                with self.subTest(fn=fn.__name__,shape=shape,stride=stride,groups=groups),self.assertRaises(Invalid):
                    fn(x,w,bias,stride,padding,dilation,False,[0],groups)

    def test_single_spatial_parameter_expands_and_bias_is_complete(self):
        x=A.tensor([[[[1,2],[3,4]]]],'f64');w=A.tensor([[[[2]]]],'f64');bias=A.tensor([1],'f64')
        for fn in (A.convolution,L.l_convolution):
            out=fn(x,w,bias,[1],[0],[1],False,[1],1)
            self.assertEqual(out.shape,(1,1,2,2))
            for idx in np.ndindex(out.shape):
                actual=out.get(idx) if isinstance(out,L.Lazy) else out.a[idx]
                self.assertEqual(actual.v,2*x.a[idx].v+1)

    def test_zero_dilation_stays_unestablished(self):
        for n in (0,1):
            x=A.tensor(np.ones((n,1,3)),'f64');w=A.tensor([[[1]]],'f64')
            for fn in (A.convolution,L.l_convolution):
                with self.assertRaises(NotEstablished):fn(x,w,None,[1],[0],[0],False,[0],1)

if __name__=='__main__':unittest.main()
