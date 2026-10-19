import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A,lazy as L
from kernel_spec.num import NotEstablished,Unsupported

class ConvolutionDtypes(unittest.TestCase):
    def test_integer_forward_preserves_dtype_and_exact_large_values(self):
        big=2**53+1
        for dtype in ('i32','i64'):
            values=[1,2,3] if dtype=='i32' else [big,big+1,big+2]
            x=A.tensor([[values]],dtype);w=A.tensor([[[1]]],dtype);b=A.tensor([2],dtype)
            for fn in (A.convolution,L.l_convolution):
                out=fn(x,w,b,[1],[0],[1],False,[0],1)
                self.assertEqual(out.dtype,dtype)
                for i,v in enumerate(values):
                    result=out.get((0,0,i)) if isinstance(out,L.Lazy) else out.a[0,0,i]
                    self.assertIsInstance(result,int)
                    self.assertEqual(result,v+2)

    def test_mixed_dtypes_are_not_silently_reinterpreted(self):
        for n in (0,1):
            x=A.tensor(np.ones((n,1,3)),'f32');w=A.tensor([[[1]]],'f64')
            for fn in (A.convolution,L.l_convolution):
                with self.assertRaises(NotEstablished):fn(x,w,None,[1],[0],[1],False,[0],1)
                with self.assertRaises(NotEstablished):fn(x,A.tensor([[[1]]],'f32'),A.tensor([1],'f64'),[1],[0],[1],False,[0],1)

    def test_backward_does_not_invent_float_gradients_for_integer_inputs(self):
        t=A.tensor([[[1]]],'i64')
        with self.assertRaises(Unsupported):A.convolution_backward(t,t,t,[1],[1],[0],[1],False,[0],1,[True]*3)
        with self.assertRaises(NotEstablished):A.convolution_backward(A.tensor([[[1]]],'f64'),A.tensor([[[1]]],'f32'),A.tensor([[[1]]],'f32'),[1],[1],[0],[1],False,[0],1,[True]*3)

    def test_boolean_convolution_is_not_a_float_convolution(self):
        t=A.tensor([[[True]]],'b8')
        for fn in (A.convolution,L.l_convolution):
            with self.assertRaises(Unsupported):fn(t,t,None,[1],[0],[1],False,[0],1)

if __name__=='__main__':unittest.main()
