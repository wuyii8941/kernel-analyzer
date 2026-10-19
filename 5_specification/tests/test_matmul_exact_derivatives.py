"""Independent exact derivatives of documented matrix-product objectives.

NumPy object matmul evaluates the forward formula using Fraction arithmetic.
Holding other operands fixed, <G, f> is affine in each coordinate; therefore
(F(x+1)-F(x-1))/2 is the exact partial derivative, not a numerical estimate.
No reference VJP, transpose-gradient or broadcast-reduction routine is used
when constructing the expected derivatives.
"""
import sys,unittest
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A


def values(shape,offset=0):
    out=np.empty(shape,dtype=object)
    for n,index in enumerate(np.ndindex(shape)):
        out[index]=Q((n+offset)%7-3,8)
    return out


def exact_derivatives(forward,operands,upstream):
    def objective(args):
        output=np.asarray(forward(*args),dtype=object)
        assert output.shape==upstream.shape
        return sum((output[i]*upstream[i] for i in np.ndindex(output.shape)),Q(0))
    derivatives=[]
    for operand_index,operand in enumerate(operands):
        if operand is None:
            derivatives.append(None)
            continue
        result=np.empty(operand.shape,dtype=object)
        for index in np.ndindex(operand.shape):
            plus=list(operands);minus=list(operands)
            plus[operand_index]=operand.copy();minus[operand_index]=operand.copy()
            plus[operand_index][index]+=1;minus[operand_index][index]-=1
            result[index]=(objective(plus)-objective(minus))/2
        derivatives.append(result)
    return derivatives


class ExactMatrixDerivatives(unittest.TestCase):
    def check_case(self,forward,vjp,operands,kwargs=None):
        output=np.asarray(forward(*operands),dtype=object)
        upstream=values(output.shape,4)
        expected=exact_derivatives(forward,operands,upstream)
        args=[None if x is None else A.tensor(x,'f64') for x in operands]
        actual=vjp(A.tensor(upstream,'f64'),*args,**(kwargs or {}))
        self.assertEqual(len(actual),len(expected))
        for result,target in zip(actual,expected):
            if target is None:
                self.assertIsNone(result)
                continue
            self.assertEqual(result.dtype,'f64')
            self.assertEqual(result.shape,target.shape)
            for i in np.ndindex(target.shape):
                self.assertEqual(result.a[i].v,target[i],(vjp.__name__,target.shape,i))

    def test_matmul_vectors_broadcasts_and_empty_dimensions(self):
        shapes=[((3,),(3,)),((3,),(3,2)),((2,3),(3,)),
                ((3,),(2,3,2)),((2,2,3),(3,)),
                ((2,1,2,3),(3,3,2)),((2,3),(2,3,2)),
                ((0,),(0,)),((2,0),(0,3)),((0,2,3),(1,3,2))]
        for sa,sb in shapes:
            with self.subTest(a=sa,b=sb):
                self.check_case(np.matmul,A.vjp_matmul,(values(sa),values(sb,2)))

    def test_mm_and_bmm_complete_operand_gradients(self):
        for fn,sa,sb in [(A.vjp_mm,(2,3),(3,2)),(A.vjp_mm,(2,0),(0,3)),
                         (A.vjp_bmm,(2,2,3),(2,3,2)),(A.vjp_bmm,(0,2,3),(0,3,2))]:
            with self.subTest(fn=fn.__name__,a=sa):
                self.check_case(np.matmul,fn,(values(sa),values(sb,2)))

    def test_linear_matrix_vector_and_scalar_bias(self):
        for sx,sw,sb in [((2,3),(2,3),(2,)),((2,2,3),(2,3),()),
                         ((3,),(2,3),None),((2,3),(3,),()),
                         ((3,),(3,),None),((0,3),(2,3),(2,))]:
            with self.subTest(x=sx,w=sw,bias=sb):
                def forward(x,w,b):return np.matmul(x,w.T)+(0 if b is None else b)
                self.check_case(forward,A.vjp_linear,
                                (values(sx),values(sw,2),None if sb is None else values(sb,1)))

    def test_addmm_scales_and_broadcast_addend(self):
        for sc in [(),(2,),(1,2),(2,1),(2,2)]:
            for alpha,beta in [(Q(-3,4),Q(1,2)),(Q(0),Q(1)),(Q(1),Q(0))]:
                with self.subTest(c=sc,alpha=alpha,beta=beta):
                    def forward(c,a,b):return beta*c+alpha*np.matmul(a,b)
                    self.check_case(forward,A.vjp_addmm,
                                    (values(sc,1),values((2,3)),values((3,2),2)),dict(alpha=alpha,beta=beta))

if __name__=='__main__':unittest.main()
