import sys,unittest
from fractions import Fraction
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A,lazy as L
from kernel_spec.num import Invalid

class LazyArithmeticAlpha(unittest.TestCase):
    def test_invalid_alpha_is_rejected_before_reading_even_for_empty_tensors(self):
        for size in (0,2):
            x=L.Lazy((size,),'i64',lambda _:self.fail('unexpected value read'))
            for fn in (L.l_add,L.l_sub):
                for alpha in (Fraction(3,2),1.5,1.0,True):
                    with self.subTest(fn=fn.__name__,size=size,alpha=alpha),self.assertRaises(Invalid):fn(x,x,alpha)

    def test_boolean_zero_scaling_does_not_include_second_operand(self):
        a=A.tensor([False,True],'b8');b=A.tensor([True,False],'b8')
        for alpha in (0,False):
            out=L.l_add(a,b,alpha)
            self.assertEqual([out.get((i,)) for i in range(2)],[False,True])
        for alpha in (1,-1,2,True):
            out=L.l_add(a,b,alpha)
            self.assertEqual([out.get((i,)) for i in range(2)],[True,True])

    def test_subtraction_rejects_boolean_operands_including_scalar_forms(self):
        for a,b in [(A.tensor([1],'i64'),True),(1,A.tensor([True],'b8')),
                    (A.tensor([],'f32'),A.tensor([],'b8'))]:
            for fn in (A.sub,L.l_sub):
                with self.subTest(fn=fn.__name__),self.assertRaises(Invalid):fn(a,b)

    def test_valid_scaling_matches_exact_formula(self):
        for fn,sign in ((L.l_add,1),(L.l_sub,-1)):
            for dtype,alpha in [('i64',-2),('f32',Fraction(1,3))]:
                out=fn(A.tensor([2,-3],dtype),A.tensor([3,6],dtype),alpha)
                for i,(a,b) in enumerate(((2,3),(-3,6))):
                    v=out.get((i,));self.assertEqual(v.v if hasattr(v,'v') else v,a+sign*alpha*b)

if __name__=='__main__':unittest.main()
