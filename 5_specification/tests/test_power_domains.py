import sys,unittest
from pathlib import Path
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A,lazy as L
from kernel_spec.num import Invalid,NotEstablished,Unsupported

class PowerDomains(unittest.TestCase):
    def test_integer_negative_scalar_power_is_rejected_before_values(self):
        for values in ([],[2]):
            for fn in (A.pow_,L.l_pow):
                with self.subTest(fn=fn.__name__,values=values),self.assertRaises(Invalid):fn(A.tensor(values,'i64'),-1)

    def test_integer_tensor_exponents_keep_exact_integer_values_and_dtype(self):
        for fn in (A.pow_,L.l_pow):
            out=fn(A.tensor([2,3,2**53+1],'i64'),A.tensor([3,2,1],'i64'))
            self.assertEqual(out.dtype,'i64')
            values=[out.get((i,)) if isinstance(out,L.Lazy) else out.a[i] for i in range(3)]
            self.assertEqual(values,[8,9,2**53+1])
            self.assertTrue(all(isinstance(v,int) for v in values))

    def test_float_dtype_is_not_widened_by_integer_tensor_exponents(self):
        for fn in (A.pow_,L.l_pow):
            out=fn(A.tensor([2,4],'f16'),A.tensor([-1,2],'i64'))
            self.assertEqual(out.dtype,'f16')
            self.assertEqual([(out.get((i,)) if isinstance(out,L.Lazy) else out.a[i]).v for i in range(2)],[Fraction(1,2),16])

    def test_negative_integer_tensor_exponent_is_not_given_a_float_answer(self):
        x=A.tensor([2],'i64');b=A.tensor([-1],'i64')
        with self.assertRaises(NotEstablished):A.pow_(x,b)
        out=L.l_pow(x,b)
        with self.assertRaises(NotEstablished):out.get((0,))

    def test_boolean_power_has_no_declared_real_boolean_semantics(self):
        for fn in (A.pow_,L.l_pow):
            with self.assertRaises(Unsupported):fn(A.tensor([True],'b8'),A.tensor([True],'b8'))

if __name__=='__main__':unittest.main()
