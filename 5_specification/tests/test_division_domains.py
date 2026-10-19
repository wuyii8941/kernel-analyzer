import sys,unittest
from pathlib import Path
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A,lazy as L
from kernel_spec.num import Invalid,Unsupported
from kernel_spec.real import R

class DivisionDomains(unittest.TestCase):
    def test_unknown_modes_are_rejected_including_empty_results(self):
        for values in ([],[3]):
            for mode in ('bad','round',0,False,[]):
                for fn in (A.div,L.l_div):
                    with self.subTest(fn=fn.__name__,mode=mode,values=values),self.assertRaises(Invalid):
                        fn(A.tensor(values,'i64'),2,rounding_mode=mode)

    def test_scalar_true_division_is_real_and_floating(self):
        for fn in (A.div,L.l_div):
            out=fn(3,2)
            self.assertEqual(out.dtype,'f32')
            value=out.get(()) if isinstance(out,L.Lazy) else out.a.item()
            self.assertIsInstance(value,R)
            self.assertEqual(value.v,Fraction(3,2))

    def test_integer_operand_does_not_widen_an_existing_float_dtype(self):
        for dtype in ('f16','bf16','f32','f64'):
            for fn in (A.div,L.l_div):
                out=fn(A.tensor([3],dtype),A.tensor([2],'i64'))
                self.assertEqual(out.dtype,dtype)
                value=out.get((0,)) if isinstance(out,L.Lazy) else out.a[0]
                self.assertEqual(value.v,Fraction(3,2))

    def test_rounding_direction_and_integer_zero(self):
        for fn in (A.div,L.l_div):
            for mode,expected in [('trunc',[-3,3]),('floor',[-4,3])]:
                out=fn(A.tensor([-7,7],'i64'),2,rounding_mode=mode)
                self.assertEqual([out.get((i,)) if isinstance(out,L.Lazy) else out.a[i] for i in range(2)],expected)
                if fn is A.div:
                    with self.assertRaises(Invalid):fn(A.tensor([1],'i64'),0,rounding_mode=mode)
                else:
                    out=fn(A.tensor([1],'i64'),0,rounding_mode=mode)
                    with self.assertRaises(Invalid):out.get((0,))

    def test_rounded_boolean_division_is_not_an_integer_result_with_bool_label(self):
        for fn in (A.div,L.l_div):
            for mode in ('trunc','floor'):
                with self.assertRaises(Unsupported):fn(A.tensor([True],'b8'),A.tensor([True],'b8'),mode)

if __name__=='__main__':unittest.main()
