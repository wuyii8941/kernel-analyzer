"""Cross-check of the independent rounding oracle against hardware rounding (numpy) and against the documented edge values."""
import unittest, random
from fractions import Fraction as F
import numpy as np
from oracle_rounding import round_to
class T(unittest.TestCase):
    def test_rne_matches_numpy(self):
        rng = random.Random(3)
        for _ in range(4000):
            x64 = rng.uniform(-6e4, 6e4) * rng.choice([1.0, 2.0**-20, 2.0**-30, 2.0**-40])
            self.assertEqual(round_to(F(x64), 'f32'), F(float(np.float32(x64))))
            self.assertEqual(round_to(F(x64), 'f16'), F(float(np.float16(x64))))
    def test_bf16_ties_and_subnormal(self):
        self.assertEqual(round_to(1 + F(1, 256), 'bf16'), 1); self.assertEqual(round_to(1 + F(3, 256), 'bf16'), 1 + F(4, 256))
        self.assertEqual(round_to(F(2) ** -150, 'f32'), 0)                                   # below half the smallest subnormal
        self.assertEqual(round_to(F(2) ** -149, 'f32'), F(2) ** -149)                         # smallest subnormal
    def test_directed_overflow(self):
        self.assertEqual(round_to(F(70000), 'f16', 'rz'), 65504); self.assertEqual(round_to(F(70000), 'f16', 'rd'), 65504)
        self.assertEqual(round_to(F(-70000), 'f16', 'ru'), -65504); self.assertEqual(round_to(F(70000), 'f16', 'ru'), 'inf')
        self.assertEqual(round_to(F(70000), 'f16', 'rne'), 'inf')
    def test_fp8(self):
        self.assertEqual(round_to(F(448), 'f8e4m3fn'), 448); self.assertEqual(round_to(F(1000), 'f8e4m3fn'), 'nan')
        self.assertEqual(round_to(F(1000), 'f8e4m3fn', saturate=True), 448); self.assertEqual(round_to(F(57344), 'f8e5m2'), 57344)
        self.assertEqual(round_to(F(100000), 'f8e5m2'), 'inf')
if __name__ == '__main__': unittest.main()
