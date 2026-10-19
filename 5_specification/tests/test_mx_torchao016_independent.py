import struct
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel_spec import mx
from test_mx_core import candidates


INFO = {"e4m3": (8, Q(448)), "e5m2": (15, Q(57344))}


def source_exponent(value):
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return ((bits >> 23) & 0xFF) - 127


def independent_block(values, fmt):
    maximum_power, maximum = INFO[fmt]
    amax = max(map(abs, values))
    exponent = max(-127, min(128, source_exponent(amax) - maximum_power))
    code = exponent + 127
    divisor = Q(2) ** max(-126, exponent)
    points = candidates(fmt)
    elements = []
    for value in values:
        normalized = max(-maximum, min(maximum, value / divisor))
        sign = -1 if normalized < 0 else 1
        chosen = min(points, key=lambda point: (abs(abs(normalized) - point[0]), point[1]))[0]
        elements.append(sign * chosen)
    return code, elements


class TorchAO016MXIndependent(unittest.TestCase):
    def test_scale_bits_and_element_enumeration_are_independent(self):
        blocks = [
            [Q(500), Q(1), Q(-3, 4)] + [Q(0)] * 29,
            [Q(2) ** -127] * 32,
            [Q(2) ** -118, Q(2) ** -120, Q(-2) ** -121] + [Q(0)] * 29,
            [Q(65536), Q(32768), Q(-1024)] + [Q(0)] * 29,
        ]
        for fmt in INFO:
            for block in blocks:
                expected_code, expected_elements = independent_block(block, fmt)
                actual = mx.torchao016_floor_block(block, fmt)
                self.assertEqual(actual["scale_code"], expected_code)
                self.assertEqual(actual["elements"], expected_elements)

    def test_two_dimensional_block_membership_and_all_coordinates(self):
        x = []
        for row in range(3):
            x.append([Q((row + 1) * (column + 3), 8) for column in range(64)])
        for fmt in INFO:
            actual = mx.torchao016_floor_tensor(x, fmt)
            expected_scales, expected_elements = [], []
            for row in x:
                row_scales, row_elements = [], []
                for start in (0, 32):
                    code, elements = independent_block(row[start:start + 32], fmt)
                    row_scales.append(code)
                    row_elements.extend(elements)
                expected_scales.append(row_scales)
                expected_elements.append(row_elements)
            self.assertEqual(actual["scale_codes"], expected_scales)
            self.assertEqual(actual["elements"], expected_elements)

    def test_e4m3_midpoint_uses_even_retained_significand(self):
        block = [Q(256), Q(17, 16), Q(19, 16)] + [Q(0)] * 29
        actual = mx.torchao016_floor_block(block, "e4m3")
        self.assertEqual(actual["scale_code"], 127)
        self.assertEqual(actual["elements"][:3], [256, 1, Q(5, 4)])


if __name__ == "__main__":
    unittest.main()
