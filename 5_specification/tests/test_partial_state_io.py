"""Persistence must not turn outward enclosures into decimal point estimates."""
import json
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

from mpmath import iv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import attention_partials as P
from kernel_spec.numeric_io import encode_finite, decode_finite
from kernel_spec.num import Invalid, ends, from_interval


class StateIOTests(unittest.TestCase):
    def test_exact_rational_survives_json_without_float_or_decimal_rounding(self):
        for value in (F(1, 3), F(-7, 13), F(1, 2**20000), F(2**20000, 3), F(0), 0.1):
            encoded = json.loads(json.dumps(encode_finite(value)))
            self.assertEqual(decode_finite(encoded), value)

    def test_lower_precision_restore_encloses_original_endpoints(self):
        saved = iv.prec
        try:
            iv.prec = 512
            original = from_interval(F(1, 3), F(2, 3))
            before = ends(original)
            record = json.loads(json.dumps(encode_finite(original)))
            iv.prec = 32
            after = ends(decode_finite(record))
            self.assertLessEqual(after[0], before[0])
            self.assertGreaterEqual(after[1], before[1])
            self.assertLessEqual(after[0], F(1, 3))
            self.assertGreaterEqual(after[1], F(2, 3))
        finally:
            iv.prec = saved

    def test_complete_state_roundtrip_and_followup_merge(self):
        state = P.scaled(0, 3, [1, 5, -2])
        restored = P.loads_state(P.dumps_state(state))
        self.assertEqual(restored, state)
        result = P.finish(P.merge([restored, P.loads_state(P.dumps_state(P.empty(3)))]))
        self.assertEqual(result['out'], [F(1, 3), F(5, 3), F(-2, 3)])
        original = P.merge([P.scaled(-2, 2, [4, -2, 10]), P.scaled(3, 3, [-12, 21, 9])])
        restored = P.loads_state(P.dumps_state(original))
        for field in ('shift', 'mass', 'numerator'):
            before = original[field] if field == 'numerator' else [original[field]]
            after = restored[field] if field == 'numerator' else [restored[field]]
            self.assertEqual(len(before), len(after))
            for b, a in zip(before, after):
                self.assertLessEqual(ends(a)[0], ends(b)[0])
                self.assertGreaterEqual(ends(a)[1], ends(b)[1])

    def test_invalid_encodings_are_rejected_not_reinterpreted(self):
        good = encode_finite(F(1, 3))
        for bad in (None, 0.333, '0.333', True, {}, dict(good, extra=0),
                    dict(good, denominator_hex='0x0'), dict(good, denominator_hex='-0x3'),
                    dict(good, numerator_hex=True), dict(good, numerator_hex='0.333'),
                    {'kind': 'interval', 'lower': good, 'upper': encode_finite(0)}):
            with self.subTest(bad=bad), self.assertRaises(Invalid):
                decode_finite(bad)
        for bad in ('{}', 'null', '{"version":1,"version":1}',
                    '{"version":NaN}', P.dumps_state(P.empty(2)).replace('"version": 1', '"version": true')):
            with self.subTest(bad=bad), self.assertRaises(Invalid):
                P.loads_state(bad)
        record = json.loads(P.dumps_state(P.empty(2)))
        record['state']['numerator'].pop()
        with self.assertRaises(Invalid):
            P.loads_state(json.dumps(record))


if __name__ == '__main__':
    unittest.main()
