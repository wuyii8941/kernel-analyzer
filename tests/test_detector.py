from __future__ import annotations

import math
import unittest

from forkcert.detector import (
    REGION_BUG,
    REGION_FRAGILE,
    REGION_STABLE,
    classify_region,
    clip_active,
    clip_boundary,
    detect_clipping_fork,
)


class DetectorTest(unittest.TestCase):
    def test_boundaries(self) -> None:
        self.assertAlmostEqual(clip_boundary(1, 0.2), math.log(1.2))
        self.assertAlmostEqual(clip_boundary(-1, 0.2), math.log(0.8))

    def test_clip_active_positive_advantage(self) -> None:
        old = -2.0
        self.assertFalse(clip_active(old + math.log(1.1), old, 1, 0.2))
        self.assertTrue(clip_active(old + math.log(1.3), old, 1, 0.2))

    def test_clip_active_negative_advantage(self) -> None:
        old = -2.0
        self.assertFalse(clip_active(old + math.log(0.9), old, -1, 0.2))
        self.assertTrue(clip_active(old + math.log(0.7), old, -1, 0.2))

    def test_actual_fork(self) -> None:
        boundary = clip_boundary(1, 0.2)
        old = -1.0
        cert = detect_clipping_fork(
            case_id="case",
            token_index=0,
            logp_ref=old + boundary - 1e-4,
            logp_alt=old + boundary + 1e-4,
            old_logp=old,
            advantage_sign_value=1,
            eps=0.2,
            delta_bound_legal=1e-3,
        )
        self.assertTrue(cert.fork_possible)
        self.assertTrue(cert.actual_fork)
        self.assertEqual(cert.region, REGION_FRAGILE)

    def test_regions(self) -> None:
        self.assertEqual(classify_region(margin=2e-3, delta=1e-4, bound=1e-3), REGION_STABLE)
        self.assertEqual(classify_region(margin=5e-4, delta=5e-4, bound=1e-3), REGION_FRAGILE)
        self.assertEqual(classify_region(margin=5e-4, delta=2e-3, bound=1e-3), REGION_BUG)


if __name__ == "__main__":
    unittest.main()

