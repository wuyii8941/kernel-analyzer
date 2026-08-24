import unittest

import numpy as np

from scripts.analyze_adamw_response_components import decompose_step


class AdamWResponseComponentsTest(unittest.TestCase):
    def test_components_reconstruct_total(self):
        gc = np.array([0.2, -0.4, 0.01], dtype=np.float64)
        gr = np.array([0.1, -0.3, -0.02], dtype=np.float64)
        m = np.array([0.03, -0.01, 0.0], dtype=np.float64)
        v = np.array([0.2, 0.4, 0.01], dtype=np.float64)
        total, numerator, denominator = decompose_step(
            gc, gr, m, v, 4, lr=1e-4, beta1=0.9, beta2=0.95, epsilon=1e-8,
        )
        np.testing.assert_allclose(total, numerator + denominator, rtol=1e-13, atol=1e-18)

    def test_equal_gradients_have_zero_contrast(self):
        gradient = np.array([0.2, -0.4], dtype=np.float64)
        total, numerator, denominator = decompose_step(
            gradient, gradient, np.zeros(2), np.zeros(2), 1,
            lr=1e-4, beta1=0.9, beta2=0.95, epsilon=1e-8,
        )
        np.testing.assert_array_equal(total, np.zeros(2))
        np.testing.assert_array_equal(numerator, np.zeros(2))
        np.testing.assert_array_equal(denominator, np.zeros(2))


if __name__ == "__main__":
    unittest.main()
