import unittest
from fractions import Fraction as F
from relations import *
class T(unittest.TestCase):
    def test_softmax_sum(self):
        self.assertTrue(softmax_row_sum_is_one([F(1), F(-1), F(4)])); self.assertTrue(softmax_row_sum_mutation([F(1), F(-1), F(4)]))
    def test_moe_weights(self):
        self.assertTrue(moe_renormalised_weights_sum_one([F(1), F(1, 2), F(-1)], 2)); self.assertTrue(moe_raw_weights_mutation([F(1), F(1, 2), F(-1)], 2))
    def test_ignore_index(self): self.assertTrue(ignore_index_irrelevant([[2, 0, 1], [1, 1, 0], [0, 3, 1]], [0, -100, 2]))
    def test_include_self(self):
        args = ([F(1), F(5), F(2)], [0, 0, 2], [F(3), F(4), F(-1)])
        self.assertTrue(include_self_false_irrelevant(*args)); self.assertTrue(include_self_true_mutation(*args))
    def test_token_weighted(self):
        self.assertTrue(token_weighted([[1, 2, 3], [4]])); self.assertTrue(mean_of_means_mutation([[1, 2, 3], [4]])); self.assertFalse(mean_of_means_mutation([[1, 2], [3, 4]]))
    def test_online_logsumexp(self): self.assertTrue(online_logsumexp([F(1), F(3), F(-2), F(5, 2), F(0)], 2))
    def test_ce_sum_split(self): self.assertTrue(ce_sum_split([[2, 0, 1], [0, 1, 3], [1, 1, 1], [3, 0, 0]], [0, 2, 1, 0]))
    def test_avgpool_adjoint(self): self.assertTrue(avgpool_adjoint([[1, 2, 3, 4]], [[F(1), F(-2)]], kernel=2, stride=2))
    def test_adam_decay(self): self.assertTrue(adam_zero_effective_grad_decay([F(1), F(-2)], [F(1, 10), F(3, 10)]))
if __name__ == '__main__': unittest.main()
