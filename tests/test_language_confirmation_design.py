import math
from scripts.run_liger_language_confirmation import N, tail


def test_exact_null_tail_and_power_design():
    assert N == 16
    assert tail(0) == 1.0
    assert tail(N) == 2 ** (-N)
    assert tail(11) > .05
    assert tail(12) <= .05
    assert .79 < tail(12, .8) < .81
    assert math.isclose(tail(12), sum(math.comb(16, k) for k in range(12, 17)) / 2**16)
