from scripts.run_adamw8bit_error_compensation_probe import draws


def test_compensation_probe_draws_are_frozen_and_distinct():
    assert draws(8192) == draws(8192)
    assert len(draws(8192)) == 16
    assert len({tuple(row) for row in draws(8192)}) == 16
