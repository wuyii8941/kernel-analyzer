from scripts.run_adamw8bit_population_direction import independent_draws


def test_direction_confirmation_draws_are_deterministic_and_independent_rows():
    first = independent_draws(8192)
    second = independent_draws(8192)
    assert first == second
    assert len(first) == 32
    assert all(len(row) == 8 for row in first)
    assert len({tuple(row) for row in first}) == 32
