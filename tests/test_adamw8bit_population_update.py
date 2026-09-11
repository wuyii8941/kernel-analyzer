from scripts.run_adamw8bit_population_update import independent_draws


def test_population_draws_are_deterministic_independent_draw_records():
    first = independent_draws(8192)
    second = independent_draws(8192)
    assert first == second
    assert len(first) == 32
    assert all(len(unit) == 8 for unit in first)
    assert all(0 <= value < 8192 for unit in first for value in unit)
    # With-replacement sampling must not be silently changed to partitioning.
    assert len({tuple(unit) for unit in first}) == len(first)
