import random

from scripts import run_adamw8bit_iid_training_confirmation as confirmation


def test_start_draws_are_direct_iid_with_replacement_draws():
    expected = []
    generator = random.Random(confirmation.DESIGN_SEED)
    width = confirmation.START_POPULATION[1] - confirmation.START_POPULATION[0]
    for _ in range(confirmation.STREAMS):
        expected.append(confirmation.START_POPULATION[0] + generator.randrange(width))
    assert confirmation.selected_starts() == expected
    assert all(confirmation.START_POPULATION[0] <= value < confirmation.START_POPULATION[1]
               for value in expected)


def test_new_start_population_is_disjoint_from_development_population():
    assert confirmation.START_POPULATION[0] >= 500000
