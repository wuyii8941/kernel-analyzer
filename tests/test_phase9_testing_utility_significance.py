from scripts.phase9_testing_utility_significance import exact_mcnemar_pvalue, exact_sign_flip_pvalue


def test_exact_mcnemar_is_symmetric_and_one_for_tie():
    assert exact_mcnemar_pvalue(4, 4) == 1.0
    assert exact_mcnemar_pvalue(2, 8) == exact_mcnemar_pvalue(8, 2)


def test_exact_sign_flip_is_one_for_small_total_against_large_components():
    assert exact_sign_flip_pvalue([3, 2, -5]) == 1.0
