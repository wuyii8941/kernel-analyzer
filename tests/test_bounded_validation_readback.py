import numpy as np
from scripts.validate_bounded_population_equivalence import _population_artifact, _population_protocol
from kernel_analyzer.training_numerical_analysis import analyze_bounded_population_artifact


def test_real_sgd_readback_enters_full_analysis():
    raw = _population_artifact(np.zeros((2048, 1)), np.ones((2048, 1)))
    result = analyze_bounded_population_artifact(raw, _population_protocol(x_max=0, b_max=1))
    assert result['equivalence_decision'] == 'EQUIVALENT'
    assert raw['parameter_write_protocol']['implementation'] == 'torch.optim.SGD'


def test_legacy_unverified_readback_still_rejected():
    raw = _population_artifact(np.zeros((16, 1)), np.ones((16, 1)))
    raw['parameter_write_protocol'] = {'version':'adamw-readback-v2','synthetic':True}
    result = analyze_bounded_population_artifact(raw, _population_protocol(x_max=0, b_max=1))
    assert result['equivalence_decision'] == 'NOT_ASSESSED'


def test_unobserved_rare_boundary_does_not_pass():
    raw = _population_artifact(np.zeros((64, 1)), np.ones((64, 1)))
    result = analyze_bounded_population_artifact(raw, _population_protocol(x_max=163.84, b_max=1))
    assert result['equivalence_decision'] == 'INCONCLUSIVE'
