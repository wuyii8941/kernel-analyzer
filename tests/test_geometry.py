import torch

from kernel_analyzer.geometry import GeometryAnalyzer, _basis_from_rows, normalized_gram


def _row(value):
    return {"local": {"value": torch.as_tensor(value, dtype=torch.float32)}}


def test_basis_reconstruction_and_gram():
    rows = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    gram = normalized_gram(rows)
    assert torch.allclose(gram, torch.ones(3, 3), atol=1e-6)
    basis, eig = _basis_from_rows(rows, 1)
    assert basis.shape == (1, 2)
    assert abs(float(basis[0, 0])) > 0.99
    assert eig[0] > 2.9


def test_rank_report_distinguishes_sign_cancellation():
    calibration = [_row([1.0, 0.0]), _row([-1.0, 0.0])] * 8
    evaluation = [_row([1.0, 0.0]), _row([-1.0, 0.0])] * 8
    report = GeometryAnalyzer(ranks=(1,)).rank_report(calibration, evaluation)
    rank = report["ranks"]["1"]
    assert rank["odd_even_subspace_overlap"] > 0.99
    assert rank["evaluation"]["persistence"] < 0.1


def test_rank_two_recovers_two_dimensional_subspace():
    calibration = ([_row([1.0, 0.0]), _row([1.0, 0.0]),
                    _row([0.0, 1.0]), _row([0.0, 1.0])] * 4)
    evaluation = ([_row([1.0, 0.0]), _row([0.0, 1.0])] * 8)
    report = GeometryAnalyzer(ranks=(1, 2)).rank_report(calibration, evaluation)
    assert report["ranks"]["2"]["odd_even_subspace_overlap"] > 0.99
    assert report["ranks"]["2"]["evaluation"]["energy_capture"] > 0.99


def test_window_shuffle_is_reported():
    rows = [_row([1.0, 0.0])] * 16
    report = GeometryAnalyzer(windows=(2,)).window_report(rows)
    assert report["original"][0]["next_step_projection_mean"] > 0.99
    assert report["shuffled"][0]["next_step_projection_mean"] > 0.99
