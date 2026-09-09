import torch
from kernel_analyzer.recurrence_error_transport import reconstruct


def test_transport_closes_and_separates_coefficient_error():
    ac = torch.tensor([.5, .75, .5], dtype=torch.float64)
    ar = torch.tensor([.25, .5, .5], dtype=torch.float64)
    bc = torch.tensor([1., 2., 1.], dtype=torch.float64)
    br = torch.tensor([.5, 1., 1.], dtype=torch.float64)
    rr = torch.tensor([.125, 0., -.125], dtype=torch.float64)
    hc, hr = torch.tensor(2., dtype=torch.float64), torch.tensor(1., dtype=torch.float64)
    previous, actual = [], []
    for i in range(3):
        previous.append(hr)
        hc = ac[i]*hc+bc[i]+rr[i]
        hr = ar[i]*hr+br[i]
        actual.append(hc-hr)
    result = reconstruct(ac, ar, torch.stack(previous), bc-br, rr, torch.tensor(1., dtype=torch.float64))
    assert torch.equal(result['reconstructed_difference'], torch.stack(actual))
    assert not result['population_bias_proved']


def test_zero_source_cannot_grow_with_contractive_decay():
    a = torch.full((8,), .5, dtype=torch.float64)
    z = torch.zeros_like(a)
    result = reconstruct(a,a,z,z,z,torch.tensor(1.,dtype=torch.float64))
    assert torch.equal(result['reconstructed_difference'], .5**torch.arange(1,9,dtype=torch.float64))
