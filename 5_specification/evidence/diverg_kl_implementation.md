# PyTorch KL batchmean implementation review — 2026-10-19

## Reviewed task

The active `DIVERG` contract selects one public PyTorch 2.10 call: rank-2 physical float32 `input` and strictly
positive physical float32 probability-space `target`, `reduction="batchmean"`, `log_target=false`, and a physical
float32 scalar upstream. It promises the scalar loss and every coordinate of the VJP with respect to `input`.
`input` and `target` are read-only. Other reductions, log-space targets, target gradients, other ranks and dtypes,
and nonpositive targets remain outside this profile.

The previous contract mixed PyTorch KL, Liger KL, Jensen-Shannon divergence and total variation under one role. It is
retained as `contracts/draft/DIVERG_liger_jsd_tvd_20261019.yaml`, together with its adapter, and is not a runtime
fallback.

## Source and formula review

The retained PyTorch 2.10 `torch.nn.functional` source is byte-for-byte equal to the installed 2.10.0+cu128 file. Its
public `kl_div` documentation fixes the target interpretation, lists the reductions, states that `batchmean` divides
the sum by the batch size, and forwards `log_target` to `torch.kl_div`. The existing retained PyTorch KL documentation
gives the probability-target point formula. Together they establish

`loss = sum(target * (log(target) - input)) / batch_size`.

Differentiating that selected expression with `target` held fixed gives the complete input VJP

`grad_input = -target * upstream / batch_size`.

This derivation is specific to the selected input VJP. It does not claim that another KL orientation, reduction or
target representation is wrong.

## Independent implementation checks

`tests/test_diverg_kl_independent.py` recomputes four cases with independent 768-bit `python-flint` logarithms and
exact rational gradients. Every independent loss interval is contained in the adapter's rigorous enclosure, every
gradient coordinate is equal, and the reference leaves its input records unchanged.

The same test compares 12 calls covering four matrix shapes and three upstream values with public PyTorch 2.10
`torch.nn.functional.kl_div` plus autograd. Losses and complete input gradients agree within the ordinary float32
execution error, while both public inputs remain unchanged. These comparisons support the translation of the public
API; they do not replace the exact formula checks.

The output checker separately rejects a missing gradient, wrong dtype, wrong shape and wrong coordinate. The
contract checker accepts all 12 required examples and rejects all six required wrong formulas, including mean
division, omission of the target logarithm, gradient sign and divisor errors, target renormalisation, and treating a
probability target as a log target. Unsupported public choices and nonphysical input records are refused before
evaluation.

## Remaining boundary

This review validates the contract and adapter only. A Triton call may use it for bias measurement only after its
complete inputs, loss, input VJP and unchanged input state are bound and separately retained under `validated/`.
