# PyTorch mean-MSE implementation review — 2026-10-19

## Reviewed task

The original mean profile of `MSE` selects the public PyTorch 2.10 call: unweighted
`torch.nn.functional.mse_loss(input, target, reduction="mean")` on finite, nonempty, same-shape physical float32
tensors. Its forward task promises the scalar loss. Its VJP task receives an explicit physical float32 scalar upstream
and promises both complete gradients in the order `[grad_input, grad_target]`. The input, target and upstream are
read-only.

The public API also permits `sum` and `none` reductions, an optional weight, and broadcasting after a warning. Those
are distinct legal calls outside this profile. The contract does not report them as wrong.

## Source and formula review

The installed official PyTorch 2.10 `MSELoss` documentation defines every unreduced element as
`(input_i - target_i)^2` and says that `mean` divides their sum by the total number of elements. The retained installed
`torch.nn.functional.mse_loss` wrapper fixes the selected public argument boundary and shows that the unweighted path
calls the native MSE operation. The packaged native schema separately confirms the lower-level forward and input
backward operations.

For `N` elements, these sources establish

`loss = sum_i (input_i - target_i)^2 / N`.

Differentiating this selected real formula for scalar upstream `g` gives

`grad_input_i = g * 2 * (input_i - target_i) / N`,

`grad_target_i = -grad_input_i`.

The complete public-call VJP includes both gradients even though the lower-level native backward schema returns only
the gradient for its `self` argument.

The required localhost:27891 proxy refused the versioned documentation page during this review. No direct network
route was used. The retained documentation snapshot, functional wrapper, native schema and license all come from the
installed PyTorch 2.10.0+cu128 distribution.

## Independent implementation checks

`tests/test_mse_independent.py` calculates the loss and both gradients with independent `Fraction` loops over several
one-, two- and three-dimensional cases. It does not call the MSE functions in `kernel_spec.aten`. The adapter returns
the same exact rational results and does not alter its input records.

The same test compares 12 calls covering four shapes and three upstream values with public PyTorch 2.10
`torch.nn.functional.mse_loss` plus autograd. The scalar loss and every coordinate of both gradients agree within the
ordinary float32 execution error, and both public inputs remain unchanged. These execution comparisons support the
public API translation; the independent exact calculations establish the formula used by the reference.

The complete-output checker separately rejects a nonscalar loss, either omitted gradient, a wrong dtype, a wrong
shape and a wrong coordinate. The contract checker accepts all 13 required examples and rejects all six required
wrong formulas: division by the leading dimension, missing mean division, missing factor two, wrong target-gradient
sign, ignored upstream and omitted target gradient. Unselected parameters and nonphysical inputs are refused before
evaluation.

## Remaining boundary

This review validates the contract and adapter. A Triton call may use it for bias measurement only after the complete
input, target, upstream, scalar loss, both gradients and unchanged input state are bound and retained under
`validated/`.

## 2026-10-19: complete reduction choices required by real public calls

The attorch public modules and tests promise none/sum/mean, while the original profile admitted only mean.
A second explicitly named profile, `torch210_unweighted_f32_same_shape`, now covers the three reductions with the
same nonempty, finite physical float32 and unweighted same-shape boundary. The original mean profile and existing
bindings retain their restrictions. Both profiles use the same exact arithmetic; the selected profile controls admission.

The retained installed module formula specifies every element, sum without division, and mean divided by all elements.
`TORCH210_LOSS_REDUCTIONS` points to the already retained PyTorch 2.10 refs file: `_apply_loss_reduction` directly
implements those three choices. The installed L1 prose contains a sentence saying that sum divides by N; its displayed
formula, subsequent sum sentence and executable reference all distinguish sum from mean. The formula and explicit
helper resolve that wording discrepancy; no candidate output was used to choose the task.

For none, the upstream must have exactly the full input/output shape; each element derivative is multiplied by its
own upstream. For sum, a scalar upstream scales all derivatives without division. Only mean divides by numel.
The target gradient remains the negative input gradient. Scalar inputs are valid for all three choices. Weight,
broadcasting, deprecated overrides, other dtypes and empty/special-value tasks remain outside this reviewed range.

`tests/test_loss_reductions.py` independently computes rational element losses and full derivatives, checks PyTorch
public autograd with nonuniform upstream, rejects scalar substitution for a none result and omission of target gradient,
and verifies original mean-profile refusal and value equivalence. The existing independent mean tests also pass.
The checker additionally rejects collapsing none to a scalar and dividing sum by numel. These are semantic/shape errors,
not numerical tolerance comparisons. Source, formula and implementation review here is by Codex, not a third party
or a claim of full-domain formal verification. No production binding is added by admitting this reference profile.
