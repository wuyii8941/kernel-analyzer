# PyTorch mean-L1 implementation review — 2026-10-19

## Reviewed task

The original mean profile of `L1` selects unweighted `torch.nn.functional.l1_loss(input, target, reduction="mean")` on
finite, nonempty, same-shape physical float32 tensors. The forward task returns the scalar loss. The VJP task receives
an explicit physical float32 scalar upstream and returns both complete gradients in the order
`[grad_input, grad_target]`. All inputs are read-only.

Broadcasting, weighting, other reductions, deprecated overrides, complex tensors and special values are outside this
profile. They are not declared wrong.

## Source and formula review

The installed official PyTorch 2.10 `L1Loss` documentation defines every unreduced element as
`|input_i-target_i|` and says that `mean` divides by the number of output elements. The installed public wrapper fixes
the argument boundary and distinguishes the unweighted native path from the separately written weighted path. The
installed PyTorch refs implementation independently computes `abs(input-target)` before applying the reduction.

For `N` elements, the selected task is therefore

`loss = sum_i |input_i-target_i| / N`.

Away from equality, differentiation for scalar upstream `g` gives

`grad_input_i = g * sign(input_i-target_i) / N`,

`grad_target_i = -grad_input_i`.

At equality, absolute value is locally convex. PyTorch's official autograd note selects the minimum-norm subgradient
at locally convex nondifferentiable points, hence zero; its public `sign` definition also gives `sign(0)=0`. This fixes
the otherwise ambiguous boundary without inferring it from the candidate kernel.

The required localhost:27891 proxy had refused the versioned documentation page earlier in this review session. No
direct network route was used. The new documentation and source snapshots come from the installed PyTorch
2.10.0+cu128 distribution; the already retained PyTorch 2.10 autograd, abs and sign pages supply the boundary rule.

## Independent implementation checks

`tests/test_l1_independent.py` calculates the loss and both gradients with independent `Fraction` loops over several
one-, two- and three-dimensional cases, including equal coordinates. It does not call the L1 functions in
`kernel_spec.aten`. The adapter returns the same exact results and leaves its inputs unchanged.

The same test compares 12 calls across four shapes and three upstream values with public PyTorch 2.10 L1 plus
autograd. Every call forces one equal input/target coordinate; both public gradients are exactly zero there. The loss
and all other gradient coordinates agree within ordinary float32 reduction/division error. Complete-output checks
reject a nonscalar loss, either omitted gradient, wrong dtype, wrong shape and wrong coordinate.

The contract checker accepts all 14 required examples and rejects six wrong formulas: leading-dimension division,
signed rather than absolute difference, positive gradient at equality, wrong target-gradient sign, ignored upstream
and omitted target gradient. Unselected calls and nonphysical values are refused before evaluation.

## Remaining boundary

A Triton call may use this contract for bias measurement only after its complete input, target, upstream, scalar
loss, both gradients and unchanged input state are bound and independently retained under `validated/`.

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
