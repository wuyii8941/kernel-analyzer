# REDUCE implementation review — 2026-10-19

This review covers the finite floating `real` profile in
`contracts/REDUCE.yaml` and the current `adapters/REDUCE.py`. It does not
approve a concrete Triton call.

## Sources and selected task

The retained PyTorch 2.10 pages define sum, mean, product, extrema and their
tie behavior, cumulative sum, log-sum-exp, corrected variance, standard
deviation, the ordered variance/mean pair and the public vector norm. The retained installed
`native_functions.yaml` fixes the selected default functional call forms. In
particular, sum and mean may take a dimension list, product takes one dimension,
cumulative sum takes one dimension, log-sum-exp requires dimensions, and the
available dtype arguments are keyword-only. For variance, this review selects
the `var.correction` form, whose public formula divides the sum of squared
deviations by `N - correction`; `std.correction` takes the nonnegative square
root of the same value; `var_mean.correction` returns the complete ordered
pair `(variance, mean)`. An omitted correction means one. For norm, this
review does not revive deprecated `torch.norm`; it selects public
`torch.linalg.vector_norm` only for real f32/f64 order-two calls with dtype
omitted.

The reviewed profile admits finite f16, bf16, f32 and f64 tensor values. It
uses exact real arithmetic after any explicit floating input conversion. The
VJP operations differentiate those forward tasks and require a complete
same-dtype upstream tensor. Device accumulation order, intermediate rounding,
and final rounding remain in the program comparison.

The variance, standard-deviation and var-mean admission is deliberately small:
a finite floating tensor; all dimensions, one integer dimension, or a nonempty
list of distinct integer dimensions; literal correction zero or one; and
positive `N - correction`.
Each VJP requires a complete same-dtype upstream tensor. Standard-deviation
VJP additionally requires positive std, except for a one-element correction-zero
slice whose forward function is identically zero and therefore has zero derivative.
For a larger constant slice the real standard deviation is not differentiable,
so the query is undefined instead of adopting a framework-specific zero gradient.
The selected vector-norm VJP similarly requires positive norm in every reduced
slice; the zero vector is a nondifferentiable point and is not assigned the
framework's zero-gradient convention. Other vector-norm orders, f16/bf16 and
complex inputs, explicit dtype conversion, deprecated `torch.norm`, the
empty-list spelling for dimensions, other correction values, and legacy,
named-dimension and out-buffer overloads remain outside the admission. The
full earlier draft remains under `contracts/draft` and is not loaded by the
runtime registry.

## Formula and implementation correspondence

- Sum and product use their zero and one identities. Mean rejects an actual
  empty reduced slice, while an output with zero rows remains a valid empty
  tensor.
- Amax and amin reduce one or more dimensions. Argmax, argmin, and the selected
  max/min overload reduce one dimension and retain the documented first index.
  Empty reduced axes are rejected before coordinate evaluation.
- Cumulative sum converts first when dtype is supplied and computes every
  prefix along one checked dimension. Log-sum-exp uses max subtraction for a
  finite slice and returns negative infinity for an empty reduced slice.
- Variance first computes the exact mean over the selected slice, then divides
  the exact sum of squared deviations by `N - correction`. A nonpositive
  denominator is outside the exact-real task. With positive denominator its
  VJP is `upstream * 2 * (x - mean) / (N - correction)`, including an exact
  zero gradient for a constant slice.
- Standard deviation is the rigorous nonnegative square root enclosure of that
  variance. At positive std its VJP is
  `upstream * (x - mean) / ((N - correction) * std)`. A larger constant slice
  is rejected before division; the one-element correction-zero case returns
  the derivative of its constant-zero forward function.
- Var-mean returns both variance and mean in the documented order. Its input
  VJP receives two complete upstream tensors and adds the corresponding
  variance and mean VJPs; neither output or upstream is optional.
- The selected public vector 2-norm is the rigorous nonnegative square root of
  the sum of squares. At positive norm its input VJP is the broadcast upstream
  times `x/norm`; a zero-norm slice is rejected before division.
- Sum, mean, product and cumulative-sum VJPs follow directly from their forward
  formulas. Product multiplies all other elements, so zero inputs do not cause
  division by zero. Amax/amin split ties as documented; max/min send the
  gradient to the returned first index. Log-sum-exp uses the derivative of its
  sourced formula.
- Complete-output comparison checks dtype, explicit shape, every coordinate,
  both tensors returned by max/min, and empty outputs. A candidate inside a
  nonpoint transcendental enclosure is not reported as equal.

The review corrected three admission errors in the broad draft adapter: it had
accepted dtype positionally, allowed a dimension list for product, and allowed
log-sum-exp without a nonempty dimension selection. It also had no finite-type
boundary, no same-dtype VJP rule, and no complete-output comparison.

## Independent checks

`tests/test_reduction_independent.py` builds sum, mean, product, variance,
standard deviation, vector 2-norm and cumulative sum directly with Python `Fraction` and
100-digit square-root calculations. It reconstructs polynomial VJPs from the
exact scalar upstream objective by symmetric differences, without calling the
reference reduction or broadcast-back code, and separately calculates the
standard-deviation VJP from the sourced forward formula. Var-mean is checked as
an ordered pair and differentiated from an independent scalar objective using
both upstream tensors. Vector norm uses independent 100-digit square roots and
the directly differentiated sourced formula. It separately checks the retained
extrema tie rules and exact equal-logit log-sum-exp weights.

The profile-domain tests exercise floating types, nonfinite refusal, malformed
dimensions, keyword-only dtype, complete upstream shape and dtype, and explicit
boolean keepdim. Complete-output tests reject a changed coordinate, dtype,
shape, missing extrema index, or incomplete empty tensor. Existing empty-domain,
lazy-reduction and cumulative-sum tests remain applicable.

As a same-source implementation cross-check, 434 finite float64 forward
shape/value cases and 51 autograd cases agreed with the installed PyTorch 2.10
behavior within the earlier declared domain. Another 24 paired variance and
24 paired standard-deviation forward/VJP cases cover two shapes, three
dimension selections, correction zero and one, and both keepdim choices. A
further 24 paired var-mean cases check both ordered outputs and the combined
VJP over the same cross-product. Twelve paired vector-norm forward/VJP cases
cover two shapes, three dimension selections and both keepdim choices. Direct
boundary probes also confirmed the
installed behavior for scalar and one-element inputs, empty reduced axes,
fractional and negative corrections, empty dimension lists, and the selected
floating dtypes. These same-source checks helped find boundary mistakes; they
are not the independent basis for the formula. The contract checker establishes
all 55 examples and rejects all 20 listed wrong implementations, including
ignoring correction, omitting the variance factor two, returning variance in
place of std, omitting division by std in its VJP, reversing the var-mean pair,
dropping either var-mean upstream, omitting the vector-norm square root, and
omitting division by the norm in its VJP.

## Limits

This review does not establish standard-deviation VJPs on larger zero-variance
slices, vector-norm VJPs at zero, other vector-norm orders or dtype choices,
deprecated `torch.norm`, unselected variance, standard-deviation or var-mean
corrections and overloads, integer or boolean reductions, fixed-width overflow,
complex or float8 values, nonfinite arithmetic, named dimensions, out buffers,
or native device rounding. A real kernel still needs a separate correspondence
review for every input, output, dimension and state effect.
