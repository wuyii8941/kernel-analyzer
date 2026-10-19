# MATMUL implementation review — 2026-10-18

This review covers the `real` profile in `contracts/MATMUL.yaml`, the current
`adapters/MATMUL.py`, and the matrix-product functions they call in
`kernel_spec/aten.py`. It does not approve a concrete Triton call binding.

## Sources and selected boundary

The retained PyTorch 2.10 pages define the shapes and formulas for `mm`, `bmm`,
`matmul`, `addmm`, `baddbmm`, and `linear`. The full installed PyTorch 2.10
`native_functions.yaml` is retained as `sources/TS_NATIVE_FUNCTIONS.yaml.gz`.
It declares separate default, dtype, and out overloads: the default `addmm` and
`baddbmm` schemas have three tensor arguments and keyword-only `beta` and
`alpha`. The retained Inductor post-grad source defines `mm_plus_mm` as the sum
of two matrix products.

The admitted profile is deliberately limited to the named default functional
operations on finite f16, bf16, f32, or f64 tensor values. Public `out_dtype`,
out buffers, sparse tensors, integers, booleans, float8, complex values, and
infinite product operands require other profiles. This avoids replacing
fixed-width integer behavior with an unsourced unbounded-integer task.

TF32, input precision, product rounding, and accumulation order remain in the
program interpretation and the K-G comparison. They do not change the exact
finite-real task value selected here.

## Formula and implementation correspondence

- `mm` and `bmm` check ranks, batch counts, inner dimensions, and matching
  dtypes before iterating. Empty inner dimensions produce the correctly shaped
  zero result; empty batches do not bypass metadata checks.
- `matmul` implements the documented vector promotion/removal and batch
  broadcasting. Its VJP differentiates the product and sums broadcast
  dimensions back to each original operand shape.
- `addmm` and `baddbmm` first fix the product output shape and require the
  addend to broadcast to that shape without enlarging it. `beta=0` skips only
  the addend values, as the documentation states; its dtype and shape are still
  checked. No similar exception is inferred for `alpha=0`.
- `linear` computes `x @ weight.T + bias`. Matrix and vector weights, scalar
  bias, documented output-feature bias, and absent bias have distinct output
  and VJP shapes.
- Complete-output comparison checks every coordinate, dtype, explicit shape,
  gradient position, and the explicit `None` for an absent bias gradient.

The adapter rejects nonfloating tensor records before evaluation. Numeric and
textual infinities in used operands are unsupported; NaN is invalid. The only
nonfinite exception is addend storage proven unused by exact `beta=0`.
Unsupported overloads and unknown keywords are refused rather than ignored.

## Independent checks

`tests/test_matmul_exact_derivatives.py` builds each scalar objective directly
with NumPy object arrays and `Fraction`. Since the objective is affine in each
operand coordinate, `(F(x+1)-F(x-1))/2` is its exact derivative. The test does
not use the reference VJP, transpose-gradient, or broadcast-reduction code. It
checks mm, bmm, broadcast matmul, linear, addmm, vector cases, empty dimensions,
all operand gradients, scaling, and absent bias.

The domain, dtype, linear, scaling, lazy-evaluation, ignored-addend, and output
tests additionally compare all coordinates of small exactly representable
cases with PyTorch/autograd. `tests/test_matmul_profile_domains.py` checks the
admission boundary, nonfinite handling, keyword-only scales, and refusal of the
separate dtype overload. The contract checker finds all 25 examples consistent and
rejects all nine listed wrong implementations.

### Default argument names — 2026-10-19

Local source/implementation review compared all six default signatures in the
retained `TS_NATIVE_FUNCTIONS.yaml.gz` with the internal functions in `aten.py`.
The explicit adapter mappings preserve operand order: self/mat2 for mm and bmm,
self/other for matmul, self/mat1/mat2 for addmm, self/batch1/batch2 for baddbmm,
and input/weight/bias for linear. Normalization happens before signature binding
and decoding, so named self reaches the same beta=0 ignored-addend check as a
positional addend. Duplicate arguments still fail; dtype/out overloads and the
synthetic VJP interfaces are unchanged. Output checking uses the same mapping.
This is a local review of the interface correction, not external certification;
the retained adapter and contract were updated only after this comparison.

## Limits

This review establishes the declared ideal task and its executable reference.
It does not certify device rounding, integer overflow, sparse behavior,
out-buffer state, dtype overloads, undocumented bias shapes, or a real Triton
kernel's input/output/state correspondence. Those cases must be refused or
bound to a separately reviewed profile.
