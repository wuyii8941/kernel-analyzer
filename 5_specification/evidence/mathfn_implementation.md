# MATHFN implementation review — 2026-10-18

## Admitted task

The active contract covers the functional one-tensor forms of `exp`, `log`, `log1p`, `expm1`, `sqrt`, `rsqrt`,
`sin`, `cos`, `tanh`, `sigmoid` and `erf`. Inputs are dense finite f16/bf16/f32/f64 values with `|x| <= 100` and
must satisfy the function's ordinary real domain. The reference applies the sourced mathematical formula to the exact
stored input value. Device approximation and output rounding remain in the separate implementation comparison.

The former draft also accepted integer tensors and infinities. It described integer promotion as exact and selected
several special-value conventions without enough source support. That draft is retained as
`contracts/draft/MATHFN_extended_20261018.yaml`; it is not a runtime fallback. Zero is admitted for `sqrt`, but `log`,
`log1p` and `rsqrt` use their open real domains, so their singular endpoints are undefined in this profile.

## Source and schema check

The retained PyTorch 2.10 pages give each formula. `T_SPECIAL` supplies the full `expm1`, sigmoid/expit and erf
definitions where the public pages are aliases. The retained installed `native_functions.yaml` declares a separate
functional `Tensor -> Tensor` schema and distinct in-place/out schemas for all eleven operations. The active profile
therefore rejects `out=`, extra arguments, sparse/layout effects and types outside its stated floating range.

## Executable reference check

`adapters/MATHFN.py` validates dtype, finiteness, point-valued input and the magnitude bound before evaluation. It
adds explicit open-domain checks for `log`, `log1p` and `rsqrt`, preserves the input tensor's dtype and shape, and uses
the common complete-output checker so empty dimensions cannot be reconstructed from nested values alone.

The following checks are independent of the contract examples:

- `tests/test_mathfn_independent.py` evaluates ten function families with 768-bit Arb operations and checks the
  reference enclosure against those results. Exact rational results such as `rsqrt(9)=1/3` must lie inside Arb's
  interval; non-exact reference intervals must contain Arb's higher-precision interval.
- `tests/test_reference_guarantees.py` checks erf against a rational integral series with an explicit remainder,
  rather than checking the Arb evaluator against itself.
- `tests/test_mathfn_profile_domains.py` exercises all four admitted dtypes, empty shapes, every function-specific
  domain boundary, rejected integer/boolean/float8/complex inputs, infinities, NaN, excessive magnitude and call
  schema errors.
- `tests/test_mathfn_outputs.py` rejects missing output metadata, wrong dtype, wrong shape, wrong coordinates and
  swapped empty dimensions.
- A direct local PyTorch 2.10 check covered all eleven operations in f16, bf16, f32 and f64 (44 calls): every admitted
  call preserved dtype and shape, returned finite values for the selected interior inputs, and left the input
  unchanged. This supports schema correspondence; it is not an independent proof of the formulas.
- The contract checker establishes 22/22 examples and rejects 8/8 independent wrong formulas.

These checks support the declared finite task. They do not certify special values, larger magnitudes, backward
formulas, storage aliasing, device accuracy or any concrete Triton call; those require a matching call binding.
