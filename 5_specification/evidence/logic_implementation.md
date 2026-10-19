# LOGIC physical comparison and boolean implementation review — 2026-10-18

## Reviewed task

The `real` profile covers functional comparisons, logical operations, boolean-only bitwise operations and
all-elements or one-integer-dimension `any`/`all`. Inputs are physical finite f16, bf16, f32, f64 or boolean tensor
records. Floating values and comparison scalars must be exactly representable in the declared tensor dtype;
Tensor/Tensor calls use one dtype. Every result is a complete boolean tensor.

The separate `ieee_values` profile covers only `isnan` on physical f16, bf16, f32 or f64 records. It accepts finite
values, both infinities and NaN, and observes only whether each coordinate is NaN.

The retained PyTorch 2.10 public pages define the comparison, truth, boolean bitwise, reduction and isnan values. The
retained `native_functions.yaml` distinguishes the selected functional schemas from scalar-first, integer bitwise,
dimension-list, named-dimension, out and in-place overloads. The review checked these retained files directly.

## Errors removed from the former active draft

The former profile accepted mathematical `1/3` under an f32 label, combined finite floating, infinities, booleans and
fixed-width integer bitwise tasks without validating physical representation, and had no complete-output checker.
The exact former contract is retained as `contracts/draft/LOGIC_extended_20261018.yaml` and is not loaded at run time.

The reviewed adapter validates the selected call schema before evaluation, requires exact physical floating values
and literal boolean records, rejects mixed tensor dtypes, and keeps `isnan` special-value handling in its own profile.
Complete-output comparison requires boolean dtype, shape and every coordinate; integer values such as 0 and 1 cannot
stand in for booleans.

## Independent checks

The contract has 31 examples and nine required wrong implementations. All examples are consistent and every wrong
implementation is rejected. Independent NumPy broadcasting and truth reductions reproduce the comparison and
reduction values. Explicit truth tables check logical and boolean bitwise operations. PyTorch 2.10 cross-checks
f16, bf16, f32 and f64 comparison, logical and truth-reduction results, plus boolean bitwise results. Separate tests
check NaN versus infinity, empty identities, scalar outputs, empty shapes and complete boolean output records.

## Excluded scope

Integer and float8 comparisons, integer bitwise rules, mixed-dtype promotion, complex values, NaN comparisons,
dimension-list and named-dimension reductions, out/in-place forms, storage, device placement, gradients, NaN payloads
and signalling state are outside these profiles.
