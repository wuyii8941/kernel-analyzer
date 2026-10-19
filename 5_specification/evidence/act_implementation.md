# ACT implementation review — 2026-10-18

This review covers the two ACT profiles in `contracts/ACT.yaml` and the current
`adapters/ACT.py`.  It does not approve a particular Triton call binding.

## Sources and task scope

- The retained PyTorch 2.10 ReLU, GELU, SiLU and Softplus pages state the
  forward formulas.  The Softplus pages explicitly make the strict
  `beta*x > threshold` linear branch part of the API behavior.
- The retained PyTorch 2.10 decomposition source defines threshold, GELU,
  SiLU, Softplus, tanh and sigmoid backward operations.  The saved file is the
  same source as the installed PyTorch 2.10 file used for this review.
- The retained GLU paper defines GEGLU and SwiGLU as componentwise products.
  The retained Liger 0.7 sources define the concrete tanh-GELU and SiLU forms
  and their two gradients; both files agree with the installed 0.7 package.

The admitted domain is deliberately narrower than every possible direct ATen
call: finite floating tensors only, with one matching shape for all operands of
a backward or GLU call.  Integer, boolean, complex, infinite and broadcasted
variants require another reviewed profile.

## Corrections made during review

1. `softplus_formula` is now described as a separately selected mathematical
   task.  It is not presented as another reading of the PyTorch API, whose
   threshold branch is explicit.
2. At `beta*x == threshold`, PyTorch uses the formula branch in both forward
   and backward.  The piecewise forward has a small jump at this finite
   boundary, so the returned backward value there is a defined PyTorch
   backward convention rather than an ordinary derivative.
3. `tanh_backward` and `sigmoid_backward` apply their formulas to a value
   interpreted as the saved forward output.  They are VJPs only when that
   correspondence actually holds.  The separately named `vjp.aten.tanh` and
   `vjp.aten.sigmoid` tasks receive the original input and differentiate the
   sourced public forward functions, so an end-to-end autograd region does not
   have to mislabel `x` as a saved output.
4. Numeric infinities from runtime tensor observations can no longer bypass
   the finite-domain check.  NaN is invalid, and non-floating tensor records
   are refused by these profiles.
5. Complete output comparison checks dtype, shape and both GLU gradients.

## Independent implementation checks

`tests/test_act_independent.py` differentiates the documented GELU (both
forms), SiLU, formula-Softplus, public tanh and sigmoid, GEGLU and SwiGLU forward expressions with dual
numbers.  It evaluates those independent expressions with high-precision Arb
ball arithmetic and checks that the reference's strict enclosures contain the
independent results.  It does not call the adapter's closed-form backward
functions to derive the expected gradients.

`tests/test_act_domains.py` checks numeric and textual infinities, NaN,
non-floating records, mismatched backward shapes, the exact Softplus threshold
branch, the separation of the two Softplus profiles, and complete structured
outputs.  The contract checker establishes all 37 examples and rejects all 20
listed wrong implementations.

## Limits

- This is exact finite-real task semantics.  Device casts, transcendental
  instruction approximations and overflow remain part of the measured program.
- A binding to Liger must match the actual installed Liger version and select
  tanh GEGLU explicitly.  This review alone does not resolve any older-package
  binding.
- Direct ATen behavior outside the admitted dtype and shape domain is not
  classified by these profiles.
