# PyTorch Numerical Contract Records v0.1 — 2026-07-16

> Purpose: instantiate the numerical catalog for representative PyTorch operator semantics without inventing a universal tolerance. Each record separates documented API semantics, analytical accuracy assumptions and policy-only compatibility.

## 1. P5 — `torch.sum`

### Documented semantics

PyTorch states that `torch.sum` returns the sum of the input elements. If `dtype` is supplied, the input is cast to that dtype before summation; otherwise result dtype may be promoted. Source: [torch.sum 2.13](https://docs.pytorch.org/docs/2.13/generated/torch.sum.html), retrieved 2026-07-16.

The API does not state a universal ULP/relative-error guarantee or a fixed reduction tree.

### Contract split

```text
S1 exact fields:
    output structure/dtype and reduction axes

S2 numerical target:
    exact real sum of the values after the documented input cast

S2 arithmetic domain:
    finite inputs, declared accumulation dtype/rounding,
    no unhandled overflow/underflow, n terms, proven candidate realization
```

### Quantitative envelope

For unit roundoff `u`, `k=n-1`, `k*u<1`, exact casted-input sum `s*`, and absolute scale `a=sum_i |x_i|`:

```text
gamma_k = k*u/(1-k*u)
S_P5(x) = [s* - gamma_k*a, s* + gamma_k*a]
```

The contract adds a separately certified output-cast term if the accumulator and output representations differ. A sharper tree-specific bound may replace this only when the realized tree/precision is established.

### Verdict

- outside the envelope: `REJECT S2_NUMERICAL_CONFORMANCE`;
- inside: `ACCEPT` for the declared arithmetic model and covered input;
- unknown accumulation precision/tree assumptions needed by the chosen bound: `INVALID` or use the conservative admitted model;
- special/overflow case outside domain: not silently passed; route to a separate contract or `UNINSTANTIATED`;
- eager mismatch inside the envelope: compatibility difference only.

### Limits

The conservative bound may be too wide to detect accuracy regression. It is a sound coarse conformance contract, not a claim of best achievable accuracy.

## 2. P6 — `torch.mm`

### Documented semantics and configuration

`torch.mm` performs matrix multiplication and supports TensorFloat32; some device/dtype combinations use different backward precision. Source: [torch.mm 2.13](https://docs.pytorch.org/docs/2.13/generated/torch.mm.html), retrieved 2026-07-16.

PyTorch's matmul-precision API documents:

- `highest`: float32 internal computation;
- `high`: TF32 or a two-bfloat16 representation/algorithm when available, otherwise highest;
- `medium`: bfloat16 internal computation when available, otherwise high;
- the setting changes internal computation, not output dtype.

Source: [torch.set_float32_matmul_precision 2.13](https://docs.pytorch.org/docs/2.13/generated/torch.set_float32_matmul_precision.html), retrieved 2026-07-16.

Therefore precision mode and realized algorithm are contract identity, not runtime variance.

### Coordinate reference

For output coordinate `ij` with inner dimension `k`:

```text
c*_ij = sum_l a_il*b_lj                  exact real dot product
s_ij  = sum_l |a_il*b_lj|                conditioning scale
```

### Full-float envelope

Under a proven homogeneous model, use an appropriate dot-product bound. Conservative examples are:

```text
FMA accumulation:          |c_ij-c*_ij| <= gamma_k*s_ij
separate rounded mul/add:  |c_ij-c*_ij| <= gamma_{2k}*s_ij
```

with any output-cast term included and standard-model assumptions enforced.

### Reduced-input-precision envelope

If execution evidence proves a concrete input quantizer `q` such as TF32-style mantissa reduction, decompose:

```text
|c_ij - c*_ij|
<= |dot(q(a_i),q(b_j)) - dot(a_i,b_j)|       input-quantization effect
 + B_accum(q(a_i),q(b_j))                    accumulation bound
 + B_out                                      output-cast bound
```

All terms can be computed or conservatively enclosed from the input and declared arithmetic model.

If `high`/`medium` permits several algorithms and the realized one is unknown, either use the union/enclosure of every documented possibility or return `UNINSTANTIATED` for a sharper numerical claim. Selecting the narrowest mode after observing the output is forbidden.

### Verdict

- output outside the configuration-correct envelope: numerical-conformance rejection;
- inside: acceptance for that configuration/model;
- bitwise mismatch with eager but inside: not correctness failure;
- a stricter “candidate no worse than eager truth-error” contract is separate S4 non-degradation.

## 3. P7 — `torch.nn.functional.softmax`

### Documented semantics

PyTorch defines softmax by normalized exponentials along `dim`; the elements are rescaled to `[0,1]` and sum to one. If `dtype` is supplied, input is cast before the operation, which can prevent overflow. Source: [functional.softmax 2.13](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.softmax.html), retrieved 2026-07-16.

The API page does not supply a quantitative forward-error/ULP guarantee.

### Contract split

```text
S1 exact fields:
    output shape/dtype/dim relation and documented domain

mathematical target:
    high-precision normalized exponential vector p*(x)

necessary diagnostics:
    finite/domain behavior, range, normalization residual

primary numerical geometry:
    componentwise error and TV(p_C,p*) = 0.5*||p_C-p*||_1
```

Range and sum-to-one are not sufficient for acceptance because many wrong probability vectors satisfy them.

### Envelope status

- if an operator/backend accuracy bound is supplied, instantiate S2 around `p*`;
- if an application supplies a categorical-law tolerance, instantiate S3 using TV or task loss;
- if only eager compatibility is required, instantiate S4 with a predeclared probability geometry/margin;
- otherwise numerical acceptance is `UNINSTANTIATED`, although truth-relative error and exact-core violations are still reported.

### Impact separation

Argmax/top-k/sampling changes are downstream impact contracts. A tiny TV difference can flip a boundary case; a larger probability difference can leave the decision unchanged. Neither substitutes for numerical conformance.

## 4. P8 — `torch.nn.LayerNorm`

### Documented semantics

PyTorch documents the LayerNorm formula, epsilon, optional elementwise affine parameters, normalization dimensions, and use of the biased variance estimator equivalent to `torch.var(..., correction=0)`. Source: [torch.nn.LayerNorm 2.13](https://docs.pytorch.org/docs/2.13/generated/torch.nn.LayerNorm.html), retrieved 2026-07-16.

The API does not give one quantitative floating error budget across dtype/backend/fused implementations.

### Contract split

```text
S1 relation:
    normalized axes, correction=0 variance convention,
    epsilon and elementwise affine semantics

truth reference:
    high-precision evaluation of that documented formula

geometry:
    componentwise/vector truth error,
    mean/variance residual before affine as diagnostics,
    downstream direction/projection only as impact
```

### Envelope status

An S2 envelope requires certified error propagation through mean reduction, centered subtraction, variance reduction, square root/division and affine transformation under the realized precision schedule. If that schedule is not known or the resulting enclosure is too weak, use an externally justified S3/S4 contract or return `UNINSTANTIATED`.

Checking only output mean/variance is insufficient because a wrong permutation or directional error can preserve both.

## 5. What these records establish

| Record | Direct quantitative numerical envelope? | Why |
|---|---|---|
| P5 sum | yes, restricted conservative S2 | standard reduction error bound after documented cast |
| P6 mm | yes when realized precision algorithm is known | input quantization + dot accumulation + output cast |
| P7 softmax | not from API alone | mathematical formula exists, quantitative allowance does not |
| P8 LayerNorm | only with certified composite propagation | formula is specified but precision schedule/budget is not |

This is the intended behavior of a reliable Oracle: it gives determinate answers where semantics and arithmetic support them, and explicitly refuses unsupported acceptance elsewhere.

## 6. Interaction with bias and variance

For every record, after conformance is judged:

- decompose candidate-baseline discrepancy into average shift, input/signature heterogeneity and exact-input runtime variability;
- when truth exists, separately decompose truth-relative candidate and baseline errors;
- stratify by conditioning and precision mode;
- preserve tails and contract violations even when average relative bias cancels.

The decomposition explains implementation behavior; it does not generate these analytical envelopes.
