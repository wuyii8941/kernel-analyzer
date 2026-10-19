# ARITH finite floating and selection-autograd implementation review — 2026-10-19

## Reviewed task

The active `real` profile covers twenty common default functional operations: same-dtype floating add, sub, rsub,
mul, three forms of div, neg, reciprocal, abs, sign, Inductor fma, floor, ceil, scalar clamp variants,
maximum/minimum, where, scalar masked_fill and floor_divide. Inputs are dense finite f16, bf16, f32 or f64 values.
Every recorded tensor value and scalar must be exactly representable in the declared data dtype. Data tensors in one
call have one dtype; normal broadcasting is supported except that the internal fma profile requires equal shapes.

The retained PyTorch 2.10 public pages state the formulas, broadcasting and public arguments. The retained
`native_functions.yaml` distinguishes the selected functional tensor/scalar overloads from in-place and out forms.
The retained refs source supplies rsub and functional masked_fill behavior. The retained Inductor source defines fma
as `a*b+c` without rounding after the multiplication. The review checked these retained files directly.

The task value is the exact real formula. Floating instruction and output-storage rounding remain in K-G. This is why
the contract can distinguish a wrong formula from ordinary floating execution. Division or reciprocal by zero has no
real task value. Integer overflow, mixed dtype promotion, pow and special floating values remain outside this profile.

## Errors removed from the former active draft

The old single profile treated integer arithmetic as unbounded exact arithmetic even though ATen integer operations
have fixed-width behavior. It also accepted arbitrary rationals such as mathematical `1/3` while labelling them f16
or f32 inputs, combined unresolved pow and infinity choices with routine arithmetic, and had no complete output
checker. That exact contract is retained as `contracts/draft/ARITH_extended_20261018.yaml` and is not loaded at run
time.

The reviewed adapter validates the selected call schema before decoding, checks physical input representability,
requires the first input of the selected arithmetic tensor/scalar overloads to be a tensor, rejects mixed tensor
dtypes and unsupported overloads, and checks the complete output dtype, shape and every value.

## Independent checks

The contract has 44 examples and twenty required wrong implementations. All examples are consistent and every wrong
implementation is rejected. Separate tests do not call the adapter's arithmetic helpers: they use `Fraction`, manual
formulas and NumPy broadcasting over 160 generated input pairs. They cover add/sub/rsub/mul/div, trunc and floor
division, all unary operations, clamp, extrema, selection, masked fill and fma. A separate PyTorch 2.10 cross-check
confirms dtype, broadcast shape and rounded stored values for f16, bf16, f32 and f64. Output tests reject wrong dtype,
shape, value, missing coordinates and additional fields.

The implementation review does not certify a concrete Triton call. A call becomes usable for bias measurement only
after its arguments, state, complete output and every launch are bound separately.

## Excluded scope

Mixed floating dtypes, integer and boolean arithmetic, float8, complex, NaN and infinity behavior, signed zero, pow,
gradients outside the selected masked-fill and Tensor/Tensor where data VJPs, tensor clamp bounds, fma broadcasting,
out/in-place variants and memory or aliasing effects remain refused.

## PyTorch 2.10 scalar `masked_fill` forward and input VJP — 2026-10-19

A later audit began from the real public call `Tensor.masked_fill(mask, Python scalar)`. The original `real` profile
already defined the selected forward values, but it could not answer a training call because it omitted the input
gradient and did not distinguish the public functional call from its in-place and Tensor-value overloads. The new
`torch210_masked_fill_scalar_autograd` profile reuses the same selection formula and adds only this missing interface.

The retained PyTorch 2.10 native schema separates `masked_fill.Scalar` from the Tensor, out and in-place overloads.
The retained official reference converts the scalar to the input dtype, applies `where(mask,value,input)` under
ordinary broadcasting and returns a new contiguous tensor. Ordinary broadcasting matters in both directions: a
`[1,3]` input and `[2,3]` mask legally return `[2,3]`. From that formula, an output coordinate contributes its
upstream value to the corresponding input coordinate exactly when the mask is false. Contributions are summed over
axes on which the input was broadcast. The profile returns every forward and input-gradient coordinate and rejects
an absent gradient. Output contiguity and unchanged input state are checked by the real-call runner because they are
physical call effects rather than extra numerical arrays.

Four independent tests use a separate coordinate loop, including cases where only the input expands, only the mask
expands and both operands expand. Across f16, bf16, f32 and f64 they agree with the adapter and PyTorch 2.10 CPU
autograd. The complete contract now has 39 consistent examples and rejects 16 deliberately wrong implementations;
the added errors invert the selection, pass gradient through masked coordinates, omit broadcast-gradient reduction,
or omit the gradient result.

The GPU batch was fixed before execution. FlagGems 5.0.2 was the development implementation. Four of ten calls
matched all values and the selected output layout. Four legal calls were rejected because its wrapper requires the
mask to broadcast to the existing input shape, which is narrower than the public functional rule. Two calls with a
noncontiguous input returned the correct values and input gradient but preserved noncontiguous output strides; the
versioned functional reference returns a new contiguous tensor. The candidate source retained in each report shows
both the restricted shape check and the pointwise output path.

The separately fixed TorchInductor split used unseen shapes, ranks, values, mask patterns and seeds. All ten calls,
including input expansion, rank four and a noncontiguous input, matched every exact forward and gradient coordinate,
returned contiguous outputs and captured Triton launches in both phases. FlagGems supplies no dedicated backward
operator; backward-phase launches in its report arise when public autograd evaluates its gradient using another
masked-fill operation. These finite results do not establish a formal kernel binding or a statistical bias verdict.
The retained records are `masked_fill_development_v1.json.gz`, `masked_fill_acceptance_v1.json.gz` and
`masked_fill_validation_v1.json.gz` under `corpus/external_batch/results`.

## PyTorch 2.10 Tensor/Tensor `where` forward and both data VJPs — 2026-10-19

The next real call was `torch.where(condition,input,other)`, which FlagGems 5.0.2 maps to its Triton
`aten.where.self` implementation. The existing `real` profile already had the official forward selection, so the
missing part was narrow: a complete training call must return gradients for both data tensors under independent
three-way broadcasting. The new `torch210_where_tensor_autograd` profile selects only the Tensor/Tensor overload;
scalar, out and one-argument variants remain separate tasks.

The retained PyTorch 2.10 page defines the output coordinate as input when the broadcast condition is true and other
otherwise, and states that all three tensors broadcast. The native schema fixes the selected overload. Transposing
that coordinate map gives the two VJPs: upstream passes to input on true coordinates and to other on false
coordinates, then each is summed back to its own source shape. The Boolean condition has no requested gradient.

Four independent tests implement this with their own broadcast-index loop and do not call the specification's
selection or reduction helpers. Sixteen combinations of f16, bf16, f32 and f64 with different condition/input/other
shapes agree exactly with PyTorch 2.10 CPU autograd and the adapter. The complete-output checker requires out,
grad_input and grad_other together. Added wrong implementations exchange branches, omit input-gradient reduction,
send both gradients through the same branch or omit grad_other; all are rejected.

Twenty public calls were frozen before GPU execution and first cross-checked on CPU. All ten FlagGems development
calls matched exactly, including independent data broadcasts, an all-true condition and a noncontiguous input; the
reports cover 612 forward, 416 input-gradient and 40 other-gradient coordinates. All ten separately selected
TorchInductor calls also matched, covering 1,434 forward, 846 input-gradient and 410 other-gradient coordinates with
unseen ranks, broadcast directions, an all-false condition and two noncontiguous data inputs. Every call captured
forward and backward Triton phases. The reports do not establish a formal kernel binding or statistical bias verdict.
They are retained as `where_development_v1.json.gz`, `where_acceptance_v1.json.gz` and
`where_validation_v1.json.gz` under `corpus/external_batch/results`.

## PyTorch 2.10 scalar-bound `clamp` forward and input VJP — 2026-10-19

The real public call is functional `torch.clamp(input,min,max)` with Python scalar bounds. The old `real` profile
already supplied exact forward values, but it could not judge a training call and did not fix the derivative at an
active boundary. This is not determined by the ordinary derivative: PyTorch 2.10's retained `derivatives.yaml`
explicitly says the derivative is undefined at min and max and deliberately selects subgradient one there. The new
`torch210_clamp_scalar_autograd` profile therefore belongs to the versioned PyTorch task, rather than claiming a
universal derivative convention.

The retained public documentation and native schema fix scalar optional bounds, lower-then-upper clipping and the
rule that `min > max` returns max everywhere. The retained autograd source fixes the inclusive VJP. An independent
coordinate implementation first applies the lower bound and then the upper bound; it passes upstream exactly when
`x >= min` for an existing lower bound and `x <= max` for an existing upper bound. Thus reversed bounds give a zero
input VJP, while equal bounds pass upstream only where the input equals that common bound.

Four independent tests cover 24 dtype/shape/bound combinations across f16, bf16, f32 and f64 and agree exactly with
PyTorch 2.10 CPU autograd and the adapter. They also reject tensor bounds, missing bounds, malformed physical input,
wrong upstream shape or dtype, undeclared fields and incomplete output. The complete ARITH contract has 51
consistent examples and rejects all 24 selected wrong implementations, including strict rather than inclusive
boundary gradients, ignoring the upper bound, returning min for reversed bounds and omitting the VJP.

Twenty calls were fixed before GPU execution and first cross-checked on CPU. All ten FlagGems 5.0.2 development
calls matched the exact reference: 690 forward and 690 VJP coordinates. Every call captured its selected Triton
forward. FlagGems has no dedicated clamp Triton backward in this path, so the exact public VJP supplied by autograd
is call-context evidence rather than a result attributed to that forward kernel.

The separate TorchInductor batch used unseen ranks, shapes, scalar bounds, boundary density and layout. Eight calls
matched completely. Two calls with bounds `-0.75,1.25`, following an equal-bound `0.25,0.25` case in the same
process, returned `0.25` at almost every forward coordinate: 110/115 and 112/115 coordinates differed, while both
VJPs remained exact. The original report is retained without changing the specification.

A minimal program that imports neither the analyzer nor its recorder reproduced the state dependence. Compiling
only `-0.75,1.25` in a fresh process is exact. Compiling `0.25,0.25` first, resetting Dynamo and then compiling
`-0.75,1.25` gives 110 wrong forward coordinates and an exact VJP. The failing generated Triton forward embeds
`0.25` for both clipping bounds, whereas its derivative mask reads the current runtime scalar bounds. This supports
a compiled-call reuse error in the tested PyTorch 2.10 context; it does not establish statistical bias or a formal
kernel binding. The retained records are `clamp_development_v1.json.gz`, `clamp_acceptance_v1.json.gz`,
`clamp_compile_fractional_fresh_v1.json.gz`, `clamp_compile_sequence_failure_v1.json.gz` and
`clamp_validation_v1.json.gz` under `corpus/external_batch/results`.

## PyTorch 2.10 Tensor/Tensor `maximum` and `minimum` with both VJPs — 2026-10-19

FlagGems 5.0.2 publicly replaces the Tensor/Tensor `maximum` and `minimum` forward calls with Triton pointwise
kernels. The existing `real` profile already defined the finite forward extrema, so the missing specification was
the complete training call: two independently broadcast data operands, complete upstream and a VJP for each source.
Scalar, reduction, out, fmax/fmin and special-value variants are different tasks and remain outside this profile.

The retained PyTorch 2.10 public pages and native schemas fix the forward calls. The retained versioned
`derivatives.yaml` supplies the nontrivial choice at equality: a strict selected operand receives all of upstream,
whereas equal operands each receive one half. Ordinary autograd broadcasting then requires each operand's gradient
to be summed back to its own source shape. An independent implementation uses its own output-coordinate to
source-coordinate map and exact `Fraction` accumulation; it does not call the adapter's maximum, minimum or gradient
reduction helpers.

Four independent tests cover 24 combinations of operation, dtype and shape across f16, bf16, f32 and f64. They
force equality coordinates, independent broadcast directions and rank-three broadcasting, and agree exactly with
both PyTorch 2.10 CPU autograd and the adapter. The complete-output checker requires forward, input VJP and other VJP
together. The ARITH contract now has 58 consistent examples and rejects all 28 selected wrong implementations,
including assigning a tie wholly to the first operand, exchanging maximum and minimum, omitting broadcast-gradient
reduction and omitting the other VJP.

Twenty calls were fixed before GPU execution and cross-checked on CPU. All ten FlagGems development calls match the
exact reference: 822 forward, 502 input-gradient and 402 other-gradient coordinates. Every call captures the
selected FlagGems Triton forward. Its public VJPs execute through autograd without a FlagGems Triton backward and
are retained as call-context results.

The separately fixed TorchInductor batch uses unseen ranks, sizes, broadcast directions and layouts, plus a complete
all-equal case. All ten calls match the exact reference: 1,712 forward, 748 input-gradient and 532 other-gradient
coordinates. Every call captures forward and compiled backward Triton phases. These finite results establish the
profile and the selected test outcomes, but do not establish a formal kernel binding, statistical bias verdict or
coverage of the excluded variants. The retained records are `extrema_development_v1.json.gz`,
`extrema_acceptance_v1.json.gz` and `extrema_validation_v1.json.gz` under `corpus/external_batch/results`.
