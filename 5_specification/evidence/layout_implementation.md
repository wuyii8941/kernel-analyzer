# LAYOUT finite floating, constant-pad and cat autograd implementation review — 2026-10-19

## Reviewed task

The active `real` profile covers common logical-value operations: view/reshape, axis permutations, expand and
squeeze, select/slice/narrow, cat/stack/split, constant padding, functional select/slice scatter, clone/lift/alias,
and triu/tril. Inputs are dense finite f16, bf16, f32 or f64 tensor records. Every value and scalar pad must be
exactly representable in the declared dtype, and multi-tensor calls use one dtype.

The retained PyTorch 2.10 public pages define the value and shape transformations. The retained
`native_functions.yaml` distinguishes the selected integer-dimension functional schemas from named-dimension,
dtype-view, tensor-start narrow, out and in-place overloads. Retained refs and decompositions define negative
constant padding and the functional scatter/lift values. The retained PyTorch 2.10 `derivatives.yaml` separately
selects `cat_tensors_backward` with the upstream, every input size and dtype, and `dim`. The review checked these
retained files directly.

This contract deliberately specifies logical values and shapes. It does not promise storage identity, strides,
aliasing, memory format or device placement. A concrete kernel binding must separately account for all input states,
public outputs and memory effects. A view call whose validity depends on physical strides therefore needs a wrapper
or binding boundary that establishes the selected row-major input precondition.

## Errors removed from the former active draft

The former profile combined pure layouts with tensor creation, dtype conversion, copy and fill state effects. It
accepted arbitrary rationals such as mathematical `1/3` under an f32 label, admitted nonfinite and integer data in
one broad scope, labelled numeric outputs as discrete, and had no complete output checker. The exact former contract
is retained as `contracts/draft/LAYOUT_extended_20261018.yaml` and is not loaded at run time.

The reviewed adapter validates the exact call schema before evaluation, requires physical finite floating records,
rejects negative zero because signed-zero preservation is outside the logical-value task, checks same dtype for
multi-tensor calls, and validates exact scalar padding. Creation, copy/fill, dtype views, named dimensions, implicit
expand, nondefault clone memory format and state or storage effects remain refused.

## Public constant-pad training task

A real `torch.nn.functional.pad(input, pad, mode='constant', value=...)` training call exposed a narrower gap than
the broad layout profile: the old contract covered `aten.constant_pad_nd` forward values but could not judge the
public call together with its input gradient. The new `torch210_functional_constant_pad_autograd` profile fixes the
public operation, all public options, complete physical input and upstream, full output, unchanged inputs and the
complete input VJP. It does not extend the claim to reflect, replicate or circular padding.

The retained PyTorch 2.10 public documentation fixes pair order and default fill. Its retained official reference
implementation supplies the validation order that matters for negative padding: crop first, reject a crop beyond an
input dimension, then create the positive-padding result and copy the retained region. The input VJP is derived
independently as the transpose of that coordinate copy: retained coordinates read the corresponding upstream value,
cropped input coordinates receive zero, and filled output coordinates have no input predecessor.

## Public cat training task

The real FlagGems 5.0.2 `aten.cat` replacement showed a second narrow gap. `LAYOUT.real` already had the correct
logical forward copy, but it could not judge the public call together with one ordered gradient for every input.
`torch210_cat_autograd` fixes the Tensor-list overload, its dimension, all complete inputs and upstream, the complete
output and every input VJP. It deliberately excludes `out`, named dimensions, mixed dtypes and storage or aliasing
claims.

The public PyTorch 2.10 definition fixes input order, the concatenation dimension, shape agreement and the special
shape-`(0,)` input. Independently of PyTorch autograd, each nonempty input occupies a consecutive output interval, so
the transpose copies the matching upstream interval back to that input; a documented one-dimensional empty input
keeps its position in the gradient list and receives an empty gradient. This agrees with the retained official
`cat_tensors_backward` selection.

## Independent checks

The contract has 44 examples and eighteen required wrong implementations. All examples are consistent and every wrong
implementation is rejected. Separate tests reconstruct shapes and coordinates with NumPy for reshape, permutations,
ranges, joins, splits, padding, scatters and triangles. PyTorch 2.10 cross-checks f16, bf16, f32 and f64 output dtype,
shape and stored values. Constant-pad checks separately implement the crop/fill coordinate map without calling the
specification evaluator, then compare both forward and autograd results for twenty dtype/case combinations, including
positive padding, negative cropping, mixed axes and an empty output. Complete-output tests reject a wrong dtype,
shape, coordinate or missing gradient as well as missing or reordered split results. Cat checks independently build
the coordinate copy and upstream partition, then compare 24 dtype/shape cases against PyTorch 2.10 autograd,
including unequal input widths, negative dimensions, zero-width axes and shape-`(0,)` inputs. Its complete-output
checker rejects a missing, reordered, wrongly shaped or numerically wrong input gradient.

Twenty frozen float32 public calls then cross-checked exactly on CPU. FlagGems 5.0.2 produced exact forward values
for all ten development calls (776 coordinates and a captured forward Triton launch for every call), but its public
pad replacement supplied no usable autograd edge, so all ten calls remain incomplete because `dx` is absent. The
original ten autograd failures and a second report that preserves the exact partial forward results are both retained.
TorchInductor completed all ten separately selected calls, including unseen ranks, mixed crop/fill combinations,
default fill and a noncontiguous input; all 1,444 forward and 1,052 input-gradient coordinates are exact, with forward
and backward Triton launches captured. These runs do not establish a formal kernel binding or a statistical bias
verdict.

A separately frozen cat batch first cross-checked twenty float32 public calls on CPU. FlagGems completed all ten
development calls exactly: 440 forward and 440 gradient coordinates, including five inputs split over two Triton
launch groups, a shape-`(0,)` input and noncontiguous inputs. TorchInductor completed all ten held-out calls exactly:
1,052 forward and 1,052 gradient coordinates across unseen ranks, dimensions, six-input grouping, zero-width
positions, another empty-input position and noncontiguous combinations. Every call captured a forward Triton launch.
The VJPs are graph slices of upstream and require no separate arithmetic kernel. These runs do not establish a formal
kernel binding or a statistical bias verdict.

The implementation review does not certify a concrete Triton call or a composition of multiple layout operations.

## Excluded scope

Integer, boolean, float8, complex and nonfinite data; signed zero; creation; dtype conversion; copy/fill state
effects; named dimensions; nondefault memory format; out/in-place forms; gradients other than the selected
constant-pad and cat VJPs; strides, storage and aliasing are outside these profiles.
