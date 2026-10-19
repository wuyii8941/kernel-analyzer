# CONV implementation review — 2026-10-18

This review covers the CONV `real` profile in `contracts/CONV.yaml` and the
current `adapters/CONV.py`. It does not approve a particular Triton call.

## Sources and selected task

The retained PyTorch 2.10 Conv1d and Conv2d pages define valid
cross-correlation, symmetric zero padding, stride, dilation, groups, output
shapes, weight shapes, and one bias value per output channel. The retained
installed `native_functions.yaml` gives the exact low-level default functional
schemas for `aten.convolution` and `aten.convolution_backward`.

The reviewed profile selects batched, non-transposed one- and two-dimensional
operations on finite f16, bf16, f32 or f64 tensors of one dtype. Its task value
is the exact real cross-correlation. The backward is the VJP derived from that
forward formula. Device accumulation order, TF32, reduced-precision products,
and final casts remain numerical effects.

## Corrections made during review

1. The previous reference invented a default `output_mask` for
   `aten.convolution_backward`. The installed schema requires an explicit
   `bool[3]`; the adapter now rejects calls that omit it.
2. Every backward result keeps exactly three positional entries. An
   unrequested gradient is an explicit null entry, distinct from a zero tensor
   or an omitted position. The shared strict-result validation now preserves
   explicit nulls so contract-specific comparison can check them.
3. The previous scope included 3-D convolution without a retained Conv3d task
   definition. Admission is limited to the retained 1-D and 2-D formulas.
4. Fixed-width integer convolution is no longer equated with unbounded integer
   arithmetic. Integer, boolean, complex, float8, nonfinite, transposed and
   mixed-dtype calls remain outside the reviewed answer.
5. Complete comparison now checks dtype, explicit shape, every coordinate and
   all three backward positions, including empty tensors and null entries.

## Independent implementation checks

`tests/test_convolution_independent.py` directly enumerates the documented
cross-correlation with `Fraction` arithmetic. It covers one- and
two-dimensional, grouped, strided, padded and dilated cases. Holding the other
operands fixed makes the scalar upstream objective affine in each input,
weight and bias coordinate, so the symmetric unit difference reconstructs
each partial derivative exactly. These independently computed outputs and
gradients are compared with every coordinate of the executable reference.

The existing domain tests check complete upstream shapes, channel groups,
positive spatial parameters, bias shape, mixed dtypes and unresolved zero
dilation before coordinate evaluation. An independent PyTorch float64 autograd
case checks a grouped dilated VJP. The complete-output tests exercise all eight
output masks and reject missing positions, invented tensors, wrong dtype,
wrong shape and changed values. The contract checker establishes all 13
examples and rejects all five listed wrong implementations.

## Limits

This review does not establish 3-D, transposed, unbatched, complex, integer or
nonfinite convolution, asymmetric padding, padding strings, dtype/out
overloads, or hardware rounding. Zero dilation, zero input channels and
inconsistent `bias_sizes` remain unestablished rather than receiving a guessed
answer. A concrete kernel still needs a separate binding for every input,
output and state effect.
