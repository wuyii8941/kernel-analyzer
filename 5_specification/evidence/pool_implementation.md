# POOL implementation review — 2026-10-19

## PyTorch 2.10 unique-maximum max_pool2d complete task

The `torch210_unique_max2d_autograd` profile was added because real public calls return three results that the
older profile could not judge together: every pooled value, every int64 argmax index and the complete input VJP.
It selects `torch.nn.functional.max_pool2d(..., return_indices=True)` for PyTorch 2.10 and requires all effective
windows to have one unique maximum. Tied windows remain unsupported. The public documentation names an argmax but
does not establish a universal tie choice, and the ordinary derivative of max is not unique at a tie; choosing one
silently would make the reference less reliable.

The documented window starts at `output*stride-padding`, samples kernel coordinates at the declared dilation, and
uses negative-infinity padding. Floor/ceil output dimensions and the final-window correction are applied before
evaluation. For a unique maximum at `(h,w)`, the exact forward value is that input value and the ordinary VJP adds
the output upstream to that coordinate. Contributions from overlapping windows are summed. This value and gradient
rule follows independently from the max function. The flattened index `h*input_width+w` is treated differently: it
is an explicit PyTorch 2.10 task convention, requires a task binding, and is cross-checked against CPU and CUDA
public calls. It is not claimed as an implementation-independent encoding.

The adapter requires complete physical input and upstream records plus every public option. It rejects wrong ranks,
dtypes, shapes and parameter types before evaluation; `return_indices=false`, tied maxima and undeclared tasks do
not receive an answer. Its complete-output checker separately enforces floating dtype/shape/value for output and
gradient and exact int64 dtype/shape/value for indices. Six mutations cover floor in place of ceil, ignored padding,
column-major indices, overwritten overlapping gradients, ignored dilation and an omitted indices result.

Independent validation enumerates every real input coordinate in each window without calling the adapter. Across
rank-3/rank-4 inputs, all four physical dtypes and default, rectangular, padded, dilated and ceil-mode parameter
sets, it agrees exactly with the adapter and PyTorch 2.10 CPU for values, indices and gradients. The full POOL
contract now has 33 consistent examples and rejects all 17 declared mutations. The reviewed contract and adapter
are retained under `validated/`; this admits the reference profile but does not establish a particular Triton call
binding.

A second frozen batch then used unique float32 inputs and parameter sets not used by the contract examples. All 16
CPU calls matched the independent reference exactly. The separate Inductor acceptance split obtained a complete
reference and matching public results for all eight calls; six captured both Triton phases, while two dilated/ceil
calls captured only a Triton forward and retained the missing backward coverage. The FlagGems development split
had six matching rank-4 calls with only Triton forward coverage and two rank-3 public calls rejected by its four-rank
unpack. These limitations remain in the evidence rather than being converted into specification failures. Raw and
summarized records are `corpus/external_batch/results/pool_max_unique_*.json.gz`.

## Explicit two-dimensional average-pooling task

The new `finite_avg2d_autograd` profile covers full forward values and the input VJP on positive rank-3/rank-4
physical finite f16/bf16/f32/f64 inputs. It requires complete input/upstream tensor records and all options.
The original max-values profile and its output interface are unchanged. This is a local source/formula review
performed by Codex, not third-party approval; the independently implemented matrix checks are supporting evidence.

The retained PyTorch 2.10 AvgPool2d/functional documents provide the public parameters, zero padding, output sizes,
ceil correction and divisor override. The retained `PH1_DECISIONS.POOL_A1` records a deliberate divisor choice:
count explicit padding but exclude overhang beyond the explicit padded domain. Because that source itself records
ambiguity across pooling conventions, `PA.DIVISOR` remains `needs_binding`: the registry requires an explicit task
statement. Native CPU/CUDA comparisons support this choice for PyTorch 2.10 two-dimensional pooling; they do not
turn it into the only legal definition of pooling. Candidate Triton results do not define the reference.

For output o, let W_o be its integer window, D the real input coordinates and P the explicit padded coordinates.
The selected divisor d_o is the nonzero override when supplied, otherwise |W_o ∩ P| if padding counts and
|W_o ∩ D| otherwise. Then y_o = sum_{i in W_o ∩ D} x_i / d_o. Geometry and divisor are independent of x, so
A_oi = 1/d_o on real members and zero elsewhere defines a linear map y=A x. Its ordinary VJP is A^T upstream.
There is no nonsmooth choice at equal input values. Negative nonzero overrides are valid; zero is rejected.

Source review traced `e2_pool.evaluate`'s average branch: windows follow the documented integer floor/ceil formula;
the right-start correction is applied before allocating outputs; actual and padded positions are distinguished;
one exact divisor is shared by forward and every gradient contribution; overlap contributions are accumulated.
The admitted adapter first rejects extra/missing fields, malformed parameter types, incompatible physical dtype,
nonfinite or unrepresentable inputs, and inconsistent upstream. It selects only this branch of the existing
evaluator, and verifies that its extra historical argmax field is empty before excluding it from the public task.
No other part of the old broad draft is admitted. Exact rational arithmetic prevents reference rounding; observed
output checking requires both complete physical tensors, matching dtype/shape and exact values.

Independent checks construct the full rational matrix by scanning input-coordinate membership against independently
enumerated output origins, then compute its forward product and transpose product. Sixty-four combinations cover
rectangular kernels, strides, padding, floor/ceil mode, both counting choices and a negative override; the same
tasks are cross-checked with native PyTorch float64. A hand witness uses a 4x4 all-ones input and upstream only on
the last output: for kernel=3, stride=2, padding=0, ceil=true the last four input gradients are exactly 1/4.
The original captured FlagGems and Inductor results are not used to generate these expected values.

The contract has 25 examples across both profiles and 11 rejected mutations. Average mutations distinguish wrong
forward divisor, wrong backward-only divisor, ignored override, wrong padding count and missing gradient. These
checks do not automatically activate the profile. After reviewing the sources, exact computation and independent
matrix algorithm above, the current contract/adapter are retained under `validated/`; call binding is still pending.

The previous max-only contract, adapter and unchanged exact window evaluator are retained in
`pool_before_average_20261019.json.gz`. Historical batch records are never overwritten. A complete mathematical
reference does not establish a Triton-call binding, a program reference, statistical bias, or general coverage.

## Original maximum-pooling values scope

The active POOL contract covers complete forward values for PyTorch 2.10 functional two-dimensional max pooling on positive rank-3 or rank-4 physical finite f16, bf16, f32 or f64 tensors. Padding is zero, ceil mode is false, and the call returns values only. Kernel size, stride and dilation may each be a positive integer or pair; an omitted stride defaults to the kernel size.

The retained PyTorch 2.10 documentation defines the dilated window, maximum value and floor-mode output shape. The retained native schema file confirms that `aten.max_pool2d` returns one tensor and separates it from the with-indices, out and backward forms. The public functional documentation separately exposes `return_indices`. These sources support the selected task without using the result of a Triton kernel.

The adapter validates the exact call form, physical dtype, rank, positive dimensions, finite values, exact representability and every pooling parameter before constructing every output coordinate. It refuses nonzero padding, ceil mode, returned indices, backward, average or adaptive pooling, special values, signed-zero distinctions, integer or complex data and unselected overloads. Ties remain supported because they do not change a forward maximum value.

Validation comprises 18 contract examples, six mutated implementations, complete dtype/shape/value checks and 24 generated parameter cases. The generated cases cover all four admitted dtypes, rank 3 and rank 4, rectangular windows, nonunit strides and dilation. Each was recomputed by an independent coordinate loop and compared with PyTorch 2.10. The previous broad pooling draft remains in `contracts/draft/POOL_extended_20261019.yaml`; it is not an admitted ground truth.
