# Inductor softmax call binding review — 2026-10-18

This review covers `corpus/calls/inductor_softmax.py` and
`corpus/call_bindings/inductor_softmax.json`. It applies only to the public
rank-2, last-dimension call declared there and does not approve other softmax
calls by name or shape similarity.

## Task correspondence

The wrapper receives exactly three named inputs: tensor `x`, `dim=-1`, and
`half_to_float=False`. It rejects any other input set or task parameters and
returns the complete public softmax tensor. The binding maps these three inputs
to `aten._softmax(x, dim, half_to_float)`, maps the whole returned tensor to the
contract output, and requires `x` to remain unchanged.

On the installed PyTorch 2.10.0+cu128 package, an independent `make_fx` trace of
the public expression `torch.softmax(x, dim=-1)` contains exactly
`torch.ops.aten._softmax.default(x, -1, False)`. The installed schemas are
`aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None)` and
`aten::_softmax(Tensor self, int dim, bool half_to_float)`. This establishes the
selected internal operation and the false conversion flag for this call; it is
not inferred from the generated Triton result.

The admitted SOFTMAX `real` profile supplies the corresponding exact-real task
and rigorous result enclosure. Device exponent approximations, reduction order,
and output rounding remain in the program comparison rather than being added to
the task definition.

## Input, output, state, and launch evidence

The call wrapper has no tensor parameters, buffers, or closed-over tensor
state. Its compiled function depends on the received `x`; `dim` and
`half_to_float` are fixed task parameters checked before the call. The reviewed
plan accounts for every received input, the complete return, and the only tensor
pre/post-state.

The isolated measurement under the same wrapper and mapping used eight
development and sixteen confirmation inputs of shape `[16, 37]`, dtype f32.
For all 24 calls:

- the installed torch version matched the declared version;
- one captured TTIR launch had complete IR coverage;
- the only loaded tensor source was the declared input `x`;
- the complete public output was written by Triton and was not changed later;
- `x` remained unchanged;
- the admitted reference covered every output coordinate and met its requested
  enclosure resolution;
- no binding, hidden-source, budget, or output-coverage failure was reported.

The producer trace is expected not to align for compiled code because the
unmeasured trace deliberately runs under an eager stance. This call has no
cross-launch carry-over, and declared-input identity, bytes, coverage, and
version counters establish its single input provenance.

Two controlled variants added an undeclared global floating tensor and an
undeclared global integer tensor to the output path. Both were rejected as
unaccounted tensor sources before a bias conclusion was produced. A separate
`softmax(x)+0.01` variant was detected through the task difference while the
ordinary program-numerics comparison remained unconfirmed; that experiment is
useful detection evidence but is not the basis for this binding review.

After the reviewed plan and call were retained, the repository's production
measurement declaration was run twice without a temporary registry override.
The two reports agree exactly on the expanded declaration, all 24 binding
results, all 14,208 reference coordinates, numerical statistics, task
statistics, complete-output statistics, provenance, and failure classes; only
elapsed time differs. Both runs have complete call-level reference coverage and
no failure class. The task residual excludes zero at zero coordinates and its
fixed-direction mean is below `4e-19` in magnitude. The program-numerics
comparison detects a small negative effect on R1, with mean projection about
`-6.58e-9` and Holm-adjusted p about `0.00545`; complete `K-f` has the same
finding. This is evidence for the declared normal-input population and rules,
not an assertion about every input distribution.

## Scope and limits

The retained binding permits the runtime to evaluate this exact wrapper against
the admitted task for each sampled call. The current measurement declaration
uses finite f32 inputs of shape `[16, 37]`; other shapes or floating dtypes still
undergo the same per-call checks, but have not been included in the recorded
population experiment. All-masked rows, positive infinity, NaN, dtype-changing
public calls, backward operations, safe softmax, log-softmax, and unrelated
generated kernels are outside this reviewed call boundary or are refused by the
contract for the sampled task.

One reviewed binding establishes only this call correspondence. It does not
make the SOFTMAX role, a kernel name, or an arbitrary softmax-shaped computation
universally certified.
