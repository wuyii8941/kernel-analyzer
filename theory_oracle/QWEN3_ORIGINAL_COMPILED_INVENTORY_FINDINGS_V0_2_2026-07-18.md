# Qwen3 original compiled operator/kernel inventory findings v0.2

## Validity

The v0.2 inventory reproduced both frozen Dynamo graph identities and the target
compiled scorer tensor exactly. Both dynamic target repetitions were bit-exact.
Two forward specializations each materialized readable/transformed FX,
pre/post-fusion IR and generated output code. No real tensors were saved.

The prior v0.1 observer also reproduced graph/scorer identity, but ordinary cache
hits left its trace directories empty. It is retained as valid graph identity but
an incomplete kernel inventory, not counted as the final result.

## Descriptive result

For the target dynamic specialization:

- 20 generated Triton kernel families account for 482 calls;
- external kernels include 197 `mm` calls and 56 `bmm` calls;
- several transformer kernel families repeat 28 times, once per decoder layer;
- one cross-layer kernel family repeats 27 times, once between adjacent decoder
  layers.

That 27-call family combines ATen operations corresponding to:

- the preceding layer's attention/MLP residual adds;
- the next layer's input RMSNorm chain: `pow`, reduction `mean`, epsilon `add`,
  `rsqrt`, scaling `mul` and cast toward the following Linear.

This is direct evidence that a source-level `operator` and a generated `kernel`
are not interchangeable units in this subject. The candidate has no standalone
compiled realization of those constituent operations at that boundary.

## Implication for the failed layer-27 split

The invalid split cut through a repeated fused treatment. Its changed candidate
hash is therefore expected and is not runtime noise. Its mixed-arm effects cannot
be transported to the original candidate.

## What remains unknown

The inventory does not show that reduction, residual add, RMSNorm, cast or Linear
caused the scorer discrepancy. It only identifies their fused membership and
intervention constraints. The next causal experiment must use an explicit
barrier-control arm and report both boundary disturbance and target repair; a
single repaired arm is insufficient.
