# Qwen3 layer-16 fused-kernel operation-hypothesis mediation

## Scope

This is a controlled follow-up to the valid layer-16 generated-kernel slice at
state `calibration-0-late-step236`.  The intervention keeps the original
compiled execution, captures the same input to runtime call 16, replaces only
that call's post-attention norm output, and runs the same original suffix.

The generated kernel is
`triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12`.  Static
code evidence shows a fused add, FP32 conversion, square/mean reduction,
`libdevice.rsqrt`, weight multiply, and store.  The static list is provenance,
not a causal decomposition.

## Operation hypotheses tested

Four declared alternatives were replayed on the captured kernel input:

- `sum_fp32`: FP32 sum/mean and rsqrt;
- `high_precision`: FP64 intermediate normalization;
- `input_fp16`: FP16 input cast followed by FP32 mean/rsqrt;
- `rsqrt_fp64`: FP32 variance with FP64 rsqrt input.

Each alternative is an intervention on the fused output computation.  It is
not a literal replacement of one Triton instruction in the original kernel,
so it cannot by itself identify a unique constituent operation.

## Evidence

The independent runner report and verifier both pass:

- [runner result](</data1/tzh/forkcert/results/operator_oracle/qwen3_step236_layer16_attention_mlp_slice_v0_3/result.json>);
- [independent audit](</data1/tzh/forkcert/results/operator_oracle/qwen3_step236_layer16_attention_mlp_slice_v0_3/audit.json>).

The original kernel-only intervention changes one clipping decision (one
compiled-off to reference-on coordinate).  Every declared operation
alternative also produces a continuous difference from the compiled output
and changes the same clipping coordinate under the fixed suffix.  Thus all
four alternatives are sufficient to mediate this one observed event in this
state.

## Interpretation

This experiment succeeds at a narrower goal than root-cause localization:

1. it shows that the generated kernel call is a reproducible local producer
   and a fixed-suffix semantic mediator;
2. it shows that the current state does not discriminate among the listed
   internal precision hypotheses;
3. it therefore does **not** establish that reduction, cast placement, rsqrt,
   or any individual ATen/Triton operation is the root cause.

The result is not evidence that all four mechanisms are present in the actual
kernel.  They are counterfactual computations that all move the decision
margin across the same boundary.  A stronger constituent-op claim would need
stage-preserving IR/kernel intervention (or additional states/endpoints where
the hypotheses predict different outcomes), while preserving the original
fusion, layout, and launch context.

## Claim level

The strongest supported claim is:

> The declared layer-16 generated-kernel call is implicated as a call-level
> local discrepancy producer and fixed-suffix semantic mediator under the
> matched-state protocol.  Internal operation hypotheses remain
> non-discriminated candidates.

This remains implementation-relative, forward-only, and state-conditioned;
it is not a correctness proof, a unique root-cause claim, or a backward/update
claim.
