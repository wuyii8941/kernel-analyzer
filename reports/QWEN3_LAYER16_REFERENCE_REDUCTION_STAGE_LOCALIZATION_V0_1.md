# Layer-16 reference-reduction stage localization

## Result

The stage-preserving `reference_reduce` intervention was run in two matched
states and independently audited:

- [step236 report](</data1/tzh/forkcert/results/operator_oracle/qwen3_step236_layer16_attention_mlp_slice_v0_5/result.json>) and [audit](</data1/tzh/forkcert/results/operator_oracle/qwen3_step236_layer16_attention_mlp_slice_v0_5/audit.json>);
- [step097 report](</data1/tzh/forkcert/results/operator_oracle/qwen3_step097_layer16_attention_mlp_slice_v0_5/result.json>) and [audit](</data1/tzh/forkcert/results/operator_oracle/qwen3_step097_layer16_attention_mlp_slice_v0_5/audit.json>).

The intervention changes only the RMSNorm variance expression from the
compiled fused-kernel expression to the reference expression
`pow(2).mean(-1, keepdim=True)`.  It retains the compiled kernel input,
rsqrt, normalization, weight multiply, output dtype and the original suffix.

In both states:

1. the kernel has a reproducible same-input discrepancy;
2. `reference_reduce` changes the one relevant clipping event in the same
   direction as full kernel repair;
3. the `reference_reduce` output tensor hash is exactly equal to the
   same-input eager RMSNorm reference output hash;
4. the independent verifier reports `reference_reduce_stage_exact: true`.

This is materially stronger than replacing the complete kernel output: the
observed discrepancy and endpoint change can be reproduced by a single,
declared stage substitution while the other stages stay on the compiled-side
calculation.

## What this does and does not localize

The strongest supported statement is:

> For this generated kernel call, under the two tested matched states and the
> fixed original suffix, the reference-vs-compiled discrepancy is fully
> reproducible by the variance-reduction expression. The reduction stage is a
> strong stage-level localization candidate.

This is not yet a compiler root-cause proof. `reference_reduce` is a
stage-preserving counterfactual implemented outside the generated Triton
kernel; it does not prove which compiler pass selected the reduction tree, nor
does it exclude an equivalent arithmetic explanation at another IR stage.
It also covers only a forward clipping endpoint and two states. The claim is
implementation-relative, not a correctness verdict.
