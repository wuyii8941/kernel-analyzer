# Qwen3 step236 operator/kernel slice — v0.4

This report records the first valid longitudinal slice that reaches a generated
kernel while keeping the claim deliberately narrower than “root cause”.  The
case is one matched state (`calibration-0-late-step236`) and the forward PPO
clipping endpoint.

## Evidence artifacts

- Runner result: `results/operator_oracle/qwen3_step236_layer15_attention_mlp_slice_v0_4/result.json`
- Independent audit: `results/operator_oracle/qwen3_step236_layer15_attention_mlp_slice_v0_4/audit.json`
- Manifest: `theory_oracle/QWEN3_STEP236_LAYER15_ATTENTION_MLP_SLICE_V0_1.json`
- Kernel inventory and forward observability gate: `results/operator_oracle/qwen3_step236_whole_model_trace_v0_1/compiled_1/`

The runner and independent audit both report `valid: true`; all 28 runner
gates pass and the full Python suite passes (`432 passed`).  The audit also
checks the baseline fork coordinates against independent eager/compiled
transition artifacts.

## What is established

1. **Kernel-local production.**  At runtime call 15 of
   `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12`, the
   original kernel inputs (residual, attention output, and weight) are kept
   fixed.  An eager post-attention RMSNorm is evaluated on those exact inputs.
   The generated kernel's post-norm output differs reproducibly from that
   same-input reference.  The local discrepancy is small and affects 22 output
   elements in this state.

2. **Kernel-only endpoint mediation.**  Replacing only that kernel call's
   post-norm output with the same-input eager reference changes the continuous
   scorer, but does not change the clipping decision at the selected fork.
   Therefore this kernel is a demonstrated numerical producer and a
   context-dependent continuous mediator, but not a demonstrated source of the
   observed clipping fork.

3. **Region-level mediation.**  Under the same eager layer-14 entry boundary,
   replacing the attention-side region does not change the clipping decision.
   Replacing the layer-15 MLP/layer-exit boundary does change it (`on → off` at
   the selected coordinate).  This is valid fixed-suffix mediation for a
   composite MLP/layer-exit region, not a unique op or kernel attribution.

## Claim boundary

The strongest current claim is:

> In this state and forward suffix context, the declared generated kernel has
> a reproducible same-input numerical discrepancy, while the layer-15 MLP
> contextual region mediates the observed clipping fork.  The kernel-only
> repair does not mediate that fork.

This does not identify a compiler pass, ATen source line, or unique root cause.
It is implementation-relative evidence; eager is not treated as mathematical
truth.  The result is one state, one compiler realization, and one forward
endpoint.  It must not be generalized to training-wide bias or operator
importance without additional matched states and contexts.

## Why this slice is useful

It separates three questions that a single repair or a maximum delta cannot
separate:

```text
same-input local production
        + fixed-suffix mediation
        + provenance/context invariance
        = auditable operator/kernel evidence
```

The next valid extension is to repeat this exact pair of tests over independent
matched states and additional kernel calls.  If the kernel-only mediation stays
null while the MLP mediation persists, the appropriate conclusion is a
propagation/boundary effect rather than a kernel root-cause claim.
