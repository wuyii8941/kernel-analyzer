# Cross-state operation-hypothesis result

Two independent matched states were run through the same operation-hypothesis
slice and independently audited:

- [step236 result](</data1/tzh/forkcert/results/operator_oracle/qwen3_step236_layer16_attention_mlp_slice_v0_3/result.json>) and [audit](</data1/tzh/forkcert/results/operator_oracle/qwen3_step236_layer16_attention_mlp_slice_v0_3/audit.json>);
- [step097 result](</data1/tzh/forkcert/results/operator_oracle/qwen3_step097_layer16_attention_mlp_slice_v0_2/result.json>) and [audit](</data1/tzh/forkcert/results/operator_oracle/qwen3_step097_layer16_attention_mlp_slice_v0_2/audit.json>).

The same runtime call (call 16 of
`triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12`) produced a
local same-input discrepancy and mediated one clipping event in both states.
The direction reversed, matching the earlier call-level result: step236 was
compiled-off to reference-on, while step097 was compiled-on to reference-off.

## Operation hypotheses

| hypothesis | step236 | step097 |
|---|---:|---:|
| `sum_fp32` | mediates 1 event | mediates 1 event |
| `high_precision` | mediates 1 event | mediates 1 event |
| `input_fp16` | mediates 1 event | mediates 1 event |
| `rsqrt_fp64` | mediates 1 event | mediates 0 events |

These are not literal single-instruction kernel replacements; each is a
declared alternate post-norm computation on the captured kernel input. The
table therefore gives mechanism evidence, not a unique operation attribution.

The cross-state result is useful because it rejects a simplistic statement
that “changing rsqrt precision alone always explains the semantic effect.” It
also shows that one state is insufficient to distinguish candidate mechanisms.
At present, `sum_fp32`, `high_precision`, and `input_fp16` remain
non-discriminated candidates; `rsqrt_fp64` is state-dependent and is not a
necessary mediator in the early state.

## Claim boundary

The strongest supported conclusion remains call-level generated-kernel
localization under a fixed suffix and matched-state protocol. We cannot yet
name reduction, cast, rsqrt, or any individual ATen/Triton operation as the
unique root cause. To make that claim credible, the next intervention must
preserve the actual fused kernel stages and vary one stage at a time, or use
additional states/endpoints where the hypotheses make distinct predictions.
