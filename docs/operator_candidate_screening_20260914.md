# Operator-family candidate screening (2026-09-14)

This note records the first small screening pass for the current objective:
find a real implementation difference, confirm a reproducible bias, isolate one
computational choice, and only then consider a training run. Positions from the
same problem group are not counted as separate problems.

## Decision rule

The primary screen uses the common-input local replacement protocol and reports
local output, parameter gradient, and actual parameter write. A candidate is not
advanced merely because its write RMS is large: an advance requires a reproducible
mean direction or repair-aligned scaling on held-out states, together with a
source intervention whose prediction is fixed before the intervention run.

## Screening record

| Problem group | Candidate and intervention | Evidence | Decision |
|---|---|---|---|
| Reduction / accumulation | Liger fused linear-CE chunk addition; same-FP32 order change | The additive and residual profile is reproducible, but the update effect is about `3e-9` of the repair RMS. | Keep as same-precision arithmetic mechanism; not a training candidate. |
| Reduction / accumulation | Mamba exponential-weighted reduction; reverse FP32 feature order | Common-input capture is complete, but update effect is about `7e-9` relative and direction is not reproduced. | Reject for the current search. |
| Normalization | Gemma RMS backward; common-input reference | Write RMS reaches about `2.24%` and `3.83%` in the two selected locations, but all three direction views fail to reproduce a direction. | Reject as a bias candidate; retain as energy/state evidence. |
| Softmax backward | DeepSeek softmax backward; common-input reference | Complete 36-location family records have local and gradient effects below about `2.4e-4` relative in the inspected case; no material direction. | Reject for the current search. |
| GELU backward | Gemma fused tanh-GELU product; native tanh reference | Development case `952`: gradient RMS about `0.52%`, repair-aligned gradient scaling about `-0.038%`; cold-start write RMS about `5.03%`. A disjoint 32-state confirmation gives gradient RMS about `0.10%` and aligned scaling about `-0.0012%`, while write RMS remains about `4.34%`. | Reject as a stable natural bias; write effect is predominantly optimizer-state response. |

## GELU single-factor source interventions

The candidate is a real generated Triton backward product. On identical declared
operands, the following references changed only the indicated arithmetic choice:

* native `tanh` with the source expression order;
* explicit exponential reconstruction of `tanh`;
* native `tanh` with fused multiply-add expressions for the cubic argument and
  derivative polynomial.

The explicit exponential version changed the development gradient aligned
scaling to about `-1.97%`, but on the disjoint confirmation states the change was
only about `-0.0025` percentage points relative to native tanh. The fused
multiply-add version was bitwise indistinguishable from the native reference in
the measured profile. Therefore the experiment demonstrates state-sensitive
response to a source choice, but does not identify a stable GELU root bias.

The capture outputs are retained under:

* `results/property/numerical_coverage_v1/gemma_gelu_tanh_exp_intervention_v3/`
* `results/property/numerical_coverage_v1/gemma_gelu_tanh_native_confirmation_v3/`
* `results/property/numerical_coverage_v1/gemma_gelu_tanh_exp_confirmation_v3/`
* `results/property/numerical_coverage_v1/gemma_gelu_fma_confirmation_v1/`

The confirmation state bank is disjoint from the original GELU development
bank. These are fixed-suite results, not population prevalence or training-loss
claims.

## Consequence for the search

The first three requested families did not yield a new, non-micro, reproducible
same-precision natural bias in this pass. The strongest already-closed case in
the repository remains the AdamW8bit moment-residual problem, which has a
separate causal and training-outcome record. The next search should therefore
move to a new problem group or deepen a source-isolating comparison for an
existing group; it should not add more model positions to the rejected GELU,
RMS, or softmax candidates.

## RMS order intervention follow-up

The Gemma RMS pair was then rerun with `FP32_NATIVE` versus
`FP32_REVERSE_FEATURE_ORDER` on the real compiled graph. Each endpoint was observed
separately because both cases share one carrier parameter. Across 32 states,
`forward:116:out_ptr0` showed only `5.3948e-6` direct endpoint RMS and no confirmed
directional structure; `forward:200:out_ptr0` was unchanged (direct RMS `0`). A larger
effect seen when both endpoints were replaced together was an upstream region effect,
not evidence for a second RMS root cause. The detailed record is
`docs/gemma_rms_order_intervention_20260914.md`.

## Disjoint same-dtype Liger confirmation

To check whether the order result was specific to the original length-128 bank,
the same predeclared source prediction was run on all 32 length-64 records.
These records are disjoint from the length-128 bank. Both primary
implementations used FP32 inputs, FP32 dW accumulation, and the same Liger
computation; only the order of the 64 chunk additions differed. The forward
loss and hidden-state gradient were bitwise equal for every state, and a
same-order sham reproduced the candidate exactly.

The run completed 32/32 states without execution or non-finite errors. On the
untouched 16-state confirmation half, the predeclared schedule predictor had a
positive projection in 14/16 states. The conditional studentized profile was
confirmed for parameter-gradient additive and residual-direction endpoints,
and for all three AdamW first-step update endpoints:

| stage | estimate on confirmation units | 95% interval | status |
|---|---:|---:|---|
| parameter gradient, additive | `1.8345e-8` | `[3.5657e-9, 3.3124e-8]` | confirmed |
| parameter gradient, residual direction | `1.8227e-8` | `[3.9745e-9, 3.2480e-8]` | confirmed |
| AdamW update, additive | `6.0798e-10` | `[5.3167e-10, 6.8429e-10]` | confirmed |
| AdamW update, aligned | `4.0107e-10` | `[2.3171e-10, 5.7044e-10]` | confirmed |
| AdamW update, residual direction | `4.4054e-10` | `[3.1051e-10, 5.7058e-10]` | confirmed |

This is a disjoint sequence-length confirmation of a real, same-dtype
reduction-order effect. The dW vector is summarized with the predeclared
8192-coordinate sketch, so the intervals are conditional on that measurement
geometry and the declared independent-unit approximation; they are not an
all-coordinate finite-sample guarantee. No long training-loss consequence was
measured for this order-only effect. The machine record is
`results/property/liger_fp32_chunk_order_v1/length64_confirmation.json`,
produced by `scripts/run_liger_fp32_chunk_order_length.py`.

## Length-256 source intervention follow-up

An additional 32-state length-256 run compared the original order with reverse,
even-then-odd, a frozen permutation, and an FP32 Kahan outer-sum variant. The
run completed all states and preserved bitwise-equal forward loss and
hidden-state gradients. Original-versus-reverse again confirmed all three
update profile branches. Even-then-odd removed confirmation of all three
directional update branches, while increasing total update RMS; Kahan reduced
the raw gradient L2 difference but did not reduce update RMS. This is evidence
that the source choice controls the signed update component, not evidence of a
uniformly better repair or of a loss outcome. The full record and scope are in
`docs/liger_fp32_order_intervention_20260915.md`.
