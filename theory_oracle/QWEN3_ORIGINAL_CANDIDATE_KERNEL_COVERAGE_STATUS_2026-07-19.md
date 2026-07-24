# Qwen3 original-candidate generated-kernel coverage status

## Current audited denominator

The frozen forward contains 22 generated treatment families: 20 Triton families
and two external GEMM families (`mm`, `bmm`).  This denominator is distinct from
the 24 high-level semantic-role classes and 536 high-level invocations.

## Current original-candidate evidence

- 20/20 Triton families have valid representative repairs;
- 2/2 external GEMM families have valid shared-path reexecution evidence;
- 58 selected live Triton invocations have valid treatment-integrity evidence;
- 19 repairs have nonzero scorer effects;
- 39 repairs are exact null effects;
- among the 19 nonzero repairs with direction measured, 9 reduce and 10 increase
  whole-eager L2 distance.
- 28 predeclared external reexecutions are exact null effects (six `bmm`, 22
  `mm`).

The twenty partially covered Triton families include:

1. residual plus next-layer input RMSNorm (`3/27`, all nonzero);
2. MLP SiLU plus gate multiplication (`3/28`, all nonzero);
3. mask addition plus safe-softmax (`3/28`, all nonzero);
4. post-attention residual plus RMSNorm (`3/28`, all nonzero);
5. q/o weight FP32-to-FP16 copy (`6/56`, all null);
6. k/v weight FP32-to-FP16 copy (`6/56`, all null);
7. gate/up/down weight FP32-to-FP16 copy (`9/84`, all null).

The fused query RMSNorm+rotary family and fused key RMSNorm+rotary family now
also have valid early/middle/late original-candidate repairs. All six repairs
are nonzero. Query repairs are farther, farther and closer to eager; key repairs
are closer, farther and closer. This position-dependent direction is retained
rather than summarized as a uniformly corrective operator effect.

The key head-repeat/layout/scaling family has valid early/middle/late repairs,
all exact null at the scorer endpoint.

The additional valid families are embedding+input RMSNorm (nonzero and closer
to eager), causal mask construction (null), zero safe-softmax buffer (null),
final RMSNorm `rsqrt` (null), corrected lm-head cast/transpose (null), final
normalization+slice (null), logits cast (null), query clone (three nulls), value
head repeat/layout/cast (three nulls), and attention-output
transpose/cast/materialization (three nulls).

No generated treatment family is yet fully covered because state-distribution
transport has not been validated. All 20 Triton families now have at least one
valid original-candidate representative repair, and both external families
have valid early/middle/late role-aware reexecution evidence. The external
results are reported separately because eager ATen and the generated wrapper
may dispatch to the same CUDA library; they are shared-path evidence, not a
genuinely different arithmetic repair.

One instructive invalid treatment is retained: the first lm-head cast repair
passed runtime gates but ignored the non-contiguous transpose-view storage
mapping and created an artificial enormous effect.  Semantic audit invalidated
it; the corrected treatment respected `destination.t()` and produced an exact
null.  This demonstrates why call execution and anchor restoration alone are
not sufficient treatment-integrity criteria.

## Interpretation

This evidence already falsifies two tempting shortcuts:

- a valid nonzero repair does not necessarily shrink eager--compiled drift;
- sharing the broad label “cast/copy” is not sufficient to merge treatment
  families before checking their roles and contexts.

It improves operator attribution coverage but does not make the full training
operator analysis complete.  Backward, gradient control, AMP, optimizer and
multi-state population coverage remain separate required domains.
