# Backward-sensitive F+B rescreen

## Conclusion

The historical T1–T4 campaign executed backward graphs, but its selection
funnel underrepresented backward-formed bias. T1 required a directionally
coherent local endpoint error before downstream causal/carrier analysis. It
therefore excluded the transition

`LOCAL_CENTERED -> PARAMETER_GRADIENT_BIASED`

by construction. Phi `lm_head dX` is an existing strict example of this
transition.

The experimental unit remains one complete forward plus its actual backward.
No backward node is tested in isolation: candidate and exact endpoint repair
start from the same forward state, then compare the complete declared
parameter-gradient carrier and stateless effective update.

## Denominator and scale

- Historical exact tasks: 51,493 forward and 43,379 backward.
- New four-model, three-shape semantic inventory: 41,324 backward rows.
- Executable exact endpoints: 29,528.
- The first leaf-only reduction produced 724 cells, but this was not the full
  F+B semantic denominator.
- Corrected compiler-bound semantic equivalence cells: 791. After choosing an
  exact public endpoint whenever it represents the same F+B semantic cell, 67
  representatives require a semantic-region boundary.
- The previous 699/724 carrier figure therefore applies only to the leaf-only
  subset and must not be reported as full backward coverage.
- Seq64 dynamic carrier reach: Qwen 68/68, Phi 36/36, DeepSeek 64/64,
  Mamba 56/66.
- Seq128 dynamic reach completed so far: Qwen 68/68, Phi 36/36,
  DeepSeek 64/64.

A later carrier audit found another coverage bug in the Mamba denominator.
Twenty-four bias-gradient cells (eight per shape) were marked unresolved because
the binder combined a leaf `dt_proj`/`conv1d` module with its parent `mixer` and
treated same-shaped parameters as ambiguous. Selecting the deepest exact module
stack binds these cells to the actual `dt_proj.bias` or `conv1d.bias`. The
corrected static graph binds 790/791 cells; the remaining Phi seq256 `lm_head
dX` crosses an AOT graph-break boundary and is measured dynamically at the
known downstream `model.norm.weight` carrier. It is retained in the denominator
rather than silently counted as a negative.

The rescreen fixes three scales that could hide a case:

1. local directionality is no longer an entry gate;
2. complete semantic regions are retained alongside atomic endpoints;
3. the measured carrier is an exact reachable parameter gradient, stratified
   by sequence shape rather than inferred from a similar name or tensor shape.

## Sensitivity failure found in the first rescreen

The known Phi-4 seq64 `lm_head dX` positive is `backward:497:output_0`. Its
compiler-bound internal root is `backward_g0__mm_2`, but the generic task has
no public `exact_aot_endpoint_id`. The first rescreen therefore discarded it
before measurement and instead screened `backward:496`, a different CE edge.

This is not a statistical-threshold problem. It is a boundary-scale problem:
an exact compiler-bound semantic region can be executable and causally closed
even when its internal buffer is not exposed as a leaf endpoint. After fixing
the denominator, all 41,324 backward rows are represented by 791 cells. The 67
remaining semantic-region representatives concentrate in loss/CE,
normalization, and Mamba recurrent backward. Of these, eight have a directly
measurable internal port; the other 59 have an exact downstream semantic
closure, so their survival into training gradients is screened there without
pretending that the downstream endpoint localizes the internal arithmetic.

Consequently, the completed leaf screens remain valid for their measured
endpoints, but their negative result alone could not exclude these internal
regions. The corrected pass therefore recovered the Phi anchor, measured the
eight executable internal ports, and audits the 59 exact downstream closures
across Qwen, Phi, DeepSeek, and Mamba.

## Semantic-region sensitivity and first results

The corrected capture now passes the required sensitivity control. For Phi-4
seq64 `backward:497`, a one-state repair changed 174,895 coordinates of the
compiler-bound `mm_2` output and reached `model.norm.weight`. In the four-state
screen, the local endpoint remained centered (cross-state ratio 0.0031), while
the parameter-gradient ratio was 0.1181. This reproduces the expected
`LOCAL_CENTERED -> GRADIENT_DIRECTIONAL` screening signature without using the
old T1/T4 verdict as a label.

The audit also separated two kinds of internal region that must not be
conflated:

1. Phi `mm_2` is a one-to-one analytic VJP output and supports exact AOT replay.
2. Qwen/DeepSeek normalization `out_ptr0` is an FP32 split-reduction buffer,
   not the final BF16 AOT reduction output. Its mathematical partial-sum
   schedule was reconstructed from the bound `[token, head, feature]` input;
   directly reshaping the final AOT tensor would have been an invalid repair.

Across Qwen and DeepSeek, 18 such q/k-normalization regions were screened at
seq64/128/256. They contain real FP32 reduction-order differences, but the
following reduction and BF16 writeback erased the perturbation: all tested
parameter-gradient deltas were exactly zero. This supplies a concrete
variance-only mechanism: an internal arithmetic difference that does not cross
the next representation boundary.

Compiler-added loss-gradient external calls were also restored to the
denominator. Qwen `addmm` and Mamba loss-head MM both reach their tied embedding
gradient, but their four-state directions are centered. Qwen results remain
centered at seq64, seq128, and seq256; the largest short-screen ratio was
0.0107 at seq256. Mamba seq64 and seq128 are likewise centered. Phi seq128
`lm_head dX` is centered in the same four-state screen, unlike the known seq64
case, demonstrating that reduction geometry/shape is part of the mechanism
rather than operator name alone.

The previously unbound Phi seq256 `lm_head dX` initially showed the desired
short-screen signature (local ratio 0.0017, gradient ratio 0.2111), but failed
the frozen 16+16 confirmation: calibration gradient ratio was -0.0205 and the
disjoint confirmation ratio was 0.0979. It is not a new case. Across the three
tested shapes, only seq64 has stable transport-formed bias.

## Confirmed results so far

- Qwen seq64: 20 four-state candidates; no independent 16+16 bias case.
- Phi seq64: four meaningful candidates; no new stable case. The specialized
  known Phi `lm_head dX` case remains valid and is not equivalent to the generic
  CE regions tested here.
- DeepSeek seq64: six meaningful candidates; no independent 16+16 bias case.
- Mamba seq64: eight candidates; no independent 16+16 bias case. The strongest
  recurrent candidate changed from gradient ratio 0.194 in calibration to
  -0.056 in confirmation while its local residual stayed centered.
- Qwen seq128: four meaningful candidates completed 16+16; none formed a
  confirmed bias stage. The strongest short-screen transport ratios (0.425 and
  0.317) fell to ambiguous/centered confirmation results.
- Phi seq128: six meaningful candidates completed 16+16; none formed a
  confirmed bias stage. One attention-state candidate remained suggestive only
  in the confirmation half and was not stable across calibration.
- DeepSeek seq128 typed-FP32 attention-softmax candidate: exact ABI-visible
  repair had zero parameter-gradient reach, so it is not a causal training-bias
  case.

These negative confirmations show that the old funnel had a real structural
blind spot, but do not imply that every omitted backward endpoint contains
bias. Four-state directionality is often a state-sampling false positive.

## Resource and provenance policy

Only identity checks needed for the frozen release, input bank, exact endpoint,
common state, and carrier are retained. Temporary vector, loss, and result
SHA256 records are not generated. Full vectors are written under `/data1`,
reduced once to population Gram statistics, and deleted. Low-score loss/head
controls that duplicated existing strict evidence were stopped before costly
embedding-vector summarization.

## Latest coverage checkpoint

The current denominator audit contains 791 compiler-bound F+B cells. Of these,
727 exact representatives have a measured local/gradient screen and 40 further
semantic regions have a measured exact downstream-closure screen. The remaining
24 rows are explicitly `SEMANTIC_REGION_PENDING`; they are not treated as
centered, negative, or complete. They are concentrated in Mamba recurrent
backward and tied-embedding loss-head regions whose exact internal port is not
yet executable in the current release. The closure audit itself has 59
semantic-region entries: 40 screened through an exact downstream closure and 19
still blocked at the internal-boundary level. This is an honest boundary gap,
not a dropped denominator.

Additional completed checks since the previous checkpoint:

- Mamba seq64: 10 newly-bound direct candidates, all centered or unresolved;
- Mamba seq128: five strict 16+16 candidates, all centered in confirmation or
  unresolved, with no confirmed first bias stage;
- Mamba seq256: 69 four-state direct screens and eight newly-bound direct
  screens. High four-state transport ratios did not yet establish a case;
- DeepSeek seq256: both pending exact q-projection cells were centered;
- Phi seq256 `lm_head dX`: the specialized 16+16 confirmation remained
  non-replicating (calibration centered, confirmation unresolved).

The Qwen seq64, seq128, and seq256 CE/lm-head semantic closures were then
completed with independent 16+16 open-loop populations. All three are CENTERED
at all three formation layers in both partitions, so they close three pending
loss-path rows without adding a case.

Across the 727 direct/semantic screens, the candidate map reports 69 short-screen
signals but zero new strict confirmations. The only strict formation case
currently remains the previously established Phi seq64 `lm_head dX`
(`LOCAL_CENTERED -> PARAMETER_GRADIENT_BIASED`). No short-screen ratio is
counted as a case.

The targeted Mamba seq256 top-three confirmation was attempted under the same
16+16 open-loop F+B protocol. The original attempt was rejected before
measurement because its stale release failed the wrapper/graph provenance
gate. A matching `mamba_seq256_r2` release was rebuilt; a one-state
engineering preflight then passed all three exact endpoint and reference-cut
gates. The subsequent formal run completed state 0 but was stopped after the
sequential Mamba fallback demonstrated roughly 20 minutes per additional
state with negligible GPU utilization. It is recorded as an execution
blocker/partial run, not as a negative or positive scientific result. A new
case is accepted only after its complete F+B boundary, formation stage, and
disjoint confirmation are closed; short-screen scores never count as cases.

The search direction is now frozen in
[`bias_formation_search_matrix.md`](bias_formation_search_matrix.md): loss and
attention/normalization backward semantic bottlenecks are prioritized, with
source-directional, transport-amplified, optimizer, and variance-only
transitions kept as separate hypotheses. This prevents the Phi anchor from
being mistaken for a complete oracle.
