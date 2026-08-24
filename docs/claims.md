# Scientific claim ledger

Evidence snapshot: current organized mainline (2026-08-23). The historical
stateless-SGD headline contains three operator/backward persistence records.
Under the corrected common-AdamW comparison, Liger and Phi remain locally
persistent, Qwen does not, and one result-blind sampled Phi row is a
small-margin positive. Historical six- and eight-row registries are retained
only as audit history.

This file is organized by paper claim, not experiment chronology.  Each row
states the evidence threshold, the authoritative artifact, and what may be
claimed now.  `UNRESOLVED` and `NOT_APPLICABLE` are never converted into
negative labels.

## Status vocabulary

- `SUPPORTED`: the stated threshold is met under the declared scope.
- `BOUNDED`: useful evidence exists, but a stronger or broader wording is not
  supported.
- `OPEN`: the required evidence has not been collected.
- `NOT_APPLICABLE`: the implementation has no nonzero, parameter-reachable
  contrast under the declared boundary.

## Historical counting resolution

The historical stateless-SGD headline is three cases. The corrected AdamW
comparison has its own measured labels and must not inherit that count. The
counts below explain old audit artifacts and must not be used as alternate
headline counts.

The repository contains several legitimate counts with different denominators:

| count | meaning | authoritative artifact |
|---:|---|---|
| 6 | project-generated strict Flash-style cases passing complete F+B, causal repair, real carrier, and paired trajectory | `results/coverage/existing_case_reaudit.json` |
| 8 | unique closed F+B paired-separation records in the Bias Formation roster | `case.md` |
| 7/8 | those roster records with ordered-trajectory directional persistence, including feedback-sustained persistence | `case.md` |
| 6/8 | those roster records with a matched formation mechanism | `case.md` |
| 4/8 | those roster records whose formation and persistence use the same contrast or declared closed semantic-region superset | `case.md` |

Therefore the historical `4/8 -> 6` change was not a gate relaxation or a promotion. `4/8` is the
same-contrast subset of one eight-record formation roster.  `6` is the
project-wide strict Flash-style registry; it includes Qwen `lm_head dX`, which
is not in that formation roster, and it admits two causally closed semantic
regions while keeping them ineligible for single-kernel property claims.

The six historical records are Qwen seq128 `lm_head dX`, Liger fused CE, Phi-4 seq64
`lm_head dX`, Mamba seq64 `in_proj`, Qwen layer-27 saved-P, and the layer-23
q-projection semantic region.  Gemma-4 feedback drift and Qwen3-VL SiLU
feedback persistence are real trajectory phenomena but are not additional
Flash-style persistent-local-source cases.

## Method and coverage claims

| claim | evidence threshold | supporting artifact | status |
|---|---|---|---|
| Every actual invocation remains in the denominator. | Actual forward/backward invocations, unresolved rows, and fused many-to-one bindings are retained rather than sampled away. | `docs/denominator.md`; `results/coverage/cases/full_coordinate_audit.json.gz` | `SUPPORTED` for the frozen four-model × three-shape discovery scope: 1,562/1,562 directional endpoints have a T1 disposition. This is not an all-model or all-backend claim. |
| One forward plus its actual backward is the mathematical unit. | Exact saved values, cotangent, output edge, shapes/dtypes, non-tensor arguments, and executed VJP arithmetic are bound. | `docs/method.md`; `results/coverage/cases/directional_candidate_math_registry.json.gz` | `SUPPORTED` for the frozen registry. Analytic closure does not substitute for T1–T4 numerical evidence. |
| Repeated layers may be represented by one deep measurement without losing coverage. | Exact callsite/ABI, tile/reduction topology, fusion semantics, routing behavior, and repair boundary are identical; all sibling invocations stay in the denominator. | `docs/tcmp_progress.md` (“Counting rule”) | `SUPPORTED_WITH_SCOPE`. Report `deeply_tested` and `represented_by_same_pattern` separately. The Ministral 26-layer region is one repeated implementation pattern; Qwen layer-23 is a different closed semantic region and remains an independent case. |
| Screening has measured recall. | A random screen-negative sample must contain residual-nonzero, parameter-reachable rows and receive the complete downstream protocol. | `results/property/joint_bias_formation_v1/screen_negative_control_audit.json`; `results/property/joint_bias_formation_v1/consequence_summary.json` | `SUPPORTED_FOR_THE_FROZEN_12_ROW_SAMPLE`. All 12 residual-nonzero, parameter-reachable rows completed 32-step consequence. Eleven have diffusive local increments with feedback-sustained actual drift; one is mixed. This estimates behavior in the frozen sample, not universal recall. |
| Matched repair isolates the declared F+B boundary. | Candidate/repair share state; repair is type/ABI compatible; sham and non-target controls pass; the declared carrier is reached. | `docs/bias_protocol.md`; per-case gates in `results/coverage/existing_case_reaudit.json` | `SUPPORTED` for the six strict cases and explicitly named bounded controls. It is not inherited by sibling endpoints without a bound repair. |

## Measurement claims

| claim | evidence threshold | supporting artifact | status |
|---|---|---|---|
| Temporal amplification `A_X(T)` measures drift relative to diffusive energy. | Report prefix curves and lag-resolved inner products; calibrate against sign-flip or structure-matched nulls. | `docs/persistence_property_protocol.md`; `src/kernel_analyzer/short_persistence.py` | `SUPPORTED` as a continuous statistic and screening feature. `A=2` is historical engineering guidance, not a universal scientific constant. |
| Horizon comparisons are valid. | Use the same horizon or compare each path to its horizon-matched empirical null; do not compare raw `A` values across different ceilings. | strict 32-step trajectories; `results/property/bias_property_search/short_screen_protocol_v3.json` | `BOUNDED`. Strict cases use 32 steps; the low-cost Oracle uses 16-step prefixes and its own null. Cross-horizon raw magnitudes must not be ranked directly. |
| The orbit-mean estimator is not biased toward the plug-in mean. | Default schedule held out; eight non-default variants split 4+4; cross-half statistics; FP64 mathematical target; replay-time operand permutation only. | `src/kernel_analyzer/reduction_orbit.py`; `src/kernel_analyzer/persistence_property.py`; `tests/test_persistence_property.py` | `SUPPORTED` for the corrected implementation. Llama/Ministral corrected values are retrospective because the first prospective pilot used a nonconforming orbit roster. |
| Short-lag persistence is measured, not inferred from final `A`. | Positive lag-1 plus at least two positive lags, prefix growth, and sign-flip-null exceedance. | `src/kernel_analyzer/short_persistence.py`; `results/property/bias_property_search/short_screen_protocol_v3.json` | `SUPPORTED` for Oracle v3. Null-like output means “no escalation under this screen,” never SAFE. |

## Case and taxonomy claims

| claim | evidence threshold | supporting artifact | status |
|---|---|---|---|
| Six historical M0--M6 project records exist. | Complete F+B, causal repair/sham, real carrier, and paired directional trajectory. | `results/coverage/existing_case_reaudit.json`; `case.md` | `SUPPORTED_AS_AUDIT_HISTORY`: four root-arithmetic records plus two closed semantic-region records. Only three enter the historical stateless-SGD operator/backward headline. |
| Three bounded records support the historical stateless-SGD direct operator/backward persistence headline. | A valid one-operator repair reaches a real parameter and the resulting SGD update difference persists on an ordered trajectory. | `results/property/joint_bias_formation_v1/source_persistence_reclassification.json`; `results/property/joint_bias_formation_v1/headline_case_evidence_scope_v1.json`; `case.md` | `SUPPORTED_WITH_SCOPE`: Liger fused CE, Phi `lm_head dX`, and the Qwen `lm_head dX` family. The new exact seq128 Qwen rerun closes its output/gradient/update/live-run identity under AdamW and finds canceling direct AdamW updates. All three historical rows now have an AdamW direct/feedback/actual split; do not transfer the historical SGD label across optimizers. |
| Source-persistent and feedback-sustained drift are distinct regimes. | Four-arm local/feedback recurrence closes; optimizer ablation changes feedback without relabeling the source. | Gemma section of `docs/tcmp_progress.md`; `results/property/bias_property_search/heldout_validation_v3_gemma_disjoint.json` | `SUPPORTED_CASE_LEVEL`. Gemma local is null-like while feedback/actual persist; stateless SGD and moment reset collapse the feedback. A general feedback predictor is still `OPEN`. |
| Negative evidence is nontrivial and correctly classified. | Nonzero local residual, parameter reach, complete F+B, and centered/canceling gradient/update population. | `results/property/tcmp_allop_v1/final_decision_matrix.json`; `results/property/tcmp_allop_v1/scope_extension_20260822.json` | `SUPPORTED`. YaRN exact-zero and parameter-inaccessible softmax/mask regions are `NOT_APPLICABLE`; compiler/provenance failures are `UNRESOLVED`; neither group is counted as a negative. |
| Error magnitude alone does not predict source persistence. | Compare backward-visible residuals with different persistence outcomes under a common statistic and control boundary. | `results/property/joint_bias_formation_v1/rms_persistence/rms_persistence.json` | `SUPPORTED_FOR_FORMATION_POPULATION`. Across 32 reachable nonzero rows, Pearson is `0.018` (`p=0.921`) and Spearman is `0.243` (`p=0.178`) under 10,000 two-sided permutations. RMS explains essentially none of the linear variation, and neither association is distinguishable from zero at this sample size. |

## Mechanism and causal claims

| claim | evidence threshold | supporting artifact | status |
|---|---|---|---|
| Repeated drift is not explained by a single random kick. | Repeated, support/norm-matched perturbations at every step, multiple frozen seeds, and drift-subspace analysis. | `results/property/tcmp_allop_v1/repeated_orbit_null_summary.json`; `docs/tcmp_progress.md` | `SUPPORTED`, but the result is not source-specific: repeated orbit variants retain 93.4–101.8% of natural drift with high cosine and effective rank 1.28/1.57. It supports low-dimensional closed-loop feedback and rejects “fixed schedule alone is the anchor.” |
| Phi source persistence exceeds a repeated diffusive null. | Inject a per-step support/RMS-matched sign-scrambled local effective update while retaining closed-loop feedback, using multiple frozen seeds. | `results/property/joint_bias_formation_v1/four_scale_arms/phi_repeated_random_null.json` | `SUPPORTED_ON_FINAL_NORM_CARRIER`: natural common-state local `A=4.701`; five random arms span `0.870--1.037` with mean `0.959`. |
| Phi persistence is carrier-selective. | Repeat the identical four-arm protocol on a second reachable parameter carrier. | `results/property/joint_bias_formation_v1/four_scale_arms/phi_layer26_post_attention_norm.json` | `SUPPORTED_CASE_LEVEL`: final norm has operator `A=4.488`, while layer-26 post-attention norm has `A=1.114`. The endpoint does not impose equal persistence on all reachable parameters. |
| The final drift direction is operator-specific. | Natural drift must separate from multiple matched perturbation directions beyond the shared low-dimensional feedback subspace. | same repeated-orbit artifacts | `NOT_SUPPORTED`. The five-seed cosine/effective-rank result shows strong shared structure; current evidence does not isolate an operator-unique direction. |
| Liger persistence is caused by a step-invariant rounding/schedule anchor rather than error magnitude. | On the real kernel, preserve or increase error energy while removing rounding bias (RN→SR), and independently randomize chunk/order across steps. | `results/property/joint_bias_formation_v1/liger_chunk_host_certificate.json`; `results/property/joint_bias_formation_v1/liger_sr_intervention.json` | `BOUNDED`. The real 24-state chunk-geometry intervention confirms BF16 24/24 directional versus FP32 13/11 and RMS ratio 2.63384. The separate 16-state RN→SR default-residual run is `INCONCLUSIVE_NO_POSITIVE_BASELINE`: RN `A=0.942` is already diffusive, while SR is `0.975/1.001`. It cannot establish suppression of a confirmed persistent baseline. |
| Feedback is a separate causal channel. | Local source is null-like; feedback recurrence is persistent; stateless/moment-reset optimizer controls remove it. | Gemma artifacts in `docs/tcmp_progress.md` | `SUPPORTED_CASE_LEVEL`. Predictor F is not frozen; feedback remains outside the source Oracle’s domain. |
| OLMoE routing flips do not contaminate its control. | Record route indices/weights in both arms for every measured state. | OLMoE section of `docs/tcmp_progress.md` | `SUPPORTED`: routing indices and weights matched in all 26 states. The result is a backward-visible canceling control, not a case. |

## Prediction and Oracle claims

| claim | evidence threshold | supporting artifact | status |
|---|---|---|---|
| Transported conditional mean predicts persistence. | Predictor frozen before consequence, protocol-conforming orbit estimate, and disjoint states. | Llama/Ministral records in `docs/tcmp_progress.md`; `results/property/tcmp_allop_v1/final_decision_matrix.json` | `BOUNDED`: prediction ordering was prospective, but the first orbit roster deviated from the frozen 4+4 protocol; corrected values are retrospective. Evidence supports one `lm_head dX` family on new operands, not confirmatory cross-implementation generalization. |
| Held-out axes are reported honestly. | Label each row `SEEN_IMPL / NEW_OPERANDS` or `NEW_IMPL`. | `docs/heldout_property_validation.md`; `docs/heldout_property_validation_v3.md` | `SUPPORTED`. Llama/Ministral are `SEEN_IMPL / NEW_OPERANDS`; Gemma is `NEW_IMPL` but source-negative and feedback-sustained. New-implementation source-positive count is zero. |
| The low-cost Oracle safely triages exact trajectories. | Fixed 256-dimensional CountSketch, empirical null, fail-closed selector, and exact confirmation only for risks. | `docs/short_persistence_oracle_v3.md`; `scripts/select_short_screen_escalations.py`; `results/property/bias_property_search/gemma_v3_disjoint_escalation_manifest.json` | `SUPPORTED_AS_WORKFLOW`. It is not yet an accuracy or safety certificate. |
| The Oracle has measured screening behavior. | Report flagged fraction, eligible denominator, exact-confirmation recall, and false escalations under one optimizer. | `results/property/joint_bias_formation_v1/oracle_repair_v3/same_optimizer_oracle_v3.json`; `docs/oracle_repair_v3.md` | `SUPPORTED_FOR_RETROSPECTIVE_15_ROWS`: all rows use AdamW; 5/15 are flagged, recall is 3/3, false positives are 2/12, and precision is 3/5. The old 14-row metrics are withdrawn because they mixed optimizers and dropped a sampled mixed-positive row. |
| The predictor beats a same-level energy baseline. | Compare 16-step local-update directionality with 16-step local-update RMS under the same AdamW protocol. | same corrected 15-row comparison | `SUPPORTED_FOR_RETROSPECTIVE_15_ROWS`: prefix directionality AUROC `0.944`; prefix local-update RMS AUROC `0.528`. This is not unseen-implementation accuracy. |
| Four candidate properties were tested. | Positive and centered-control measurements under the same property definition. | `development_property_profile.json`; `development_property_separation_audit_v1.json`; `property_freeze_v1.json` under `results/property/bias_property_search/` | `SUPPORTED_WITH_DECISIONS`: source asymmetry separates 3 known rows from 5 centered controls and is retained as a conditional prior; source–transport has one positive and no valid controls, so remains case-level; concentration overlaps controls and is supporting-only; carrier stability has no measured centered-control trajectories and is only a consequence screen. |
| A universal all-operator property has been established. | Prospective positives and residual-nonzero negatives across unseen implementation classes, with frozen thresholds and baselines. | `docs/final_conclusion.md`; `docs/bias_property_search_completion.md` | `NOT_SUPPORTED`. The correct result is a family-scoped transported-mean hypothesis plus a fail-closed persistence screen and a separate feedback taxonomy. |

## Claims that are safe to write now

1. Kernel Analyzer provides an auditable F+B/repair/trajectory chain for a
   frozen multi-model denominator and retains unresolved rows.
2. Six project cases pass the historical strict F+B/repair/carrier/trajectory
   registry. Three support the historical stateless-SGD headline. Under one
   common AdamW protocol, two of those three remain locally persistent and one
   result-blind sampled row is a small-margin positive. Feedback-only,
   mismatch, and unresolved rows are reported separately.
3. Backward-visible implementation residuals often cancel; nonzero local error
   and low-dimensional concentration are insufficient by themselves.
4. A short sequence of local parameter-update differences is a useful,
   fail-closed persistence screen on the corrected 15-row same-AdamW
   evaluation set.
5. Feedback-sustained drift is a distinct case-level regime demonstrated by
   Gemma’s Adam-state intervention.
6. The short Oracle is a fail-closed prioritization workflow, not a safety
   certificate or a validated universal classifier.

## Optional evidence for stronger future claims

The current bounded paper claim does not require the items below. They are
needed only to claim cross-implementation generalization, complete tolerance
coverage, or a deployable automatic repair workflow.

1. Complete wall-clock cost accounting after the Mamba and uncontended
   headline timing runs.
2. Phi now has an executable source intervention under the same cold-start
   AdamW protocol as its direct-persistence result. Deterministic BF16 gives
   `A=1.02959`; four real stochastic-rounding repeats give `A=1.00045`,
   `1.00004`, `1.00005`, and `1.00182`, all inside their own sign-flip nulls.
   The no-op sham exactly reproduces the natural arm. The earlier stateless-SGD
   result remains separate evidence rather than support borrowed across
   protocols. Exact per-step update-energy matching remains optional; the
   first three AdamW SR arms already have path energy close to or above the
   natural arm. The Liger RN arm is diffusive and cannot support a suppression
   claim.
3. A prospective, protocol-conforming `NEW_IMPL` source-positive confirmation,
   or an explicit decision to publish the narrower family-scoped result.
4. Full-parameter RNG/data-order/precision scale controls.  The current Phi
   four-arm campaign is explicitly carrier-scale.

## Three-factor formation follow-up

The current follow-up keeps the operator-output error, backward/optimizer
response, and later training-state propagation as separate steps. The auditable
inventory is in `results/property/joint_bias_formation_v1/`.  It reports ten
records: eight formal v2.1 roster entries plus two previously measured
exploratory semantic records. Saved-P and Qwen3-VL SiLU have exact
positive/negative error-response replays; Phi has a case-level backward-pairing
intervention; Liger has a source-event feasibility screen.  Missing raw
`epsilon`/`+epsilon`/`-epsilon` vectors are explicitly `UNRESOLVED`.

The synthetic parity tests pass (`41` targeted tests total).  A real 16-step
stateless-SGD Phi SR intervention is now complete: RN has amplification `3.325`, while four
SR repeats have mean amplification `0.956` and carrier capture below `0.02`.
This Phi run has a confirmed coherent RN baseline; it must not be conflated
with the separate Liger RN→SR screen below.
The endpoint noise is not smaller, but effective-update energy is not exactly
norm-matched, so this is causal source evidence with an explicit boundary,
not a perfect matched-norm proof.  A host-GPU 32-state Liger formation rerun
also completed; its calibration local population was directional but the
confirmation population stayed unresolved under the frozen gate.  Separately,
the real 24-state Liger chunk-geometry intervention confirmed BF16 `24/24`
same-sign projections versus FP32 `13/11` and a BF16/FP32 residual-RMS ratio
of `2.63384`.  A separate real 16-state RN→SR run on the default residual was
completed: RN `A=0.9419`, SR repeats `0.9750/1.0010`.  Because the RN arm is
already at the diffusive boundary on this state/contrast split, this is
`INCONCLUSIVE_NO_POSITIVE_BASELINE`, not a failed suppression result.  It
cannot be compared to a historical Liger amplification unless horizon, state
bank, endpoint contrast, and denominator are proven identical.  Existing Phi
schedule randomization reduces amplification from
`3.3253` to `3.1123` (ratio `0.936`), so schedule changes alone do not remove
persistence.  A half-learning-rate Phi trajectory still passes SEUP with
local accumulation `2.37477e-5`, feedback `2.04859e-6`, and recurrence error
`1.01e-8`; this is a propagation sensitivity result, not a pure fixed-update
intervention.  A fixed real 16-step Phi error sequence injected into an
alternate checkpoint gives drift/direct-sum ratio `1.0104`, with no extra
feedback amplification in that probe.  This is a positive propagation-closure
result: direct accumulated error predicts the observed drift to within 1.04%.
The positive/negative error-response decomposition is complete for the two
cases with exact replay artifacts. Both saved-P and SiLU contain material
symmetric and sign-changing contributions rather than a single dominant channel; SiLU's symmetric-response
energy is almost entirely concentrated in its first two steps.  The 16→32
backtest and the 12-row full consequence audit are also complete.  Held-out
Gemma source-negative confirmation is complete. The generic three-part
predictor was evaluated for input eligibility on five cases; none had all
required inputs, so it correctly abstained on all five and has no accuracy
claim.

The same-optimizer Oracle correction is now complete. All 15 rows use AdamW;
16-step directionality has AUROC `0.944`, while 16-step local-update RMS has
AUROC `0.528`. Liger remains directly persistent (`A=1.720`), Phi is a
small-margin direct positive (`A=1.029`), and exact Qwen seq128 direct updates
cancel (`A=0.957`) even though its gradient differences are more aligned
(`A=1.343`). Liger, Phi, and Qwen now all have 32-step direct/feedback/actual
exports. In a separate 16-state stateless-SGD experiment, the exact Phi
norm-matched analysis gives mean stochastic `A=0.984` at the deterministic
arm's per-step update norm. It is not the intervention for the AdamW
`A=1.029` result.

The v4 reanalysis now calls the short workflow the `Cold-start AdamW Direct
Persistence Screen`. It applies Holm as the primary correction separately to
the three predeclared rows and the twelve result-blind rows. The small-margin
`0543` row is retained as `UNRESOLVED_CANDIDATE`, not relabeled negative. The
signed direct/feedback/actual contribution table is derived from exported
resultants and explicitly records that complete per-step cross-Grams were not
saved. Four same-state optimizer ablations, Qwen early/middle/late natural
phase response, and three fresh Gemma target checks are complete. The fresh
pool has no direct positive, so prospective recall and catch-and-fix remain
undefined or not applicable. The v4 package is not a universal safety
classifier. The identity-complete v4.1 entry is frozen but intentionally not
started for the current bounded paper claim.

The optimizer boundary is explicit: current data show that AdamW can suppress
the Phi gradient directionality and that optimizer state can maintain Gemma
feedback, but they do not identify the optimizer as the universal root cause.
See `docs/direct_persistence_optimizer.md`.
