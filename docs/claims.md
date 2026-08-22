# Scientific claim ledger

Evidence snapshot: `75a3fba` (2026-08-22).

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

## Counting resolution

The repository contains several legitimate counts with different denominators:

| count | meaning | authoritative artifact |
|---:|---|---|
| 6 | project-generated strict Flash-style cases passing complete F+B, causal repair, real carrier, and paired trajectory | `results/coverage/existing_case_reaudit.json` |
| 8 | unique closed F+B paired-separation records in the Bias Formation roster | `case.md` |
| 7/8 | those roster records with ordered-trajectory directional persistence, including feedback-sustained persistence | `case.md` |
| 6/8 | those roster records with a matched formation mechanism | `case.md` |
| 4/8 | those roster records whose formation and persistence use the same contrast or declared closed semantic-region superset | `case.md` |

Therefore `4/8 -> 6` is not a gate relaxation or a promotion.  `4/8` is the
same-contrast subset of one eight-record formation roster.  `6` is the
project-wide strict Flash-style registry; it includes Qwen `lm_head dX`, which
is not in that formation roster, and it admits two causally closed semantic
regions while keeping them ineligible for single-kernel property claims.

The six strict cases are Qwen seq128 `lm_head dX`, Liger fused CE, Phi-4 seq64
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
| Screening has measured recall. | A random screen-negative sample must contain residual-nonzero, parameter-reachable rows and receive the complete downstream protocol. | `docs/tcmp_progress.md` (“Frozen new-implementation audit”) | `BOUNDED`. A 12-family audit was completed, but ten rows were `NOT_APPLICABLE`; of two reachable rows, one was canceling and one provenance-unresolved. This does not yet give a useful numerical recall estimate. |
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
| Six strict Flash-style project cases exist. | Complete F+B, causal repair/sham, real carrier, and paired directional trajectory. | `results/coverage/existing_case_reaudit.json`; `case.md` | `SUPPORTED`: four root-arithmetic cases plus two closed semantic-region cases. The literature FlashAttention anchor is excluded from the six. |
| Source-persistent and feedback-sustained drift are distinct regimes. | Four-arm local/feedback recurrence closes; optimizer ablation changes feedback without relabeling the source. | Gemma section of `docs/tcmp_progress.md`; `results/property/bias_property_search/heldout_validation_v3_gemma_disjoint.json` | `SUPPORTED_CASE_LEVEL`. Gemma local is null-like while feedback/actual persist; stateless SGD and moment reset collapse the feedback. A general feedback predictor is still `OPEN`. |
| Negative evidence is nontrivial and correctly classified. | Nonzero local residual, parameter reach, complete F+B, and centered/canceling gradient/update population. | `results/property/tcmp_allop_v1/final_decision_matrix.json`; `results/property/tcmp_allop_v1/scope_extension_20260822.json` | `SUPPORTED`. YaRN exact-zero and parameter-inaccessible softmax/mask regions are `NOT_APPLICABLE`; compiler/provenance failures are `UNRESOLVED`; neither group is counted as a negative. |
| Error magnitude alone does not predict source persistence. | Compare backward-visible residuals with different persistence outcomes under a common statistic and control boundary. | Liger and Qwen128/vision-control records in `case.md` and `docs/tcmp_progress.md` | `BOUNDED`. The qualitative separation is supported, but no frozen cross-case regression establishes a universal magnitude-independent law. SiLU cannot be described as “no drift”: its local source cancels while feedback persists. |

## Mechanism and causal claims

| claim | evidence threshold | supporting artifact | status |
|---|---|---|---|
| Repeated drift is not explained by a single random kick. | Repeated, support/norm-matched perturbations at every step, multiple frozen seeds, and drift-subspace analysis. | `results/property/tcmp_allop_v1/repeated_orbit_null_summary.json`; `docs/tcmp_progress.md` | `SUPPORTED`, but the result is not source-specific: repeated orbit variants retain 93.4–101.8% of natural drift with high cosine and effective rank 1.28/1.57. It supports low-dimensional closed-loop feedback and rejects “fixed schedule alone is the anchor.” |
| The final drift direction is operator-specific. | Natural drift must separate from multiple matched perturbation directions beyond the shared low-dimensional feedback subspace. | same repeated-orbit artifacts | `NOT_SUPPORTED`. The five-seed cosine/effective-rank result shows strong shared structure; current evidence does not isolate an operator-unique direction. |
| Liger persistence is caused by a step-invariant rounding/schedule anchor rather than error magnitude. | On the real kernel, preserve or increase error energy while removing rounding bias (RN→SR), and independently randomize chunk/order across steps. | current FP32-accumulator and chunk-geometry results in `case.md` | `OPEN`. FP32 accumulation and chunk geometry show a precision × schedule interaction, but the norm-matched SR and per-step anchor-breaking interventions have not been completed. |
| Feedback is a separate causal channel. | Local source is null-like; feedback recurrence is persistent; stateless/moment-reset optimizer controls remove it. | Gemma artifacts in `docs/tcmp_progress.md` | `SUPPORTED_CASE_LEVEL`. Predictor F is not frozen; feedback remains outside the source Oracle’s domain. |
| OLMoE routing flips do not contaminate its control. | Record route indices/weights in both arms for every measured state. | OLMoE section of `docs/tcmp_progress.md` | `SUPPORTED`: routing indices and weights matched in all 26 states. The result is a backward-visible canceling control, not a case. |

## Prediction and Oracle claims

| claim | evidence threshold | supporting artifact | status |
|---|---|---|---|
| Transported conditional mean predicts persistence. | Predictor frozen before consequence, protocol-conforming orbit estimate, and disjoint states. | Llama/Ministral records in `docs/tcmp_progress.md`; `results/property/tcmp_allop_v1/final_decision_matrix.json` | `BOUNDED`: prediction ordering was prospective, but the first orbit roster deviated from the frozen 4+4 protocol; corrected values are retrospective. Evidence supports one `lm_head dX` family on new operands, not confirmatory cross-implementation generalization. |
| Held-out axes are reported honestly. | Label each row `SEEN_IMPL / NEW_OPERANDS` or `NEW_IMPL`. | `docs/heldout_property_validation.md`; `docs/heldout_property_validation_v3.md` | `SUPPORTED`. Llama/Ministral are `SEEN_IMPL / NEW_OPERANDS`; Gemma is `NEW_IMPL` but source-negative and feedback-sustained. New-implementation source-positive count is zero. |
| The low-cost Oracle safely triages exact trajectories. | Fixed 256-dimensional CountSketch, empirical null, fail-closed selector, and exact confirmation only for risks. | `docs/short_persistence_oracle_v3.md`; `scripts/select_short_screen_escalations.py`; `results/property/bias_property_search/gemma_v3_disjoint_escalation_manifest.json` | `SUPPORTED_AS_WORKFLOW`. It is not yet an accuracy or safety certificate. |
| The Oracle has measured engineering efficiency. | Report flagged fraction, eligible denominator, exact-confirmation recall, false escalations, GPU time, and memory versus full trajectories. | — | `OPEN`. The current Gemma manifest is one validation record, not a stable efficiency estimate. |
| The predictor beats simple baselines. | Frozen comparison against local RMS, reduction extent, dtype, and transport/concentration-only rules on positive and residual-nonzero negative rows. | `results/property/bias_property_search/development_property_separation_audit_v1.json` | `OPEN`. Concentration is already shown not to separate; the complete baseline comparison is not yet available. |
| Four candidate properties were tested. | Positive and centered-control measurements under the same property definition. | `development_property_profile.json`; `development_property_separation_audit_v1.json`; `property_freeze_v1.json` under `results/property/bias_property_search/` | `SUPPORTED_WITH_DECISIONS`: source asymmetry separates 3 known rows from 5 centered controls and is retained as a conditional prior; source–transport has one positive and no valid controls, so remains case-level; concentration overlaps controls and is supporting-only; carrier stability has no measured centered-control trajectories and is only a consequence screen. |
| A universal all-operator property has been established. | Prospective positives and residual-nonzero negatives across unseen implementation classes, with frozen thresholds and baselines. | `docs/final_conclusion.md`; `docs/bias_property_search_completion.md` | `NOT_SUPPORTED`. The correct result is a family-scoped transported-mean hypothesis plus a fail-closed persistence screen and a separate feedback taxonomy. |

## Claims that are safe to write now

1. Kernel Analyzer provides an auditable F+B/repair/trajectory chain for a
   frozen multi-model denominator and retains unresolved rows.
2. Six project cases pass the strict Flash-style chain, but they do not imply
   six independent physical mechanisms or one universal property.
3. Backward-visible implementation residuals often cancel; nonzero local error
   and low-dimensional concentration are insufficient by themselves.
4. A transported conditional orbit mean is a falsifiable, family-scoped
   persistence predictor for the measured `lm_head dX` reduction/VJP family.
5. Feedback-sustained drift is a distinct case-level regime demonstrated by
   Gemma’s Adam-state intervention.
6. The short Oracle is a fail-closed prioritization workflow, not a safety
   certificate or a validated universal classifier.

## Highest-priority open evidence

1. A residual-nonzero, parameter-reachable screen-negative recall sample large
   enough to estimate recall.
2. Triage efficiency and frozen simple-baseline comparisons.
3. Liger RN→SR and per-step chunk/order randomization on the real kernel.
4. A prospective, protocol-conforming `NEW_IMPL` source-positive confirmation,
   or an explicit decision to publish the narrower family-scoped result.

