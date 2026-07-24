# Qwen3 full-training operator coverage status — 2026-07-19

## Scope and verdict

Scope is the frozen Qwen3-0.6B GRPO heldout-transport-B step-29 transition under
the declared FP16, SDPA-math and tracked-Inductor realization.

`ORACLE ENDPOINTS VALID FOR DECLARED QUERIES; FORWARD FAMILY EVIDENCE COMPLETE AT ONE STATE; BACKWARD SELECTED-STATE REPAIR PARTIAL 10/41; TWO MECHANISTICALLY DISTINCT BACKWARD REPAIRS TRANSPORTED TO THREE STATES AND FOUND STATE-CONDITIONAL; FULL-TRAINING CAUSAL COVERAGE INCOMPLETE`

Oracle validity and operator coverage are separate ledgers. The existing
Oracle can validly estimate implementation-relative scorer/event/transition
effects in its declared samples even though local causal coverage is missing.

| Training domain | Descriptive/runtime denominator | Current strongest evidence | Local causal coverage | Main missing condition |
|---|---|---|---|---|
| Model forward | 20 Triton + `mm`/`bmm`; 735 calls; 24 semantic-role classes and 536 high-level invocations | 20/20 Triton families have valid representative repair; both external families have role-aware shared-path reexecution | Representative family evidence at one state; no family is fully transported | injection and effect compatibility across matched states/shapes |
| Scorer/postprocess | `selective_log_softmax` gather/logsumexp/subtract path | paired B/H/N profile over 20 frozen states; exact repeats | no constituent scorer-op repair/injection | local scorer-op attribution |
| GRPO loss and clipping event | ratio, clamp, minimum, mask, reductions and predicate | clipping event distribution over 20 states; one valid branch-function repair with controlled update effect | branch-conversion attribution only | primitive/operator attribution and held-out repair/injection |
| Model backward | 39 Triton + external `mm`/`bmm`; 1,857 calls in one valid natural transition | all 41 families dynamically observed; 8 singleton families, one repeated family at early/middle/late positions, and one three-role cast family have valid repairs; middle SiLU, singleton final-norm and a null cast control were repeated at A/B/C | 10/41 families have selected-state repair evidence; two mechanistically distinct treatments have three-state transport evidence and both are state-conditional; 0/41 are fully covered | role mapping for 9 remaining multi-role families, other repeated/external representatives, injection and distribution-level transport |
| Gradient control | unscale, global norm, clip coefficient and gradient scaling execute in natural transition | eager/compiled gradient vectors and clip trigger measured; both triggered clipping at the selected state | shared propagation/boundary path, not a separate eager/compiled treatment in this subject | cross-state boundary-conditioned propagation and a separate treatment only if alternative implementations are introduced |
| AMP/optimizer/scheduler | GradScaler, fused AdamW and linear scheduler execute from captured state | full next-state discrepancy measured; both arms finite, scale unchanged, no skip, identical post scaler/scheduler state | shared propagation/state-transition path, not a separate eager/compiled treatment in this subject | near-overflow/skip states and a separate treatment only if alternative implementations are introduced |
| Population transport | scorer/event bank has 20 states; natural transition and most operator repairs use selected states | scorer B/H/N and clipping prevalence are instantiated for the frozen bank; one SiLU repair plus null cast control were tested at A/B/C with exact repeats | three deliberately selected states establish state dependence, not population prevalence | sample a declared state distribution and transport multiple operator interventions |
| Correctness | no independent mathematical/specification authority | implementation-relative impact only | uninstantiated | high-precision/specification/wrong-code authority |

## Numerators that must not be mixed

- Forward **family evidence**: 22/22 at one selected state.
- Forward **fully covered families**: 0/22, because transport and injection are
  absent.
- Backward **runtime-observed families**: 41/41 at one selected state.
- Backward **selected-state repaired families**: 10/41. This includes eight
  singleton families, one repeated family with three position-specific repairs,
  and one multi-role family with three role-specific repairs.
- Backward **transport-tested selected invocations**: 2 non-null-at-B candidates
  plus 1 exact-null intervention control across A/B/C.  This does not increment the
  fully covered family numerator because distribution-level transport and
  injection remain absent.
- Backward **fully covered families**: 0/41, because injection and population
  transport are absent and repeated-call equivalence is not established.
- Full-training **operator causal coverage**: incomplete.

Reporting only 22/22 would therefore be misleading. It means every frozen
forward generated family has some audited representative evidence, not that
every invocation, state, backward operation or optimizer transition has a
causal explanation.

## Evidence anchors

- forward denominator and interventions:
  `QWEN3_OPERATOR_EQUIVALENCE_LEDGER_V0_1_2026-07-19.md` and
  `QWEN3_ORIGINAL_CANDIDATE_KERNEL_COVERAGE_STATUS_2026-07-19.md`;
- backward static/runtime denominator:
  `QWEN3_BACKWARD_GENERATED_KERNEL_INVENTORY_FINDINGS_V0_1_2026-07-19.md`,
  `QWEN3_BACKWARD_RUNTIME_CENSUS_FINDINGS_V0_1_2026-07-19.md` and
  `QWEN3_BACKWARD_CAUSAL_COVERAGE_LEDGER_V0_3_2026-07-19.md`;
- backward repair extensions:
  `QWEN3_BACKWARD_SINGLETON_REPAIR_FINDINGS_V0_2_2026-07-19.md`,
  `QWEN3_BACKWARD_REPEATED_FAMILY_REPAIR_FINDINGS_V0_1_2026-07-19.md` and
  `QWEN3_BACKWARD_MULTIROLE_CAST_REPAIR_FINDINGS_V0_1_2026-07-19.md`;
- three-state operator-attribution transport:
  `QWEN3_OPERATOR_ATTRIBUTION_TRANSPORT_FINDINGS_V0_1_2026-07-20.md` and
  `QWEN3_FINAL_NORM_ATTRIBUTION_TRANSPORT_FINDINGS_V0_1_2026-07-20.md`;
- shared post-backward propagation:
  `QWEN3_POST_BACKWARD_SHARED_PATH_AUDIT_V0_1_2026-07-19.md`;
- multi-state scorer/event Oracle:
  `QWEN3_GRPO_GRAD_EVENT_BANK_UNIFIED_EVIDENCE_V0_4.json`;
- natural full-step transition:
  `QWEN3_GRPO_NATURAL_TRANSITION_FINDINGS_V0_2_2026-07-18.md`;
- branch-function attribution:
  `archive/QWEN3_GRPO_GRAD_BRANCH_REPAIR_FINDINGS_V0_9_2026-07-17.md`.
