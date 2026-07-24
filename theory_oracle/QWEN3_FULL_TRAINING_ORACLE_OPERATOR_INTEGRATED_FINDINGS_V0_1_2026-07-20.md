# Qwen3 full-training Oracle and operator analysis — integrated findings v0.1

> Interpretation update (2026-07-20): this document is authoritative for the recorded
> selected-state measurements and intervention results, but not for a target-population
> training-bias claim. The next-stage population calibration and bias-contributor gates are
> defined in `QWEN3_BIAS_ORACLE_CALIBRATION_PLAN_V0_1_2026-07-20.md`.

## Executive verdict

For the declared Qwen3-0.6B GRPO FP16/SDPA-math/Inductor subject:

- the matched-state discrepancy Oracle is instantiated and valid for its
  declared state samples;
- the complete forward/backward runtime denominator is accounted for;
- one-step transition impact is instantiated;
- two mechanistically distinct backward-region repairs plus an exact-null
  control have valid three-state transport evidence;
- full source-operator causal attribution and compiler correctness remain
  uninstantiated.

The correct combined verdict is:

`DECLARED MEASUREMENT AND DENOMINATOR AUDIT COMPLETE; ORACLE VALID FOR DECLARED MATCHED STATES; SELECTED REGION ATTRIBUTION STATE-CONDITIONAL; FULL SOURCE-OPERATOR CAUSAL ATTRIBUTION INCOMPLETE; CORRECTNESS UNINSTANTIATED`

## 1. What the Oracle now produces

The Oracle has three linked endpoints rather than one scalar verdict.

### Numerical/event profile over matched states

The 20-state event bank reports implementation-relative average shift,
state-conditioned heterogeneity, within-state runtime variability, directional
semantic shift and semantic disagreement.  It observed two stable clipping
direction changes among 4,608 applicable tokens.  Both trajectories had
nonzero variation of state means, while eager and compiled repeats were exact.

This is the cleanest empirical B/H/N separation:

- average shift exists under the declared state weighting;
- state heterogeneity exists because state means vary and signs can differ;
- runtime variability was not detected under the deterministic protocol.

The bank is a deterministic selected-state sample, not a population estimate.

### One-step state transition

At heldout-B step 29, eager and compiled self-repeats were exact but their
clipped-gradient, parameter-update and complete next-state endpoints differed.
The declared clip/AMP/skip events did not differ.  Therefore numerical/update
impact can exist without a discrete-event fork at the selected state.

### Operator-attribution transport

The same local treatment was repeated at A/B/C with exact within-arm repeats:

| treatment | A | B | C | transport conclusion |
|---|---|---|---|---|
| middle SiLU backward call | exact-null | non-null | non-null | state-conditional; B/C effect vectors nearly orthogonal |
| singleton final-RMSNorm backward region | exact-null | non-null | non-null | state-conditional; B/C effect vectors nearly orthogonal |
| FP16-to-FP32 cast control | exact-null | exact-null | exact-null | intervention control passes |

Neither non-null treatment changed the declared semantic events.  Both altered
continuous clipped-gradient/update endpoints at B/C.  At C, a positive
projection toward eager could coexist with a larger total distance from eager;
directional projection and distance reduction are therefore distinct
estimands.

## 2. Bias, heterogeneity and runtime variability in this evidence

“Bias” is used only as implementation-relative average shift under a declared
state distribution or finite-state weighting.  It is not trueness and does not
say eager is correct.

The evidence rejects a simple model in which one operator has one fixed bias
vector that transports across states.  A is null, while B/C are non-null, and
the B/C repair-effect vectors are almost orthogonal for both tested mechanisms.
The dominant operator-attribution finding is state-conditioned heterogeneity,
not a stable global direction.

Within-state repeats were exact throughout these campaigns.  Hence no runtime
variance was detected under this protocol.  This does not prove GPU runtime
noise is universally zero; it bounds the claim to these deterministic
executions.

## 3. Complete denominator versus causal coverage

The units are deliberately separated.

| unit | denominator evidence | causal evidence |
|---|---|---|
| source graph targets | forward: 34 types / 3,835 calls; backward: 40 types / 9,471 calls | not decomposed per target after fusion |
| forward generated/external families | 22 families, 735 runtime calls | all 22 have partial representative original-candidate evidence at one state; 0 fully covered |
| backward runtime families | 39 Triton + external mm/bmm; 41 families, 1,857 calls | 10/41 have selected-state repairs; 2 candidates have A/B/C transport; 0 fully covered |
| scorer/loss/control/optimizer path | source/path inventoried and full transition executed | only selected scorer/branch/region interventions; shared propagation paths are not discrepancy-generation treatments |

Thus “all ops were considered” is true only for the descriptive inventory and
coverage ledger.  “All ops have causal effects measured” is false.

## 4. Why a fused-region effect is not an operator contribution

The tested final-norm region contains multiple operations such as add, sum,
pow, rsqrt, cast and view.  Replacing the whole generated callable estimates
the effect of that intervention in its compiled context.  It cannot identify
which constituent source operator generated, propagated or converted the
discrepancy.  Assigning the region effect to each ATen operator would double
count and commit a unit-of-analysis error.

Likewise, a shared generated family name is not an equivalence class.  Earlier
early/middle/late repairs of the same SiLU family had different effect sizes
and directions, and A/B/C transport showed further state dependence.

## 5. What remains before full source-operator causal attribution

Five gaps are substantive rather than bookkeeping:

1. 31/41 backward runtime families still lack a valid selected-state repair;
2. no valid operator injection exists, so necessity/sufficiency language is unsupported;
3. fused-region replacement does not identify constituent source-operator effects;
4. three selected states do not estimate prevalence under a target state distribution;
5. no high-precision, specification or confirmed wrong-code authority supports correctness claims.

Brute-force repetition of the current repair design would improve the family
ledger but would not solve gaps 2–5.  A next causal phase must first define a
source-operator-preserving intervention or explicitly retain generated region
as the causal unit.

The non-identifiability argument and minimum additional capabilities are
formalized in
`QWEN3_SOURCE_OPERATOR_CAUSAL_IDENTIFIABILITY_BOUNDARY_V0_1_2026-07-20.md`.

## Evidence

- integrated machine-readable ledger:
  `results/operator_oracle/qwen3_full_training_oracle_operator_ledger_v0_2/ledger.json`;
- SiLU transport:
  `results/operator_oracle/qwen3_operator_attribution_transport_v0_1/evaluation.json`;
- final-norm transport:
  `results/operator_oracle/qwen3_final_norm_attribution_transport_v0_1/evaluation.json`;
- coverage status:
  `QWEN3_FULL_TRAINING_OPERATOR_COVERAGE_STATUS_2026-07-19.md`.
