# Existing Evidence Reclassification under Operator Oracle v0.1 — 2026-07-16

> Purpose: determine which existing ForkCert results validate the new Oracle, which remain discrepancy/impact measurements, and what empirical evidence is genuinely missing. No historical result is retroactively promoted to a stronger claim.

## 1. Summary verdict

Existing work strongly validates the need for:

- fail-closed execution identity;
- separation of average shift, input heterogeneity, runtime variability and sampling uncertainty;
- event-specific geometry and stochastic-law comparison;
- one-step transition as a distinct endpoint;
- intervention-identity downgrades.

It does **not** yet validate a general operator conformance Oracle because the studies lack per-operator semantic envelopes, per-operator realization correspondence and independently labeled conforming/violating controls.

## 2. Evidence map

| Existing artifact | v0.1 classification | What it validates | What it cannot support |
|---|---|---|---|
| early online scan with recompile-limit fallback | invalid-control witness | path label is insufficient; fallback must invalidate evidence | any denominator/rate after fallback |
| CUDA GPU pilot | calibrated full-program/composite measurement | candidate calls, self-pairs, deterministic heterogeneity, event-map differences | operator conformance, correctness, natural event prevalence |
| BERT/ResNet/Qwen multisubject study | full-program compatibility/impact measurement | global mean insufficiency, checkpoint/state conditioning, zero repeat noise in protocol | operator verdict or mathematical error |
| centered-logit diagnostic | semantic-invariance/impact negative control | common-mode raw delta can be decision-invariant; centering does not automatically improve detection | correctness acceptance of logits |
| Qwen sampling study | distributional compatibility measurement | law distance differs from coupled token disagreement and RNG variability | target-law correctness or unacceptable drift without margin |
| BERT/Qwen one-step transition | matched full-program transition discrepancy | equal forward output can hide gradient/update difference | operator source, harm, historical optimizer correctness |
| output-boundary factorial | `I2` boundary-value/Jacobian intervention | context-specific boundary value versus derivative-path effects and interaction | source-operator localization or unique causal share |
| BERT segmented attribution | `I1` region intervention, failed original-program parity | region responses are intervention dependent; exact parity gate works | constituent operator cause in monolithic graph |
| prior mutation evaluation | inconclusive baseline comparison | fork was not shown superior to raw delta | validation of the new semantic-envelope Oracle |

## 3. Realization-identity audit

### Full-program studies

The GPU pilot, multisubject, sampling and transition studies record one stable compiled graph and candidate runtime calls. This establishes execution of the declared **full compiled subject**. It does not establish `R4` correspondence for constituent semantic operators.

Correct label:

```text
full-program/composite realization valid
constituent operator realization not identified
```

### Output-boundary study

Exact boundary-value splicing and anchored downstream arms support `I2` boundary-value causal effects. They do not prove which internal operator generated the value/Jacobian discrepancy.

### Segmented BERT study

The segmented endpoints fail exact monolithic parity. It is a strong negative control for operator attribution and correctly downgrades to `I1` region intervention. It should be retained as validation that the framework refuses a tempting but invalid root-cause claim.

## 4. Contract-verdict audit

| v0.1 verdict capability | Existing evidence |
|---|---|
| `INVALID` on fallback/treatment failure | supported by early fallback and segmented-parity cases |
| `INDETERMINATE` when no tolerance exists | supported conceptually; existing real-model findings already state measurement-only |
| exact-core `REJECT` on known specification witness | missing |
| exact/set-valued `ACCEPT` on independently labeled allowed behavior | missing as a formal validation control |
| S2 numerical `ACCEPT/REJECT` using certified envelope | missing |
| stochastic target-law conformance | missing; current TV is baseline-relative |
| operator `R4` behavior verdict | missing |
| Oracle superiority to raw delta | missing |

The existing data can be re-rendered through v0.1 as `VALID measurement`, `INVALID`, `NOT_IDENTIFIABLE` or `UNINSTANTIATED`; they cannot be retrospectively turned into correctness pass/fail.

## 5. Existing evidence that already challenges raw delta

Three results demonstrate non-equivalence of numerical magnitude and semantic/transition impact:

1. common-mode logit delta contributes substantial raw energy while being invariant to softmax/ranking;
2. boundary distance ranks argmax/top-k exposure much better than raw/centered magnitude;
3. equal logits/loss can coexist with nonexact gradients.

These support the need for structured geometry. They are not yet detector-validation ground truth because they do not label compiler correctness.

## 6. Minimal empirical validation kernel

The first new validation should demonstrate verdict mechanics, not claim family coverage.

### Kernel K1 — exact/set-valued semantics

- P1 `argmax` on identical finite operands with first-tie rule;
- P2 `topk` with explicitly allowed tied alternatives;
- positive controls outside the exact/set relation;
- negative controls with large representation/reference difference but permitted set membership.

Purpose: validate exact `ACCEPT/REJECT`, set-valued false-positive avoidance and same-operand identity.

### Kernel K2 — certified numerical envelope

- P5 sum and P6 one known matmul precision mode;
- exact/high-precision reference;
- inputs spanning cancellation/conditioning and ordinary scale;
- positive controls outside the analytical envelope;
- negative controls from different legal evaluation orders inside the envelope;
- raw-delta baselines.

Purpose: validate that conformance is not raw eager distance and that deterministic bias can be legal or illegal depending on the envelope.

### Kernel K3 — stochastic/indeterminate behavior

- P3 multinomial target-law relation;
- structural/support violation positive controls;
- same-law/different coupling negative controls;
- deliberately insufficient sample budget that must yield `INDETERMINATE`.

Purpose: validate law/coupling separation and correct abstention.

### Kernel K4 — subject identity

- one isolated realization labeled `R2`;
- one fused composite labeled `R3`;
- one observation-changing-compilation invalid control;
- no operator-level promotion unless `R4` correspondence is independently established.

Purpose: validate scope/downgrade behavior rather than force an operator attribution.

## 7. Why this kernel is not “testing only four examples”

K1--K4 test the Oracle's semantic mechanisms and verdict states. They are a gate before external-validity expansion, not the evaluation population. Passing them licenses only the statement that the decision procedure can work.

The subsequent coverage matrix must still span structural, elementwise/cast, reduction, contraction, normalization, selection/routing, stochastic, backward and stateful operators across signatures/configurations.

## 8. Reusable assets

Existing work can contribute:

- fail-closed compiled-call and stable-graph canaries;
- self-pair and exact-repeat controls;
- discovery/confirmation state clustering;
- state-level uncertainty and tail reporting;
- sampling-law/coupling separation;
- transition endpoint handling;
- negative region-intervention integrity case.

It cannot supply without new evidence:

- operator semantic operands and realization correspondence;
- certified truth/envelopes for numerical contracts;
- directly evaluated positive/negative operator controls, although the DeepOPFuzz confirmed-bug corpus now supplies strong candidate labels and reproducers;
- held-out operator-family validation.

## 9. Decision

Do not discard the existing studies: they are strong validation of measurement, impact and fail-closed attribution rules. Do not use them to claim the new operator conformance Oracle is empirically validated. K1--K4 should reuse the confirmed-bug/fixed-control mapping in `CONFIRMED_BUG_ORACLE_VALIDATION_MAPPING_V0_1_2026-07-16.md`, followed by coverage-balanced held-out evaluation under the validation standard.
