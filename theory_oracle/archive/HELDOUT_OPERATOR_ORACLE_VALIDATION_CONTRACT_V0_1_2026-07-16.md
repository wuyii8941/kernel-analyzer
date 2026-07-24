# Held-out Operator Oracle Validation Contract v0.1 — 2026-07-16

> Frozen after W1/W2 discovery and before executing the confirmation cases below. W1 wrong-stride and W2 ignored-alpha are excluded from confirmation scoring.

## 1. Validation question

Does Operator Oracle v0.1 correctly distinguish independently confirmed semantic violations, allowed/fixed differences, invalid evidence and genuinely unresolved contracts—and does it add information beyond raw numerical delta/allclose?

This is detector/contract validation on stress-selected controls. It does not estimate natural workload prevalence.

## 2. Environments and version roles

```text
broken candidate pool: PyTorch 2.11.0+cu126 and case-specific historical environment when required
fixed candidate pool:  PyTorch 2.13.0.dev20260609+cu126 where the mapped fix is present
hardware:              Tesla T4
```

For each case, version applicability is checked before semantic scoring. API absence, unsupported hardware or failure to instantiate the historical bug condition is `INAPPLICABLE`, not `ACCEPT`.

## 3. Frozen confirmation positives

### H1 — explicit cast preservation

```text
case: paper confirmed bug #179561, fp32 -> bf16/fp16 -> fp32 cast elision
contract: explicit representation conversion must remain observable
subject: fused composite program; constituent fusion pass attribution only if independently evidenced
expected broken verdict: REJECT
expected fixed verdict: no cast-elision witness
```

### H2 — exact indexing/expanded-family semantics

```text
case: paper confirmed bug #183986
contract: exact index/gather/scatter relation on admitted inputs
expected broken verdict: REJECT
expected fixed verdict: covered-case ACCEPT/no witness
```

### H3 — slice-scatter backward

```text
case: paper confirmed bug #180164 or its independently mapped family case
contract: exact scatter placement plus backward/state relation
expected broken verdict: REJECT
expected fixed verdict: covered-case ACCEPT/no witness
```

### H4 — hardtanh bf16 boundary gradient

```text
case: paper confirmed bug #185472
contract: documented/decomposition boundary convention for admitted bf16 value
expected broken verdict: REJECT on trigger values
expected non-trigger/fixed verdict: ACCEPT for covered cases
```

### H5 — higher-order gradient metadata

```text
case: create_graph/requires_grad metadata-drop family when version-applicable
contract: observable differentiation state/grad_fn relation
expected broken verdict: REJECT exact state relation
expected fixed verdict: covered-case ACCEPT/no witness
```

### H6 — eager-wrong/compiled-right numerical case

```text
case: confirmed TensorFlow SELU/ELU cancellation or complex-sign underflow
contract: high-precision mathematical relation
expected result: truth-relative Oracle favors the correct implementation;
                 eager disagreement alone must not mark compiled wrong
```

H6 is cross-framework and may be executed separately if the historical TensorFlow environment is unavailable. Its absence must be reported as missing cross-framework coverage.

## 4. Frozen confirmation negatives

### N1 — matched fixed versions

Run the identical H1--H5 relation on the mapped fixed/current version. A fixed run is a covered negative only when the trigger input and execution identity remain valid.

### N2 — non-trigger strata

Use mechanism-derived non-triggers, not random easy cases:

- representable values for cast round-trip where applicable;
- shapes/strides not entering the bad indexing path;
- nearby hardtanh values outside the problematic equality boundary;
- valid metadata/control configurations.

### N3 — set-valued top-k ties

Construct identical operands with tied boundary values. Different documented-valid tied indices must not be correctness rejections.

### N4 — legal floating differences

Use a sum/matmul case where two admitted evaluation strategies differ but both lie within the declared analytical envelope. Nonzero/large raw delta alone must not reject.

If a certified envelope cannot be constructed for N4, report `UNINSTANTIATED`; do not improvise a tolerance.

## 5. Frozen invalid and indeterminate controls

### V1 — invalid execution identity

Reuse the documented recompile-limit fallback pattern or an equivalent case where candidate execution cannot be verified. Expected verdict: `INVALID`.

### V2 — wrong subject level

Use the BERT segmented-parity failure as a fixed negative attribution control. Expected result: region `I1`, operator causal claim `NOT_IDENTIFIABLE`.

### U1 — no floating tolerance

Use a softmax/LayerNorm comparison with mathematical truth measurement but no documented/application accuracy margin. Expected floating acceptance verdict: `UNINSTANTIATED`.

### U2 — insufficient stochastic-law resolution

Use a finite multinomial sample budget whose confidence set overlaps the declared equivalence boundary. Expected verdict: `INDETERMINATE`.

## 6. Hard raw-delta quadrants

The confirmation set must realize all quadrants before aggregate comparison:

| Quadrant | Frozen construction |
|---|---|
| small + conforming | fixed/non-trigger ordinary floating case |
| large + conforming | top-k tied alternatives or certified legal numerical variants |
| small + violating | scale an exact option/index/cast violation toward small floating magnitude without changing its semantic label |
| large + violating | wrong-index/ignored-option/stride confirmed case |

Input scaling is predeclared as powers of two that remain finite/normal in the tested dtype. The exact scales are selected from dtype-safe analytical ranges before outputs are inspected and are shared across Oracle and baselines.

## 7. Oracle outputs

For every observation emit:

```text
case_id and independent label
contract source/relation
input/signature/configuration
candidate realization level
validity/applicability
semantic-envelope membership or distance
verdict and claim level
raw max_abs, relative, norm and conventional-allclose baselines
relative bias/heterogeneity/runtime fields when multiple inputs/repeats exist
impact status if tested
```

Crash, missing, nonfinite and fallback observations remain in the denominator according to their predeclared semantic/validity rule.

## 8. Primary validation metrics

```text
positive-control detection rate
negative-control false rejection rate
invalid-control refusal rate
indeterminate-control abstention correctness
selective error among determinate verdicts
coverage / inapplicable / uninstantiated rates
```

Raw-delta/allclose comparisons report classification/ranking across the hard quadrants. No threshold is tuned on confirmation cases; conventional tolerances and any discovery-selected thresholds are reported separately.

## 9. Claim-level scoring

A result counts as correct only if both verdict and scope are correct:

- detecting a fused-region bug but claiming a unique constituent operator is a scope error;
- calling a fixed nonzero floating delta wrong is a false rejection;
- returning `ACCEPT` when the contract is missing is an error;
- returning `INDETERMINATE` on every case avoids false positives but fails useful coverage.

## 10. Confirmation success gate

The small confirmation validates mechanics only if:

1. all applicable exact confirmed positives are rejected;
2. all applicable fixed/set-valued negatives avoid correctness rejection;
3. V1/V2 are correctly invalidated/downgraded;
4. U1/U2 abstain for the declared reason;
5. all four raw-delta quadrants are populated;
6. no confirmation-derived threshold or contract change is used in scoring.

Failure narrows or revises v0.1 before expanding coverage. Passing does not establish generality; it opens the coverage-balanced evaluation stage.

## 11. Kill criteria

- confirmed labels cannot be reconstructed as independent contracts;
- most cases are inapplicable on available broken/fixed environments;
- the Oracle catches violations only when raw delta is already large;
- set-valued/legal differences are falsely rejected;
- subject identity is routinely overclaimed;
- uninstantiated cases are silently scored as passes;
- post-hoc thresholds are needed to obtain favorable results.

## 12. Frozen status

This contract is now the authoritative next empirical gate. Any case substitution, scale selection, tolerance or metric change after observing confirmation output requires a versioned deviation record and cannot be counted as preregistered confirmation.
