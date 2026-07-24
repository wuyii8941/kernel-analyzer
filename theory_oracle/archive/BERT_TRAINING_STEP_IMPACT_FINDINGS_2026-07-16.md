# BERT Training-Step Impact Findings — 2026-07-16

Contract: `BERT_TRAINING_STEP_IMPACT_CONTRACT_V0_1_2026-07-16.md`.

## 1. Bank identity

The new bank was extracted offline from the cached SST-2 validation split after the contract was frozen:

```text
source range:    [256,384)
states:          128
identity SHA256: 20719973839e917394adafecdb2886779daad45475d9f4efacfb723018f8fd29
labels:          59 negative, 69 positive
```

It does not overlap the earlier discovery `[0,128)` or confirmation `[128,256)` banks.

## 2. Execution and exact-core validity

```text
candidate identity valid: 128/128
backend compiles:          1
stable graph hash:         1aed318f…cae943
exact-core accept:         128/128
repeat-stable states:      128/128
missing/nonfinite rows:    0
```

## 3. Prediction-impact verdict

No eager/compiled argmax prediction disagreement occurred in either repeat:

```text
impact_fail states: 0/128
strict covered-bank compatibility verdict: ACCEPT
```

This means the candidate preserved the declared prediction endpoint on these 128 states. It does not prove universal prediction equivalence or mathematical correctness.

## 4. Other endpoints remain separate

The materialized next parameters still differed:

```text
mean next-state L2 discrepancy: 1.583381e-07
maximum coordinate discrepancy: 1.192093e-07
numerical transition verdict:   UNINSTANTIATED
```

Thus a covered impact `ACCEPT` coexists with unresolved numerical transition conformance. Loss and update impact have no application margin and remain descriptive.

## 5. Meaning

The result demonstrates the required separation:

```text
exact transition core:             ACCEPT
prediction compatibility impact:   ACCEPT on covered bank
numerical transition correctness:  UNINSTANTIATED
operator source attribution:        NOT_IDENTIFIABLE below R3 region
```

No axis substitutes for another.
