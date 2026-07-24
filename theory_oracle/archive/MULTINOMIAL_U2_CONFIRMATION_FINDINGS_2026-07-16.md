# Multinomial U2 Confirmation Findings — 2026-07-16

Contract: `MULTINOMIAL_U2_CONFIRMATION_MANIFEST_V0_1_2026-07-16.md`.

## Result

The frozen compiled CUDA call produced 100 supported samples:

```text
token 0 count                  53
token 1 count                  47
support valid                  true
p_hat(token 0)                0.53
95% Hoeffding radius           0.13581015157406195
confidence interval            [0.39418984842593807, 0.665810151574062]
acceptable interval            [0.49, 0.51]
Dynamo calls captured          1
Dynamo unique graphs           1
```

The confidence interval overlaps the acceptable interval but is not contained in it.

```text
verdict: INDETERMINATE
```

## Meaning

The observed 53/47 split is neither an acceptance proof for a plus/minus 1% equivalence contract nor evidence that the law violates it. The sample budget is too small for that claim.

This validates the distinction between:

- an input-level target law;
- finite-draw randomness;
- uncertainty about the candidate law;
- a population acceptance boundary.

A single token mismatch or a nonsignificant difference would not provide this verdict logic.

## Scope

The control validates stochastic abstention mechanics for a binary categorical law and one deliberately strict margin. It does not establish the compiled implementation's distributional correctness, because `INDETERMINATE` deliberately withholds that conclusion; nor does the chosen 1% margin represent an application requirement.
