# Qwen3 GRPO B/H/N analysis contract v0.2

Date: 2026-07-18
Status: frozen while held-out trajectory A was incomplete (6/10 state records)

## Subject

- reference: grad-enabled eager Trainer scorer;
- candidate: grad-enabled tracked Inductor scorer;
- state cluster: one pre-minibatch rollout state;
- observable: aligned 4×128 current-token log-probability field;
- repeats: two measured calls per implementation after candidate warm-up;
- target bank: the predeclared A/B/C fixed-start held-out trajectories.

This is a whole-scorer observable profile. It is not a source-operator profile.

## Frozen estimands

- signed global mean shift, retained as a descriptive scalar;
- `B`: RMS norm of the elementwise mean effect, preventing sign cancellation;
- `H`: across-state conditional-mean variance with `N/R` correction;
- `N`: paired-difference repeat variance, plus separate eager and candidate repeat
  variances;
- `U`: state-cluster normal-approximation interval for relative B;
- per-trajectory and combined profiles;
- state signed-mean range and SD.

Tokens are coordinates inside a state field, not independent state samples.

## Verdict boundary

No numerical or application acceptance envelope is declared. Therefore the B/H/N
profile must return `UNINSTANTIATED` for acceptance even when every observed repeat
is exact or B is numerically small. Semantic clipping events, natural transition,
correctness and attribution remain separate ledgers.

## Kill conditions

- any trajectory lacks exactly ten states or any state lacks flat indices 0..511;
- eager/candidate repeat shapes differ;
- the evaluator averages tensor elements before constructing sign-safe B;
- same-state repeat absence is reported as N=0;
- token count is used as the independent sample size for U;
- zero B is called safe, or nonzero B is projected linearly across training steps;
- the whole-scorer profile is called operator causality.

