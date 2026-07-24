# Synthetic TrainingOracle result reclassification

Date: 2026-07-18

Artifact: `results/training_oracle_experiment.json`

## Verdict

This artifact is a valid record of one synthetic twin-trajectory run, but it is
**not valid evidence that B/H/N predicts operator-substitution safety**.

## Why

- The candidate perturbations are manually added hooks, not compiler
  implementations.
- After the first update, reference and candidate parameters differ. Later module
  output differences combine the current hook perturbation with accumulated state
  divergence, so their temporal variance is not matched-state H or N.
- There are no same-state repeats. Runtime variance is not identified.
- Scenario C is called symmetric noise because the generator was constructed that
  way; the Oracle did not discover this property.
- Scenario C's final parameter divergence is nonzero. Lack of REJECT only means
  the chosen point-estimate bias rule did not cross its chosen threshold; it does
  not demonstrate SGD absorption or harmlessness.
- Growth exponents in nonlinear feedback training do not uniquely identify a
  persistent shift or random walk.

## Permitted use

The file may be used as a trajectory-visualization or negative methodological
example. It must not be cited as real eager/compiled validation, correctness,
benign-noise evidence or proof that raw delta is inferior for downstream impact.

The replacement evidence path is the frozen Qwen held-out matched-state bank plus
the independently selected natural step-29 transition.

