# Sampling-distribution Oracle Contract — 2026-07-16

Frozen before reading outputs from this stage.

## Purpose

Separate three quantities that can all appear as “sampling variance”:

1. exact implementation-relative change in the categorical token distribution;
2. algorithmic RNG variability even when the distribution is fixed;
3. uncertainty from estimating probabilities with finitely many states or draws.

Single sampled-token disagreement is not the Oracle.

## Subject and state population

- model: `data/phase0_policy_final`;
- reference: eager CUDA; candidate: tracked Inductor full graph;
- state: one frozen sequence from `data/phase0_grpo_samples.jsonl` and the final valid response prediction position under length 166;
- discovery: rows `[0, 32)`;
- held-out confirmation: rows `[32, 64)`;
- logits converted to categorical probabilities at temperature `0.7` in FP64 analysis arithmetic;
- deterministic model execution; sampling RNG is introduced only after logits are frozen.

The final response position is selected by protocol, not by discrepancy or boundary distance.

## Exact distribution endpoints

For each state, compute without Monte Carlo:

- total variation distance `0.5 * sum |p_compiled - p_eager|`;
- Jensen–Shannon divergence;
- eager-to-compiled and compiled-to-eager KL when finite;
- entropy shift;
- probability shift for the eager top-1 token and candidate top-1 token;
- top-k probability-mass shift for `k=5`;
- exact independent-draw disagreement probability `1 - dot(p_eager, p_compiled)`;
- eager and candidate within-distribution two-independent-draw disagreement `1 - sum p²`;
- minimal possible paired disagreement under maximal coupling, equal to total variation.

Total variation is a marginal distribution estimand. A realized paired disagreement depends on the chosen coupling.

## RNG calibration

For each state and implementation:

- four independent replicate streams;
- 1024 categorical draws per replicate;
- fixed, recorded and non-overlapping seeds;
- report empirical top-1 event frequency and its between-replicate variability;
- report deviation from the exact top-1 probability;
- run a common-uniform inverse-CDF coupling as a coupling-dependent diagnostic, not as the distribution distance itself.

Finite-draw variability is reported separately from deterministic eager/compiled logit discrepancy.

## Inference and confirmation

- state is the bootstrap unit;
- draw-level observations remain nested within state and replicate;
- exact distribution endpoints and Monte Carlo calibration are reported separately;
- qualitative scale and direction of probability/entropy shifts must reproduce before being called persistent;
- zero sampled disagreement cannot establish equal distributions;
- no sampling correctness claim is made without a normative target distribution.

## Interpretation gates

1. Large within-implementation sample disagreement with tiny TV is algorithmic stochasticity, not compiler drift.
2. Nonzero TV with few realized mismatches can still be a distribution shift hidden by finite draws.
3. Common-random-number mismatch is coupling-dependent and cannot replace TV.
4. State-bootstrap uncertainty is not execution or RNG variance.
5. Temperature is part of the Oracle definition; results do not generalize automatically to another temperature.

