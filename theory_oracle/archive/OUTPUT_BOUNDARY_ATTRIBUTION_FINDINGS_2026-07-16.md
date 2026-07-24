# Output-boundary Transition Attribution Findings — 2026-07-16

## Question

When eager and compiled produce different one-step gradients, how much is associated with the numerical value arriving at the final-logit boundary, how much remains when that value is repaired, and do the two mechanisms interact?

The four arms and state populations were frozen in `ORACLE_ATTRIBUTION_CONTRACT_2026-07-16.md`.

| Arm | Forward logits | Differentiation/Jacobian path |
|---|---|---|
| A | eager | eager |
| I | compiled | eager |
| R | eager | compiled |
| B | compiled | compiled |

Mixed arms use an exact stop-gradient splice. Every mixed boundary was bitwise equal to its intended value. All candidate-call, graph-stability, formal-transition anchor and same-arm repeat gates passed.

## BERT: most states are Jacobian-only at the output boundary

| Endpoint | Discovery, 32 states | Confirmation, 32 states |
|---|---:|---:|
| Total `||G_B-G_A||`, mean | 4.16e-3 | 8.09e-3 |
| Value injection `||G_I-G_A||`, mean | 1.58e-3 | 9.53e-5 |
| Residual after value repair `||G_R-G_A||`, mean | 3.57e-3 | 8.02e-3 |
| Interaction norm, mean | 1.09e-3 | 3.69e-5 |
| States with zero value-injection effect | 29/32 | 30/32 |
| Median value-injection / total ratio | 0 | 0 |
| Median residual-after-repair / total ratio | 1 | 1 |
| Median interaction / total ratio | approximately 8e-10 | approximately 7e-10 |

For the large majority of BERT states, final eager and compiled logits are bitwise equal while their full gradients differ. Replacing the compiled forward value with the eager value therefore cannot repair the gradient: the discrepancy is in the compiled differentiation/Jacobian path conditional on the same final value.

This is stronger than the earlier observation “loss is equal but gradient differs,” because the four-arm intervention verifies that output-value injection has exactly zero effect in those states.

The discovery means contain a sparse outlier: one value-injection effect is larger than the total A/B distance, with cancellation through interaction. Consequently, the ratio of means and the mean of per-state ratios tell different stories. Median/count descriptions are required; no additive percentage claim is valid.

## Qwen: value and Jacobian effects strongly interact

| Endpoint | Discovery, 4 minibatch states | Confirmation, 4 minibatch states |
|---|---:|---:|
| Total `||G_B-G_A||`, mean | 0.505 | 0.433 |
| Value injection `||G_I-G_A||`, mean | 0.217 | 0.227 |
| Compiled Jacobian at eager value `||G_R-G_A||`, mean | 0.537 | 0.420 |
| Value effect under compiled Jacobian `||G_B-G_R||`, mean | 0.292 | 0.277 |
| Compiled Jacobian at compiled value `||G_B-G_I||`, mean | 0.514 | 0.425 |
| Interaction norm, mean | 0.347 | 0.345 |
| Mean value-injection / total ratio | 0.446 | 0.550 |
| Mean residual-after-repair / total ratio | 1.053 | 0.971 |
| Mean interaction / total ratio | 0.710 | 0.827 |

All effect scales reproduce qualitatively. The interaction is comparable to the total discrepancy, and value repair does not monotonically recover the reference gradient. This is not a contradiction: vector effects can reinforce or cancel, and norm ratios are not additive shares.

The appropriate conclusion is:

> both output-value discrepancy and compiled Jacobian discrepancy matter for Qwen, but their combined transition effect is strongly non-additive and state-conditioned.

It would be incorrect to report “forward error explains 45–55%” as a contribution percentage. That ratio is one intervention-dependent norm comparison, not a Shapley value, variance share or causal fraction.

## Repair and injection answer different questions

- Injection asks whether the observed compiled boundary value can induce a gradient change through the eager Jacobian.
- Repair asks what gradient discrepancy remains when the compiled Jacobian is evaluated at the eager boundary value.
- Their disagreement is expected when the loss/model Jacobian is nonlinear or when compiled and eager differentiation paths differ.
- The interaction vector is necessary evidence; omitting it would create a false single-cause story.

The BERT result is close to Jacobian-only for most states. The Qwen result is a mixed, interacting mechanism. Thus a single universal attribution rule is already falsified.

## What this does not localize

The output boundary distinguishes **value injection** from **error propagation/differentiation**, but it does not identify where the discrepancy was created. It cannot name a reduction, cast, fusion or source operator.

The next BERT segmented-region experiment moves the prefix/suffix boundary through embeddings and encoder layers. Because segmentation changes graph partitioning, fusion and scheduling, it is audited against the monolithic endpoint. Any parity failure downgrades the result to intervention-dependent region attribution.

## Claim boundary

These results support causal effects only for the declared stop-gradient interventions. They do not prove:

- unique operator necessity or sufficiency;
- mathematical wrongness;
- that eager has the correct derivative;
- historical Adam/AdamW update impact;
- long-run optimization harm.

