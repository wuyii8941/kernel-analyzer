# Oracle Exploration Findings — 2026-07-15

## Purpose

This is a claim-limited exploration of existing ForkCert data for choosing an Oracle structure. It does not assume eager is truth, does not label implementation-relative differences as bugs, and does not use fork-selected cases to estimate a population rate.

## Finding 1: implementation identity must be part of the Oracle contract

The 100-rollout online file contains nonzero eager--compiled differences only for rollout batches 0--7. From rollout 8 onward, every recorded difference is exactly zero. The original systemd journal records at the same boundary:

```text
torch._dynamo hit config.recompile_limit (8)
```

The measurement code constructs one compiled wrapper and reuses it across changing inputs, but the output rows do not contain a generated-code or compiled-path canary. The exact alignment between the eight nonzero rollout batches and Dynamo's recompile limit is strong evidence that later calls fell back to eager execution.

Consequences:

- The full 51,200-row file must not be treated as 100 rollout batches of validated eager--compiled comparisons.
- The published denominator of 39,936 applicable decisions is not valid for an implementation-discrepancy rate unless compiled-path execution is independently recovered for every row.
- The conservative exploratory window is rollout batches 0--7 only: 4,096 applicable token decisions in 32 prompt-response cases.
- Future records need a per-state implementation-identity canary. A path label such as `path_alt=compile` is insufficient.

## Finding 2: a single global signed mean is not an adequate Oracle

Within the conservative eight-rollout window:

| Endpoint | Estimate | Moving-block 95% interval |
| --- | ---: | ---: |
| signed log-probability shift | `3.03e-6` | `[-1.15e-4, 1.39e-4]` |
| absolute log-probability difference | `2.13e-3` mean | descriptive only |
| clipping-event-oriented mean shift | `4.02e-5` | `[-2.39e-5, 1.16e-4]` |
| clipping disagreement | `5/4096 = 0.001221` | `[0.000244, 0.002686]` |
| directional clipping shift | `(4-1)/4096 = 0.000732` | `[0, 0.001465]` |

The numerical signed mean is near zero while the observed event changes are directionally imbalanced 4-to-1. With only eight serial rollout clusters this direction is not statistically stable. The data nevertheless demonstrate why signed numerical mean, absolute discrepancy, event disagreement, and directional event shift are distinct endpoints.

## Finding 3: boundary conditioning adds structure that global magnitude omits

Among the 4,096 applicable tokens:

- 36 reference decisions are within `0.01` log-margin of the clipping boundary.
- Four of the five disagreements occur within `0.01`; the fifth is in `[0.01, 0.1)`.
- The rank association between absolute numerical difference and reference boundary distance is negative (`Spearman = -0.289`) in this small window.

This supports retaining reference boundary distance and event-oriented shift in the semantic layer. It does not establish that a boundary score is superior to raw numerical delta for bug detection.

## Finding 4: observed runtime noise is zero only under the recorded deterministic protocol

Both attached eager self-pair and compiled self-pair maxima are zero in the conservative window. The independent-process Phase A1 data also report bitwise equality under the measured T4 FP16 configurations.

The supported statement is:

> Within-state runtime variability was not observed under this deterministic configuration and these observables.

It does not support a general claim that GPU execution noise is absent, or that stochastic configurations have zero runtime variance.

## Finding 5: grouping level changes the meaning of heterogeneity

For signed log-probability discrepancy in the conservative window:

| Grouping level | Between-group share of descriptive variance |
| --- | ---: |
| prompt-response case | `0.71%` |
| rollout batch | `0.16%` |

Most observed token-level dispersion occurs within a prompt-response case. This is not runtime variance: token identity and token position are part of the evaluated state. The result shows that the project must define the state unit before naming a variance component. Aggregating at checkpoint, rollout, prompt, or token level answers different questions.

## Finding 6: the endpoints are not reducible to one another

Existing complementary data already show:

- Common-random-number sampling changes 29 first-draw tokens over 1,024 fixed contexts, with zero self-sampling failures; 24 of the 29 changes occur without a candidate-set change.
- The matched step-5 case propagates a clipping disagreement into a target gradient difference and a nonzero one-step parameter-update distance.
- The 20-step counterfactual does not show that fork steps have systematically larger normalized full-gradient gaps.
- Mutation evaluation does not establish fork as a better general mutation detector than raw numerical delta.

Therefore the Oracle should be a structured profile, not one scalar endpoint:

1. numerical discrepancy endpoints;
2. state-conditioned and runtime components;
3. semantic-event distribution endpoints;
4. gradient/update/transition endpoints;
5. correctness verdict only when an independent specification exists.

## Immediate experimental requirement

The next GPU experiment should not repeat the original online scan unchanged. It needs:

- non-selected matched states;
- explicit compiled-path canaries for every state;
- fail-closed behavior on recompilation fallback;
- eager--eager, compiled--compiled, and eager--compiled rows;
- multiple observable layers through gradient, update, and next state;
- state identifiers that distinguish trajectory, checkpoint, minibatch, prompt, and token;
- execution order and randomness protocol recorded per pair.

The current environment exposed no CUDA device during this exploration, so no new GPU replay was launched. The CPU-side reanalysis artifacts are:

- `theory_oracle/exploration_existing.json` — full-file diagnostic, retained to reveal the fallback dilution;
- `theory_oracle/exploration_rollouts_0_7.json` — conservative pre-limit window;
- `theory_oracle/EXPLORATION_ROLLOUTS_0_7.md` — rendered endpoint and interval tables.

