# Phase 8 Matched-Step Counterfactual Confound Analysis

## Objective

Test whether the previously observed association between clipping-fork steps and faster parameter divergence can be explained trivially by larger full-batch gradient scale.

## Controls And Scope

- Same frozen step-5 state, batch, old logprobs, advantages, optimizer, learning rate and target token for A eager and B default compile.
- A fork step is defined before analysis as `A.target_clip_active != B.target_clip_active`.
- There are 20 matched steps and 7 target-token fork steps: 1, 3, 6, 10, 11, 18, 19.
- Parameter checkpoints exist only at steps 1, 5 and 20. Therefore this analysis does **not** reconstruct a 20-step parameter-distance increment series.

## Result

| Metric | Fork-step mean | Non-fork mean | Ratio | Mean difference | Bootstrap 95% CI | Exact permutation p |
|---|---:|---:|---:|---:|---:|---:|
| Average full-gradient norm | 7.0188455 | 6.8349616 | 1.027 | 0.18388391 | [-0.221, 0.623] | 0.3892 |
| Absolute gradient-norm gap | 0.074300544 | 0.07704436 | 0.964 | -0.0027438165 | [-0.0581, 0.0569] | 0.9316 |
| Normalized gradient-norm gap | 0.010558423 | 0.011415442 | 0.925 | -0.00085701956 | [-0.00866, 0.00728] | 0.8515 |
| Absolute loss gap | 9.8253999e-05 | 6.098816e-05 | 1.611 | 3.7265839e-05 | [-2.18e-05, 0.000101] | 0.198 |
| Absolute target-logp gap | 0.0076772656 | 0.002043504 | 3.757 | 0.0056337616 | [0.00221, 0.00879] | 0.004683 |

Fork steps have an average full-gradient norm ratio of `1.027` relative to non-fork steps. The normalized A/B gradient-norm gap ratio is `0.925`.

All `7` branch-fork rows also disagree on whether the target token has zero versus non-zero loss gradient.

## Parameter-Distance Anchors

| Step | A-B L2 | A-C L2 | A-C / A-B |
|---:|---:|---:|---:|
| 1 | 1.1049261e-05 | 4.9810227e-06 | 0.4508 |
| 5 | 3.6014474e-05 | 1.4050677e-05 | 0.3901 |
| 20 | 5.2171284e-05 | 5.5094004e-05 | 1.0560 |

## Interpretation

This analysis separates two questions. The average full-gradient norm checks the simple scale confound; the normalized A/B norm gap checks whether fork steps coincide with a larger path-dependent gradient disturbance after dividing out batch gradient scale.

It cannot validate the earlier `6.49x` parameter-divergence jump ratio because that result comes from a different 100-step twin trajectory and the current 20-step run lacks intermediate parameter checkpoints. A strict normalized jump-ratio claim requires rerunning and saving every step, or recording gradient vectors sufficient to predict each update distance.

The exact permutation and bootstrap numbers are descriptive only: the 20 observations are serially dependent and come from one replay trajectory. They must not be presented as cross-prompt or cross-checkpoint significance.

## Artifacts

- Source trajectories: `results/trajectory_step5_fusion/A_reference.json`, `B_alternative.json`
- Checkpoint anchors: `results/trajectory_step5_fusion/merged.json`
- Structured analysis: `results/phase8_matched_step_counterfactual.json`
- Analysis script: `scripts/phase8_matched_step_counterfactual.py`

## Decision

**REVISE.** Use this as a gradient-scale confound audit, not as proof of a normalized parameter-jump effect. The single-step A/B/C intervention remains the direct causal result; the long-horizon 6.49x timing result remains coupling evidence pending a fully instrumented rerun.
