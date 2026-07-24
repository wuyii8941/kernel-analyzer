# Matched One-step Transition Findings — 2026-07-16

## Bottom line

The controlled one-step study establishes a local transition discrepancy that is invisible to a forward-event-only Oracle:

- BERT and Qwen have nonzero full-gradient eager/compiled differences at every measured matched state;
- the gradient-discrepancy scale reproduces on held-out state banks;
- same-state loss and gradient-discrepancy repeat variance is zero under the deterministic protocol;
- loss direction is not persistent and prediction events need not change;
- the declared SGD next-state discrepancy is therefore real but small and protocol-relative;
- nothing here establishes historical optimizer replay, long-run harm or incorrectness.

The contract was frozen in `ORACLE_NEXT_STAGE_CONTRACT_2026-07-16.md` before formal outputs were read.

## Comparison

| Subject / split | States | Mean signed loss delta, 95% state-bootstrap CI | Mean gradient-difference L2, 95% CI | Mean relative gradient difference | Mean gradient cosine | Same-state repeat variance | Prediction event changes |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT discovery | 128 | 5.73e-6 [-2.03e-6, 1.48e-5] | 6.76e-3 [5.05e-3, 8.64e-3] | 8.31e-4 | 0.99999947 | 0 | 0 |
| BERT confirmation | 128 | -2.57e-5 [-7.33e-5, 1.26e-5] | 8.68e-3 [6.66e-3, 1.08e-2] | 7.05e-4 | 0.99999973 | 0 | 0 |
| Qwen discovery | 8 minibatches | -6.04e-5 [-3.36e-4, 1.90e-4] | 0.513 [0.434, 0.610] | 1.813e-2 | 0.9998300 | 0 | n/a |
| Qwen confirmation | 8 minibatches | -1.30e-4 [-3.55e-4, 8.30e-5] | 0.440 [0.403, 0.475] | 1.730e-2 | 0.9998489 | 0 | n/a |

All compiled-call, graph-stability, finite-gradient and self-pair gates passed. Every formal split compiled one stable graph and performed no new compilation during measurement.

## What this identifies

For a fixed checkpoint, input/target state, deterministic execution protocol and controlled optimizer map, it identifies:

> the implementation-relative difference between the full gradients and the next states induced by one declared SGD step.

The two implementations share the same model object and parameters. No update is carried from one observed state to the next, preventing feedback divergence from contaminating the local estimate.

For no-momentum SGD, update difference is exactly learning rate times gradient difference. It is reported to make the transition explicit, not as independent evidence.

| Subject / split | Mean derived SGD next-state L2 difference | Maximum next-parameter coordinate difference |
|---|---:|---:|
| BERT discovery | 6.76e-8 | 8.25e-8 |
| BERT confirmation | 8.68e-8 | 9.63e-8 |
| Qwen discovery | 5.13e-7 | 3.50e-8 |
| Qwen confirmation | 4.40e-7 | 4.00e-8 |

## Counterexamples to scalar-Oracle reasoning

### Equal or near-equal loss does not imply equal transition

The BERT smoke state had exactly equal reported losses and identical predictions, yet relative gradient L2 was approximately 1.54e-3. Across formal BERT states, no gradient vector was exactly equal.

### Global direction is not required for local transition discrepancy

The signed loss intervals include zero in all four formal splits. Nevertheless, the nonnegative full-gradient-distance intervals are bounded away from zero and reproduce. A mean loss bias and a gradient/update discrepancy are different estimands.

### Absolute gradient difference is partly scale-conditioned

In BERT, reference gradient norm and absolute gradient-difference L2 have Spearman correlations around 0.97 in both splits. Relative gradient difference is therefore the more portable scale descriptor. Qwen does not reproduce this same correlation structure, so a universal proportional-noise model is not supported.

### Raw loss delta is an incomplete proxy

The association between absolute loss delta and gradient-difference L2 is only weak to moderate and changes by split/subject. The transition endpoint is not recoverable by thresholding scalar loss delta alone.

## State heterogeneity and runtime variability

The gradient discrepancy varies substantially across states:

- BERT discovery gradient-difference L2 ranges from roughly 9.0e-5 to 5.17e-2;
- BERT confirmation ranges from roughly 9.3e-5 to 6.60e-2;
- Qwen discovery ranges from 0.390 to 0.786;
- Qwen confirmation ranges from 0.336 to 0.530.

This spread is deterministic state-conditioned heterogeneity. It must not be called runtime variance. Same-state repeated loss and gradient distances were identical in every run.

## Parameter-block observation

BERT gradient discrepancy is distributed across the model rather than isolated to the classifier. Aggregated absolute block effects are largest in embeddings, followed by encoder layer 0, encoder layer 1, classifier weight and pooler in both splits. This is a propagation description, not evidence that embeddings are the source operator: upstream parameters naturally accumulate downstream sensitivity, and absolute block norms are affected by block size and gradient scale.

## Claim boundary

The controlled optimizer is full-parameter SGD with no momentum or weight decay. Historical Adam/AdamW state was unavailable and was not reconstructed. Qwen uses a teacher-forced response-token loss, not the production GRPO surrogate. Consequently, this stage supports controlled transition sensitivity, not replay of the original training program.

One-step discrepancy does not imply divergence, slower convergence or lower final quality. Those require a coupled-kernel or long-run validation problem with additional stability/mixing assumptions.

## Attribution entry

The reproduced BERT and Qwen gradient endpoints satisfy the predeclared entry condition for repair/injection. The next stage uses an output-boundary 2x2 value/Jacobian factorial before attempting source-region localization. This avoids the concept swap “repair removed a difference, therefore one operator is the root cause.”

