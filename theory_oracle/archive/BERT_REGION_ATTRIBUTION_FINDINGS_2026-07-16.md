# BERT Segmented-region Attribution Findings — 2026-07-16

## Bottom line

Moving a compiled/eager intervention boundary through BERT produces reproducible intervention effects, but it does not establish an operator causal effect for the original monolithic compiled graph.

The segmented eager endpoint is exact. None of the segmented compiled endpoints is exactly equal to the monolithic compiled gradient endpoint. Segmentation changes graph partitioning and therefore potentially fusion, scheduling, layout and numerical order. The formal integrity contract consequently downgrades all results to **intervention-dependent region attribution**.

Within that restricted interpretation, the typical compiled-prefix injection effect is small at the embedding boundary and becomes comparable to the segmented total difference after encoder layer 0. Repairing the prefix nevertheless leaves nearly all of the total difference, and the interaction is also comparable to the total. This is evidence against unique necessity, sufficiency and additive contribution stories.

## Formal results

Each cell uses 16 states and two exactly repeated observations per state. Ratios are state-level norm ratios averaged across states; they are not additive shares.

| Boundary / split | Mean segmented total `||B-A||` | Mean prefix injection `||I-A||` | Mean residual after prefix repair `||R-A||` | Mean interaction | Mean injection / total | Mean residual / total | Mean interaction / total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Embeddings discovery | 2.98e-3 | 2.12e-3 | 2.34e-3 | 2.33e-3 | 0.361 | 0.981 | 0.576 |
| Embeddings confirmation | 4.75e-3 | 1.15e-3 | 4.84e-3 | 2.74e-3 | 0.308 | 0.972 | 0.569 |
| Layer 0 discovery | 2.28e-3 | 6.37e-4 | 2.19e-3 | 1.23e-3 | 0.634 | 0.913 | 0.794 |
| Layer 0 confirmation | 5.15e-3 | 2.91e-3 | 4.90e-3 | 3.89e-3 | 0.612 | 0.934 | 0.830 |
| Layer 1 discovery | 2.27e-3 | 6.33e-4 | 2.19e-3 | 1.25e-3 | 0.693 | 0.917 | 0.886 |
| Layer 1 confirmation | 5.13e-3 | 2.95e-3 | 4.78e-3 | 3.90e-3 | 0.669 | 0.925 | 0.875 |

All execution-identity, stable-graph, no-measurement-recompile and same-state repeat gates passed. Same-state effect variance was zero for every arm and boundary.

## Means hide state-conditioned tails

Mean ratios alone are misleading. Median injection/total ratios were:

| Boundary | Discovery median | Confirmation median |
|---|---:|---:|
| Embeddings | 0.104 | 0.0039 |
| Layer 0 | 0.886 | 0.776 |
| Layer 1 | 0.906 | 0.865 |

The embedding mean is driven by sparse large states and does not describe a typical state; even its discovery/confirmation median is unstable. Layer 0 and layer 1 show a more reproducible typical-state increase. However, their median residual-after-repair ratios remain approximately 0.98–0.99, and median interactions are approximately 0.93–1.11. Prefix injection being large does not imply prefix repair is sufficient, because vector effects interact and cancel.

This is deterministic state heterogeneity, not runtime variability.

## Monolithic-parity audit

| Boundary / split | Maximum relative error in segmented versus monolithic gradient-difference norm | Candidate endpoint exact? |
|---|---:|---:|
| Embeddings discovery | 1.817 | No |
| Embeddings confirmation | 0.188 | No |
| Layer 0 discovery | 3.19e-6 | No |
| Layer 0 confirmation | 3.30e-2 | No |
| Layer 1 discovery | 1.52e-3 | No |
| Layer 1 confirmation | 7.78e-3 | No |

Relative error can be unstable when the original discrepancy norm is tiny, but exact parity was the predeclared causal-integrity requirement and it fails in all six studies. The embedding segmentation also changes candidate loss in some states. A small average parity error would not repair the conceptual problem: the intervention operates on a changed compiled program.

## What can and cannot be said

Supported:

- under the declared segmented program, the effect exposed before layer 0 is usually smaller than the effect exposed after layer 0;
- downstream compiled/eager paths and prefix values interact strongly;
- state-conditioned tails matter and exact repeats show no runtime noise;
- a single additive region-contribution percentage is not defensible.

Not supported:

- layer 0 or layer 1 is the unique numerical-error source;
- an embedding/layer prefix is necessary or sufficient for the original graph discrepancy;
- repair removed a fraction of the original compiler error;
- the region ranking identifies a wrong operator;
- either eager or compiled derivative is mathematically correct.

## Oracle consequence

Repair/injection is useful only after its endpoint is fixed and intervention integrity is audited. A region result should be emitted with one of two labels:

1. **original-program causal attribution**, only when treatment identity and endpoint parity hold to the predeclared requirement;
2. **intervention-dependent attribution**, when the splice or segmentation changes the program being explained.

This study lands in the second category. It is a scientifically useful negative result: naive region replacement does not yet provide the operator-level causal semantics required by the Oracle plan.

## Kill-criterion audit

- The claim “segmented repair/injection locates an original-graph root cause” is killed by parity failure.
- The claim “one prefix explains an additive fraction of the gradient discrepancy” is killed by residual and interaction effects of the same order as the total.
- The restricted claim “the segmented intervention response is state-conditioned and changes across boundaries” survives held-out confirmation.
