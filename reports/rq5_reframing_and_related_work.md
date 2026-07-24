# RQ5 Reframing And Related-Work Synthesis

## Revised Claim

Fork is not a universal replacement for numerical delta. It is a parameter-free observation at a semantically privileged decision surface. Delta measures numerical infection and provides broad sensitivity; fork measures one form of semantic propagation and provides direct interpretation. Gradient/update and task-level effects are later, stronger endpoints.

The evidence now supports a staged oracle rather than a winner-takes-all comparison:

```text
numeric infection       decision kill       state kill          task kill
delta != 0          ->  branch/token fork -> gradient/update -> reward/quality
```

For clipping, a fork is an internal semantic checkpoint. It reaches state-kill only when exact replay shows a different gradient or update. For sampling, a sampled-token fork is already an externally visible rollout difference, but reward and training-data consequences remain a later endpoint.

## New RQ5 Evidence

### Paired Testing Comparison

- Fork: 420 artificial-mutation true positives and 5 legal-path alarms.
- Delta top-425: 425 artificial-mutation true positives and no legal-path alarms.
- Paired correctness: 114 fork-only correct, 124 delta-only correct.
- Exact McNemar `p=0.5597`.
- Prompt-cluster bootstrap intervals for delta-minus-fork precision and recall both contain zero.

The point estimates reject the preregistered claim that fork has higher precision. They do not establish that delta has a statistically reliable token-level advantage.

### Test-Budget Diversity

- Fork true positives cover 11/15 mutation families.
- Delta top-425 true positives cover 1/15 families.
- All delta alarms are consumed by the largest-amplitude `rmsnorm_no_upcast` mutation.

This suggests an amplitude-masking effect: global delta ranking can spend a finite inspection budget repeatedly reporting one obvious failure family. Decision events allocate alarms across more mechanisms. The current catalog is artificial, so this is a mutation-family coverage result rather than a historical-bug discovery result.

### Delayed Forks

Four mutations have zero clipping forks at the initial frozen state. None is training-equivalent:

| Mutation | First clipping fork | Step-1 update distance / legal pair | 20-step branch-fork events |
|---|---:|---:|---:|
| rotary phase in FP16 | 3 | 0.489 | 78 |
| decoder-0 BF16 round trip | 2 | 0.443 | 79 |
| FP16 log-softmax | 2 | 0.406 | 74 |
| reversed chunked logsumexp | 2 | 0.257 | 76 |

At step 1 all clipping branches agree, but each parameter update is already nonzero. Continuous within-branch gradient differences first change model state; the changed state then crosses clipping boundaries at step 2 or 3. An independent clean rerun is bitwise identical at checkpoints 1, 5 and 20.

This yields two useful definitions:

- **fork latency:** optimizer steps from an observed numerical infection to the first decision fork under a fixed matched trajectory.
- **k-step decision stability:** no decision fork occurs between two paths over the next `k` matched updates. One-step decision stability does not imply k-step stability.

## Relation To Classical Testing Theory

### PIE/RIP

Voas's PIE analysis separates execution, state infection and propagation to a failure. ForkCert instantiates a measurable propagation checkpoint for DL training. The Phase 10 result also shows that propagation may remain continuous for one update before reaching a discrete decision surface.

The mapping must not collapse all stages:

- operator/logprob delta: infection;
- clipping or sampling decision difference: decision-level propagation;
- gradient or parameter update difference: training-state propagation;
- reward or quality difference: externally visible failure.

### Weak And Strong Mutation

Howden's weak mutation requires the altered statement to create an incorrect local state. Strong mutation requires propagation to observable output. A clipping fork alone is between these endpoints: it is more semantic than arbitrary tensor infection but is still internal. Calling every clipping fork a strong kill is too broad. Exact gradient/update replay promotes a concrete clipping case to a training-state kill.

### Equivalent Mutants

An equivalent mutant cannot be distinguished from the original by any input. Four survivors of one clipping scan over one batch are not equivalent mutants. The defensible terms are `clipping-surviving mutant` or `boundary-equivalent on the observed trace`. Phase 10 directly refutes training equivalence for these four cases.

### Oracle Problem

Barr et al. define the oracle problem as deciding whether observed behavior is correct. Fork is a partial domain oracle: a decision mismatch is unambiguously a semantic difference under a frozen path contract, but it does not decide whether either implementation violates a numerical correctness contract. This distinction preserves the natural-fork result without labelling legal floating-point behavior a bug.

## Relation To DL Testing And Numerical Work

### CRADLE And TTrace

CRADLE compares backend outputs using class/MAD thresholds and localizes deviation growth in hidden states. TTrace compares distributed-training tensors with a trusted reference and derives tolerance guidance. ForkCert changes the observation point: it asks whether a deviation changes a training or rollout decision, then uses intervention and replay to connect that event to state consequences. The methods are complementary because TTrace/CRADLE retain sensitivity to continuous drift that has not yet crossed a decision surface.

### TAO/NAO

TAO constructs operator-level acceptance regions from theoretical bounds and empirical hardware profiles. ForkCert's Phase 2 shows why an end-to-end bound is not currently available for the measured Inductor graph. A future hybrid can use TAO-style local contracts for infection validity and ForkCert decision events for downstream semantic severity.

### TIM, FP16, And Bitwise Alignment

Recent TIM work shows that trainer/rollout numerical disagreement can change the effective optimization problem and cause collapse; FP16 and bitwise-aligned stacks reduce the mismatch. ForkCert is narrower and more mechanistic: it records individual boundary crossings, locates sufficient scheduling interventions, and measures exact gradient/update consequences. It should not claim discovery of TIM or general training sensitivity.

### Entropy-Preserving RL

Entropy-Preserving RL proves that finite-precision ratio computation can create an upward multiplicative bias and effective clipping asymmetry. This implies that a near-zero global signed-logprob mean does not exclude boundary-conditioned directional effects. ForkCert can test this by conditioning signed deltas on boundary side, advantage sign, probability rank and entropy change rather than searching only for a global mean.

## Strongest Additional Research Directions

1. **Temporal fork hazard:** predict fork latency from initial signed margin, signed delta and one-step update sensitivity. This extends the current instantaneous predictor into survival/hazard prediction.
2. **Boundary-conditioned bias:** test whether near-boundary delta direction systematically adds or removes gradient mass even when the global signed mean is zero.
3. **Cascaded testing:** use delta/margin for high-recall candidate selection, fork for decision-semantic triage, and matched replay for expensive state-kill confirmation.
4. **Diversity-aware test allocation:** compare unique historical bug families found per inspection budget, not only token-level mutation labels.
5. **Selective alignment:** switch only high-risk scheduling classes or recompute only low-margin decisions, comparing overhead with full bitwise alignment.
6. **Cross-decision propagation:** test whether clipping-surviving mutations already trigger sampling forks, and whether sampling forks alter reward/advantage.

## Paper Contribution Package

The defensible contribution order is:

1. Natural training-semantic forks exist in clipping and sampling.
2. Decision forks form a distinct propagation checkpoint between numerical infection and training-state consequence.
3. Five natural clipping forks have singleton scheduling-class repairs; one has a matched one-step causal update result.
4. Initially sub-boundary mutations can diverge continuously and create delayed forks, motivating fork latency and k-step decision stability.
5. Fork and delta are complementary in testing: statistically similar token-level mutation identification, broader family coverage for fork, broader pre-fork sensitivity for delta.
6. Global analytic certification is presently blocked by missing kernel contracts and graph-level arithmetic mappings.

## Primary References

- Jeffrey Voas, [The Revealing Power of a Test Case](https://doi.org/10.1002/stvr.4370020105), 1992.
- William Howden, [Weak Mutation Testing and Completeness of Test Sets](https://personal.utdallas.edu/~lxz144130/cs6301-readings/howden-tse82.pdf), 1982.
- Barr et al., [The Oracle Problem in Software Testing: A Survey](https://philmcminn.com/publications/barr2015.pdf), 2015.
- Pham et al., [CRADLE: Cross-Backend Validation to Detect and Localize Bugs in Deep Learning Libraries](https://www.cs.purdue.edu/homes/lintan/publications/cradle-icse19.pdf), ICSE 2019.
- Jiang et al., [TTrace: Lightweight Error Checking and Diagnosis for Distributed Training](https://arxiv.org/abs/2506.09280), 2025.
- Yao et al., [TAO: Tolerance-Aware Optimistic Verification for Floating-Point Neural Networks](https://arxiv.org/abs/2510.16028), 2025.
- Qi et al., [Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/abs/2510.26788), 2025.
- Zhong et al., [Diagnosing Training Inference Mismatch in LLM Reinforcement Learning](https://arxiv.org/abs/2605.14220), 2026.
- Petrenko et al., [Entropy-Preserving Reinforcement Learning](https://openreview.net/pdf?id=865bab2ae33bb19c1e5ad100ebce4e1c1ee38d3c), ICLR 2026.
