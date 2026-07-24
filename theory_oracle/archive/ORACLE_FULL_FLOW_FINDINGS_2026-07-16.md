# Matched-state Oracle Full-flow Findings — 2026-07-16

## 1. Direct answer

The project should not define one context-free quantity called “the compiler error,” then force it into bias and variance. The defensible primitive is:

> under a frozen state population, implementation pair, observable, configuration and randomness/coupling protocol, characterize the conditional eager/compiled discrepancy and its downstream consequences.

For that declared contract, the current evidence supports a multi-endpoint **implementation-relative impact Oracle profile**. It does not yet support a correctness Oracle or a single scalar pass/fail threshold.

The profile is necessary because the experiments establish non-redundancy:

- a global signed output mean can reverse across checkpoints while absolute and semantic discrepancy persist;
- a decision can remain unchanged while the full gradient changes;
- the same numerical distribution shift can be dwarfed by ordinary sampling RNG;
- repair and injection can disagree because of interactions;
- segmenting a graph can change the very compiled endpoint being attributed.

## 2. What “bias and variance” mean here

Let `D(s)` denote candidate minus reference for one declared observable, with allowed runtime randomness averaged according to the contract.

The report should use four separate labels:

1. **Average implementation shift:** the state-population average of the conditional discrepancy. It is endpoint-, checkpoint-, population- and protocol-relative. Without independent truth it is not mathematical bias.
2. **State-conditioned heterogeneity:** stable variation of the implementation effect across states. It can remain large in completely deterministic execution.
3. **Within-state runtime variability:** changes when the same implementation executes the exact same state repeatedly under the declared runtime protocol.
4. **Sampling uncertainty:** uncertainty in population summaries because only finitely many states were observed. It is not a physical noise source.

Algorithmic RNG is another modeled source, not automatically runtime noise. For example, categorical token sampling occurs after deterministic logits in the current protocol; it must be calibrated conditional on each implementation's probability law.

Fixed reduction trees, reassociation, cast placement and fusion are deterministic implementation choices until an explicitly randomized or nondeterministic protocol makes them vary. “Floating point is variance” and “reduction is bias” are therefore rejected as mechanism-by-name classifications.

## 3. Evidence ladder

| Layer | Main empirical result | Why the layer cannot be replaced by the previous one |
|---|---|---|
| Calibration | All formal candidate paths compiled one tracked graph; self-pairs were exact; no measurement-time graph proliferation | A mislabeled fallback would make every downstream estimand invalid |
| Numerical | BERT, ResNet and Qwen have deterministic nonzero discrepancy and state heterogeneity; Qwen final-checkpoint signed shift reverses at earlier checkpoints | Absolute delta and global direction answer different questions |
| Invariance/boundary | About 22% of Qwen logit-delta energy is common-mode; boundary distance ranks ranking changes far better than raw or centered magnitude | Decision-invariant translation must be separated; magnitude alone does not predict event crossing |
| Semantic | Qwen top-5 disagreement reproduces near 1.2%; argmax is rarer and less stable | A numerical difference need not alter the declared event |
| Sampling law | Mean TV is about 0.066–0.105%, while ordinary independent token disagreement is about 16.7–17.8% | A realized sampled-token fork confounds distribution shift with algorithmic RNG |
| Transition | Every measured BERT/Qwen state has a nonexact full gradient; relative discrepancy reproduces even when loss direction and predictions do not | Forward loss/event equality does not imply equal one-step transition |
| Output-boundary attribution | BERT is Jacobian-path dominated in most states; Qwen value and Jacobian effects strongly interact | The same total gradient delta can arise through different mechanisms |
| Region attribution | Region responses reproduce descriptively, but segmented compiled endpoints fail exact monolithic parity | Intervention response is not automatically an original-program operator cause |

## 4. Main empirical conclusions

### 4.1 There is discrepancy, but no universal direction bias

- BERT and ResNet signed-logit confidence intervals include zero.
- The final Qwen checkpoint has a reproduced negative raw-logit shift, but three earlier checkpoints have positive shifts under the same input bank and compiled graph hash.
- Qwen signed sampling-probability and entropy summaries reverse or include zero across discovery/confirmation.
- Absolute discrepancy, state heterogeneity and several nonnegative impact distances reproduce more reliably than a global sign.

Thus “bias exists” is meaningful only after naming endpoint and state population. The evidence rejects a universal compiler-direction claim.

### 4.2 State dependence currently dominates runtime noise

Under the deterministic formal protocols, same-state repeated inference, gradient, attribution and sampling-logit measurements were exact. Runtime variability was estimated as zero. Across-state effect ranges and tails were large.

This is not evidence that GPU execution is always deterministic. It says that, in this protocol, observed spread belongs to state-conditioned effect heterogeneity, not repeated-execution noise. Atomics, stochastic rounding, nondeterministic kernels and autotuning-selection regimes remain separate experiments.

### 4.3 Semantic and transition endpoints add information

Boundary distance was substantially more informative for argmax/top-5 exposure than raw or centered numerical magnitude. Centering remains a correct invariance decomposition but failed the stronger criterion of improving event ranking.

The transition study gives a second non-redundancy result. BERT can have bitwise-equal final logits and prediction with nonexact gradients. Qwen relative full-gradient discrepancy is around 1.7–1.8%; BERT is around 0.07–0.08%. These are controlled one-step effects, not historical optimizer replay or long-run harm.

### 4.4 Sampling requires a law-level Oracle

For Qwen at temperature 0.7, exact categorical TV is small, nonzero and highly state-conditioned. Independent token mismatch is roughly two orders of magnitude larger and would remain large even if the two laws were identical. A common-RNG mismatch is coupling-dependent and is not a marginal distribution distance.

Therefore the sampling Oracle compares probability laws first. Realized token events are secondary diagnostics under a declared coupling.

### 4.5 Attribution is conditional on the intervention

Output-boundary stop-gradient arms preserve their intended values exactly and support effects for those precise interventions. They show that BERT and Qwen do not share one mechanism and that norm ratios are not additive contribution shares.

BERT segmented-region experiments fail the stronger monolithic-parity requirement. They reveal intervention sensitivity and propagation structure but cannot identify a unique source operator in the original graph. This negative result is central: repair removing or changing an effect is insufficient without treatment identity and endpoint parity.

## 5. Recommended Oracle v1 output

For every frozen comparison contract, emit this profile rather than one number:

```text
OracleProfile = {
  contract_and_execution_validity,
  numerical_discrepancy_geometry,
  average_implementation_shift,
  state_conditioned_distribution_and_tails,
  within_state_runtime_variability,
  finite_state_uncertainty,
  semantic_event_or_distribution_shift,
  one_step_transition_discrepancy,
  attribution_with_intervention_integrity,
  specification_and_acceptance_status,
  endpoint_specific_verdicts,
  claim_level
}
```

Endpoint verdicts have distinct meanings:

- `invalid`: comparison or treatment integrity failed;
- `measurement only`: discrepancy estimated but no acceptable region exists;
- `equivalent within declared bound`: confidence set lies inside a predeclared practical bound;
- `operational drift`: confidence set violates a predeclared application bound;
- `correctness violation`: an independent specification or truth-relative bound is violated;
- `indeterminate`: current uncertainty overlaps the decision boundary.

The current real-model results are mostly `measurement only`. They are not compiler pass/fail verdicts because no independent practical tolerance or mathematical specification has been provided.

## 6. What has been falsified or downgraded

1. **One global mean as the Oracle:** falsified by checkpoint sign reversal, common-mode invariance and near-zero means with nonzero semantic/transition effects.
2. **Single fork as the Oracle:** falsified by sparse/state-conditioned event behavior and coupling dependence.
3. **Single sampled token as the Oracle:** falsified by algorithmic RNG disagreement dominating implementation TV.
4. **Loss equality implies equal training step:** falsified by nonexact gradients at equal loss/output states.
5. **Centered delta is a stronger detector:** downgraded; useful for invariance, not stable event ranking.
6. **Additive repair/injection percentages:** falsified by interactions comparable to total effects and ratios above one.
7. **Segmented replacement identifies root cause:** falsified for the current BERT intervention by monolithic-parity failure.
8. **No observed runtime noise means no variance problem:** rejected; it is only a statement about one deterministic protocol.
9. **Observed relative discrepancy is a correctness error:** unavailable without specification.

## 7. What remains scientifically open

The next work should be selected by the desired claim, not by whichever discrepancy is easiest to produce.

### If the goal is an operational impact Oracle

1. Choose the target state distribution: reference trajectory, compiled trajectory, or explicitly weighted mixture. This changes the population average and tail risk.
2. Choose the decision endpoint and consequence: event-law shift, categorical TV, full-gradient/update geometry, or an application loss over them.
3. Supply a predeclared acceptable region from deployment or training requirements. Data cannot manufacture its own threshold.
4. Validate on held-out checkpoint/trajectory clusters, not only held-out examples at one checkpoint.

### If the goal is a stochastic-variance Oracle

Open one source at a time: execution nondeterminism, autotuning selection, stochastic rounding or algorithmic RNG. Estimate eager and compiled marginal variability plus their paired covariance. Do not pool sources or subtract `2 sigma^2/R` unless independence, equal variance and balanced repeats are verified.

### If the goal is operator localization

The next intervention must preserve the original compiled context or explicitly model context changes. Source, propagation and boundary conversion should be different roles. A candidate operator effect is credible only when:

- the intended treatment actually executes;
- non-target graph choices remain fixed or are part of the treatment definition;
- the hybrid endpoint matches the original anchors under the predeclared integrity relation;
- repair, injection and interaction are all reported;
- discovery/confirmation state rankings reproduce.

Otherwise the output remains intervention-dependent attribution.

### If the goal is correctness

Add high-precision reference values, formal semantics/invariants, or confirmed wrong-code cases. Eager remains merely a baseline until then. A legal but different floating-point implementation with persistent impact is an operational or reproducibility risk, not automatically a compiler bug.

### If the goal is long-run training impact

Treat it as external validation or a separate transition-kernel problem. Matched one-step comparison identifies local effects at shared states; free-running trajectories confound local implementation effects with visitation-distribution feedback. Predicting convergence requires additional stability, contraction/mixing, or task-specific robustness assumptions.

## 8. Strongest defensible research claim

The mature components—paired comparison, variance components, differential testing, branch proximity, distribution distances and causal interventions—are not individually novel. The empirically supported research object is their disciplined relational composition:

> an execution-calibrated matched-state Oracle profile that keeps numerical discrepancy, boundary-conditioned semantic-law change, one-step transition change and intervention integrity distinct, while explicitly separating implementation-relative impact from specification-backed correctness.

This remains vulnerable to the reviewer criticism “a careful combination of known techniques.” It becomes a stronger contribution only if the profile predicts or localizes meaningful transition/semantic impact beyond raw delta, generalizes across declared state populations, and supports attribution without violating intervention integrity.

## 9. Immediate decision

The measurement/decomposition phase is mature enough to freeze an Oracle v1 schema. The immediate blocker is no longer discovering another statistic; it is choosing what counts as unacceptable.

Before producing a binary operational Oracle, the project owner must select:

1. target state population;
2. primary impact endpoint(s);
3. independent acceptance bounds or application loss;
4. whether correctness is in scope;
5. whether operator attribution must preserve the original monolithic graph exactly.

Until those choices are supplied, the scientifically correct product is a validated measurement-and-impact profile with explicit uncertainty and claim boundaries—not a bug verdict.
