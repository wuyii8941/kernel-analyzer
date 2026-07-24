# Multi-subject Matched-state Oracle Findings — 2026-07-15

## Bottom line

The experiment does **not** support the statement “there is no bias.” It supports a more structured conclusion:

1. all three subjects exhibit nonzero eager/compiled numerical discrepancy;
2. the discrepancy is deterministic under the frozen protocol: same-state repeat variability is zero in every run;
3. BERT and ResNet do not show a confirmed global mean signed-logit direction on the sampled state populations;
4. the final Qwen checkpoint does show a confirmed negative mean signed-logit shift on its discovery and held-out confirmation populations;
5. that Qwen direction reverses at three earlier checkpoints even though the compiled graph hash and input bank are unchanged, so it is model-state conditional rather than a universal compiler direction;
6. semantic effects cannot be inferred from the global numerical mean: Qwen top-5 set disagreement reproduces near 1.2%, while target log-probability direction and argmax disagreement rate are less stable;
7. no result here establishes mathematical incorrectness, a compiler bug, or long-run training harm.

The evidence therefore favors an Oracle **profile**, not a scalar pass/fail statistic:

> measurement validity; endpoint-specific average shift; state-conditioned heterogeneity; same-state runtime variability; boundary exposure; semantic-event discrepancy; and, later, one-step transition discrepancy.

## Frozen comparison

The protocol was frozen in `MULTISUBJECT_ORACLE_CONTRACT_2026-07-15.md` before reading the formal outputs.

- reference: PyTorch eager CUDA;
- candidate: tracked `torch.compile`/Inductor full graph;
- FP16, evaluation mode, deterministic algorithms, fixed shape, batch size 1;
- three repetitions of both paths per state with alternating order;
- separate discovery and held-out confirmation state banks;
- eager is a baseline, not mathematical truth.

The subjects were:

- BERT-tiny on 128 + 128 SST-2 validation states;
- ResNet-18/ImageNet graph on 128 + 128 real CIFAR-10 images, used only as numerical/ranking inputs because the label spaces differ;
- Qwen3-0.6B on 32 + 32 real GRPO rollout sequences, with token positions nested within each sequence state;
- three earlier Qwen checkpoints on the same 16-sequence stress bank, kept separate from the primary population.

Every formal run compiled exactly one graph, recorded a compiled runtime call for every candidate invocation, and had zero non-exact eager/eager or compiled/compiled self-pairs.

## Discovery and held-out confirmation

| Subject | Split | Mean signed logit delta, 95% state-bootstrap CI | Mean absolute logit delta | Top-1 margin heterogeneity variance | Same-state repeat variance | Argmax disagreement | Top-5 set disagreement |
|---|---|---:|---:|---:|---:|---:|---:|
| BERT | discovery | 2.31e-5 [-2.71e-5, 7.38e-5] | 3.78e-4 | 1.70e-6 | 0 | 0/128 | n/a |
| BERT | confirmation | -1.90e-6 [-5.55e-5, 5.15e-5] | 4.04e-4 | 1.59e-6 | 0 | 0/128 | n/a |
| ResNet | discovery | -1.74e-6 [-5.94e-6, 2.65e-6] | 1.29e-3 | 2.74e-5 | 0 | 1/128 | 3/128 |
| ResNet | confirmation | -1.85e-6 [-6.21e-6, 2.57e-6] | 1.34e-3 | 2.74e-5 | 0 | 0/128 | 0/128 |
| Qwen | discovery | -8.55e-4 [-1.24e-3, -4.46e-4] | 6.30e-3 | 2.90e-6 | 0 | 12/4092 | 49/4092 |
| Qwen | confirmation | -9.43e-4 [-1.20e-3, -6.90e-4] | 6.34e-3 | 2.11e-6 | 0 | 3/4084 | 47/4084 |

Here, “heterogeneity variance” means variance of the state-level mean top-1-margin effect. It is not runtime randomness. The bootstrap interval represents uncertainty from observing finitely many states; it is not another physical variance component.

### What reproduced

- The absolute discrepancy scale reproduced closely within each subject.
- The state-heterogeneity scale reproduced closely for BERT and ResNet and remained of the same order for Qwen.
- Runtime variability remained exactly zero under this deterministic protocol.
- Qwen top-5 set disagreement reproduced: 1.197% in discovery and 1.151% in confirmation.
- Qwen ordered top-list disagreement also reproduced: 185/4092 and 181/4084.
- Every observed argmax or top-5 set change was inside the conservative boundary-risk set; there were zero stability-condition violations.

### What did not reproduce as a single direction

- BERT global signed-logit intervals include zero in both splits.
- ResNet global signed-logit intervals include zero in both splits. Its mean signed top-1-margin delta changes sign between splits.
- ResNet's sparse argmax/top-5 set events do not repeat in the held-out bank, although ordered-list changes remain.
- Qwen target-token log-probability shift is not a persistent direction: discovery is compatible with zero, whereas confirmation is small and positive.
- Qwen argmax disagreement is present in both banks but its rate changes from 0.293% to 0.0735%; with only 32 sequence states per split, this is descriptive rather than a precise population rate.

## Checkpoint stress result

The three earlier Qwen checkpoints used the same first 16 sequences and produced the same compiled graph hash as the final checkpoint.

| Checkpoint | Mean signed logit delta, 95% state-bootstrap CI | Mean absolute logit delta | Argmax disagreement | Top-5 set disagreement |
|---|---:|---:|---:|---:|
| step 5 pre | +5.19e-4 [+1.29e-4, +8.71e-4] | 5.90e-3 | 1/2048 | 21/2048 |
| step 11 pre | +1.02e-3 [+6.22e-4, +1.40e-3] | 5.95e-3 | 1/2048 | 26/2048 |
| step 14 pre | +4.55e-4 [+1.05e-4, +8.20e-4] | 6.01e-3 | 2/2048 | 20/2048 |

The direction reversal is a direct counterexample to an unconditional claim such as “this compiler configuration systematically lowers logits.” A signed shift can be stable for one declared model-state distribution and reverse for another. In contrast, the absolute discrepancy scale and top-5 risk are much more stable across these checkpoints.

Because these checkpoints were retained around prior effects, they are a stress population. Their disagreement rates must not be presented as natural-workload prevalence.

## Why the confirmed Qwen raw-logit shift is not yet the core Oracle

Adding the same constant to every logit changes the mean logit but changes neither softmax probabilities nor rankings. Consequently, a mean over all vocabulary-coordinate deltas contains a **common-mode component** that can be semantically irrelevant.

The current evidence is consistent with that warning:

- the raw-logit mean has a confirmed negative direction at the final checkpoint;
- the target-log-probability direction does not match it consistently;
- the top-1-margin mean is near zero relative to the absolute discrepancy;
- ranking changes occur only for boundary-exposed observations.

This does not prove the shift is entirely common-mode, because the present records retain aggregate rather than full centered-delta statistics. It does establish that raw mean logit is not sufficient as the primary Oracle endpoint.

The next numerical calibration should predeclare and record both:

1. common-mode shift: the per-observation mean coordinate delta;
2. decision-relevant residual: coordinate deltas after subtracting that mean, plus event-specific margin and log-probability deltas.

This is not a cosmetic normalization. It removes a transformation under which the declared semantic maps are invariant.

## Interpretation of “bias” and “variance” after this experiment

### Endpoint-specific average shift

For an observable and target state distribution fixed in advance, the mean candidate-minus-reference difference is an implementation-relative average shift. It may be called a relative bias only if the baseline is named explicitly. It is not bias relative to mathematical truth.

The sign is not a property of “floating point” or “the compiler” alone. It depends on the observable, model weights, state distribution, implementation pair, and protocol.

### State-conditioned heterogeneity

The implementation effect varies across states and token positions even when execution is deterministic. This is real effect heterogeneity, not execution noise. It is often the dominant structure hidden by a near-zero global mean.

### Runtime variability

Under the present deterministic configuration, repeated execution of the same implementation at the same state produced identical outputs, so estimated runtime variability is zero. This result is protocol-specific; atomics, stochastic rounding, sampling RNG, autotuning choices, or nondeterministic kernels would define different protocols and could make it nonzero.

### Sampling uncertainty

Confidence intervals quantify uncertainty in generalizing from the finite state bank to its declared target population. They do not show that repeated GPU execution is random.

## Oracle consequence

A scientifically defensible output for one endpoint is not “bias yes/no.” It is the tuple:

1. **validity** — did the intended implementations actually execute and do self-pairs behave correctly?
2. **average shift** — is there a reproducible direction for this specific observable and population?
3. **conditional structure** — how does the effect vary across states, tokens, margins, and model checkpoints?
4. **runtime variability** — does identical-state repetition vary under the declared randomness protocol?
5. **semantic impact** — do declared event distributions or paired decisions change, especially near their exact boundaries?
6. **transition impact** — does the discrepancy change the one-step gradient/update/state transition?

The first five have now been calibrated at inference endpoints. The sixth remains the next major layer.

## Recommended next experiment

Run a matched **one-step transition** study on BERT and Qwen using frozen training states. Measure separately:

- loss and target-log-probability shift;
- gradient-vector norm, direction and coordinate discrepancy;
- clipping/overflow/skip events if present;
- optimizer update-vector norm, direction and parameter-block heterogeneity;
- same-state repeat variability;
- state-bootstrap uncertainty.

Do not free-run the two trajectories for this causal estimate. Long-run runs can later validate practical consequence, but they cannot identify the local implementation effect once the trajectories visit different states.

Only after a transition or semantic endpoint reproduces should operator repair/injection target it. Qwen top-5 set disagreement is currently the strongest reproduced inference endpoint; raw mean logit is unsuitable as the sole target because of common-mode invariance.

## Claim boundary

These runs establish deterministic, implementation-relative discrepancy and endpoint-dependent semantic drift on named matched-state populations. They do not establish:

- eager as truth;
- a correctness violation;
- a universal compiler direction;
- a natural fork prevalence for stress-selected checkpoints;
- a harmful one-step update;
- failed convergence, lower final accuracy, or longer training.

