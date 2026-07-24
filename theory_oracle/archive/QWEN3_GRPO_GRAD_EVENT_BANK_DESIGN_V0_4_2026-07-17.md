# Qwen3 GRPO Grad-Enabled Event-Bank Design v0.4 — 2026-07-17

## Status

Pre-execution design.  It is not frozen for GPU execution until the executor,
independent verifier, configurations and input artifacts are hashed in a manifest.

## Question

On a named finite bank of matched GRPO pre-minibatch states, does the
gradient-enabled tracked-Inductor scorer differ from the gradient-enabled eager
Trainer scorer in:

1. current-token log-probabilities;
2. sign-specific clipping decisions; and
3. stable event direction across exact same-state repeats?

This bank discovers a transition-context event eligible for a later one-step
intervention.  It does not itself estimate an update effect.

## Population and selection

- Reuse the two predeclared reference-trajectory checkpoint/prompt strata from the
  valid v0.2 confirmation, but rerun them with the corrected grad-enabled probe.
- The sampling unit is a rollout-state cluster; tokens remain nested observations.
- The bank is finite and reference-trajectory anchored.  No natural-training or
  deployment prevalence claim is licensed.
- State ordering is frozen as stratum, optimizer step, batch order and flat token
  index.
- A later repair witness, if any, is the first event in that order whose eager and
  compiled decisions disagree identically in both measured repeats.  Event
  magnitude and future repair outcome are not selection variables.

## Matched state

Each paired call must share:

- exact in-memory model parameters and buffers at the pre-minibatch point;
- prompt, completion, attention-mask and token-position tensors, including their
  padding and shapes;
- old per-token log-probabilities, advantages, epsilon and temperature;
- model training mode, gradient-checkpointing setting and SDPA MATH lock;
- FP16 autocast and the Accelerate output-to-FP32 wrapper used by Trainer;
- enabled autograd and the same `logits_to_keep`/TRL log-probability path;
- fixed compiler configuration and warm-up rule.

Optimizer state is part of the surrounding training state but is not mutated by
this forward-only bank.  The probe must verify that parameter gradients and
optimizer counters are unchanged before and after measurement.

## Implementations

- Reference: the actual grad-enabled eager Trainer scorer in its Accelerate
  execution context.
- Candidate: a tracked `torch.compile`/Inductor realization of that same scorer
  context, with runtime invocation and graph-specialization evidence.

The pair is implementation-relative.  Eager is not mathematical truth.

## Randomness and repeats

- Dropout and stochastic rounding remain disabled.
- Generated completions and all algorithmic RNG-dependent state are frozen before
  paired scoring.
- Candidate warm-up is discarded according to the frozen rule.
- Two measured eager calls and two measured compiled calls are retained per state.
- Nonzero repeat differences are reported as within-state runtime variability; they
  are not relabelled as implementation shift.  A repair witness additionally
  requires stable decision disagreement across repeats.

## Estimands

For candidate-minus-reference log-probability discrepancy, report separately:

1. token- and state-weighted average implementation-relative shift;
2. state-conditioned heterogeneity of the within-state effect summaries;
3. within-state runtime variability for each implementation;
4. finite-bank sampling scope, without a population interval unless a probability
   design is later supplied.

For the binary clipping endpoint, report:

- `0->1` and `1->0` counts;
- directional semantic shift, `(n_01 - n_10) / n_applicable`;
- semantic disagreement, `(n_01 + n_10) / n_applicable`;
- reference boundary exposure and stable-versus-repeat-unstable events.

Token pooling never substitutes for rollout-state clustering in uncertainty or
heterogeneity summaries.

## Validity gates

The bank is invalid for its declared claim if any of the following holds:

- token/state alignment or in-memory parameter identity fails;
- the eager realization does not match the actual Trainer scorer in the same call
  context;
- Accelerate output conversion, autograd enablement or compiler specialization is
  absent or unrecorded;
- the tracked compiled graph is not invoked for a claimed candidate measurement;
- the probe mutates parameter gradients, optimizer state, RNG state needed by the
  training trajectory or the next training transition;
- event definitions or the witness ordering rule are changed after results are
  read.

Exact self-repeat equality is not a universal validity requirement: failure is
reported as runtime variability.  It is, however, required for selecting a
deterministic single-state repair witness under this design.

## Output and next gate

The bank produces a structured discrepancy/semantic profile, not one pass/fail
number.  A one-step repair contract may be frozen only if at least one stable
grad-enabled event exists and can be exactly reconstructed with its complete
scorer tensor and branch decision.  Otherwise branch repair is killed or moved to
a deliberately boundary-stressed design whose frequency is not described as
natural prevalence.

Correctness, source-operator cause, long-run convergence and quality remain out of
scope without their own authorities and endpoints.
