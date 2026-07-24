# Qwen3 GRPO Training-Control Confirmation Contract v0.1 — 2026-07-17

## Status

Frozen before executing the two confirmation trajectories or inspecting their
eager/compiled signed crossings.

## 1. Claim

This contract asks whether eager and tracked Inductor execution change the actual
PPO/GRPO clipping decision on newly generated matched training states, and how the
continuous discrepancy decomposes over those states.

It is an implementation-relative training compatibility contract. It is not a
compiler correctness contract because no independent legal numerical envelope is
available for the Qwen forward transition.

## 2. State strata

Two reference-anchored trajectories are fixed:

| Trajectory | Initial model state | Prompt slice | Seed |
|---|---|---|---:|
| A | `data/r1_from240_step242_pre` | built-in arithmetic `[192,256)` | 20260718 |
| B | `data/r1_from270_step272_pre` | built-in arithmetic `[256,320)` | 20260719 |

The initial model weights differ. Both trajectories deliberately create a new empty
optimizer/scaler state and execute 30 GRPO optimizer steps. They are not historical
optimizer replays.

With `num_iterations=3`, every trajectory supplies ten policy-iteration-2
pre-minibatch matched states. Eager training determines subsequent model states;
compiled scoring is observational and does not feed the update. The state population
is therefore `Q_R`, reference-trajectory anchored.

The two prompt slices must have zero hash overlap with all retained discovery, R1
held-out and Qwen confirmation prompt banks, and zero overlap with one another. The
eligibility manifest is frozen before CUDA execution.

## 3. Matched state

At every scored pre-minibatch state, both paths share:

- identical in-memory model parameters and buffers;
- identical prompt and generated response token IDs;
- identical old-policy token log-probabilities;
- identical group-normalized advantages and signs;
- identical masks, position semantics, training mode and dropout-disabled setting;
- FP32 master parameters with CUDA FP16 autocast;
- SDPA MATH attention;
- identical hardware, seed and framework configuration.

Optimizer state is part of the trajectory state but is not touched by observational
compiled scoring. It affects later eager-anchored states.

## 4. Candidate realization identity

The candidate is `torch.compile(model, backend=tracked_inductor)` with graph breaks
permitted. It is a compiled-region implementation, not a promised single full graph.

Validity requires:

1. at least one backend compile and one graph-code hash;
2. positive compiled runtime invocation counts for both measured candidate calls in
   every scored state;
3. one discarded compiled warmup call before the two measurements;
4. identical candidate outputs in the two measured calls under the deterministic
   protocol;
5. exact eager self-repeat equality;
6. complete token alignment and finite log-probabilities.

Failure is `INVALID`, not compatibility `ACCEPT`.

## 5. Continuous observable and decomposition

For every applicable token,

```text
D = logp_compiled - logp_eager
```

Report separately:

- finite-bank average signed shift;
- rollout-state/checkpoint-conditioned shift distribution and both signs;
- within-state runtime variability from eager and compiled self repeats;
- boundary distance and its relationship to event crossing;
- finite-bank sampling-unit counts.

The trajectories are deterministically generated confirmation strata, not
probability samples from all Qwen training. No deployment prevalence interval is
permitted.

## 6. Training semantic event

For nonzero advantage sign `a` and `epsilon=0.2`, clipping uses the declared TRL
PPO/GRPO sign-specific boundary. Zero-advantage tokens are `INAPPLICABLE`.

Per token report:

- eager unclipped, compiled clipped (`0 -> 1`);
- eager clipped, compiled unclipped (`1 -> 0`);
- disagreement, the sum of both directions;
- directional clipping shift, `count(0->1) - count(1->0)`;
- eager boundary distance and signed implementation delta.

The strict finite-bank compatibility verdict is:

```text
REJECT  if at least one valid applicable token changes clipping decision
ACCEPT  if all valid applicable tokens preserve clipping decision
```

This is intentionally stricter than a practical-risk threshold. It says whether the
implementations are decision-identical on the frozen bank, not whether a difference
is harmful or illegal.

## 7. One-step consequence rule

If one or more events occur, choose the first event by the fixed ordering
`trajectory, optimizer_step, case_id, token_index`. Reproduce that state through a
deterministic rerun and require exact token/old-logp/advantage/logp matching before a
branch-repair one-step experiment.

The repair changes only that token's clipping branch while retaining the compiled
log-probability. It estimates an intervention-specific contribution to gradient and
next update. It does not establish numerical correctness or an operator root cause.

If no event occurs, this endpoint is `INAPPLICABLE`; a controlled midpoint boundary
may only be reported as stress calibration in a separately labelled stratum.

## 8. Confirmation success and kill criteria

The confirmation mechanics succeed if both trajectories finish, all candidate
identity/self/token gates pass, all twenty expected rollout states are represented,
and the evaluator applies the frozen endpoint without post-hoc exclusions.

Narrow or invalidate the claim if:

- fewer than two distinct initial checkpoint hashes or twenty rollout states appear;
- prompt overlap is nonzero;
- candidate invocations are absent or graph identity is unrecorded;
- compiled scoring influences the training update;
- row-level confidence intervals are reported as population uncertainty;
- a stress-constructed crossing is counted as natural prevalence;
- any disagreement is called a compiler bug without independent specification.

