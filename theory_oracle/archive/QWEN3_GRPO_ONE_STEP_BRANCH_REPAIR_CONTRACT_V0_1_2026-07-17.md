# Qwen3 GRPO One-Step Branch-Repair Contract v0.1 — 2026-07-17

## Status

Frozen after v0.2 semantic scoring selected the first event by the predeclared order,
and before executing the branch-repair one-step arms.

## Selected event

The selection rule was fixed in the parent confirmation contract. It selected:

```text
trajectory A2
optimizer step 11, rollout batch 3
case grpo_000003_dc31e4bafb66
token index 100, token id 18395
advantage sign -1
eager clipped -> compiled unclipped (1 -> 0)
```

No event was selected by delta size or expected recovery.

## State reconstruction gate

Rerun trajectory A2 from its original initial weights, empty optimizer state, prompt
slice and seed. Save the pre-minibatch model at optimizer step 11. Before the
counterfactual, require:

- exact selected case/token identity;
- exact old log-probability and advantage from the original state rows;
- exact eager and compiled selected-token log-probabilities from the original v0.2
  confirmation;
- identical 4-response/512-token batch order;
- tracked candidate identity and zero self-repeat delta in the reconstruction scan.

Failure invalidates the follow-up.

## Arms

Starting from the reconstructed identical weights and a new empty SGD state with
`lr=1e-5`:

| Arm | Numerical path | Target clipping branch |
|---|---|---|
| A | eager | ordinary eager decision |
| B | tracked Inductor | ordinary compiled decision |
| C | same tracked Inductor log-probabilities as B | forced to A's branch only at the selected token |

All other token objectives use the ordinary sign-specific GRPO clipping expression.
Because every response has 128 valid tokens, the implemented flat token mean equals
the mean of per-response token means for this batch.

Candidate identity requires at least one tracked compiled graph invocation in B and
C. B and C target log-probabilities, old log-probabilities and advantages must be
exactly equal. C may differ only in the selected objective branch after log-probability
materialization.

## Endpoints and estimand

Report:

- selected-token `dLoss/dlogp`;
- loss and full-gradient norm;
- actual next-parameter L2 distances `A-B`, `A-C` and `B-C` after SGD;
- `||A-C|| / ||A-B||` when the denominator is nonzero.

The repair effect is intervention-specific. `A-C < A-B` shows that setting this
branch to A's decision moves the chosen next-state endpoint toward A under the
compiled numerical path. It does not make the branch the whole cause, prove
necessity/sufficiency, establish long-run benefit, or identify a source operator.

Numerical correctness remains uninstantiated.

