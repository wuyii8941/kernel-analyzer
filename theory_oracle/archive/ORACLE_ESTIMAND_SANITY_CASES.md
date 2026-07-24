# Estimand Sanity Cases

These are constructed counterexamples used to check whether a proposed Oracle definition makes invalid logical implications. They are not empirical evidence about a compiler.

Let `g_R` be a signed reference event margin, `d = g_C - g_R`, and event `E_I = 1[g_I > 0]`. States are equally weighted unless noted.

## S1. Large average numerical shift, no event change

```text
g_R = [10, 20]
d   = [ 1,  1]
```

- average shift: `1`;
- disagreement: `0`;
- directional event shift: `0`.

Therefore a large average numerical shift is not sufficient for semantic drift.

## S2. Zero average numerical shift, directional event change

```text
g_R = [-0.01, -10]
d   = [+0.02, -0.02]
```

- average shift: `0`;
- first state changes `0 -> 1`;
- second state remains `0`;
- disagreement: `1/2`;
- directional event shift: `+1/2`.

The two numerical shifts cancel globally, but only the positive shift is near the boundary.

## S3. Zero directional shift, maximal paired disagreement

```text
g_R = [-0.01, +0.01]
d   = [+0.02, -0.02]
```

- average shift: `0`;
- one `0 -> 1` change and one `1 -> 0` change;
- directional event shift: `0`;
- paired disagreement: `1`.

Directional shift cannot replace disagreement. The implementations have equal marginal event rates but disagree on every paired state.

## S4. Zero conditional mean with runtime-noise-induced semantic drift

Use one fixed state with `g_R=-0.01`. Let compiled runtime shift be

```text
d = +0.02 with probability 1/2
d = -0.02 with probability 1/2.
```

- conditional mean shift: `0`;
- within-state shift variance: `0.0004`;
- reference event rate: `0`;
- compiled event rate: `1/2`;
- paired disagreement under this coupling: `1/2`.

A zero conditional mean does not prevent a nonlinear boundary from converting symmetric runtime variability into marginal semantic drift.

## S5. Small aligned shift versus large irrelevant shift

```text
state A: g_R=-0.001, d=+0.002  -> event changes
state B: g_R=-100,   d=+1      -> event unchanged
```

Raw delta magnitude ranks state B as more extreme, while event impact ranks state A as relevant. Numerical magnitude and boundary alignment answer different questions.

## S6. Same semantic event, different transition impact

Suppose two states both have one event disagreement, but downstream transition losses are:

```text
state A: event disagreement, update loss = 0
state B: event disagreement, update loss = 1
```

Event disagreement alone cannot determine update relevance. A transition endpoint or consequence-weighted event cost is needed.

## S7. Equal sampling marginals, high shared-RNG disagreement

Let `U ~ Uniform(0,1)` and define

```text
E_R = 1[U < 0.5]
E_C = 1[U >= 0.5].
```

Both marginal distributions are Bernoulli `(0.5)`, so total variation and directional marginal shift are zero. Under this shared-RNG coupling, disagreement is one.

This proves that a coupled sampling fork rate is not, by itself, a test of marginal distribution change.

## Required logical behavior of the Oracle profile

| Sanity case | Numerical endpoint | Semantic marginal endpoint | Coupled disagreement | Runtime endpoint | Transition endpoint |
| --- | --- | --- | --- | --- | --- |
| S1 | detects shift | no change | no change | zero | application-dependent |
| S2 | global cancellation | detects direction | detects change | zero | application-dependent |
| S3 | global cancellation | no directional change | detects instability | zero | application-dependent |
| S4 | zero conditional mean | detects event-rate change | detects coupling effect | detects variability | application-dependent |
| S5 | ranks large far-boundary delta | detects small aligned change | detects small aligned change | zero | application-dependent |
| S6 | insufficient | detects event change | detects event change | unspecified | separates consequences |
| S7 | not naturally ordered | no marginal change | detects coupling-specific mismatch | algorithmic RNG | application-dependent |

Any proposed universal scalar endpoint that claims to preserve all these distinctions is suspect. A scalar is defensible only after an application supplies a loss function that explicitly chooses the tradeoffs.

