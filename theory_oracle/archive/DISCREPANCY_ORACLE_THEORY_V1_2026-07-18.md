# Discrepancy Oracle Theory v1 — 2026-07-18

## 1. What the Oracle compares

The Oracle is not a test of whether two floating-point tensors are equal. It is a
scoped relational decision procedure over matched execution states:

> Under a declared state distribution, implementation pair, execution-history
> protocol and randomness coupling, characterize implementation-relative
> discrepancy; determine whether it changes declared semantic events or one-step
> transitions; issue correctness only when an independent specification exists.

One single fork, mean delta or parameter norm is evidence for one endpoint, not
the Oracle itself.

## 2. The complete query

A query is the tuple

```text
O = (Q, X, H, I, U, Pi, Y, E, T, G, A, L)
```

where:

- `Q` is the target distribution or frozen bank of sampling units;
- `X` is program state: parameters, buffers, optimizer/scaler state, batch,
  counters, RNG state and modes;
- `H` is implementation/execution state: compiler caches, graph specialization,
  autotuning selections, wrapper/dispatch state, hardware and configuration;
- `I={R,C}` is the fixed reference/candidate implementation pair;
- `U` names randomness left open within a matched state;
- `Pi` is the coupling of randomness between implementations and repeats;
- `Y` is the continuous numerical observable hierarchy;
- `E` is the semantic event map or event law;
- `T` is the declared one-step next-state observable or transition kernel;
- `G` gives endpoint geometries/costs;
- `A` gives endpoint-specific acceptance relations and authorities;
- `L` is the requested claim level and explicit non-claims.

If changing a field can change the realized outcome, that field must be in `X`,
`H`, or `U`. Leaving it unnamed makes the query invalid. The Qwen v0.9 result
demonstrates why `H` cannot be reduced to `compile=True`: cold and history-warmed
Inductor executions produced different candidate values at identical model state
and input.

## 3. The four distinct uncertainty objects

For a scalar observable, define the paired discrepancy under the declared
coupling:

```text
D(s,u) = Y_C(s,u_C) - Y_R(s,u_R).
```

Then, relative to the declared `Q` and runtime law:

```text
m(s)   = E[D | S=s]
mu_Q   = E_Q[m(S)]
h(s)   = m(s) - mu_Q
eta    = D - m(s)
```

The report uses:

- **average implementation-relative shift**: `mu_Q`;
- **state-conditioned heterogeneity**: the distribution of `h(S)`;
- **within-state runtime variability**: the conditional distribution of `eta`;
- **sampling uncertainty**: uncertainty in estimating the above from finite
  trajectories, state clusters and repeats.

These are not interchangeable variances. In deterministic execution, `eta=0`
may hold while `h(S)` remains large. A fixed reduction tree, reassociation or cast
placement contributes to `m(s)` and possibly `h(s)`; it is not runtime variance
merely because floating point is involved. Compiler/autotuning history belongs to
`H` when conditioned on, and to `U` only when deliberately sampled as runtime
randomness.

Without independent truth, `D`, `m` and `mu_Q` are discrepancies or shifts, not
correctness error or compiler bias.

## 4. Boundary-conditioned semantic impact

For a binary event `E`, report both:

```text
directional semantic shift = P(E_C=1) - P(E_R=1)
semantic disagreement      = P(E_C != E_R) under Pi.
```

The first compares marginal event rates. The second is coupling-dependent
reproducibility. Large disagreement with zero directional shift is still important:
the implementations may flip equally often in opposite directions.

Every threshold/ranking endpoint must retain its reference boundary geometry and
direction. A global mean numerical shift is insufficient because effects far from
the boundary may dominate it while contributing no event change.

For categorical, ranking, top-k and routing events, replace the binary sign by:

- marginal outcome-law distance;
- coupled disagreement;
- an application-defined cost or set/assignment geometry.

For stochastic sampling, a single shared-RNG token is only a coupling-specific
observation. The primary semantic object is the selection law or transition
probability, with shared-RNG disagreement reported separately.

Signed decision margin is related to branch-distance testing but is not identical
to traditional nonnegative, target-branch fitness. This Oracle is relational and
conditions an implementation discrepancy on boundary geometry.

## 5. One-step transition impact

Each implementation induces a one-step kernel

```text
K_i(. | x,h) = Law(T_i | X=x, H=h).
```

The transition ledger may contain next-parameter/update distance, optimizer and
scaler state, skip/overflow events, gradient geometry and task projections. It
must name the compared object and geometry; “the training differs” is not an
estimand.

Numerical, semantic and transition endpoints form a dependency ladder, not logical
equivalences:

```text
numerical discrepancy -> possible event change -> possible update change.
```

Each arrow can fail because of margins, dead effects, clipping, cancellation or
later transformations. Conversely, a tiny numerical discrepancy can have a large
event effect when aligned with a boundary.

## 6. Five separate verdict ledgers

The Oracle returns a structured result, never an unexplained weighted scalar:

1. **Validity** — were `X`, `H`, implementation identity and coupling realized?
2. **Discrepancy** — what shift, heterogeneity and runtime variability were
   estimated under `Q`?
3. **Semantic/transition impact** — were declared events or one-step behaviors
   changed beyond a predeclared application rule?
4. **Correctness** — did the candidate violate an independently authoritative
   exact relation, numerical envelope or stochastic law?
5. **Attribution** — what does a valid repair/injection intervention contribute
   to an already-defined endpoint?

`ACCEPT`, `REJECT`, `INDETERMINATE`, `UNINSTANTIATED`, `INVALID` and
`INAPPLICABLE` remain distinct. Failure to detect a difference is not equivalence;
disagreement is not correctness; impact is not correctness.

## 7. Correctness boundary

Correctness requires an authority independent of the candidate observation, such
as documented exact semantics, a high-precision reference, certified error
envelope, invariant, or confirmed wrong-code relation. Eager may be the baseline
for impact without being truth.

A legal alternative floating-point implementation with persistent semantic drift
can be a reproducibility or operational-risk finding while correctness remains
uninstantiated. A project name must therefore avoid promising that every relative
drift is a compiler bug.

## 8. Attribution interface

The endpoint is fixed before attribution.

- **Repair** estimates the endpoint change when a candidate-context component is
  replaced by its reference counterpart.
- **Injection** estimates the endpoint change when the corresponding candidate
  behavior is introduced into the reference context.
- Their asymmetry is evidence of context dependence or interaction, not an error
  in the framework.

Total, repair, injection and interaction contrasts must be reported separately.
Repair supports neither unique necessity nor sufficiency when alternative causes
or higher-order interactions exist.

An operator-causal claim additionally requires that the intervention preserve
fusion, layout, dispatch and downstream compiler choices, and that the semantic
operator has realization-level correspondence. Otherwise report
**intervention-dependent attribution**.

Roles remain separate:

- discrepancy source;
- discrepancy propagation/exposure;
- boundary conversion into a semantic branch.

The Qwen v0.9 clipping repair identifies the third role only. It does not identify
the source operator.

## 9. Population and long-run scope

`Q` is part of the estimand. Reference-trajectory, candidate-trajectory, external
and stress-enriched states must remain separate unless weights are justified.
Changing `Q` can change or reverse `mu_Q`; that is not evidence that the estimator
is invalid, but it prohibits an unqualified universal shift claim.

Population rates require probability/cluster sampling at the actual state unit.
Token counts nested inside a few trajectories do not create independent states.

Matched-state one-step comparison estimates a local relation. Free-running
training changes future states after the first discrepancy and is not a direct
estimate of the local implementation effect. A local-to-long-run claim requires
additional stability, coupling, coverage and mixing/contraction assumptions;
otherwise long-run training is a separate validation endpoint.

## 10. Three meanings of completeness

### 10.1 Definition completeness

The schema is definition-complete when every outcome-relevant quantity has a
declared place in `(Q,X,H,I,U,Pi,Y,E,T,G,A,L)`, all ledgers have fail-closed
verdict rules, and no ledger is used as a proxy for another.

### 10.2 Query/subject completeness

A particular query is complete only when every ledger it marked `REQUIRED` has
valid evidence. It may correctly leave correctness or population inference out of
scope. “Complete” must always name this declared scope.

### 10.3 External-validation completeness

A general-purpose claim additionally needs held-out state distributions, multiple
models/implementation families, stochastic regimes, negative and legal-difference
controls, and evidence that the semantic ledger contains boundary/direction
information absent from raw magnitude. No current experiment establishes this
level. If predictive prioritization is claimed, its held-out score must not use
the candidate delta or event observed on the held-out state; the paired semantic
detector itself is post-execution and is not a predictor.

## 11. Minimal validity and kill gates

A result is invalid or its claim must be narrowed when:

- a relevant state, compiler-history or randomness variable is omitted;
- the claimed candidate silently falls back or changes graph/dispatch identity;
- an event is transported from no-grad/logging context without reproducing it in
  the transition context;
- a tolerance is learned from candidate outputs;
- state heterogeneity, runtime variability and sampling uncertainty are pooled;
- a single stochastic draw is treated as a law comparison;
- repair changes fusion/layout or lacks B/C non-target identity;
- finite stress frequency is reported as operational prevalence;
- local one-step drift is claimed to prove long-run harm;
- the ranking of cases is indistinguishable from raw numerical delta and no
  additional semantic, transition or diagnostic information is demonstrated.

## 12. Current evidence and remaining work

The executable decision core and existing controls support definition completeness
for deterministic exact/numerical, discrepancy, semantic impact, controlled
transition and intervention-dependent attribution queries. Qwen v0.9 adds direct
evidence that execution history belongs in the state contract.

The framework is not externally complete. Highest-priority missing evidence is:

1. a probability/cluster-sampled multi-checkpoint state population;
2. a natural optimizer/scaler one-step endpoint on the same real training subject;
3. repair plus injection and interaction under realization-preserving treatments;
4. an independent real-model numerical authority;
5. stochastic full-step law validation and any justified local-to-long-run bridge;
6. independently selected states showing that discrepancy, event and transition
   ledgers remain interpretable beyond the construction bank. The v0.4 diagnostic
   already establishes the algebraic finite-bank fact that raw magnitude omits
   boundary and direction. A separate held-out comparison is required only for a
   predictive prioritization claim, whose score must not use held-out candidate
   delta.

These are separate validation obligations. They should not be collapsed into more
examples of the same clipping witness.
