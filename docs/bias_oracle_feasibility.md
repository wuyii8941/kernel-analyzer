# Bias Oracle Feasibility

## Decision

A single flat oracle is not currently both cheap and universal. The viable
design is a **selective cascade**:

1. screen all instrumentable F+B units together;
2. escalate uncertain or interacting groups;
3. issue a scientific bias certificate only from the existing exact
   conditional-antithetic measurement;
4. keep unsupported units in the denominator as `ABSTAIN`.

The exact mechanism remains

\[
\mathbb E[F(\epsilon)\mid c]
=
\int p_a(\epsilon\mid c)F_o(\epsilon,c)d\epsilon
+
\int p_s(\epsilon\mid c)F_e(\epsilon,c)d\epsilon.
\]

The first term is transported source/event/pairing asymmetry. The second is
response rectification. The feasibility work below changes how these terms
are screened; it does not redefine bias or alter any existing case verdict.

## Experiments

### 1. First/second-moment response screen

For local residual mean \(\mu\), raw second moment \(M_2\), and effective
update response \(F\), the smooth local approximation is

\[
\mathbb E[F(\epsilon)-F(0)]
\simeq J\mu + \tfrac12 H:M_2.
\]

Five frozen synthetic controls were used: centered linear variance, biased
source through linear transport, centered variance through a quadratic
response, a mixed case, and a nonsmooth support switch. The transported-mean
plus curvature sketch recovered the first four exactly. A half-amplitude gate
correctly returned `ESCALATE_NONLOCAL_OR_NONSMOOTH_RESPONSE` for the support
switch. This makes the sketch a valid smooth-region screen, not a universal
certificate.

### 2. Shared all-block HVP

One global Hessian-vector product can return an unbiased blockwise curvature
estimate for every declared local injection block at once. Cross-block terms
cancel over independent or coded block probes. A coupled two-block synthetic
response recovered both exact block traces with one shared forward graph, one
first reverse pass, and eight coded HVPs.

The same implementation was then run on actual differentiable training
semantic cuts:

| semantic cut | 4-probe time / ordinary F+B | peak CUDA memory |
|---|---:|---:|
| loss head + cross entropy backward | 11.78× | 17.26 MB |
| RMSNorm backward | 9.87× | 17.23 MB |
| attention backward, two local blocks | 5.47× | 17.23 MB |

All three supported double backward. The extra peak memory over their ordinary
cut was about 0.14–0.17 MB. These tiny cuts are dominated by launch overhead,
so the ratios are not full-model runtime predictions. They do show that probe
count, rather than local block count, can be shared inside one differentiable
graph.

This route has two hard limitations:

- compiled/custom backward kernels may not support second differentiation;
- one HVP experiment measures one scalar projection of the parameter update.

The second limitation was tested directly from retained complete Grams.
Gaussian output sketches of 4 dimensions reproduced only 36.9%–55.0% of
full-vector conditional verdicts; most remaining comparisons became
`UNRESOLVED`. At 64 dimensions, agreement was 79.4%–94.4%. No projected
comparison produced an opposite resolved verdict, but the abstention rate is
too high for a standalone oracle. HVP is therefore a conservative prioritizer,
not the final verdict path.

### 3. Fewer exact repair repeats

The retained conditional-debias Grams for Qwen64 `v_proj`, Qwen128 `v_proj`,
and Mamba64 input projection were re-evaluated using unchanged margins and
2,000 bootstrap draws. Each budget used a prefix, an evenly spaced subset, and
a deterministic random subset.

| case | 4-repeat agreement | 6-repeat | 8-repeat | first full agreement |
|---|---:|---:|---:|---:|
| Qwen64 `v_proj` | 92.9% | 97.5% | 97.5% | 12 repeats |
| Qwen128 `v_proj` | 93.3% | 97.5% | 100% | 8 repeats |
| Mamba64 input projection | 87.1% | 90.0% | 91.7% | 16 repeats |

There was no opposite resolved verdict at any budget. All lost decisions were
unresolved, apart from a few comparisons whose full Mamba result was itself
unresolved. This supports a sequential screen: start at four repeats and add
6/8/12/16 only when unresolved. It does **not** justify declaring an early
centered result to be a complete safety certificate without an anytime-valid
stopping rule.

### 4. Coded group interventions

As a non-HVP alternative, ordinary F+B runs can switch random groups of units
between candidate and repair while retaining the complete parameter-response
vector. A vector-valued sparse recovery model was tested on 64 units with four
active sources over 20 independent trials.

| mechanism | training mask runs | mean active recall | held-out response residual |
|---|---:|---:|---:|
| sparse additive | 16 | 97.5% | 0.062 |
| sparse additive | 24 | 100% | 0.008 |
| sparse with interaction | 24 | 90.0% | 0.947 |
| dense additive, four-source fit | 24 | 20.0% | 0.738 |

The held-out mask residual cleanly rejects the interaction and dense cases
instead of returning a false localization. This is promising because its run
count depends on sparsity and desired confidence, not directly on the number
of units, and it requires only ordinary F+B. It remains a synthetic result and
requires a controllable candidate/repair switch for every included unit.

## Relation to existing methods

- Stochastic-arithmetic/CESTAC-style operator analysis commonly uses three
  samples and batched stochastic representatives, but still reports a minimum
  cost near 3× and does not certify the real training backward when a surrogate
  gradient is used ([Noisefloat](https://arxiv.org/html/2607.25494)).
- Hutchinson trace estimation supplies the matrix-free curvature primitive
  used here ([HAWQ-V2](https://arxiv.org/abs/1911.03852)). Recent layerwise
  analysis shows why one global HVP can yield estimates for every parameter
  block simultaneously ([layerwise Hessian trace](https://arxiv.org/abs/2605.25674)).
- Dimension-independent simultaneous perturbation motivates the coded global
  screen ([SPSA](https://www.sciencedirect.com/science/article/pii/S0005109896001495));
  sparse recovery provides the sublinear localization hypothesis
  ([compressed-sensing SPSA](https://jmlr.csail.mit.edu/papers/v18/15-592.html)).

These methods provide useful estimators, not the missing scientific label.
Kernel Analyzer's contribution remains the exact F+B boundary and the
conditional odd/even bias-formation semantics.

## Recommended cascade

### Stage 0 — capability and cheap analytic checks

- preserve the full F+B invocation denominator;
- derive deterministic rounding/event moments from reference operands and the
  declared schedule where possible;
- compute exact SGD/Adam response parity offline once a gradient residual is
  available;
- classify unavailable repair, source fidelity, and nondifferentiability
  before spending GPU time.

### Stage 1 — model-level coded ordinary-F+B screen

Run balanced candidate/repair masks over all switchable units. Retain complete
gradient/update Gram information. Fit a sparse vector-valued response and
check it on held-out masks. A small residual permits group localization; a
large residual escalates because sparsity/additivity failed.

### Stage 2 — optional shared HVP prioritization

Use this only on smooth, double-differentiable reference/AOT semantic regions.
It can distinguish transported mean from curvature rectification and rank
groups, but low-dimensional output projections cannot certify safety.

### Stage 3 — exact conditional certificate

Run the existing matched antithetic/repair experiment only on localized groups.
Use four repeats as the first look and append repeats when unresolved. Do not
change the frozen statistic or margins, and do not convert abstention into
`CENTERED`.

## Claim boundary

This feasibility study supports an implementable research direction, not a
finished universal oracle. The proposed cascade has acceptable *screening*
cost because expensive work is shared across units or reserved for hits. Its
generality comes from fail-closed routing across ordinary F+B, differentiable
HVP, and exact local repair paths—not from pretending every kernel supports
one perturbation primitive.

The next decisive engineering experiment is a Qwen model-level masked-repair
runner covering several already bound F+B units in one campaign. It should
measure whether the sparse/additive assumption survives natural compiler
interactions. If held-out mask closure fails, coded localization is rejected
and the project retains the sequential exact oracle only.
