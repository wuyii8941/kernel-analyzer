# Effective antithetic symmetry: working bias-formation hypothesis

## Question

Why do some implementation differences become directional training bias while
other, sometimes larger, numerical differences cancel?

The working answer is **not** “any numerical stage may be biased.”  The common
object is the implementation-error contribution after the real forward,
backward, and optimizer map.

For a fixed training condition `c`, let `z_r` be the repaired semantic-boundary
value, `epsilon=z_c-z_r`, and let `U_s(z)` denote the complete downstream
backward, gradient processing, and optimizer update at the same pre-state.
For a differentiable path, the fundamental theorem of calculus gives

```text
Delta u(c)
  = U_s(z_r + epsilon) - U_s(z_r)
  = [integral_0^1 D U_s(z_r + t epsilon) dt] epsilon
  = A_bar_s(epsilon) epsilon.
```

This is an exact path-averaged influence, not a small-error linearization.  It
includes the real VJP, nonlinear regions, clipping, and optimizer response.  A
piecewise or support-switching program uses the corresponding finite secant
contribution and is marked separately when the path crosses a discontinuity.
For multiple elementary events, define

```text
w_j(c) = A_bar_j(c, epsilon_j) epsilon_j.
```

An error-event ensemble has **effective antithetic symmetry** when its
transported contributions are closed under sign reversal: positive and
negative `w` contributions have equal conditional mass.  This implies a zero
effective-update mean.  A **symmetry defect** is a nonzero
conditional resultant:

```text
B(c) = E[sum_j w_j(c) | c] != 0.
```

The definition is state-conditioned.  It does not require unrelated training
states to share a fixed parameter direction.  A global fixed carrier is one
strong special case, not the definition of training bias.

### Exact parity budget

The statement above can be made more specific than “some stage is biased.”
Fix a semantic condition `c` and a predeclared antithetic operation
`tau(epsilon)=-epsilon`.  Let `p(epsilon|c)` be the conditional event density
and let

```text
F_c(epsilon) = U_s(z_r + epsilon) - U_s(z_r)
```

be the actual endpoint-induced update, including the real backward and
optimizer.  Split both the event distribution and the response into their
even and odd parts:

```text
p_s(epsilon) = [p(epsilon) + p(-epsilon)] / 2
p_a(epsilon) = [p(epsilon) - p(-epsilon)] / 2

F_e(epsilon) = [F(epsilon) + F(-epsilon)] / 2
F_o(epsilon) = [F(epsilon) - F(-epsilon)] / 2.
```

Integration over the antithetic domain gives the exact identity

```text
E[F_c(epsilon) | c]
  = integral p_s(epsilon) F_e(epsilon) d epsilon
  + integral p_a(epsilon) F_o(epsilon) d epsilon.
```

The two omitted cross terms vanish by parity.  Therefore, relative to a
predeclared semantic antithetic operation, bias has only two irreducible
formation channels:

1. **event-pairing asymmetry** (`p_a F_o`): opposite residual events do not
   occur with matched conditional mass.  This includes source/schedule
   asymmetry and residual--transport pairing asymmetry;
2. **response rectification** (`p_s F_e`): even perfectly matched
   `+epsilon/-epsilon` events are mapped to updates that are not opposites.
   Nonlinear backward regions and stateful optimizers can create this term.

This is the proposed Bias Formation Map.  It is not an enumeration of places
where error can occur.  It is an exact two-term budget explaining how a
zero-centered variation can fail to cancel.  The antithetic operation and
condition must be fixed from the mathematical F+B boundary before inspecting
the measured drift; otherwise the decomposition is merely post-hoc.

### What the property predicts

The map gives a concrete safe null and two different risk signatures:

```text
safe null:
  joint event population is antithetically closed
  AND
  complete downstream response is odd on the residual support

risk channel 1:
  antithetic event/pair mass imbalance

risk channel 2:
  large response-even component under an exact +/- residual pair.
```

For a stateful optimizer, the second signature can be measured without a
carrier direction:

```text
K(g,m,v,delta_g)
  = ||Opt(g+delta_g,m,v) + Opt(g-delta_g,m,v) - 2 Opt(g,m,v)||
    / (||Delta u_plus|| + ||Delta u_minus||).
```

For stateless SGD, `K=0` identically.  For AdamW it can be large where the
reference gradient and stored moments put `g+delta_g` and `g-delta_g` on
different sides of the adaptive normalization geometry.  The automated trace
therefore records the fraction of active coordinates that cross gradient sign,
the residual energy on those coordinates, and the fraction of response-even
energy they carry.  These are prospective susceptibility features; unlike the
exact parity budget, their predictive sufficiency is not assumed in advance.

### Sufficient cancellation condition

For every predeclared condition `c`, suppose

```text
epsilon | (A_bar, c)  has the same distribution as  -epsilon | (A_bar, c),
```

and the downstream secant response is sign-even around the repaired boundary:

```text
A_bar(-epsilon, c) = A_bar(epsilon, c).
```

Then the two effective contributions are antithetic and
`E[Delta u|c]=0`.  Error variance may be arbitrarily large.  This is a
sufficient safety property, not an assumption that natural residuals are
Gaussian or independent.

There are correspondingly only two ways to violate this exact safety
argument: the joint event/pairing distribution is not antithetically closed,
or the downstream response is not odd.  These are exactly the two terms in the
parity budget above.

### Relation to Flash Attention

The Flash Attention derivation has

```text
Delta dW_q = sum_T a_T R_T,
```

where `a_T` is the scalar delta error and `R_T` is the transported rank-one
matrix.  This is exactly `w_T=A_T epsilon_T`.  Predominantly one-signed
`a_T` breaks source antithetic symmetry; similarity among `R_T` concentrates
the resultant into a persistent low-rank carrier.  Low rank amplifies and
organizes the drift, but the sign-symmetry defect is what prevents cancellation.

## Why this is more specific than a stage taxonomy

The exact path-averaged term has a mean/coupling decomposition:

```text
E[A_bar epsilon | c]
  = E[A_bar | c] E[epsilon | c]
  + E[(A_bar-E[A_bar|c])(epsilon-E[epsilon|c]) | c].
```

This gives a complementary first-moment view of two physically different ways
to break the same antithetic symmetry:

1. **source asymmetry**: the residual is already conditionally non-centered;
2. **error-influence coupling**: the residual is marginally centered, but its
   sign/magnitude is coupled to the VJP or optimizer influence.

Nonlinear and optimizer rectification are not extra catch-all stages: they
change the path-averaged influence `A_bar`.  The parity budget is the sharper
form: joint event/pairing imbalance contributes through `p_a F_o`, while
non-odd response contributes through `p_s F_e`.  These are not an unrestricted
list of places where errors might occur.

## Decisive experiment

For a candidate event ensemble, preserve both marginal objects:

```text
{epsilon_j} and {A_j}
```

but destroy only their real pairing.  A permutation or semantic-orbit
intervention must preserve the residual multiset, support, and norm.  If the
natural arm has a nonzero accumulated gradient/update resultant and the
pairing-broken arm centers or strongly suppresses it, then the coupling—not
error magnitude—is causal.

Conversely, if pairing destruction leaves the resultant unchanged, this
hypothesis does not explain that case.  It must not be rescued by choosing a
new direction or threshold after seeing the result.

## Current evidence boundary

### Liger fused CE: source/schedule symmetry defect

Adding mathematically ignored zero rows changes only physical chunk geometry.
With a BF16 `dW` accumulator, all 24 confirmation-state carrier projections
have the same sign (mean `0.07419`, 95% CI `[0.06107, 0.08840]`).  With an FP32
accumulator, the same semantic orbit has 13 positive and 11 negative
projections (mean `0.000182`, CI `[-0.000827, 0.001316]`).  Loss and active
`dH` remain exact.  This is a matched example in which finite-precision
schedule breaks orbit centering.

Importantly, the BF16 residual's raw coordinate signed mean is only about
`-5.2e-10`; therefore raw tensor mean is not the marker.  Direction appears in
the semantically transported `dW` contribution.

### Phi lm-head input VJP: coupling defect

The local MM residual population is centered.  Its natural F+B pairing gives a
biased parameter-gradient population (cross-state ratio `0.675`, bootstrap
interval `[0.541, 0.825]`).  Row-pairing permutation preserves every local
residual norm but produces a centered gradient population (ratio `0.108`,
interval `[0.076, 0.145]`).  This rejects local error magnitude and local mean
as sufficient explanations.

The analytic RMSNorm-only reconstruction is incomplete, so this supports a
composite backward coupling mechanism, not one uniquely named Jacobian factor.

### Qwen saved-P: head pairing rejected; optimizer oddness active

The saved/reconstructed-probability repair closes the exact softmax
forward/backward semantic region and changes q/k parameter updates along a
training trajectory.  Rolling the natural `dS` residual across attention heads
preserved its complete multiset, causal support, and pre-cast norm.  It reduced
the gradient resultant by only 2.5% and increased the effective-update
resultant by 13.7%.  The head-specific residual/transport-pairing explanation is
therefore rejected for this case.

The same run exposes a new mechanism at the optimizer boundary.  For repair
gradient `g_r` and natural residual `delta_g`, the active follow-up computes

```text
O_adam = U(g_r + delta_g) + U(g_r - delta_g) - 2 U(g_r).
```

The two gradient perturbations have exactly equal norm and opposite sign.  A
linear or locally odd update map gives `O_adam=0`; nonzero accumulated
`O_adam` is direct optimizer rectification.  The measured accumulated oddness
ratio is `0.6817`, the mean per-step ratio is `0.1998`, and the oddness
resultant aligns `0.6871` with the natural update resultant.  This is a direct
measurement of the `p_s F_e` term: source imbalance is eliminated by
construction, yet the effective updates do not cancel.

The coordinate geometry localizes the effect.  Averaged over 32 steps, only
`1.87%` of active gradient-residual coordinates cross sign between
`g_r+delta_g` and `g_r-delta_g`, but those coordinates carry `81.49%` of the
Adam response-even energy under an unweighted step average.  Weighting steps
by their response-even energy raises the concentration to `99.48%`; `99.51%`
of the total even energy occurs in steps 1--2.  At step 1, `1.94%` of
coordinates carry `99.81%` of that step's even energy.  Thus the measured rectification is concentrated at the
adaptive optimizer's small-gradient/sign boundary rather than spread uniformly
over the residual norm.

For a smooth coordinatewise update map `F`, the local expansion is

```text
0.5 * [F(g+delta) + F(g-delta) - 2F(g)]
  approximately 0.5 * H_F(g)[delta, delta].
```

Thus a centered gradient residual can acquire a nonzero mean through the
optimizer curvature contracted with its covariance.  Stateless SGD has zero
curvature and is an exact null.  AdamW can have large curvature near small
reference gradients or sign-crossing coordinates because both its first moment
and square-root second-moment denominator depend on the perturbed gradient.

### Qwen3-VL SiLU: independent optimizer-response replication

The same exact antithetic experiment was repeated on a different closed F+B
case: Qwen3-VL layer-0 SiLU, where the candidate is the actual decomposed AOT
VJP and the repair is native `aten.silu_backward`.  Across a 32-step
repair-driven trajectory:

```text
accumulated optimizer-oddness ratio       0.6956
mean per-step optimizer-oddness ratio     0.0354
cosine(natural, antithetic resultants)   -0.0323
natural update persistence                0.7458
antithetic update persistence             0.7457
```

The gradient residuals are exactly `+delta_g/-delta_g`, all forward losses are
equal, and both arms use the same weights and Adam moments.  Nevertheless the
two accumulated Adam update residuals are almost orthogonal rather than
opposite.  This independently reproduces a nonzero response-even term in a
different model and operator region.  Energy weighting shows that `99.87%` of
the even component lies on sign-crossing coordinates and more than `99.99%`
is generated in steps 1--2.  This is therefore a cold-start Adam rectification
impulse, not evidence that Adam generates the same amount of bias at every
step.  The later trajectory result asks whether training retains that impulse.

### Prior-art boundary for optimizer rectification

This project must not claim that adaptive optimizers reacting nonlinearly to
noise is itself new.  Adam is defined through adaptive first/second moments;
DP-AdamBC identifies bias in Adam's second-moment estimate under independent DP
noise; and stochastic-rounding work analyzes low-precision training under Adam.
The narrower prospective contribution here is an exact operator-local chain:

```text
one closed F+B implementation residual
  -> exact +delta_g/-delta_g matched pair
  -> measured Adam even component
  -> effective-update bias and trajectory consequence.
```

Primary references: [Adam](https://arxiv.org/abs/1412.6980),
[DP-AdamBC](https://arxiv.org/abs/2312.14334), and
[Stochastic Rounding for LLM Training](https://proceedings.mlr.press/v258/ozkara25b.html).

## Falsified or insufficient alternatives

- **Error magnitude** is insufficient: Phi preserves local norm while changing
  the bias verdict.
- **Raw signed tensor mean** is insufficient: Liger's raw mean is tiny while
  its effective direction is unanimous.
- **A fixed global rank-1 carrier** is not necessary: saved-P and other live
  trajectories can separate without cross-state rank-1 stability.
- **BF16 alone** is not an explanation: BF16 is a source condition; whether its
  errors cancel depends on effective antithetic symmetry.
- **SEUP** is downstream: it explains whether formed effective-update bias
  persists, not how bias first forms.

## Current supportable claim

> Across distinct operator-local mechanisms, training bias is generated by a
> defect in conditional antithetic cancellation.  The defect can occur because
> the joint residual/event population is not antithetically paired, or because
> the real F+B/optimizer response has a nonzero even component.  Liger and Phi
> support the event/pairing channel; the exact saved-P `+delta_g/-delta_g`
> experiment and its Qwen3-VL SiLU replication support optimizer response
> rectification.

This is an exact organizing equation and a matched, multi-case mechanism
result, not yet a universal predictor for unseen operators.  Saved-P remains a
negative for the head-specific transport-pairing hypothesis and a positive for
optimizer response rectification.  Qwen3-VL SiLU supplies the independent
operator/model replication of that channel; held-out prediction remains open.
