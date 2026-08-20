# Bias Formation v2.2: conditional, trajectory, and global claims

The previous v2.1 population statistic was too strong to serve as the
definition of training bias.  It pooled states with different activations,
cotangents, Jacobians, operand regimes, and parameter-update directions, then
required the resulting error vectors to share one global sign.  That statistic
is meaningful as a **global/state-invariant bias** test, but failure of that
test cannot imply harmless training variance.

The corrected object is a three-level map:

```text
implementation difference
        ↓
conditional local/gradient/update effect
        ↓
closed-loop trajectory separation
        ↓
optional global/state-invariant direction
```

## 1. Conditional bias

Declare the condition before looking at the candidate result.  A condition may
contain shape, semantic region, token/head/chunk schedule, operand regime, or a
matched training-phase bucket.  Within each condition, candidate and repair
must use the same complete forward/backward state and declared parameter
coordinates.

For repeated observations in condition `c`, estimate:

\[
m(c)=E[\Delta u\mid C=c].
\]

The estimator is the off-diagonal cross-state U-statistic **within that
condition**.  Conditions are never pooled to manufacture one direction.  A
condition with fewer than the preregistered replicate count is
`CONDITIONAL_UNRESOLVED`, not `CENTERED`.  A single deterministic state
contrast is an observed effect, not a statistical conditional-bias verdict.

## 2. Trajectory consequence: separation versus persistence

For a paired candidate/repair run, let

\[
D_{t+1}=D_t+L_t+B_t+r_t,
\]

where `D` is the actual parameter-state separation, `L` is the same-state
endpoint-mediated local effect, and `B` is state/optimizer feedback.  A causal
separation claim requires a closed repair/sham boundary, a complete forward
plus actual backward at every step, and live parameter separation above the
matched null.  It does **not** require one sign across unrelated states.
However, growth of `||D_t||` alone cannot distinguish signed drift from
diffusion, so it is called `TRAJECTORY_SEPARATION`, not trajectory bias.

The primary trajectory quantities are basis-free:

* final separation `||D_T||`;
* separation path and its area `Σ_t ||D_t||`;
* local/feedback contributions from the recurrence;
* matched-sham or repeat-run null.

Flash-style directional persistence is a separate, stronger claim.  It
requires a predeclared or calibration-frozen trajectory-local direction or
subspace whose signed contribution does not cancel.  This carrier need not be
one fixed global direction across unrelated natural inputs, but some
trajectory-local persistence witness is required.

## 3. Global bias

The v2.1 cross-state statistic is retained under the explicit label
`GLOBAL`.  It may be called `GLOBAL_BIAS` only when a preregistered
state-comparability certificate exists.  Otherwise its output is
`GLOBAL_NOT_IDENTIFIABLE`, `GLOBAL_CENTERED`, or an unresolved global
directional result.  A global null is not evidence against conditional or
trajectory bias.

## Consequence for existing cases

The old v2.1 labels are not deleted.  They are reinterpreted as global-scope
measurements.  Existing complete paired trajectories can be reaudited without
rerunning all 791 F+B cells.  In particular, a case that fails the directional
gate but has complete causal repair, exact sham, and growing basis-free
parameter separation is a valid causal-separation observation.  It remains
unresolved for directional persistence rather than being promoted to a
persistent-bias case.

Conditional claims still require new condition metadata and repeated units;
they must not be inferred from the old mixed-state artifacts.

## Property analysis

These three levels are observation labels, not a replacement property.  After
the level is identified, mechanism analysis must still use the original causal
theory:

```text
local residual
  → source asymmetry / backward transport / F+B numerical contract
  → optimizer update residual
  → SEUP temporal persistence
  → parameter drift
```

The P1--P6 list has now been reduced to an exact two-channel formation map.
For a predeclared semantic antithetic operation, split the conditional event
population into `p_s/p_a` and the complete F+B/optimizer response into
`F_e/F_o`:

```text
E[F(epsilon)|c] = integral p_s F_e + integral p_a F_o.
```

`p_a F_o` is event/pairing asymmetry; `p_s F_e` is response rectification.
Source schedule, backward transport and numerical-contract effects change the
first channel's joint event population.  Nonlinear and optimizer response can
create the second channel.  This replaces a loose list of possible stages with
one falsifiable cancellation property: antithetically paired events plus an
odd response have zero conditional bias regardless of variance.

The property question is consequently not “which level wins?” but:

> Given a declared condition or a closed trajectory, which source/transport/
> contract/optimizer feature determines whether the local variation becomes a
> directional effective update?

## Eight-case systematic audit

The current evidence-bounded audit is generated by
`scripts/build_systematic_bias_audit.py` under
`results/property/bias_formation_systematic/`.  It applies the same conditional
decomposition to eight complete paired F+B separation artifacts and records
the exact missing intervention for each case.  Eight is the audit denominator,
not the persistent-bias count.  Its safeguards are deliberate:

* global-centered saved-P evidence is not relabeled as variance-only;
* the Qwen seq128 output-rounding source is not joined to an accumulation-only
  trajectory repair;
* every supported mechanism has a causal intervention and exact sham;
* trajectory and SEUP fields cannot create a formation-stage label.
* parameter-distance growth cannot create a directional-persistence label;
* formation and persistence are joined only when their repair contrasts align.

The unified audit records 8/8 causal separations, 6/8 directional-persistence
positives, 6/8 matched formation-mechanism positives, and 4/8 same-contrast
full chains.  These sets intentionally differ.  This supports the formation
map as a cross-case working property, not yet as a zero-shot predictor for
unseen operators.
