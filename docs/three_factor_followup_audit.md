# Three-factor follow-up audit

This note freezes what the latest GPU artifacts do and do not establish.  It
prevents a diagnostic run from being promoted to a causal intervention.

## 1. Liger RN versus SR is not yet an intervention result

The tracked artifact
`results/property/joint_bias_formation_v1/liger_sr_intervention.json`
uses a separate 16-state default-residual split.  Its values are:

```text
RN:       A = 0.9418938921
SR #1:    A = 0.9749723104
SR #2:    A = 1.0010329325
```

The RN arm is already at the diffusive boundary on this split.  Therefore the
experiment has no confirmed positive baseline against which SR could suppress
persistence.  The correct label is
`INCONCLUSIVE_NO_POSITIVE_BASELINE`; it is not a negative causal result.

The older Liger chunk-geometry artifact is a different contrast: it reports
24/24 same-sign BF16 projections versus 13/11 FP32 projections and a residual
RMS ratio of 2.63384.  It does not report the same `A` statistic on the same
states and endpoint.  A historical value quoted as `A≈2.315` cannot be joined
to the RN→SR result until its horizon, state bank, endpoint contrast,
denominator, and estimator are bound to a tracked artifact.  Until then the
two values must not be compared.

## 2. Phi fixed-update propagation is positive closure evidence

`phi_fixed_update_propagation.json` injects the same real 16-step
candidate-minus-repair effective-update sequence into an alternate checkpoint.
The direct accumulated error is `4.4677257e-5`; the resulting drift is
`4.5143686e-5`; their ratio is `1.01043996`.

This is not a failed null.  It shows that, in this probe, the observed drift is
explained by direct local-error accumulation to within 1.04%, with no extra
closed-loop amplification.  It supports a clean source-persistent propagation
regime, while not proving a universal source mechanism.

## 3. The generic offline parity batch is not complete

`execution_status.json` previously called the artifact aggregation stage
`COMPLETE`.  That was too strong.  The repository contains exact
case-specific antithetic replays for saved-P and SiLU and a Phi pairing
intervention, but most roster entries do not retain raw `epsilon`, `+epsilon`,
and `-epsilon` response vectors.  Consequently the generic quantities

```text
F(+epsilon) + F(-epsilon)
mu_even, mu_odd
16-step prefix -> 32-step prediction
```

cannot be reconstructed without a new capture.  Missing vectors are
`UNRESOLVED`, never zero-filled.

## 4. Remaining required work

1. Reproduce a positive Liger RN baseline under the same state/endpoint/horizon
   before repeating RN→SR or order-breaking interventions.
2. Capture raw event and response vectors for the cases with valid replay
   artifacts, then compute the frozen even/odd decomposition.
3. Complete the 32-step consequence audit for the 12 mechanically selected
   residual-nonzero, parameter-reachable screen negatives.

Until these are complete, the three-factor map is a bounded mechanism
framework, not a universal low-cost oracle.
