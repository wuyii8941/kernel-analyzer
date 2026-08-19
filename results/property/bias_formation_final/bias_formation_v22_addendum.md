# v2.2 correction addendum

The original v2.1 formation table answered a narrower question than the paper
needs: whether a numerical difference is globally directional across a pooled
set of states. Because activations, cotangents, Jacobians, and update
coordinates vary with state, a global centered result cannot be used as a
training-bias negative.

The corrected observation levels are:

1. `CONDITIONAL`: repeated comparable conditions have a directional conditional
   effect;
2. `TRAJECTORY`: a complete paired candidate/repair F+B trajectory separates
   in live parameter state under exact repair/sham controls, without requiring
   one fixed carrier across unrelated states;
3. `GLOBAL`: the old v2.1 cross-state statistic, reported only as a stronger
   state-invariant statement.

The artifact audit in
`results/property/bias_formation_v22/trajectory_reclassification.json`
currently classifies eight existing paired trajectories as
`TRAJECTORY_BIAS`. This is a corrected case ledger, not eight independent
physical mechanisms and not a property result.

Mechanism/property analysis remains the original P1–P6 program: source
asymmetry, source–transport alignment, forward/backward numerical consistency,
nonlinear rectification, optimizer rectification, and semantic-orbit
centering. SEUP remains the persistence layer. A trajectory-level case must
still receive endpoint-level causal decomposition and matched intervention
evidence before it supports a mechanism property.
