# Qwen3 candidate-preserving kernel-15 repair findings v0.4

The live Inductor generated module was resolved after the frozen candidate
reproduced its exact graph family and scorer anchor.  Calls 0, 13 and 26 of the
27-call cross-layer fused family were repaired separately.  Every arm invoked
the family 27 times, entered the eager reference replacement exactly once,
repeated exactly, caused no backend compilation, and restored to the original
candidate anchor afterward.

All three selected repairs changed selected-token log probabilities.  L2 effect
magnitudes were approximately `0.07789`, `0.06782` and `0.02263` for calls 0,
13 and 26 respectively.

This is original-candidate-preserving generated-kernel invocation evidence.  It
does not identify RMSNorm, reduction, residual addition, cast or any other
constituent as the cause, because the repaired unit performs all of them.  It
also does not yet show that a repair reduces distance to eager; v0.4 measured
whether the observable changed, not the direction of drift reduction.
