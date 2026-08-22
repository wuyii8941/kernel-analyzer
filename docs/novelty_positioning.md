# Novelty boundary

The contribution is not that numerical deviation in optimized attention has
never been measured.  Golden et al., *Is Flash Attention Stable?* (2024),
already compare Flash Attention deviation with a low-precision-training
baseline and estimate the former to be 2--5 times less significant under their
weight-impact proxy: <https://arxiv.org/abs/2405.02803>.

The narrower methodological gap addressed here is that an unmatched numerical
comparison cannot isolate the causal contribution of one implementation
boundary from ordinary low-precision training, data order, RNG variation, or
closed-loop state feedback.  Kernel Analyzer therefore binds one executed F+B
endpoint and a matched repair, and measures the resulting effective-update and
paired-trajectory difference.

The corresponding claim must remain two-part:

1. **Magnitude context.** Compare operator, RNG, data-order, and precision
   perturbations on the same parameter coordinates.  Absolute distance alone
   is not a safety verdict.
2. **Dynamical character.** Report temporal coherence in addition to distance.
   A smaller but coherent operator perturbation can be qualitatively different
   from a larger diffusive baseline.  Any long-horizon crossover is reported
   only when the measured prefix curves support linear versus square-root
   scaling; it is never inferred from a single endpoint value.

The current Phi four-arm experiment updates one causally closed final-norm
carrier while running full-model F+B.  It is a controlled carrier-scale
comparison, not a full-parameter replication of Golden et al.  A full-training
precision comparison remains a separate, substantially more expensive claim.
