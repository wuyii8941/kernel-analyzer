# Bias Formation Map — mechanism discovery result

## Core answer

The current study validates two independent training-semantic bottlenecks that can turn implementation variation into directional parameter-gradient/update effects:

1. **Phi MM: composite backward transport pairing.** Local numerical variation is centered in the strict v2.1 formation capture, but the parameter-gradient and effective-update populations are biased. A norm-preserving residual/transport pairing intervention removes the gradient bias. The complete analytic transport decomposition is still open, so this is a validated empirical composite mechanism, not a universal source–transport law.
2. **Qwen layer-23 attention state: semantic backward-state transport.** The exact F+B equations close at the semantic region. Restoring `S_bwd` closes the q_proj carrier direction while restoring `K` alone does not; the sham is exact. This is an independently validated attention-state mechanism, explicitly bounded as a semantic region rather than a single kernel.

Qwen saved-P remains a centered boundary case. Liger remains unresolved at formation despite prior persistence. No optimizer-induced mechanism has been observed.

## Formation versus persistence

Formation labels come only from open-loop common-state measurements. SEUP and live-weight trajectories answer a separate question: whether an already measured mechanism persists into parameter drift. The consequence summary records this separation for Phi and layer-23.

## Scientific scope

The evidence supports a taxonomy of training-semantic bottlenecks, not a single universal property and not an endpoint-count claim. The remaining endpoint population is retained in `bias_population_matrix.csv`; rows without formation capture are explicitly unresolved/not captured. Legacy T1--T4 and SEUP roles are provenance only.
