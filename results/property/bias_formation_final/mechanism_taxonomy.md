# Bias Formation Mechanism Taxonomy

The current evidence validates two independent mechanism boundaries while keeping formation and persistence separate.

| Mechanism | Case | Validation | Boundary |
|---|---|---|---|
| Composite backward transport pairing | Phi MM | empirical matched intervention: natural gradient biased, row-paired shuffle centered, local norm preserved | analytic transport decomposition incomplete; no universal P2 claim |
| Attention-state semantic transport | Qwen layer-23 q_proj tile | complete F+B semantic equations, S_bwd-only repair closes direction, K-only does not, exact sham | semantic region, not one kernel; strict v2.1 layer formation labels not captured |
| Source-generated bias | Liger | unresolved confirmation | no source intervention |
| Numerical contract bias | Qwen saved-P | not supported in this case; all three v2.1 layers centered | case-level negative only |
| Optimizer-induced bias | none | not observed | no optimizer intervention |

These are mechanism validations, not a universal property. The two positive mechanisms share a downstream training bottleneck—backward transport into a parameter gradient—but arise from different semantic regions.
