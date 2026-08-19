# Phi MM anchor

Phi has the strict v2.1 transition `LOCAL_CENTERED -> GRADIENT_BIASED -> UPDATE_BIASED`. A matched row-pairing intervention preserves the local residual norm and changes the gradient population from `BIASED` to `CENTERED`. This validates a case-level composite backward-transport mechanism. The current analytic RMSNorm-only reconstruction has 0.32--0.60 relative error, so no universal transport property is claimed.

Evidence: `phi_transport_mechanism.json`, `intervention_results/phi_mm_transport_pairing.json`.
