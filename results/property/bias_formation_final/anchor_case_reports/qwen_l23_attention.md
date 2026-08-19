# Qwen layer-23 attention-state anchor

This is a closed semantic-region mechanism, not a single-kernel claim. The exact forward/backward equations are `Y=H W^T`, `dW=G_q^T H`, `S_bwd=alpha J_softmax(P)^T(D V^T)`, and `G_q=S_bwd K`. Restoring `S_bwd` closes the directional q_proj tile, restoring K alone does not, and the matched sham is exact. The live-weight trajectory is a separate persistence result.

Evidence: `qwen_l23_attention_mechanism.json`, `intervention_results/qwen_l23_attention_state.json`.
