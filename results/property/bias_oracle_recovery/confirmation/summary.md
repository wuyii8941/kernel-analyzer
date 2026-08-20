# Prospective moving-frame confirmation

## Result

The frozen screen confirmed 1/3 promoted candidates on 32 disjoint natural states. It directly flagged deepseek8b_seq64_l35_attention_dv. None of the 2 sign-changing controls produced a directional hit.

| case | role | mean coefficient | 95% bootstrap CI | result |
|---|---|---:|---:|---|
| `qwen_seq64_l27_q_norm_vjp` | CANDIDATE | -0.00461198 | [-0.00977001, 0.00041722] | `REJECTED_ON_HELDOUT` |
| `qwen_seq64_lm_head_dx` | SIGN_CHANGING_CONTROL | -0.00270788 | [-0.00550484, 4.55216e-05] | `CONTROL_NOT_FLAGGED` |
| `deepseek8b_seq64_ce_dlogits` | SIGN_CHANGING_CONTROL | -0.000563468 | [-0.00267081, 0.00131664] | `CONTROL_NOT_FLAGGED` |
| `deepseek8b_seq64_l35_attention_dv` | CANDIDATE | -0.00369834 | [-0.00663691, -0.000591755] | `CONFIRMED_DIRECTIONAL_RISK` |
| `deepseek8b_seq64_l35_softmax_vjp` | CANDIDATE | -0.000715803 | [-0.00810598, 0.0062426] | `REJECTED_ON_HELDOUT` |

## Scientific interpretation

The new DeepSeek case is the layer-35 attention value-gradient boundary:

`O = P V`, `dV = P^T dO`, `dW_v = dV^T H`.

Replacing only the actual compiled BF16 `dV` BMM output with its exact FP32-recomputed, BF16-ABI reference changes the complete `v_proj.weight` gradient. Across unseen states, the candidate-minus-repair gradient has a negative mean component in the same-state repair-gradient frame. Thus the implementation systematically contracts this update component even though the absolute parameter-space direction changes with the state.

This is a new conditional training-bias case, not proof that every attention BMM is biased. Qwen q_norm and DeepSeek softmax failed the frozen confirmation and remain unresolved/non-replicating rather than positives.

## Oracle boundary

The moving-frame statistic is now a validated sufficient risk witness. A miss is not a safety certificate: Phi requires complete-vector population coherence, while saved-P and SiLU require the exact antithetic optimizer-response witness. The practical oracle is therefore a fail-closed multi-witness cascade, not one universal scalar.
