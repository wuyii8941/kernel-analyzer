# Phase 1 HF-vLLM Teacher-Forcing Pair

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- exact shared local checkpoint: PASS
- tokenized prompt and response IDs supplied directly: PASS
- response scored only through prompt_logprobs: PASS
- generated token excluded: PASS
- temperature and penalties disabled: PASS
- two independent HF processes/contexts: PASS
- two independent vLLM processes/contexts: PASS
- self gate: PASS
- empirical batch-order invariance: PASS
- raw/processed identity equivalence: PASS

## Delta Self Control
HF p99=0; vLLM independent-process p99=0.

## Raw/Processed Identity
Equivalent completion: 1320 tokens compared bitwise with 0 mismatches after all request-level logits processors were explicitly disabled.

## Summary
| requests | tokens | delta_mean | delta_p50 | delta_p95 | delta_p99 | delta_max | self_ref_p99 | self_alt_p99 | self_gate | batch_order_p99 | batch_order_max | batch_order_invariant | vllm_version | vllm_engine | hf_self_is_independent_process |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 1024 | 0.3472222759676866 | 0.1132480800151825 | 1.4693503677845 | 3.1581260317563995 | 7.898191928863525 | 0.0 | 0.0 | True | 0.0 | 0.0 | True | 0.9.2 | V0 | True |

## External Validity
This pair uses vLLM V0 on Tesla T4 FP16. It does not establish native BF16 or V1 processed-logit behavior.
