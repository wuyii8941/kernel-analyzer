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
- explicit raw/processed mode: FAIL / vLLM 0.9.2 API limitation

Local-source audit shows V0 computes prompt logprobs after the generic sampler pipeline. For this experiment, penalties and minimum-token constraints are disabled, temperature processing is identity for prompt scoring, and top-p/top-k are unrestricted. This establishes an identity processed pipeline for the configured request, but the API still does not expose or independently compare explicit raw and processed modes.

## Delta Self Control
HF p99=0; vLLM independent-process p99=0.

## Summary
| requests | tokens | delta_mean | delta_p50 | delta_p95 | delta_p99 | delta_max | self_ref_p99 | self_alt_p99 | self_gate | batch_order_p99 | batch_order_max | batch_order_invariant | vllm_version | vllm_engine | hf_self_is_independent_process |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 1024 | 0.3472222759676866 | 0.1132480800151825 | 1.4693503677845 | 3.1581260317563995 | 7.898191928863525 | 0.0 | 0.0 | True | 0.0 | 0.0 | True | 0.9.2 | V0 | True |

## External Validity
This pair uses vLLM V0 on Tesla T4 FP16. It does not establish native BF16 or V1 processed-logit behavior.

The pair is composite: HF uses Transformers 5.13 MATH SDPA, while vLLM 0.9.2 uses its Qwen3 implementation and xFormers on T4. The measured cross delta must not be attributed to xFormers alone.
