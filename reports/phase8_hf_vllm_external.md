# HF-vLLM External Path Pair

## Objective

Complete the T4 FP16 serving-engine external-validity pair without allowing it to block the core eager/compile evidence.

## Controls

- Same Qwen3-0.6B step-5 model tensor hash and tokenizer vocabulary hash.
- Exact prompt and response token IDs supplied to both paths.
- Teacher forcing through vLLM `prompt_logprobs`; generated token excluded.
- HF parameters and vLLM parameters both stored as FP16 in the matched comparison.
- Two independent HF processes, two independent vLLM processes and one reversed-request vLLM process.
- HF self p99, vLLM self p99 and vLLM request-order p99 are all zero.

## Result

Across 8 requests and 1,024 response tokens, cross-path absolute logprob delta has mean `0.3472`, p50 `0.1132`, p99 `3.1581` and max `7.8982`. The self gate and empirical request-order invariance gate pass.

When aligned with the 512 real step-5 clipping states, the composite HF-vLLM pair changes 139 clipping branches. This is an external sensitivity count, not a canonical single-variable fork rate.

## Interpretation

The result establishes that the measured HF and vLLM T4 execution stacks are not interchangeable for training logprobs at this checkpoint. It does not isolate a root operator: the paths differ in Transformers/vLLM model implementation, MATH SDPA versus xFormers, engine execution and software version. vLLM V0 lacks an explicit raw/processed selector; local source audit only establishes that configured processors are semantically identity.

## Artifacts

- Matched rows: `results/phase1_hf_vllm_fp16_matched.jsonl`
- Clipping alignment: `results/phase4_hf_vllm_step5.jsonl`
- Independent process payloads: `results/phase1_hf_fp16_for_vllm_a.json`, `results/phase1_hf_fp16_for_vllm_b.json`, `results/phase1_vllm_a.json`, `results/phase1_vllm_b.json`, `results/phase1_vllm_c_reordered.json`

## Next Decision

**STOP on T4 expansion.** Native BF16, FlashAttention and vLLM V1 require Ampere-or-newer hardware. Do not interpret this composite pair as operator attribution.
