# Phase 9 Refined Artificial Mutation Evaluation

## Objective

Evaluate decision-level fork detection on a broader set of executed model mutations spanning normalization, RoPE, attention, MLP, materialization and final vocabulary reduction.

## Claim Boundary

These are artificial model-level mutations motivated by compiler/kernel failure mechanisms. They are neither historical bugs nor certified violations of a floating-point contract. A nonzero canary proves that the altered execution reached the measured logprobs; it does not prove realism of the mutation frequency.

## Controls

- Frozen checkpoint and replay batch; `512` aligned response tokens per mutation.
- Baseline maximum mismatch against canonical eager certificates: `1.66892e-06`.
- Baseline clipping-branch mismatches against canonical certificates: `0`.
- Each mutation is independently installed and restored on the same eager model object.
- Mutations with zero observed output delta are marked invalid and excluded from `all_mutation_rows.jsonl`.

## Results

| Mutation | Mechanism | Modules | Canary max delta | p99 delta | Branch forks | Fork rate | Valid |
|---|---|---:|---:|---:|---:|---:|---|
| rmsnorm_no_upcast | missing FP32 accumulation | 113 | 11.931 | 11.931 | 356 | 0.69531 | True |
| rmsnorm_eps_wrong | constant/configuration corruption | 113 | 0.091016 | 0.029076 | 2 | 0.0039062 | True |
| rmsnorm_nminus1 | reduction denominator off by one | 113 | 0.26334 | 0.066632 | 9 | 0.017578 | True |
| rotary_tail_unrotated | partial rotary-kernel write | 1 | 0.98501 | 0.35827 | 35 | 0.068359 | True |
| rotary_phase_fp16 | missing phase upcast | 1 | 0.057185 | 0.024729 | 0 | 0 | True |
| attention_scale_plus_0p1pct | miscompiled scalar constant | 28 | 0.029996 | 0.015573 | 2 | 0.0039062 | True |
| q_projection_bf16_round | unexpected projection materialization | 1 | 0.022552 | 0.015401 | 1 | 0.0019531 | True |
| attention_output_bf16_round | unexpected attention materialization | 1 | 0.037163 | 0.015724 | 3 | 0.0058594 | True |
| mlp_gate_bf16_round | unexpected MLP materialization | 1 | 0.032879 | 0.015808 | 3 | 0.0058594 | True |
| mlp_output_bf16_round | unexpected fused-MLP output cast | 1 | 0.044917 | 0.014547 | 1 | 0.0019531 | True |
| embedding_bf16_round | wrong embedding output dtype | 1 | 0.020986 | 0.014656 | 1 | 0.0019531 | True |
| decoder0_output_bf16_round | wrong residual-stream materialization | 1 | 0.052545 | 0.016688 | 0 | 0 | True |
| lm_head_bf16_round | wrong logits materialization | 1 | 0.084793 | 0.062231 | 7 | 0.013672 | True |
| logsoftmax_fp16 | missing final-logits upcast | 1 | 0.0038452 | 0.0021982 | 0 | 0 | True |
| logsumexp_chunked_reverse | different final reduction partition/order | 1 | 1.9073e-06 | 1.4305e-06 | 0 | 0 | True |

## Interpretation

Fork rate measures whether an executed mutation changes the frozen PPO/GRPO clipping branch. Delta magnitude and fork rate remain separate signals: a large mutation can miss all boundaries, while a smaller directed change can cross one.

Cross-format BF16 round-trip mutations are controlled sensitivity probes on T4 FP16, not claims about native BF16 kernel behavior.

## Artifacts

- `results/phase9_mutations/summary.json`
- `results/phase9_mutations/all_mutation_rows.jsonl`
- `results/phase9_mutations/<mutation>.jsonl`
- `scripts/phase9_mutation_catalog.py`
