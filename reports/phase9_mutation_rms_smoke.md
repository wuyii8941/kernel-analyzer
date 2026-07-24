# Phase 9 Refined Artificial Mutation Evaluation

## Objective

Evaluate decision-level fork detection on a broader set of executed model mutations spanning normalization, RoPE, attention, MLP, materialization and final vocabulary reduction.

## Claim Boundary

These are artificial model-level mutations motivated by compiler/kernel failure mechanisms. They are neither historical bugs nor certified violations of a floating-point contract. A nonzero canary proves that the altered execution reached the measured logprobs; it does not prove realism of the mutation frequency.

## Controls

- Frozen checkpoint and replay batch; `512` aligned response tokens per mutation.
- Baseline maximum mismatch against canonical eager certificates: `1.66892e-06`.
- Each mutation is independently installed and restored on the same eager model object.
- Mutations with zero observed output delta are marked invalid and excluded from `all_mutation_rows.jsonl`.

## Results

| Mutation | Mechanism | Modules | Canary max delta | p99 delta | Branch forks | Fork rate | Valid |
|---|---|---:|---:|---:|---:|---:|---|
| rmsnorm_no_upcast | missing FP32 accumulation | 113 | 11.931 | 11.931 | 356 | 0.69531 | True |

## Interpretation

Fork rate measures whether an executed mutation changes the frozen PPO/GRPO clipping branch. Delta magnitude and fork rate remain separate signals: a large mutation can miss all boundaries, while a smaller directed change can cross one.

Cross-format BF16 round-trip mutations are controlled sensitivity probes on T4 FP16, not claims about native BF16 kernel behavior.

## Artifacts

- `results/phase9_mutations/summary.json`
- `results/phase9_mutations/all_mutation_rows.jsonl`
- `results/phase9_mutations/<mutation>.jsonl`
- `scripts/phase9_mutation_catalog.py`
