# Native BF16 External Replay Protocol

## Objective

Replay the canonical Qwen3-0.6B GRPO clipping experiment on an Ampere-or-newer GPU using native BF16 compute. This closes the current FP16/T4 external-validity limitation without silently changing the model, training recipe, or decision rule.

## Fail-Closed Hardware Gate

`run_phase15_bf16_external.sh` requires exactly one visible CUDA device, compute capability SM80 or newer, and `torch.cuda.is_bf16_supported() == True`. It has no FP16 fallback. A T4 run exits before model loading.

## Fixed Comparison

- model: `Qwen/Qwen3-0.6B`
- training: 300-step TRL GRPO, 64 prompts, 3 policy iterations, epsilon 0.2
- reference: HF eager BF16 with SDPA MATH locked
- alternative: `torch.compile` BF16 with SDPA MATH locked
- state: policy iteration 2, before each minibatch update
- coverage target: 51,200 online token records

The single comparison variable is eager versus compiled execution. The shared in-memory model, token IDs, attention mask, position-ID convention, old logprob, advantage, and optimizer state are frozen within each comparison.

## Command

```bash
cd /data1/tzh/forkcert
./run_phase15_bf16_external.sh
```

The runner may select one compatible GPU automatically. A scheduler can instead expose one device through `CUDA_VISIBLE_DEVICES`.

## Outputs

- `results/bf16_external/preflight.json`: hardware and BF16 capability gate
- `data/bf16_external/grpo_dump.metadata.json`: actual training dtype and environment
- `results/bf16_external/online_compile.jsonl`: state-aligned path measurements
- `results/bf16_external/certificates.jsonl`: clipping fork certificates
- `results/bf16_external/audit.json`: final fail-closed consistency audit
- `reports/bf16_external/`: margin, online-scan, natural-fork, and audit reports

## Claim Boundary

A passing replay supports external validity of natural decision forks under native BF16 on the recorded hardware. It does not prove that every FP16 fork persists in BF16, establish a legal end-to-end floating-point bound, or classify a path difference as an implementation bug.
