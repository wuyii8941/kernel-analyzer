# Qwen3 Training-Step Oracle Contract v0.1 — 2026-07-17

## Status

Frozen before scoring. This is the second Training-Step Oracle subject and the
first decoder-only language-model subject. It is an architecture and transition
coverage extension, not a claim that Qwen3-0.6B is the newest available model.

## 1. Why this subject

Subject: the local `Qwen3ForCausalLM` checkpoint at
`data/phase6_policy_step5_pre`, derived from Qwen3-0.6B training.

The official model card describes Qwen3-0.6B as a 0.6B-parameter causal LM with
28 layers and grouped-query attention (16 query heads, 8 key/value heads):
<https://huggingface.co/Qwen/Qwen3-0.6B>.

It adds mechanisms absent from the BERT/SST-2 contract:

- causal teacher-forced token loss;
- decoder attention with GQA and RoPE;
- RMSNorm and gated SiLU/SwiGLU-style MLPs;
- tied input/output embeddings;
- materialized AdamW moment-state creation;
- sequence-level and token-level impact observables.

Newer Qwen3.6 open models are 27B dense or 35B-A3B according to the official
repository and cannot instantiate an unquantized full-training-step contract on
the current T4. Quantization or LoRA would define a different transition. Gemma
3 270M is a plausible later independent family, but its Hugging Face files
require license acceptance and no such acceptance is assumed here.

## 2. State population and sample bank

The first mechanics run uses frozen rows from
`data/phase6_step5_replay_samples.jsonl`. A row supplies already materialized
`prompt_ids` and `response_ids`; no tokenizer-dependent re-encoding is allowed.

For row `s`, the token sequence is `prompt_ids || response_ids`, truncated to
the declared maximum length. Only retained response tokens contribute to the
loss. A row with fewer than two total tokens or no retained response target is
`INAPPLICABLE`.

This is a batch-conditioned bank over one frozen model checkpoint. It is not a
sample from all Qwen training, all DL training, or the historical optimizer
trajectory.

## 3. Complete declared initial state

For each matched state, both arms receive:

1. identical model parameters and buffers loaded from the local checkpoint;
2. identical tied-embedding alias relation;
3. identical single-sequence token IDs, response loss mask and position implied
   by the input;
4. identical empty AdamW state at optimizer step zero;
5. AdamW options frozen below;
6. identical CPU and CUDA RNG state;
7. model in `eval()` mode; Qwen3-0.6B has attention dropout zero;
8. `use_cache=False`, SDPA math backend, deterministic-algorithm mode;
9. identical external step counter.

The model checkpoint comes from a real GRPO run, but its optimizer state was not
preserved. Therefore this contract intentionally instantiates a new AdamW step
from the checkpoint. It must not be described as replay of the historical GRPO
optimizer transition.

## 4. Implementation pair

- reference arm: eager PyTorch execution of the frozen forward/loss core;
- candidate arm: the same core through `torch.compile(..., backend=Inductor,
  fullgraph=True, dynamic=False)`;
- both arms use the same uncompiled `torch.optim.AdamW` implementation.

Candidate identity is valid only if the tracking backend observes a compiled
graph and exactly one compiled runtime invocation for every scored candidate
arm. Compilation failure, graph fallback or a missing invocation is `INVALID`,
not a numerical reject.

## 5. Frozen transition

Forward loss is next-token cross entropy over response targets only. The model
weights remain float32 and forward execution uses CUDA float16 autocast, matching
the precision family of the prior Qwen/T4 work.

AdamW options:

```text
lr = 1e-6
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.0
amsgrad = false
maximize = false
foreach = false
fused = false
capturable = false
differentiable = false
```

The optimizer is deliberately non-fused in v0.1 so that this study changes the
model forward/backward implementation but not the optimizer implementation.
Fused AdamW is a separate candidate stratum.

## 6. Observables

Every scored arm materializes:

- scalar masked token loss;
- retained target count;
- per-target greedy token predictions;
- gradient presence, shape and dtype for every named parameter;
- next model parameters and buffers;
- next AdamW state, including `step`, `exp_avg` and `exp_avg_sq`;
- optimizer option structure;
- tied-embedding alias relation;
- next CPU/CUDA RNG state;
- next external step counter;
- compiled execution identity.

## 7. Verdicts fixed before execution

### 7.1 Exact transition core

`REJECT` if any of the following differs or violates the declared relation:

- model-state or optimizer-state key/shape/dtype structure;
- gradient presence/shape/dtype structure;
- optimizer option structure;
- AdamW `step` counters;
- non-floating model or optimizer fields;
- tied-embedding alias relation;
- next CPU/CUDA RNG state;
- external counter increment by exactly one.

An independently labelled `candidate-counter-mode=stale` control may keep the
candidate counter unchanged while leaving the numerical model transition
untouched. It must be rejected by the exact core. This is a validation control,
not evidence of a natural compiler bug.

Floating parameters and AdamW moments are not required to be bitwise equal by
the exact core.

### 7.2 Numerical transition

`UNINSTANTIATED`. No independent admissible set for the composed
autocast-forward, backward and AdamW transition is available. The run reports
loss, parameter and optimizer-moment discrepancies descriptively; observed
discrepancies may not be reused to choose a passing tolerance.

### 7.3 Semantic impact

Token-level greedy disagreement and sequence-level any-disagreement are
descriptive in the first mechanics run. They become a pass/fail impact contract
only after a separate bank, endpoint and acceptance rule are frozen.

### 7.4 Correctness language

An exact-core reject is a violation of the covered transition contract. A
floating discrepancy is implementation-relative evidence only. With numerical
transition `UNINSTANTIATED`, the study cannot call a nonzero floating delta a
compiler correctness error.

## 8. Randomness and repeats

Both arms restore the same RNG state. Every state is run at least twice. With
dropout disabled and deterministic algorithms required, any within-arm change
in the result signature is runtime variability and invalidates the claim of a
deterministic mechanics study.

State heterogeneity across token sequences and finite-bank sampling uncertainty
remain separate from within-state runtime variability.

## 9. Predeclared mechanics gates

The first run passes only if:

1. the local checkpoint and frozen sample bank are readable and hashed;
2. candidate identity is valid for every row;
3. all scored states satisfy the exact transition core;
4. all repeated signatures are stable;
5. numerical and impact verdicts remain abstentions as specified;
6. no result is described as historical GRPO optimizer replay;
7. all missing fields and runtime failures are surfaced, not silently dropped.

After the correct arm passes mechanics, one one-state stale-counter control is
run under the same contract and must produce `REJECT` without being relabelled
as a numerical failure.

Failure of a gate narrows or invalidates the mechanics claim; it does not prove
that eager is mathematically correct.

## 10. Explicit non-coverage

This v0.1 contract does not cover:

- historical AdamW moments at the Qwen checkpoint;
- fused/foreach optimizer implementations;
- gradient clipping, AMP `GradScaler`, overflow or step skipping;
- train-mode stochastic dropout;
- gradient accumulation;
- distributed collectives or sharded optimizer state;
- KV-cache training semantics;
- MoE routing (Qwen3-0.6B is dense);
- long-run convergence or quality;
- constituent-operator causality.
