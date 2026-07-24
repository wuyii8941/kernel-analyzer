# Qwen3-1.7B checkpoint scale-up gate v0.1

## Scope

This gate uses the official local `Qwen/Qwen3-1.7B` checkpoint (revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`), one deterministic token tensor of
shape `[1, 4]`, and the same immutable model instance for eager and compiled
execution.  The probe records final logits and hidden-state boundaries after
layers 0, 14 and 27.  It is a scale-up gate, not a bug-localization result.

## Result

The CPU `aot_eager` run passed:

- all four endpoints were bitwise equal between eager and compiled paths;
- two compiled repeats were bitwise equal;
- one captured FX graph contained 2,497 nodes;
- the artifact records node names, targets, users and available source/module
  metadata.

Artifact: `results/operator_oracle/qwen3_1p7b_checkpoint_probe_cpu_aot_eager_v0_1.json`.

This establishes that the larger, real checkpoint can enter the same endpoint
and provenance interface without introducing a discrepancy merely because of
model size.  It does **not** validate CUDA/Inductor behavior.

## Failed CUDA/Inductor gates

The direct CUDA run failed closed because the current execution environment had
no visible CUDA device.  The CPU Inductor run captured the same 2,497-node graph
but stopped before execution because the installed compiler rejects
`-std=c++20` (it only accepts `-std=c++2a`).  This is an environment/toolchain
failure, not evidence of a Qwen3 numerical bug.

The failed artifact is retained as:
`results/operator_oracle/qwen3_1p7b_checkpoint_probe_cpu_v0_1.json`.

## Claim limit

The checkpoint scale-up is currently validated only for CPU `aot_eager`.  A
Qwen3-1.7B CUDA/Inductor result and any operator/kernel claim require a working
CUDA device and compatible C++/Triton toolchain.  The earlier Qwen3 Attention
blind cases remain the positive blind-localization evidence; they use a
TorchTitan model implementation bug, not an independently confirmed Inductor
kernel bug.
