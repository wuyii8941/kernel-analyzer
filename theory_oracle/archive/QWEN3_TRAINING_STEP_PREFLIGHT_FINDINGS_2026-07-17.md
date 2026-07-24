# Qwen3 Training-Step Oracle Preflight Findings — 2026-07-17

> Historical preflight record. CUDA access was subsequently obtained through
> the host user-systemd execution context and scoring completed. Current results
> are in `QWEN3_TRAINING_STEP_FINDINGS_2026-07-17.md`.

## Outcome

The modern-model extension is contract-frozen and mechanically ready, but no
Qwen3 eager/compiled transition has been scored in this record. CUDA device
access is currently unavailable, so the executor correctly refused to replace
the declared CUDA implementation pair with CPU execution.

This document remains readiness evidence, not Oracle findings.

## Subject decision

Selected subject: local Qwen3-0.6B checkpoint plus frozen teacher-forced
response sequences and a newly initialized AdamW transition.

The choice adds causal-LM, GQA/RoPE, RMSNorm, gated MLP, tied embeddings and
optimizer-moment state to the BERT coverage. It does not claim that
Qwen3-0.6B is the latest released model or that one model represents all modern
DL training.

Qwen3.6 was not substituted because the current official open sizes begin at
27B dense / 35B-A3B and cannot support an unquantized complete training step on
the T4 target. Gemma 3 270M remains a useful later independent-family subject,
but its model files require license acceptance that was not assumed.

## Frozen artifacts

```text
model config:
434086f920d0b657ea845c09902fd144a8641e7ced8a361253fad31fe47144c8

model.safetensors:
f28ce3f7f7da92f7230438acae3f50f0adb83e13207558d03c1a93c3b9e31f11

sample JSONL:
9dbaa3d5940e4aa29e89529ffc0fd68fd70ab22341ee9843d48f04560a5906bd
```

The sample bank has 8 rows. All 8 retain at least one response target under the
frozen maximum sequence length of 64. The first four rows each contain 38
prompt tokens and 128 response tokens before truncation.

## Executor checks completed

- `qwen3_training_step_oracle.py` passes Python bytecode compilation;
- CPU helper tests passed for model-state discrepancy geometry;
- CPU helper tests passed for empty-to-materialized AdamW state creation and
  exact optimizer control comparison;
- the CUDA absence path returns a hard error before creating the output
  directory;
- no tolerance or semantic acceptance rule was selected from observations;
- `candidate-counter-mode=stale` is predeclared as the exact negative control.

## Runtime gate currently failing

Read-only host inspection found Tesla T4 devices on PCI and loaded NVIDIA kernel
modules, but:

```text
/dev/nvidia* absent
nvidia-smi exit code 9
torch.cuda.is_available() == False
```

No host-level repair was performed. Recreating device nodes or changing GPU
driver state is outside the workspace and needs explicit system authority.

## Claims allowed now

- a second, modern decoder-only contract is frozen;
- its state and abstention boundaries are more complete than reusing old Qwen
  log-prob rows as if they were full-step evidence;
- the executor is statically and mechanically checked up to the CUDA boundary.

## Claims not allowed now

- Qwen3 exact transition `ACCEPT` or `REJECT`;
- Qwen3 numerical shift, state heterogeneity or runtime variability estimates;
- Qwen3 semantic-impact conclusions;
- historical GRPO optimizer replay;
- any cross-model generality claim.

## Next executable actions after CUDA access returns

1. run one state, two repeats in `candidate-counter-mode=correct`;
2. require one stable compiled graph and valid candidate invocation identity;
3. run the same state in `candidate-counter-mode=stale` and require exact
   `REJECT` while numerical verdict remains `UNINSTANTIATED`;
4. if both controls behave as declared, score all 8 frozen sequence states;
5. only then freeze a separate held-out impact bank or operator ledger.
