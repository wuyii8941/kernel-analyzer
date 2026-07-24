# Kernel Analyzer repository manifest

This repository is the curated continuation of the ForkCert research workspace.
It preserves the theory, implementations, calibration results and claim
boundaries needed to continue compiler/operator localization on another
machine.

## What is tracked

- `src/`, `scripts/`, `tests/`, `configs/`: executable implementation and
  regression tests;
- `theory_oracle/`: definitions, theory notes, manifests and subject-specific
  experiment logic; historical documents are retained as historical evidence,
  not automatically current claims;
- `reports/`: stage reports plus the current authority documents:
  `LOCALIZATION_EVIDENCE_LEDGER_V0_1.md` and
  `LOCALIZATION_METHOD_PLAN_V1.md`;
- compact calibration and historical localization artifacts under `results/`.

## What is intentionally not tracked

- model/optimizer checkpoints, rollout dumps and Hugging Face caches;
- TorchInductor/Triton caches, CUDA binaries, build products and environment
  wheels;
- large raw result logs whose derived conclusions are already bound in reports.

Those objects remain on the original machine and must be reproduced or moved
through an artifact store.  Their absence does not permit a stronger claim than
the evidence ledger licenses.

## Current method state

The current method is a staged compiler-localization pipeline:

```text
semantic-contract violation
  -> stage screen
  -> symptom-preserving region reduction
  -> cross-level provenance
  -> same-input production replay
  -> fixed-suffix mediation / controlled intervention
  -> claim-gated certificate
```

The kernel and one-step training calibration cases are complete.  The generic
locator is frozen.  Independent withheld historical evaluation and the
Megatron matched-step flagship case remain open; see the localization plan and
Phase-4 asset audit.
