# ForkCert

ForkCert studies correctness and training impact of numerically different but intended-equivalent DL implementations, such as eager and `torch.compile`/Inductor.

The current project is a contract-based Oracle, not a fork counter:

```text
operator semantic contract
        ↓
operator conformance ledger
        ↓
complete one-step transition contract
        ↓
semantic/training impact
        ↓
long-run validation as a separate question
```

## Current Oracle

For a declared operator input or complete training state, first define the permitted behavior from an API/math/specification/high-precision source. Then check candidate membership.

- `REJECT`: a valid covered contract violation;
- `ACCEPT`: evidence lies inside the declared permitted set and scope;
- `UNINSTANTIATED`: no defensible contract/envelope;
- `INVALID/INAPPLICABLE`: state or candidate path is not the promised subject;
- `INDETERMINATE`: finite evidence cannot resolve the boundary.

Eager is a baseline unless an independent source makes it normative. Bias, state heterogeneity and runtime variability describe implementation discrepancy; they do not define correctness.

## Start here

1. [Oracle theory index](theory_oracle/README.md)
2. [Current discrepancy definition](theory_oracle/DISCREPANCY_ORACLE_DEFINITION_V3.md)
3. [B/H/N/U definition](theory_oracle/BIAS_VARIANCE_ORACLE_DEFINITION_V2.md)
4. [Historical-bug blind protocol](reports/HISTORICAL_BUG_BLIND_PROTOCOL_V0_1.md)
5. [Generic locator and Oracle evidence](reports/GENERIC_LOCATOR_CASE004_ORACLE_ROLE_V0_1.md)
6. [Localization evidence ledger](reports/LOCALIZATION_EVIDENCE_LEDGER_V0_1.md)
7. [Frozen localization-method plan](reports/LOCALIZATION_METHOD_PLAN_V1.md)
8. [Qwen3 subject scope and cleanup policy](theory_oracle/QWEN3_SUBJECT_SCOPE_V0_1.md)

## Current status

- Operator Oracle v0.1 has defined and empirically exercised the decision mechanics, including small violations, real large-but-conforming differences, shared-wrong zero discrepancy, invalid paths and stochastic abstention.
- Training-Step Oracle v0.1 now defines complete state, transition, operator-ledger composition and impact boundaries.
- The first two-state materialized BERT/SGD CUDA smoke passed candidate/state/exact-core gates and observed real next-state discrepancies while correctly retaining `UNINSTANTIATED` numerical conformance.
- Frozen full banks then reproduced the mechanics on 128 discovery plus 128 confirmation states: all covered exact cores accepted, all repeats were stable, and numerical conformance remained explicitly uninstantiated.
- A separately frozen 128-state impact bank accepted strict eager/compiled prediction identity while materialized parameter discrepancies remained numerically uninstantiated.
- Existing BERT/Qwen transition results remain implementation-relative measurements; they are not retrospectively labeled correct/incorrect without an independent numerical transition envelope.
- The project is not yet a universal PyTorch/compiler correctness suite. Coverage across operator families, optimizers, AMP and stochastic training remains explicit future validation scope.
- Qwen3 is a development/regression subject, not an independent localization-accuracy benchmark.  The Qwen higher-order-gradient witness is currently classified as a **blind AOT-stage/operation candidate with post-reveal issue agreement**, not external patch coverage.
- Case 001 (`adaptive_avg_pool2d → flatten/view → sum`) is a live PyTorch-2.11 Inductor regression witness with local replay and a generated-kernel intervention.  It supports intervention-dependent attribution only; it is not an independently patched historical benchmark.
- GPU execution is available on the host.  Earlier reports that describe CUDA/Inductor as unavailable are historical environment records, not current capability statements.
- The active method is being calibrated with two minimum-sufficient cases (kernel plumbing and one-step training), then frozen before hidden historical-bug evaluation and a held-out complex-training case.  The authoritative classification and artifact paths are in the [localization evidence ledger](reports/LOCALIZATION_EVIDENCE_LEDGER_V0_1.md).
- The project test collection is restricted to `tests/` so retained external TorchTitan worktrees are not collected accidentally; the full project suite passes (449 tests).
- The current requirement-to-evidence audit is recorded in [Qwen3 blind-localization completion audit](reports/QWEN3_BLIND_LOCALIZATION_COMPLETION_AUDIT_V0_1.md); it stops at operation/stage candidate evidence and does not claim a unique CUDA kernel root cause.

## Repository layout

- `theory_oracle/`: normative definitions, contract records, validation manifests and retained evidence;
- `src/`: ForkCert library code;
- `scripts/`: experiment/reproduction tools, including retained historical phase runners;
- `tests/`: existing code tests;
- `configs/`: experiment configurations;
- `data/`, `results/`, `reports/`: retained inputs, compact evidence and historical reports.

Historical phase scripts and reports are retained for reproducibility. They do not define the current Oracle or paper claim. Generated caches and stale links are removed; see the [cleanup record](theory_oracle/DEPRECATED_FILE_CLEANUP_2026-07-23.md).  The evidence ledger, rather than historical report prose, is the current claim authority.
