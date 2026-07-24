# Qwen3 blind compiler-localization completion audit v0.1

## Objective

Test Qwen3 against known implementation/compiler failures and determine, with
bug metadata hidden from the locator, how far the Oracle can localize the
affected operation and compiler stage.

## Requirement-to-evidence audit

| Requirement | Evidence | Result |
|---|---|---|
| Real Qwen3 subject | Qwen3-1.7B checkpoint, layer-0 `q_proj` weight and actual embedding → input-RMSNorm boundary in case C | satisfied |
| Known external failure | PyTorch issue #181581 and the separate TorchTitan Qwen3 issue #2223 | satisfied |
| Patch-hidden execution | Opaque case C package; protocol audit has no issue/fix/root-cause keys or source worktree | satisfied |
| Oracle beyond raw delta | Numeric scalar is exact while `requires_grad`, `grad_fn` and backward contract differ | satisfied |
| Bug-agnostic operation evidence | Generic FX inventory records stable `mm` target; no operation target is selected in locator code | satisfied |
| Stage evidence | Same-version Dynamo `eager` control preserves the reference contract while `aot_eager` does not | satisfied as stage candidate |
| Post-reveal mechanism check | Public issue #181581 agrees with the `mm`/AOTAutograd mechanism; no independently verified upstream patch artifact is bound | issue agreement only |
| Reproducibility | Repeated runs, stable operation labels, opaque artifact hashes, 41 targeted tests and 449 full project tests | satisfied |
| Unique compiler source line or CUDA kernel root cause | No source-line or CUDA/Inductor execution because the host NVIDIA driver currently times out | not established |
| Full Qwen3 training-step impact | Not part of the isolated higher-order-gradient case | not established |

## Final evidence level

```text
BACKEND_SPECIFIC_STAGE_CANDIDATE
BLIND_STAGE_OPERATION_CANDIDATE_WITH_POST_REVEAL_ISSUE_AGREEMENT
```

The current pipeline has demonstrated automatic localization to an operation
candidate and a backend/compiler-stage candidate in a development case.  It
has **not** demonstrated independently scored historical-patch coverage, a
unique compiler source line, generated kernel root cause, or full training
impact.  Those remain explicitly outside the current claim rather than being
silently inferred.  The current authoritative classification is maintained in
`reports/LOCALIZATION_EVIDENCE_LEDGER_V0_1.md`.
