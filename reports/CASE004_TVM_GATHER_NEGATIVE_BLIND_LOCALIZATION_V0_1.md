# Case 004: TVM ONNX Gather negative-index blind localization

This case was preregistered from the reproducer and independent ONNX Runtime
contract.  The fixed revision, patch, and issue discussion are outside the
locator input.  The report is frozen at the observation stage before any
repair is attempted.

## Baseline evidence

Two independent CPU/LLVM runs of the buggy checkout agree exactly.  The
positive-index control agrees with ONNX Runtime, while the negative-index
witness disagrees with maximum absolute error 12.  This separates a general
Gather failure from the declared negative-index trigger.

The converted Relax IR is `relax.take(..., mode="fast")`; the legalized TIR
reads the supplied index directly.  This is provenance and structural
evidence, not same-input numeric local production evidence.

## Frozen claim

The pre-reveal certificate allows only:

> `OBSERVATION`

It deliberately does not claim a producer, propagator, mediator, repair, or
root cause.  Those claims require the local replay and fixed-suffix tests from
the protocol.  The fixed checkout will be used only after this certificate is
frozen, as an external validation step.

## Post-reveal validation

After the pre-reveal certificate was frozen, the fixed checkout was run. It
matches the ONNX Runtime output exactly. Its Relax IR contains explicit
negative-index normalization (`less`, `add`, and `where`) before `take`, while
the buggy IR passes the raw index to `take`. This supports the hypothesis at
the ONNX frontend → Relax stage, but does not prove a unique kernel cause or
operator-level causal effect.

The post-reveal assessment therefore reports stage coverage only:

> `STAGE_LEVEL_EXTERNAL_VALIDATION_ONLY`

A separately recorded IR repair inserts negative-index normalization at the
Relax boundary and restores the exact output on the buggy checkout. Because
this intervention rebuilds Relax/TIR and does not preserve non-target kernel
context, it is reported as `INTERVENTION_DEPENDENT_ATTRIBUTION`, not as a root
cause or unique kernel localization.

The repair artifact also replays the raw and repaired regions with identical
`X/I` boundary tensors. Their outputs differ by 12, which is valid
IR-level local production evidence. A fixed non-target suffix is not available
in this single-region witness, so mediation remains uninstantiated.

Machine-readable artifacts:

- `theory_oracle/blind_cases/case_004/case_manifest.json`
- `results/operator_oracle/tvm_gather_negative/buggy.json`
- `results/operator_oracle/tvm_gather_negative/buggy_r2.json`
- `results/operator_oracle/tvm_gather_negative/case004_pre_reveal_certificate.json`
- `results/operator_oracle/tvm_gather_negative/fixed.json`
- `results/operator_oracle/tvm_gather_negative/case004_post_reveal_score.json`
- `results/operator_oracle/tvm_gather_negative/repair.json`
- `results/operator_oracle/tvm_gather_negative/case004_final_localization_certificate.json`
- `theory_oracle/tvm_gather_negative_case_v0_1.py`
- `theory_oracle/assemble_tvm_gather_blind_certificate_v0_1.py`
- `theory_oracle/score_tvm_gather_post_reveal_v0_1.py`
- `theory_oracle/tvm_gather_ir_repair_v0_1.py`
- `theory_oracle/assemble_tvm_gather_final_certificate_v0_1.py`
