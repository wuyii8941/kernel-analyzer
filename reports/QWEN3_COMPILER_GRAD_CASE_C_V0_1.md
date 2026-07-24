# Qwen3-1.7B actual-boundary compiler gradient case — v0.1

Case C uses both an actual Qwen3-1.7B layer-0 `self_attn.q_proj.weight` and an
actual layer-0 input boundary produced by the checkpoint's embedding followed
by input RMSNorm.  The isolated declared projection is then evaluated under
eager, PyTorch 2.11 `aot_eager`, same-version Dynamo `eager`, and a newer
nightly.

The blind result is unchanged from case B:

```text
numeric scalar:    exact match
requires_grad:     True → False
has_grad_fn:       True → False
backward_succeeds: True → False
```

The independent Dynamo-eager control preserves the contract, so the blind
claim is `BACKEND_SPECIFIC_STAGE_CANDIDATE`.  After reveal, the candidate
operation inventory covers `mm` and the stage evidence agrees with the
AOTAutograd mechanism documented in
[PyTorch issue #181581](https://github.com/pytorch/pytorch/issues/181581).

This is the strongest current Qwen3 evidence: actual checkpoint weights and an
actual model boundary, patch-hidden execution, semantic Oracle, operation
inventory, and a same-version backend control.  It still stops short of a
full training run, CUDA kernel localization, or a unique compiler source-line
claim.

Artifacts:

- package: `data/qwen_bug_sources/opaque_qwen3_grad_case_c/`
- blind locator: `results/operator_oracle/qwen3_compiler_grad_case_c_blind_locator.json`
- protocol audit: `results/operator_oracle/qwen3_compiler_grad_case_c_protocol_audit.json`
- post-reveal score: `results/operator_oracle/qwen3_compiler_grad_case_c_post_reveal_score.json`
