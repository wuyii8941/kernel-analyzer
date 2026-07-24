# Qwen3-1.7B checkpoint-weight compiler gradient case — v0.1

This is the stronger version of the compiler-specific case.  The input uses
Qwen3-1.7B hidden size 2048 and the weight is the actual layer-0
`self_attn.q_proj.weight` from the official checkpoint revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; only the algebra is expressed as
`x @ weight` so the declared projection is a generic Qwen3-equivalent subject.

The blind locator received only the opaque tensors and endpoint contract.  It
observed:

```text
numeric scalar:    exact match
requires_grad:     True → False
has_grad_fn:       True → False
backward_succeeds: True → False
```

The generic compiled graph inventory contained an `mm` operation.  After the
blind result was frozen, the official PyTorch issue
[#181581](https://github.com/pytorch/pytorch/issues/181581) and remediation PR
[#181606](https://github.com/pytorch/pytorch/pull/181606) were revealed.  The
post-reveal score covers the documented silent AOTAutograd metadata-loss
mechanism.  The newer nightly preserves metadata and turns the unsupported
double backward into an explicit failure.

The same input and checkpoint weight with a contiguous `F.linear` negative
control preserved `requires_grad` and `grad_fn` in PyTorch 2.11.  This is not a
claim that the negative control supports double backward; it only rules out
the specific silent metadata-loss signature for that path.

An additional same-version control compiled the same function with Dynamo's
`eager` backend rather than `aot_eager`.  It preserved `requires_grad` and
`grad_fn`, which supports localizing the semantic change to the AOTAutograd
stage rather than to Dynamo graph capture alone.  This is a stage-level
localization signal, not a unique source-line proof.

The blind locator was given this control without issue metadata and reported a
`BACKEND_SPECIFIC_STAGE_CANDIDATE`: the candidate differed from reference while
the control preserved the contract.  Only after reveal do we interpret that
stage as AOTAutograd.

Claim level:

```text
COMPILER_OPERATION_CANDIDATE_WITH_EXTERNAL_PATCH_COVERAGE
```

This is a compiler/AOTAutograd semantic localization result around a real
Qwen3 checkpoint weight, not a full Qwen3 training run or a generated CUDA
kernel root-cause proof.

Artifacts:

- package: `data/qwen_bug_sources/opaque_qwen3_grad_case_b/`
- blind report: `results/operator_oracle/qwen3_compiler_grad_case_b_blind_locator.json`
- protocol audit: `results/operator_oracle/qwen3_compiler_grad_case_b_protocol_audit.json`
- post-reveal score: `results/operator_oracle/qwen3_compiler_grad_case_b_post_reveal_score.json`
- same-version Dynamo-eager control: `results/operator_oracle/qwen3_compiler_grad_case_b_dynamo_eager_pt211.json`
