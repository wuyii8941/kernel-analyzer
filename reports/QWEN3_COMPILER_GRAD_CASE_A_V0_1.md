# Qwen3-shaped compiler higher-order-gradient case — v0.1

## Why this case matters

The previous positive Qwen3 case was a TorchTitan model implementation bug.
This case targets the compiler boundary itself.  It uses Qwen3-1.7B hidden-size
dimensions (`2048 x 2048`) and a declared query-projection-equivalent matrix
operation, followed by `autograd.grad(create_graph=True)`.  The case is small
enough to run on CPU and does not depend on a CUDA driver.

The external ground truth is PyTorch issue
[#181581](https://github.com/pytorch/pytorch/issues/181581), which documents
that `torch.compile` can silently drop higher-order gradient metadata for
selected operations including `mm`.  Its remediation is tracked by PR
[#181606](https://github.com/pytorch/pytorch/pull/181606).

## Blind protocol

The locator received only the opaque input/weight package, the eager endpoint
contract and the compiled candidate report.  It did not receive the issue,
PR, version interpretation or root-cause description.  The package audit is
the same patch-free protocol used for the attention cases.

## Results

The PyTorch 2.11.0 `aot_eager` candidate produced exactly the same numeric
scalar as eager, but changed all three semantic fields:

```text
requires_grad:    True → False
has_grad_fn:      True → False
backward_succeeds: True → False
```

The generic locator therefore reported a semantic disagreement even though a
raw numerical-delta Oracle would report zero.  Its compiled graph inventory
contained a generic `mm` node; the locator did not name `mm` as a target in
advance.

The current nightly candidate preserved `requires_grad` and `grad_fn`, while
raising an explicit unsupported-double-backward error on the final backward.
As a predeclared negative control, the same PyTorch 2.11 environment with the
same Qwen3-shaped tensors and `F.linear` preserved `requires_grad` and
`grad_fn`; it did not show the silent metadata-loss signature.  This does not
prove `F.linear` is fully higher-order-gradient correct, but it prevents the
case from being read as a blanket failure of every compiled projection path.
After the blind report was frozen, the post-reveal score found that the
operation inventory covered `mm`, the old run showed silent metadata loss, and
the newer run converted it to an explicit failure.

## Claim level

```text
COMPILER_OPERATION_CANDIDATE_WITH_EXTERNAL_PATCH_COVERAGE
```

This is stronger than the earlier model-implementation case because the
independent issue is explicitly labeled a silent compiler/AOTAutograd
correctness problem.  It still does not prove a unique compiler source line,
generated CUDA kernel, or a complete Qwen3 training failure.  It is a
Qwen3-shaped semantic compiler slice and should be described exactly that way.

Artifacts:

- opaque package: `data/qwen_bug_sources/opaque_qwen3_grad_case_a/`
- reference: `results/operator_oracle/qwen3_compiler_grad_case_a_reference.json`
- buggy candidate: `results/operator_oracle/qwen3_compiler_grad_case_a_candidate_pt211.json`
- fixed/newer candidate: `results/operator_oracle/qwen3_compiler_grad_case_a_fixed_nightly.json`
- blind locator: `results/operator_oracle/qwen3_compiler_grad_case_a_blind_locator.json`
- post-reveal score: `results/operator_oracle/qwen3_compiler_grad_case_a_post_reveal_score.json`
- negative control: `results/operator_oracle/qwen3_compiler_grad_case_a_negative_linear_pt211.json`
