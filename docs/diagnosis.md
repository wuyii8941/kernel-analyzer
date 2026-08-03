# Why the search has found few complete cases

The short answer is that the search is not failing to see local numerical
differences. It is failing at the much rarer second and third links of the
causal chain:

\[
\text{local directional error}
\;\longrightarrow\;
\text{persistent carrier}
\;\longrightarrow\;
\text{weight accumulation}.
\]

Most operator invocations have a small finite residual, but their signs and
directions vary across tokens, layers, and states. Those residuals cancel
before they become a parameter-gradient direction. A full F+B proof therefore
does not imply a FlashAttention-like bias case.

There are also two coverage gaps in the previous round:

1. **State coverage.** The main Qwen screen used frozen pretrained text states.
   It did not provide a bank of checkpoints from an evolving optimizer
   trajectory. A carrier can be absent at initialization and emerge only after
   repeated updates, so more frozen texts are not equivalent to more training
   states.
2. **Implementation coverage.** Many candidate/reference pairs used the same
   arithmetic program at the same precision. A dtype comparison can expose
   BF16 reduction error, but it cannot test a changed online-softmax schedule,
   accumulator, materialization point, approximation, or layout unless that
   implementation is actually selected. The previous matrix therefore had a
   large denominator but a small number of genuine implementation differences.

The public FlashAttention paper reference requires a particularly specific
regime: low-rank attention representations, repeated near-maximum logits, and
an online-softmax quantity reused by backward. The repository did not contain
the paper's OpenWebText binaries or checkpoints, and `flash_attn` is not
installed in the current environment. Thus the paper's natural run could not be
claimed as reproduced from the old Qwen evidence.

## What the new control establishes

`scripts/flash_control.py` writes a compact two-level control:

- `PAPER_REFERENCE_REPRODUCTION`: the public Python online-softmax algorithm,
  with its backward edge bound to the same forward tensors. A low-rank,
  repeated-near-maximum stress bank shows finite BF16 forward and backward
  differences, and a paired live-weight V-only trajectory evaluates the FP32
  reference at each arm's current weight.
- `REAL_PYTORCH_SDPA_FLASH_BACKEND`: PyTorch's CUDA flash-SDPA backend on the
  A6000, evaluated against FP32 eager on the identical inputs. This is a real
  fused CUDA control, but it is not evidence about the separate `flash_attn`
  package and it is not the paper's Python implementation.

The output is `results/final/flash_control.json`. The stress bank is a positive
control, not a natural-model case: its purpose is to verify that the complete
F+B/carrier instrumentation can detect a known mechanism when the required
geometry is deliberately present. The real CUDA results are kept separate
because the public reference uses BF16 online accumulators, whereas production
FlashAttention/SDPA kernels can use different accumulator and FMA schedules.

## Consequence for the next search

If a natural Qwen matrix still yields only the two existing cases after these
controls, the defensible conclusion is not “all other operators are safe.” It
is that local arithmetic differences are common but coherent training bias is
rare and requires a carrier-bearing implementation/state combination. The next
search must therefore cross:

\[
\text{natural checkpoint}
\times
\text{real implementation difference}
\times
\text{closed F+B unit},
\]

then apply the same carrier and live-weight gates. Only after that matrix is
exhausted should we decide whether to add a new model architecture or begin
property generalization.

