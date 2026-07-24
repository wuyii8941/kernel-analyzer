# Qwen3 historical bug cases — v0.1

This is the external ground-truth ledger for the Qwen3 blind-localization
campaign.  It must not be copied into a blind case package.

## Primary case: varlen attention layout

TorchTitan issue [#2223](https://github.com/pytorch/torchtitan/issues/2223)
describes a Qwen3 varlen attention path where the backend already transposes
the packed attention output, while the parent `Attention` implementation
applies an unconditional second transpose.  The report explains that a later
reshape can hide the layout mistake.  PR
[#2229](https://github.com/pytorch/torchtitan/pull/2229) is merged and moves the
transpose into only the flex and sdpa branches.  This gives us a fixed commit,
an explicit changed source path, a small endpoint, and a developer-written
mechanism description.

The blind case contains none of the issue/PR text or commit IDs.  The locator
is allowed to see only the Qwen3 Attention configuration, reference endpoint
artifacts, natural module boundaries, and a generic operation trace.

The same case was also instantiated at the Qwen3 1.7B flavor dimensions
(2048 hidden size, 16 query heads, 8 KV heads, 128 head dimension).  The blind
run again narrowed the difference to the `inner_attention -> wo` interval and
the generic trace mismatch covered the historical patch after reveal.  This
is an architecture-scale check, not a full 1.7B checkpoint or training run.

## Additional candidates

* PR [#2173](https://github.com/pytorch/torchtitan/pull/2173) fixes a missing
  attention scaling argument in Qwen3 flex/sdpa calls.  It is a good second
  forward case, but the first generic SDPA screening was a negative control:
  the historical buggy and fixed outputs were exact because current SDPA's
  default scale is already `head_dim**-0.5`.  A faithful reproduction needs
  the flex backend; the SDPA result is retained to prevent us from turning a
  version/backend-dependent patch into a universal bug claim.
* Issue [#2252](https://github.com/pytorch/torchtitan/issues/2252) and PR
  [#2253](https://github.com/pytorch/torchtitan/pull/2253) fix Qwen3 weight
  tying.  This is intentionally a training-semantics transfer case, not
  evidence of an Inductor compiler defect.

## What these cases can validate

The primary case can validate whether a generic Oracle plus module/operation
trace narrows a wrong endpoint to an attention-region operation interval.  It
cannot, by itself, prove that the historical patch would have been discovered
without the reference artifact, nor can it prove a unique compiler root cause.
Those claims require the blind certificate, post-reveal patch coverage, and
non-target context checks.

## Compiler-specific Qwen3 case

PyTorch issue [#181581](https://github.com/pytorch/pytorch/issues/181581)
provides a separate semantic compiler case.  The materialized case uses the
actual Qwen3-1.7B layer-0 query-projection checkpoint weight, expresses the
same projection as a declared `mm` subject, and tests the
`create_graph=True` higher-order-gradient contract.  PyTorch 2.11 loses
`requires_grad` and `grad_fn` while preserving the numeric scalar; the newer
nightly preserves metadata and reports unsupported double backward explicitly.

The blind locator identifies semantic disagreement and an operation inventory
containing `mm` without receiving issue or patch information.  This is the
first case in the ledger that directly exercises a compiler/AOTAutograd
semantic failure.  It remains a Qwen3 projection slice rather than a full
Qwen3 training run or generated CUDA-kernel localization.

Case C strengthens this by using the actual layer-0 Qwen3 input boundary
(embedding followed by input RMSNorm) together with the actual `q_proj` weight.
Its blind result and stage control are the same, so the evidence is not an
artifact of a random hidden-state choice.
