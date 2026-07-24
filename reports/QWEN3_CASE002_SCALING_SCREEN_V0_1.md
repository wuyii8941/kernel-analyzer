# Qwen3 case 002 — scaling screening result

The same blind case generator was instantiated with `attn_type="sdpa"` using
the parent and fixed revisions of TorchTitan PR #2173.  The endpoint, every
natural module boundary, and the generic operation trace were exactly equal;
the two repeated candidate executions were also exact.

This is a useful negative result, not a failed implementation.  In the
current PyTorch nightly, SDPA's omitted scale defaults to the same
`head_dim**-0.5` value that Qwen3 computes explicitly.  Therefore the PR's
behavioral difference is not visible in this SDPA configuration.  The bug
requires a flex-attention backend case, which is not silently substituted by
the current T4 harness.

The case remains in the registry as a screened control.  We do not claim that
the scaling fix is reproduced until a flex-capable environment is available.

