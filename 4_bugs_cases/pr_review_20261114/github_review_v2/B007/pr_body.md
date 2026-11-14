## Summary

Preserve the normalization key sum Z across BF16/FP16 decode calls. With unit Q/K/V, Z currently stalls at 256/2048 steps while the FP32 KV state keeps increasing, so the output no longer preserves constant values. Keep Z and its denominator in FP32 and return attention output in the incoming activation dtype.

This affects `normalize=True`: it is the default for `chunk_linear_attn` and `fused_chunk_linear_attn`, but not `fused_recurrent_linear_attn`. It also affects `LinearAttention(do_feature_map_norm=True)` and the normalized chunk/fused-chunk paths of Based/ReBased. The precision policy needs an RFC agreement before this implementation PR; the separate RFC draft includes the interface decision.

## Test plan

- New decode regressions: original 2 numerical failures / 2 explicit-FP32-state controls passing; patched 4 passes. Numeric assertions precede the state-dtype assertion.
- Complete `tests/ops/test_linear_attn.py`: **58 passed**, including all three implementations, gradients, state resume, padding, cache and empty sequences.
- `tests/layers/test_based.py`: **43 passed, 8 skipped**; skips are the existing unsupported odd expanded dimensions in 16-bit chunk kernels.
- ReBased integration: **6 passed**, covering both affected modes and three dtypes, independent quadratic attention output comparisons from physical feature-map values, and finite input/parameter gradients. This does not claim an independent ReBased gradient oracle.
- Changed files pass the repository Ruff rules and whitespace checks.

Hardware/software: RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0, `FLA_CI_ENV=0`. The complete-file run used an 80% allocator cap on an otherwise idle GPU; the earlier 16% cap caused OOMs. Earlier failed/diagnostic attempts remain in the evidence notes. No test tolerance was relaxed.

```bash
FLA_CI_ENV=0 python -m pytest tests/ops/test_linear_attn.py
FLA_CI_ENV=0 python -m pytest tests/layers/test_based.py
```

## Benchmark / NCU

Same-device normalization-helper forward timings on RTX A6000 (BF16, H=2, D=64), three warm samples; descriptive medians, not a speedup claim:

| Workload | Before ms | After ms | Incremental peak bytes before → after |
| --- | ---: | ---: | ---: |
| dense T=1 | 0.016908 | 0.023263 | 1,536 → 2,048 |
| dense T=128 | 0.025306 | 0.034516 | 98,304 → 197,632 |
| packed T=128, lengths 64/64 | 0.365197 | 0.420195 | 131,584 → 230,912 |

Baseline timing revision: `aa0f314b97628ef0f5ba1a62c733cb1ba16752ae`. Current patch base: `2705ab8ee9a9086bcfa6d079858e4281e01215a1`; the relevant normalization/cumsum source is unchanged between these revisions. Candidate is the attached patch, SHA-256 `4734e2a369f6be05cabcba8c372fc51eb6c61712546556f5c98c9d8a1bf46141`. [The bounded timing/allocation procedure and samples](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/PERFORMANCE_v1.md) are publicly linked; these are helper timings, not model latency. The FP32 calculation costs more time and memory in these measurements. NCU and pretrained-model quality were not evaluated.

## Breaking changes

Returned Z becomes FP32, doubling its FP16/BF16 storage. Cache consumers must preserve FP32; lost increments in an old rounded cache cannot be reconstructed. Signatures and state-tuple structure do not change. When the caller explicitly supplies FP32 Z, the old helper could also promote attention output to FP32; this proposal returns the activation dtype instead. That output policy and default FP32 state need maintainer agreement in the RFC.

## Checklist

- [x] I have read [CONTRIBUTING.md](https://github.com/fla-org/flash-linear-attention/blob/2705ab8ee9a9086bcfa6d079858e4281e01215a1/CONTRIBUTING.md) and follow its conventions (code style, docstrings, commit prefixes).
- [x] I have read [AGENTS.md](https://github.com/fla-org/flash-linear-attention/blob/2705ab8ee9a9086bcfa6d079858e4281e01215a1/AGENTS.md) and, where my change matches its scope, the relevant skill under [.agents/skills](https://github.com/fla-org/flash-linear-attention/tree/2705ab8ee9a9086bcfa6d079858e4281e01215a1/.agents/skills).
- [ ] Dependent tests pass locally or in CI, and new behavior is covered by tests where applicable (tick as N/A for changes with no testable code, e.g. docs-only).
- [x] Kernel changes include same-hardware before/after benchmark numbers, dense + varlen where applicable (tick as N/A when no kernel code changed).
- [ ] This PR is minor/cosmetic-only (typo, formatting, style-only tweaks) — tick only if it is, and justify below.
- [ ] I understand this PR exceeds 500 changed lines and have explained its single purpose and why it should stay together below.

The size exception is not applicable: this patch is below 500 changed lines. The dependency helper returns a broad import graph; the tests above cover the changed normalization helper, its three operator callers and its direct layer integration. Unrelated backend/model suites are not claimed as executed.

Current gating item: agree the RFC output/state dtype policy and assert both implicit and explicit-FP32 output dtypes before opening the implementation PR. The unchecked dependent-test item reflects that pending interface gate; it does not negate the [58-case numerical run](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B007_complete_file_cap80/junit.xml). [All prior failed attempts and limits](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/README.md) remain available.
