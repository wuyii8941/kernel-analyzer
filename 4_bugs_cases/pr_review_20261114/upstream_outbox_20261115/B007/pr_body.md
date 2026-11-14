## Summary

Preserve the cumulative key sum during normalized BF16/FP16 decoding. With unit Q/K/V, the current Z state stalls at 256/2048 while the FP32 KV state keeps growing, so the attention output no longer stays at 1.

Keep Z and its denominator in FP32 and return the attention output in the activation dtype. This draft implements the proposal in #1735; the default state dtype and explicit-FP32-state output compatibility remain open design decisions.

The affected paths use `normalize=True`: chunk and fused-chunk default to true, fused-recurrent defaults to false. This also covers `LinearAttention(do_feature_map_norm=True)` and normalized Based/ReBased chunk paths.

## Test plan

- Complete `tests/ops/test_linear_attn.py`: **58 passed**, including supported gradients, resumed states and cache/sequence cases.
- `tests/layers/test_based.py`: **43 passed, 8 skipped** for existing unsupported odd 16-bit chunk dimensions.
- Six additional ReBased caller checks pass output-reference and finite-gradient checks; these are not an independent gradient oracle.
- The four decode regressions also pass after adding output-dtype assertions for both implicit and explicit-FP32 initial states. The original numerical regressions fail for implicit low-precision Z; explicit FP32 Z is a working numerical control.
- Changed files pass the repository Ruff rules and whitespace checks.

```bash
FLA_CI_ENV=0 python -m pytest tests/ops/test_linear_attn.py
FLA_CI_ENV=0 python -m pytest tests/layers/test_based.py
```

Tested base: `2705ab8ee9a9086bcfa6d079858e4281e01215a1`. NVIDIA RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.

## Benchmark / NCU

Normalization-helper forward, BF16, H=2, D=64, RTX A6000; medians of three warm samples:

| Workload | Before ms | After ms | Incremental temporary bytes before → after |
| --- | ---: | ---: | ---: |
| dense T=1 | 0.016908 | 0.023263 | 1,536 → 2,048 |
| dense T=128 | 0.025306 | 0.034516 | 98,304 → 197,632 |
| packed T=128, lengths 64/64 | 0.365197 | 0.420195 | 131,584 → 230,912 |

These helper measurements show a time/memory cost; they do not measure whole-model latency. Baseline measurement revision `aa0f314b97628ef0f5ba1a62c733cb1ba16752ae` has the same normalization/cumsum code as the PR base. NCU and model-quality experiments were not run.

## Breaking changes

Returned FP16/BF16 Z becomes FP32, increasing persistent Z storage from 2 to 4 bytes per element. Cache consumers must preserve FP32 state. Signatures and state-tuple structure are unchanged.

Explicit FP32 Z can currently promote the attention output to FP32; this draft instead returns the activation dtype and tests that proposal. This compatibility decision must be agreed in #1735 before marking the PR ready.

## Checklist

- [x] I have read [CONTRIBUTING.md](../CONTRIBUTING.md) and follow its conventions (code style, docstrings, commit prefixes).
- [x] I have read [AGENTS.md](../AGENTS.md) and, where my change matches its scope, the relevant skill under [.agents/skills](../.agents/skills).
- [x] Dependent tests pass locally or in CI, and new behavior is covered by tests where applicable (tick as N/A for changes with no testable code, e.g. docs-only).
- [x] Kernel changes include same-hardware before/after benchmark numbers, dense + varlen where applicable (tick as N/A when no kernel code changed).
- [ ] This PR is minor/cosmetic-only (typo, formatting, style-only tweaks) — tick only if it is, and justify below.
- [ ] I understand this PR exceeds 500 changed lines and have explained its single purpose and why it should stay together below.
