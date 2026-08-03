# Kernel Analyzer

Concrete-invocation numerical analysis for a complete training forward and
backward. The method tests the causal chain used in the FlashAttention bias
work: directional local error, coherent gradient carrier, then weight
accumulation.

Current result: two confirmed natural precision-mediated cases: the seq128
Qwen3-1.7B `lm_head` input-gradient MM and Liger fused-linear cross-entropy
`dW` accumulation. Both now include 32-step paired baseline/repair live-weight
trajectories. BF16/FP16 eager all-op precision coverage and the primary
BF16 Inductor Triton screen at seq64/128/256 are complete in the frozen scope.
The Qwen3-VL round closes every BF16 and FP32 AOT forward/backward unit and
isolates an AOT SiLU-backward decomposition difference, but that difference
fails the cross-state directional-carrier gate. Two accepted cases are still
insufficient for property induction.

## Read

- [Method](docs/method.md)
- [Results](docs/results.md)
- [Three complete F+B cases](case.md)
- [Qwen3-VL all-op F+B result](round2.md)

## Verify

```bash
python3 scripts/check.py
```

Compact evidence is stored in `results/final/`. Raw state-by-state runs and
rebuildable compiler products are intentionally not versioned.
