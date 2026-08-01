# Kernel Analyzer

Concrete-invocation numerical analysis for a complete training forward and
backward. The method tests the causal chain used in the FlashAttention bias
work: directional local error, coherent gradient carrier, then weight
accumulation.

Current result: one confirmed natural case, the seq128 Qwen3-1.7B `lm_head`
input-gradient MM. BF16/FP16 eager all-op precision coverage is complete in the
frozen scope. The next phase studies same-precision implementation factors.

## Read

- [Method](docs/method.md)
- [Results](docs/results.md)
- [Confirmed MM case](docs/mm.md)

## Verify

```bash
python3 scripts/check.py
```

Compact evidence is stored in `results/final/`. Raw state-by-state runs and
rebuildable compiler products are intentionally not versioned.
