### PR Category

Operator, OP Test

### Type of Change

Bug Fix

### Description

Keep undropped softmax probabilities in the Q/K softmax derivative. For `P = softmax(S)`, dropout factor `R`, and `U = dO Vᵀ`, the derivative is `dS = P * (R * U - sum(P * R * U))`. Applying R to the outer P introduces an extra dropout factor. Use a separate dropped probability only where dV needs it. Correct the dense and variable-length DQ/DKV loops; the shared attention-bias derivative receives the same correction.

The direct-core regression uses identical keys, making dQ exactly zero for a fixed input-independent mask, and Q=0, making dK zero. It also checks dV against the analytic same-mask result. FP16/BF16, dense/variable-length, and dropout 0/.5 give eight GPU cases: four fail on the original source and all eight pass after the change.

The mask is reconstructed using this backward core's RNG indexing. This isolates the derivative formula; it does not establish compatibility with a different forward's RNG convention.

### Validation

RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0. Changed files pass Black, isort, project-configured Flake8, and whitespace checks.

The existing attention test invocation reports 522 passes, 300 skips, and 1 failure(s). The BF16 sliding-window case at `(B=2,H=4,M=N=512,D=64)` also fails on the unmodified source with the same mismatch: one element, absolute difference `0.0001678466796875` versus allowed `0.0001`. This run is not a clean full-suite pass; its command and JUnit are retained. Tests/tolerances were not relaxed.

```bash
python -m pytest tests/test_flash_attention_dropout_derivative.py
```

Bounded same-mask dropout-backward timings are provided separately. No speedup or general performance-neutrality claim is made; non-NVIDIA backends and end-to-end forward/backward dropout integration remain outside this validation.

### Issue

Related: [#8748](https://github.com/flagos-ai/FlagGems/issues/8748) reports dropout-gradient failures in a Hygon forward/backward pairing. This PR isolates a same-mask derivative error and must not automatically close [#8748](https://github.com/flagos-ai/FlagGems/issues/8748).

### Progress

- [ ] Maintainer review
- [x] GPU regression tests for the corrected shared formula
- [ ] Broader hardware and forward/backward RNG integration validation

Pinned base: [flagos-ai/FlagGems@714859074155592820cac77d08a006f2c48eb4fc](https://github.com/flagos-ai/FlagGems/commit/714859074155592820cac77d08a006f2c48eb4fc) (`master`). Environment: RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.
