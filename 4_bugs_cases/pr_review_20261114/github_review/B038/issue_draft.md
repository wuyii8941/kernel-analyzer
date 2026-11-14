# Attention backward drops the outer probability factor in the softmax derivative

For nonzero dropout, the dense/variable-length shared backward masks and rescales both `dp` and `p` before `ds = p * (dp - Di)`. The outer p must be the undropped softmax probability.

For `P=softmax(S)`, `R=mask/(1-dropout_p)`, `U=dO@V.T`, the chain rule gives:

```text
dS = P * (R * U - sum(P * R * U))
dV = (P * R).T @ dO
```

The existing dS instead uses `P * R` as its outer factor. Identical keys provide a direct invariant: for a fixed input-independent mask, Q only shifts all scores in a row equally, so dQ must be zero. Q=0 also gives dK=0.

Attach `test_flash_attention_dropout_derivative.py` from this review and run:

```bash
python -m pytest tests/test_flash_attention_dropout_derivative.py
```

The eight GPU cases cover FP16/BF16, dense/variable-length calls and p=0/.5. Original source: four failures at p=.5 and four p=0 controls passing; patched source: all eight pass. The test reconstructs this backward core's mask and also checks dV. It isolates the formula, not a different forward's RNG compatibility.

Related [#8748](https://github.com/flagos-ai/FlagGems/issues/8748) describes a Hygon forward/backward pairing. This report does not claim to resolve that whole integration problem or establish first discovery. If a new issue would duplicate that discussion, submit the focused PR with a related-issue link instead.

Pinned base: [flagos-ai/FlagGems@714859074155592820cac77d08a006f2c48eb4fc](https://github.com/flagos-ai/FlagGems/commit/714859074155592820cac77d08a006f2c48eb4fc) (`master`). Environment: RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.
