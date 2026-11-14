### PR Category

Operator, OP Test

### Type of Change

Bug Fix

### Description

Use the undropped probability as the outer factor in variable-length attention's softmax derivative. Dropout scales/masks dP; dS must still use `P * (dP_drop - D)`. Keep a separate dropped probability for dV.

Only `_flash_attn_bwd_varlen_dq_fused` and `_flash_attn_bwd_varlen_dkv` change. Dense corrections are already proposed in #8969 and are not included here.

This draft targets current master `714859074155592820cac77d08a006f2c48eb4fc`. The added tests exercise only the changed varlen path: **4 fail / 4 pass before, all 8 pass after**. They cover FP16/BF16, dropout 0/0.5, identical-key invariants and nonzero Q/K/V with an independent FP64 autograd reference. dV has a separate allowance for its existing probability-format conversion; this is not a bound for the entire kernel.

```bash
python -m pytest tests/test_flash_attention_varlen_dropout_derivative.py
```

Reverting only varlen DKV fails the two nonzero-Q/K/V dropout cases specifically at dK; the old Q=0 controls still pass. dBias and cross-implementation forward/backward RNG compatibility are outside this test's coverage.

NVIDIA RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0. Black/isort/Flake8 with repository options, whitespace and operator-marker checks pass.

### Issue

Related: #8969 (dense formula correction) and #8748 (separate forward/backward RNG-routing report). This PR does not close #8748. It can be reviewed as a varlen-only follow-up while the dense PR is in progress.

### Progress

- [ ] Change is properly reviewed (1 reviewer required, 2 recommended).
- [ ] Change is responded to an issue.
- [x] Change is fully covered by a UT.

### Performance

Same-mask varlen backward, Q=(128,2,32), K/V=(256,2,32), FP16, dropout=0.5; same RTX A6000, three warm samples:

| | Before | After |
| --- | ---: | ---: |
| Median ms | 0.026110 | 0.026330 |
| Incremental temporary bytes | 82,944 | 82,944 |

Mask construction is excluded. These limited samples do not establish a speedup or a statistically significant regression.
