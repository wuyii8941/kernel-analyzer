# Oracle correction: one optimizer, one measurement rule

The earlier 14-row Oracle comparison was not a fair accuracy test. Its three
known positive rows used stateless SGD, while its eleven control rows used
AdamW. Phi already showed that the same gradient differences can have
`A=4.665` before the optimizer but only about `A=1.03` after AdamW. The old
AUROC and recall numbers therefore cannot be used as Oracle performance
evidence.

This correction uses the same AdamW settings for every row:

- learning rate `1e-4`;
- betas `(0.9, 0.95)`;
- epsilon `1e-8`;
- no weight decay;
- zero initial moments;
- 16 steps for screening and 32 steps for confirmation.

It also keeps all twelve mechanically selected, result-blind sampled rows.
The earlier table silently omitted one sampled row after its full run showed a
small but significant local effect. With the three historical headline rows,
the corrected denominator is therefore 15, not 14.

## Same-optimizer case results

`A16` is the short-screen value. `A32` is the full value. A row is confirmed
only when its `A32` exceeds its own random-sign 95% bound.

| row | `A16` local update | `A32` local update | random-sign 95% bound | AdamW result |
|---|---:|---:|---:|---|
| Liger fused CE | 1.338 | 1.720 | 1.116 | persistent |
| Phi-4 `lm_head dX` | 1.013 | 1.029 | 1.004 | persistent, small margin |
| Qwen seq128 `lm_head dX` | 0.971 | 0.957 | 1.045 | canceling |
| sampled Phi seq256 row `0543` | 1.007 | 1.014 | 1.011 | persistent, small margin |

The important correction is not merely a smaller score. Qwen changes class:
it is a persistent gradient-difference example under stateless SGD, but its
direct AdamW update differences cancel under this protocol. An Oracle that
claims to predict parameter updates must therefore include the actual
optimizer.

## Corrected short-screen result

Across all 15 rows:

- 3 rows are positive under the full 32-step AdamW confirmation;
- the fixed `A16 > 1` escalation rule selects all 3 positives and 2 of 12
  negatives;
- recall is `3/3`, precision is `3/5`;
- `A16` AUROC is `0.944`;
- 16-step local update RMS AUROC is `0.528`.

These are retrospective results on declared parameter locations. They support
a cheap prioritization step, not a SAFE verdict or an accuracy claim for an
unseen implementation.

## Qwen is now one exact seq128 experiment

The old evidence combined a seq128 live run with a seq256 three-stage
companion. The new seq128 run measures the same exact endpoint, state order,
parameter, and AdamW setup from operator output through the 32-step run:

| Qwen seq128 stage | `A32` |
|---|---:|
| operator output difference | 1.005 |
| parameter-gradient difference after backward | 1.343 |
| AdamW parameter-update difference | 0.961 |

Backward makes the gradient differences more aligned, but AdamW removes most
of that alignment. In the live pair, the direct local update has `A=0.957`,
while feedback has `A=1.191` and the actual parameter separation has
`A=1.508`.

## Liger direct effect and feedback are now separated

The Liger run evaluates candidate and repair at both live states on every
step, so the measured update obeys:

```text
actual parameter separation added this step
= direct operator update difference
+ difference caused by the already-separated training states
```

At 32 steps:

- direct operator update: `A=1.720`;
- later-state feedback: `A=3.494`;
- actual parameter separation: `A=3.489`;
- maximum recurrence error: `8.5e-8`.

The operator has a real persistent direct effect under AdamW, but feedback is
much larger in this one-parameter training run. It would be incorrect to call
the entire final separation a direct operator effect.

## Phi equal-energy intervention

For Phi, deterministic BF16 rounding gives `A=3.325`. Four stochastic-rounding
repeats average `A=0.956`. After each stochastic update error is rescaled to
have exactly the same per-step L2 norm as the deterministic error, the mean is
still only `A=0.984`; the maximum relative norm mismatch is below `1.4e-7`.

The real stochastic-rounding endpoint is the causal intervention. The exact
norm match is an update-space analysis rather than a second executable kernel.
Together they show that the drop is about repeated direction, not simply less
error energy.

## Result boundary

The corrected result is narrower and stronger:

> A short sequence of update differences can prioritize persistent cases when
> every row uses the same optimizer. The optimizer is part of the question,
> not a bookkeeping detail.

It does not establish a universal all-operator Oracle. It also does not erase
the historical stateless-SGD observations; it states clearly that those
observations answer a different optimizer-specific question.

Machine-readable results:

- `results/property/joint_bias_formation_v1/oracle_repair_v3/same_optimizer_oracle_v3.json`
- `results/property/joint_bias_formation_v1/oracle_repair_v3/qwen_seq128_three_stage_adamw.json`
- `results/property/joint_bias_formation_v1/oracle_repair_v3/qwen_seq128_adamw_consequence32_with_stages.json`
- `results/property/joint_bias_formation_v1/oracle_repair_v3/liger_adamw_consequence32.json`
- `results/property/joint_bias_formation_v1/oracle_repair_v3/phi_seq64_adamw_consequence32.json`
- `results/property/joint_bias_formation_v1/oracle_repair_v3/phi_sr_update_norm_matched.json`
