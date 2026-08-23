# Joint bias round: final status

> Historical round note. Its 14-row Oracle table mixes stateless-SGD headline
> rows with AdamW controls and is superseded by `docs/oracle_repair_v3.md` for
> all screening-performance claims.

## Completed work

| item | result | artifact |
|---|---|---|
| 12 sampled screen-negative 32-step consequences | 12/12 complete; 11 feedback-sustained, 1 mixed; no new persistent local-source family | `results/property/joint_bias_formation_v1/consequence_summary.json` |
| 16-step prefix versus 32-step consequence | local side-of-1 agrees for 11/12; actual and feedback agree for 12/12 | same summary |
| exact response parity replay | saved-P and SiLU both have mixed even/odd contributions; SiLU even energy is concentrated in steps 1--2 | `results/property/joint_bias_formation_v1/mu_parity_decomposition.json` |
| held-out confirmation | Gemma NEW_IMPL source-negative confirmed; full joint predictor unresolved | `results/property/joint_bias_formation_v1/heldout_confirmation.json` |
| RMS versus directionality | 32 rows: Pearson `0.018` (`p=0.921`), Spearman `0.243` (`p=0.178`); neither differs significantly from zero | `results/property/joint_bias_formation_v1/rms_persistence/rms_persistence.json` |
| Liger `0.9419` versus historical `2.315` | not the same bound measurement; `2.315` has no canonical state/contrast/estimator binding and is retired | `results/property/joint_bias_formation_v1/metric_binding.json` |
| four comparable Phi carrier arms | complete 32-step run; details below | `results/property/joint_bias_formation_v1/four_scale_arms/summary.json` |

The frozen Oracle evaluation set contains **14 rows: 3 declared positives and
11 controls**.  The three positives are Liger fused CE, Phi `lm_head dX`, and
Qwen `lm_head dX`; they were in the frozen set before comparison, rather than
added after seeing the scores.

## Four-arm result

All four arms start from the same checkpoint and are measured on one declared
evolving carrier at a time.  Only that carrier is updated, so this is not a
full-parameter training comparison.  The table below is the final-norm run;
the layer-26 carrier is reported separately.

| arm | final distance | coherence `A` | distance / initial parameter |
|---|---:|---:|---:|
| operator candidate vs repair | `9.186e-5` | `4.488` | `1.721e-6` |
| RNG seed | `0` | not informative | `0` |
| data order | `3.548e-5` | `0.0067` | `6.649e-7` |
| BF16 vs FP32 F+B | `3.223e-4` | `1.857` | `6.040e-6` |

The operator difference is smaller than the precision difference at step 32,
but is substantially more coherent.  Its distance follows an almost linear
prefix fit (exponent `1.034`, log-space R² `0.999`).  Precision is not a pure
square-root baseline (exponent `0.690`, R² `0.953`), so no long-horizon
crossover is claimed.  The data-order arm cancels when the complete batch
multiset has been consumed.  The RNG arm is inapplicable because Phi uses zero
dropout.

The repeated-random control injects an independently sign-scrambled residual
with exactly the same per-step support and RMS on every step.  Across five
frozen seeds its `A` ranges from `0.870` to `1.037` (mean `0.959`), while the
natural common-state operator effect has `A=4.701`.  This supplies the missing
empirical diffusion scale.

A second parameter carrier, layer-26 post-attention norm, does **not** repeat
the final-norm result: operator `A=1.114`, precision `A=0.914`, and data-order
`A=0.0158`.  Persistence is therefore carrier-selective within Phi; it is not
a model-wide property of every parameter touched by the endpoint.

The five-seed random null was evaluated on all 12 declared carriers.  Its
per-seed carrier means are `0.884--1.029`, while the natural final-norm carrier
reaches `A=4.488`.  The repeated-random unseen-loss arm is complete
(`A=1.028`, random-minus-repair loss gap `8.20e-8`).  The separate AdamW replay
gives gradient `A=4.665` but effective-update `A=1.031`, compared with SGD
update `A=4.701`; this is a response-map measurement, not a new formation
label.

The fresh wall-clock rerun has 11 of the 12 screen-negative rows with a real
32-step timed output. The Mamba scientific consequence certificate is already
complete; a separate timing-only rerun is being resumed in the background with
the slow sequential implementation because the optional fast CUDA path is not
installed. Therefore no complete 12-row GPU-time saving number is claimed yet.
This affects timing only, not the already complete 12/12 scientific results.
The measured subset is recorded in
`results/property/joint_bias_formation_v1/timed_efficiency_partial_v1.json`:
11 timed rows total 4.927 one-GPU-equivalent hours. The three headline
positives have scientifically reproduced timing runs, but the final combined
14-row cost table remains pending until Mamba finishes and the positive timings
are consolidated without background CPU contention.

## Scientific boundary

This round strengthens three bounded claims:

1. error magnitude does not explain formation directionality;
2. endpoint implementation differences can be much more temporally coherent
   than data-order variation on identical coordinates;
3. closed-loop feedback persistence is common even when the local operator
   increment is diffusive.
4. source persistence can be concentrated in one parameter carrier rather
   than inherited by every reachable parameter.

It does **not** establish a universal all-operator property, a universal joint
predictor, or a full-parameter Golden-style BF16/FP32 training comparison.

## Three-stage operator-to-update check

The current round also measures the same 32-step ordered reference trajectory
at three separate points: operator output error, parameter-gradient error, and
effective parameter-update error.  The final coherence values are:

| case | operator output | parameter gradient | stateless SGD parameter update |
|---|---:|---:|---:|
| Liger fused CE | 2.984 | 2.931 | 2.931 |
| Phi lm_head dX | 2.074 | 4.701 | 4.701 |
| Qwen lm_head dX | 1.008 | 1.698 | 1.698 |

This separates two claims that were previously easy to confuse.  Liger is
already directional at the local operator boundary.  Phi and Qwen are much
less directional at that boundary, but the difference becomes directional
after the backward pass reaches the parameter gradient.  These are carrier-
scale measurements, not full-model training claims.

The four-arm final weights were also evaluated on a common unseen FP32 loss
path.  The operator, data-order, and precision arms have absolute loss gaps of
`3.74e-6`, `5.59e-8`, and `1.26e-5`, respectively; the seed arm is zero because
the tested Phi configuration has no dropout.  This is a downstream consequence
check, not a new case label.
