# Joint bias round: final status

## Completed work

| item | result | artifact |
|---|---|---|
| 12 sampled screen-negative 32-step consequences | 12/12 complete; 11 feedback-sustained, 1 mixed; no new persistent local-source family | `results/property/joint_bias_formation_v1/consequence_summary.json` |
| 16-step prefix versus 32-step consequence | local side-of-1 agrees for 11/12; actual and feedback agree for 12/12 | same summary |
| exact response parity replay | saved-P and SiLU both have mixed even/odd contributions; SiLU even energy is concentrated in steps 1--2 | `results/property/joint_bias_formation_v1/mu_parity_decomposition.json` |
| held-out confirmation | Gemma NEW_IMPL source-negative confirmed; full joint predictor unresolved | `results/property/joint_bias_formation_v1/heldout_confirmation.json` |
| RMS versus directionality | 32 rows: Pearson `0.018`, Spearman `0.243`; local RMS is not a useful linear predictor of open-loop directionality | `results/property/joint_bias_formation_v1/rms_persistence/rms_persistence.json` |
| Liger `0.9419` versus historical `2.315` | not the same bound measurement; `2.315` has no canonical state/contrast/estimator binding and is retired | `results/property/joint_bias_formation_v1/metric_binding.json` |
| four comparable Phi carrier arms | complete 32-step run; details below | `results/property/joint_bias_formation_v1/four_scale_arms/summary.json` |

## Four-arm result

All four arms start from the same checkpoint and are measured on the same
evolving final-norm carrier.  Only that carrier is updated, so this is not a
full-parameter training comparison.

| arm | final distance | coherence `A` | distance / initial parameter |
|---|---:|---:|---:|
| operator candidate vs repair | `9.186e-5` | `4.488` | `1.721e-6` |
| RNG seed | `0` | not informative | `0` |
| data order | `3.548e-5` | `0.0067` | `6.649e-7` |
| BF16 vs FP32 F+B | `3.223e-4` | `1.857` | `6.040e-6` |

The operator difference is smaller than the precision difference at step 32,
but is substantially more coherent.  Its distance follows an almost linear
prefix fit (exponent `1.034`, log-space R² `0.999`).  Precision is not a pure
square-root baseline (exponent `0.690`, R² `0.953`), so the extrapolated
crossing near step `703` is exploratory and must not be stated as a universal
prediction.  The data-order arm cancels when the complete batch multiset has
been consumed.  The RNG arm is inapplicable because Phi uses zero dropout.

## Scientific boundary

This round strengthens three bounded claims:

1. error magnitude does not explain formation directionality;
2. endpoint implementation differences can be much more temporally coherent
   than data-order variation on identical coordinates;
3. closed-loop feedback persistence is common even when the local operator
   increment is diffusive.

It does **not** establish a universal all-operator property, a universal joint
predictor, or a full-parameter Golden-style BF16/FP32 training comparison.
