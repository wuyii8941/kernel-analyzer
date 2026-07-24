# Step-5 Natural Clipping Fork: Attribution and Matched-Step Counterfactual

## Objective

This batch addresses RQ2 and RQ4 for `clip-step5-grpo_000001_2817771126c0-t80`: localize the numerical divergence without perturbing the compiled path, and test whether repairing only the observed clipping decision moves the parameter update toward the reference update.

## Intervention / Comparison

- A: HF eager, FP16 autocast, SDPA MATH, standard GRPO clipping.
- B: identical state and inputs under `torch.compile`, standard GRPO clipping.
- C: the same compiled numerical path and logprob as B, but the single audited fork token is forced onto A's unclipped branch.
- Attribution probes: component graph barriers and reference-tensor forward-hook splices.

The A/B equivalence contract is backend substitutability for the same Qwen3-0.6B weights, fixed token IDs, FP16 autocast and MATH-locked SDPA. C is a semantic counterfactual, not an equivalent implementation path.

## Controls

- Frozen checkpoint: `data/phase6_policy_step5_pre`.
- Batch: four responses and 512 aligned response tokens from rollout batch 1.
- Same old logprobs, advantages, masks, position semantics, seed and fresh SGD state.
- Reference replay error at the fork token: `2.384185791015625e-7`.
- Compile logprob remained `-1.0754941701889038` in B and C.
- Every splice was paired with an identity-hook negative control.
- Component switches retained the existing `1e-3` positive canary, but are interpreted only as compile-region barriers.

## Result

The natural fork reproduces: A gives logp `-1.0795209408` and remains unclipped; B gives `-1.0754941702` and is clipped.

| Arm | Target logp | Target dLoss/dlogp | Full gradient norm | Loss |
|---|---:|---:|---:|---:|
| A reference | -1.0795209408 | -0.0033917371 | 7.8296764046 | -0.0406679884 |
| B alternative | -1.0754941702 | 0 | 7.8147339861 | -0.0404946432 |
| C branch repair | -1.0754941702 | -0.0034054227 | 7.8240808673 | -0.0405010059 |

After one identical SGD step (`lr=1e-5`):

| Distance | L2 | Relative L2 |
|---|---:|---:|
| A-B | 1.1049261192e-5 | 7.6686670571e-9 |
| A-C | 5.0171820252e-6 | 3.4821421856e-9 |
| B-C | 1.0648649720e-5 | 7.3906253002e-9 |

`distance(A,C) / distance(A,B) = 0.4541`: repairing only the target decision removes about 54.6% of the observed one-step A/B parameter distance. The residual 45.4% is consistent with continuous backend drift and any other token-level semantic differences; it is not attributed to this fork.

## Attribution Audit

Disabling attention, MLP, RMSNorm or lm_head before whole-model compilation makes the target output equal eager. These interventions insert graph barriers and therefore establish only that breaking the original compile region removes the fork. They do not identify four independent operator causes.

All decoder-layer and layer-0 submodule forward-hook splices failed the identity-hook negative control: an identity hook alone changed the output from compiled to eager. Consequently, every hook-based splice is invalid for causal attribution. No earliest successful splice point or minimal operator repair set is claimed.

## Certificate / Artifacts

- Baseline: `results/baseline_manifest.json`
- Attribution audit: `results/attribution/clip-step5-grpo_000001_2817771126c0-t80.json`
- Matched-step result: `results/matched_step/clip-step5-grpo_000001_2817771126c0-t80.json`
- Updated A/B/C weights: `results/matched_step/clip-step5-grpo_000001_2817771126c0-t80/`
- Attribution command: `CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src python scripts/phase8_case_attribution.py`
- Matched-step command: `CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src:. python scripts/phase8_matched_step.py`

## Interpretation

This case supports a direct causal contribution of the discrete clipping fork to the one-step gradient and parameter update. It does not yet support an operator-level root cause. The observation mechanism is itself compile-path-sensitive, so RQ2 remains unresolved below the whole compile-region level.

External validity remains limited to T4 FP16. A positive FP16 fork is evidence that this path pair is not decision-stable under the measured condition; no BF16 or newer-GPU magnitude claim is made here.

## Next Decision

**REVISE.** Keep the RQ4 branch-level causal result. Reject hook-based operator attribution and freeze step-11 state to make the two additional natural forks replayable. Future compile attribution must use a non-hook observation mechanism or generated-kernel/FX-level localization.
