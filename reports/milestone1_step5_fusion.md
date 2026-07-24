# Milestone 1: Natural Fork to Fusion Repair to Parameter Recovery

## Objective

Close the first end-to-end RQ2/RQ4 chain for `clip-step5-grpo_000001_2817771126c0-t80`.

## Contract

HF eager and `torch.compile` execute the same Qwen3-0.6B step-5 checkpoint, fixed four-response batch, token IDs, masks, position semantics, FP16 autocast, FP32 master parameters and MATH-locked SDPA. The alternative is expected to preserve the GRPO clipping decision semantics.

## Intervention / Comparison

- A: eager reference.
- B: default full-graph compile.
- C: compile with the single Inductor setting `max_fusion_size=2`.

No branch is forced in C. The repair changes the compile fusion partition and lets the ordinary clipping formula select the branch.

## Controls

- Frozen pre-minibatch checkpoint, old logprobs and advantages.
- 512/512 token alignment.
- Independent Dynamo reset and Inductor cache for B and C.
- Generated training kernels: B `2055`, C `2645`; the intervention canary is non-zero.
- Full-batch inference audit: B has 1 branch fork versus eager; C has 0.
- Earlier hook-based splices and `output_hidden_states` probes were rejected because identity observation changed the compiled path.

## Result

| Arm | Target logp | Target dLoss/dlogp | Full gradient norm |
|---|---:|---:|---:|
| A eager | -1.0795209408 | -0.0033917371 | 7.8296764 |
| B default compile | -1.0754941702 | 0 | 7.8147340 |
| C fusion repair | -1.0797338486 | -0.0033910151 | 7.8232707 |

After one identical SGD step (`lr=1e-5`):

| Distance | L2 |
|---|---:|
| A-B | 1.1049261e-5 |
| A-C | 4.9810227e-6 |
| B-C | 1.0920549e-5 |

`distance(A,C) / distance(A,B) = 0.4508`. The fusion repair removes all batch branch forks and moves the parameter update materially toward A.

The fusion-size ladder is non-monotonic. Size 1 repairs the target but introduces another batch branch fork; size 2 repairs all branch forks; size 4 reproduces baseline. The responsible unit is therefore a fusion partition/schedule class, not a claim that less fusion is universally more stable.

## Artifacts

- Frozen baseline: `results/baseline_manifest.json`
- Fusion probe: `results/attribution/step5_compile_fusion_probe.json`
- Independent probe repeat: `results/attribution/step5_compile_fusion_repro.json`
- Full-batch scans: `results/attribution/batchscan_step5_base.json`, `batchscan_step5_size1.json`, `batchscan_step5_size2.json`, `batchscan_step5_size4.json`
- Valid A/B/C result: `results/matched_step_fusion_r4/clip-step5-grpo_000001_2817771126c0-t80.json`
- Invalid cache-reuse attempts are retained under `results/matched_step_fusion/`, `_r2/` and `_r3/` and must not be cited as valid interventions.

## Interpretation

Supported chain:

```text
default compile fusion partition
-> target logprob crosses clipping boundary
-> target policy gradient becomes zero
-> fusion partition repair removes all batch branch forks
-> target gradient returns
-> one-step parameter update moves toward eager
```

This identifies a minimal compile-configuration repair class, not a unique source-level operator or an implementation bug. Regions remain `unknown` because no usable analytic legal bound exists.

## Decision

**GO.** Extend the valid A/B/C protocol to 5 and 20 steps and to the two step-11 forks. Keep long-horizon and gradient-norm-matched non-fork evidence separate.
