# 20-Step Trajectory Analysis: Fusion Repair Persistence

## Key Result

The fusion repair works causally in the short term but does not persist to 20 steps.

| Step | A-B L2 | A-C L2 | Recovery ratio | Interpretation |
|---:|---:|---:|---:|---|
| 1 | 1.105e-5 | 4.981e-6 | 0.451 | C 54.9% closer to A |
| 5 | 3.601e-5 | 1.405e-5 | 0.390 | C 61.0% closer to A |
| 20 | 5.217e-5 | 5.509e-5 | 1.056 | C 5.6% farther from A |

## Clipping Pattern Alignment

| Window | A==B | A==C |
|---|---|---|
| Steps 1-5 | 3/5 (60%) | **5/5 (100%)** |
| Steps 6-10 | 3/5 (60%) | 1/5 (20%) |
| Steps 11-20 | 7/10 (70%) | 7/10 (70%) |

The repair perfectly aligns C's clipping behavior with A for the first 5 steps. By step 6, the underlying compile numerical drift at *other* tokens causes new clipping divergences, and C loses its advantage over B.

## Growth Rate Analysis

- A-C grows 3.9x from step 5 to 20 (vs A-B growing only 1.4x). The repair slows early divergence but C accumulates its own compile-path drift, eventually matching B's total displacement.
- All three arms converge to similar aggregate training statistics by step 20 (loss ≈ -0.048, grad_norm ≈ 6.4-6.6).

## Interpretation for the Research

1. **The idea works for single-step causal attribution.** Fusion partition scheduling is causally responsible for the observed fork, and repairing it recovers 55-61% of the parameter distance for the first 5 steps. This is the RQ2+RQ4 core result.

2. **Single-point repair is necessary but not sufficient for long-term trajectory alignment.** The compile path produces continuous numerical drift across all 512 tokens in the batch. Fixing one token's fork prevents that specific clipping decision error but cannot prevent the accumulation of other per-token numerical differences over subsequent steps.

3. **This is not a negative result — it is a characterization result.** It tells us that fork is a per-step phenomenon, not a one-time event. The system is continuously near decision boundaries, and the compile path continuously injects fresh perturbations. A lasting fix would require either (a) fixing the fusion schedule globally, or (b) accepting that compile introduces a certain fork rate per step.

4. **The batch scan data confirms the mechanism:** `max_fusion_size=2` reduces batch-wide branch forks from 1 to 0 at step 5, while other fusion sizes (1, 4, None) all have 1 fork. The effect is specific to a particular fusion partition class.

## Artifacts

- Merged trajectory: `results/trajectory_step5_fusion/merged.json`
- Per-arm trajectories: `results/trajectory_step5_fusion/{A_reference,B_alternative,C_fusion_repair}.json`
- Batch scans: `results/attribution/batchscan_step5_{base,size1,size2,size4}.json`
