# Cleanup Manifest — 2026-07-15

## Purpose

The project focus moved from the previous fork-centered experiments to the discrepancy decomposition Oracle. The user explicitly approved deletion of heavyweight, reproducible model/checkpoint directories while retaining reports, summaries, configurations, source code, and compact result files.

## Previously removed disposable caches

- `.pytest_cache/`
- `cache/pip/`
- `cache/Miniforge3-Linux-x86_64.sh`

Approximate reclaimed size: 4.7 GB.

## Heavy result directories approved for deletion

- `results/phase10_zero_fork_twins/` (~34 GB)
- `results/phase11_heldout_twins/` (~14 GB)
- `results/trajectory_step5_fusion/` (~34 GB)
- `results/matched_step/` (~20 GB)
- `results/matched_step_fusion/` (~20 GB)
- `results/matched_step_fusion_r2/` (~6.7 GB)
- `results/matched_step_fusion_r3/` (~6.8 GB)
- `results/matched_step_fusion_r4/` (~6.8 GB)

Reports and compact JSON/JSONL summaries corresponding to these experiments are retained.

## Old training checkpoint directories approved for deletion

- `data/logsoftmax_online_smoke_policy/checkpoint-3/`
- `data/online_smoke_policy/checkpoint-3/`
- `data/phase0_policy_final/checkpoint-240/`
- `data/phase0_policy_final/checkpoint-270/`
- `data/phase0_policy_final/checkpoint-300/`
- `data/phase4_online_full_policy/checkpoint-240/`
- `data/phase4_online_full_policy/checkpoint-270/`
- `data/phase4_online_full_policy/checkpoint-300/`
- `data/phaseA3_recovery_run/checkpoint-6/`
- `data/phaseA3_recovery_run/checkpoint-9/`
- `data/phaseA3_recovery_run/checkpoint-12/`
- `data/phaseA3_recovery_run/checkpoint-15/`
- `data/phaseA3_recovery_run_r3/checkpoint-3/`
- `data/phaseA3_recovery_run_r3/checkpoint-6/`
- `data/phaseA3_recovery_run_r3/checkpoint-9/`
- `data/phaseA3_recovery_run_r3/checkpoint-12/`
- `data/phaseA3_recovery_run_r4/checkpoint-273/`

Approximate checkpoint size before deletion: 111 GB.

## Explicitly retained

- `theory_oracle/`
- `reports/`
- `configs/`
- `src/`, `scripts/`, and `tests/`
- compact result summaries and JSON/JSONL evidence
- non-checkpoint state snapshots and rollout data
- environment manifests
- remaining caches that were not part of the approved heavy-result list

## Recoverability note

`forkcert` is not an independently recoverable Git repository in the current environment. Deleted model/checkpoint payloads can only be recreated by rerunning their originating experiments or restoring an external backup.

## Completion verification

Cleanup completed successfully on 2026-07-15.

Post-cleanup sizes:

- `cache/`: 2.0 GB
- `data/`: 36 GB (previously 147 GB)
- `results/`: 7.3 GB (previously 148 GB)
- `reports/`: 656 KB, retained
- `theory_oracle/`: retained

No `checkpoint-*` directories remain under `data/`, and none of the approved heavyweight result directories remain. Approximate total space reclaimed across cache and heavyweight-result cleanup: 256–257 GB.
