# Phase 13 Historical Bug Replays

## Objective

Test ForkCert-adjacent decision signals on independently documented upstream wrong-result bugs.

## Results

| Case | Independent runs | Status | Eager self exact | Inductor self exact | Max delta | Argmax fork | top-16 set fork |
|---|---:|---|---|---|---:|---|---|
| [pytorch/pytorch#183986](https://github.com/pytorch/pytorch/issues/183986) | 2 | fail_closed | True | None | n/a | None | None |
| [pytorch/pytorch#186577](https://github.com/pytorch/pytorch/issues/186577) | 2 | wrong_result_reproduced | True | True | 20.7031 | False | False |

## Interpretation Boundary

Independent replay of already reported upstream bugs; no new-bug discovery claim.

These operator-level output decisions are replay gates. They are not PPO clipping, sampling, reward, or task-level consequences.

## Artifacts

- `results/phase13_historical_bug_replays.json`
- `results/phase13_historical_bug_replays/`
- `scripts/phase13_historical_bug_replay_once.py`
