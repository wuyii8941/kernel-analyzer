# 证据 P05：盲测 v1 / v2 计分与校准记录（通用能力轮，2026-10-08）

作为「参照可靠」「检测可靠」的既有证据入库索引（文件本身早已入库，冻结不改）。盲测记录的 SHA256SUMS 校验：通过。

| 文件 | 支撑 | 说明 | sha256（前 16 位） |
|---|---|---|---|
| `1_experiments/blind_tests/blind_records/v1/blind_test_v1_final_scoring.md` | reference + detection | blind test v1 final scoring | `8c70b11a5f9de85a` |
| `1_experiments/blind_tests/blind_records/v1/blind_test_v1_verification_report.json` | reference | independent recomputation of K_R (v1) | `2f1dba18bbd27f33` |
| `1_experiments/blind_tests/blind_records/v1/blind_test_v1_phase2_verification_report.json` | reference + detection | v1 phase 2 verification | `8aa5349ce117a809` |
| `1_experiments/blind_tests/blind_records/v1/blind_test_v1_answer_key_UNSEALED.json` | detection | v1 answer key (unsealed after scoring) | `d346a53adde9fc97` |
| `1_experiments/blind_tests/blind_records/v2/blind_test_v2_final_scoring.md` | reference + detection | blind test v2 final scoring | `90fa201a73324993` |
| `1_experiments/blind_tests/blind_records/v2/blind_test_v2_verification_report.json` | reference | independent recomputation of K_R (v2) | `2485b769b7cee742` |
| `1_experiments/blind_tests/blind_records/v2/blind_test_v2_answer_key_UNSEALED.json` | detection | v2 answer key | `6710f1b9477d771c` |
| `1_experiments/blind_tests/blind_records/v2/blind_test_v2_answer_key_update_layer_UNSEALED.json` | detection | v2 update-layer answer key | `d17529f34b9183a8` |
| `1_experiments/blind_tests/blind_records/SHA256SUMS` | integrity | checksums of the blind records (verified: sha256sum -c) | `7f0553e35aa07f2f` |
| `1_experiments/blind_tests/blind_records/errata.md` | integrity | errata of the blind records | `078a607c6f643ec4` |
| `1_experiments/blind_tests/recount.json` | reference + detection | recount from the frozen verdict matrix: 16,908,160 elements and 974 projections, 0 violations | `7c01ea8622bf7d36` |
| `pre-reorg-20261009:results/reference_eval/blind_test_v1/coverage_check.json` | reference | v1: every program has a complete reference | `2efa2ba551de382b` |
| `pre-reorg-20261009:results/reference_eval/blind_test_v2/coverage_check.json` | reference | v2: every program has a complete reference | `4deefc2f95aeb33a` |
| `pre-reorg-20261009:results/reference_eval/calibration_equivalence.json` | detection | t / bootstrap-t coverage, TOST (4000 per cell) | `347c8ba5a835727f` |
| `pre-reorg-20261009:results/reference_eval/detector_calibration_v2.json` | detection | default detector v2 false-positive calibration | `a76be70e4c180174` |
| `pre-reorg-20261009:results/reference_eval/decision_rule_calibration.json` | detection | direction-rule false-positive and family-wise rates | `29b8c940e0507c90` |
| `pre-reorg-20261009:docs/statistics_calibration_20261006.md` | detection | calibration report (source of S0, N0, n_min) | `e4f4f24271005435` |
| `pre-reorg-20261009:docs/tool_validation.md` | reference | bit-exact emulation and counterfactual checks | `21428d2df552cb0e` |
