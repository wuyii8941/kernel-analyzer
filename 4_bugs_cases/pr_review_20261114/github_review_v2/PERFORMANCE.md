# Bounded operator measurements

RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0. Three warm timing samples per workload in isolated before/after processes on the same device; setup is excluded. Medians are descriptive, not speedup estimates. Timings cover the Python operator call using CUDA events and can include dispatch gaps. Incremental PyTorch peak allocation is measured separately and excludes driver/compiler memory. No model training or end-to-end latency claim.

| Case / workload | Before ms | After ms | Incremental bytes before → after |
| --- | ---: | ---: | ---: |
| B007 / normalization_T1_packedFalse | 0.016908 | 0.023263 | 1,536 → 2,048 |
| B007 / normalization_T128_packedFalse | 0.025306 | 0.034516 | 98,304 → 197,632 |
| B007 / normalization_T128_packedTrue | 0.365197 | 0.420195 | 131,584 → 230,912 |
| B047 / mean_l1 | 0.141348 | 0.138362 | 803,840 → 803,840 |
| B049 / mean_kl_torch.float16 | 0.297636 | 0.264479 | 2,098,176 → 4,195,328 |
| B049 / mean_kl_torch.float32 | 0.265826 | 0.271038 | 4,195,328 → 4,195,328 |
| B054 / masked_softmax_contiguous | 0.274097 | 0.273862 | 524,800 → 524,800 |
| B054 / masked_softmax_strided_mask | 0.260647 | 0.251002 | 524,800 → 524,800 |

FLA measurements are normalization-helper forward calls, not full attention. Its normalized Z storage changes even where the table only shows transient peak allocation. Trident before/after stride-mask timing compares an incorrect original result with the correction and must not imply equivalent work. Liger trades extra low-precision-teacher temporary storage for correctness. FlagGems measures the same-mask core, not forward RNG integration.

The first B038 no-dropout benchmark overlapped a separate test on GPU 2. The first Trident before/after pair used different physical GPUs. Both are retained in runs.json and excluded from this final comparison; the Trident pair was rerun serially on GPU 2. Other repositories were tested concurrently on different GPUs; these measurements do not establish system-wide isolation.

## Current FlagGems dependency-base measurements

The older B038 pair is historical and is not used for the current submission. The current pair uses PR8969 head 69caccc0f and the varlen-only delta on the same physical RTX A6000, FP16, p=.5; dense code is unchanged.

| Workload | Before ms | After ms | Incremental bytes before → after |
| --- | ---: | ---: | ---: |
| dropout_backward_dense | 0.025979 | 0.026462 | 82944 → 82944 |
| dropout_backward_varlen | 0.026114 | 0.026238 | 82944 → 82944 |

[Original samples](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B038_8969_benchmark_before/benchmark.json), [candidate samples](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B038_8969_benchmark_after/benchmark.json). No speedup or broad neutrality claim.

## Additional mode profiles

These are separate, later workloads using the published measurement code, not repetitions to pool with the first profile. Backward-only retains a prepared autograd graph. Incremental allocation is above that resident graph, not total model memory.

### B007

| Workload | Before ms | After ms | Incremental bytes before → after |
| --- | ---: | ---: | ---: |
| normalization_T1_packedFalse_forward | 0.016839 | 0.023967 | 1536 → 2048 |
| normalization_T1_packedFalse_backward | 0.165682 | 0.257163 | 3072 → 3072 |
| normalization_T1_packedFalse_full | 0.639689 | 0.758008 | 4608 → 4608 |
| normalization_T128_packedFalse_forward | 0.024862 | 0.045462 | 98304 → 197632 |
| normalization_T128_packedFalse_backward | 0.185478 | 0.284697 | 132096 → 230912 |
| normalization_T128_packedFalse_full | 0.628214 | 0.876199 | 198144 → 330240 |
| normalization_T128_packedTrue_forward | 0.369001 | 0.440312 | 131584 → 230912 |
| normalization_T128_packedTrue_backward | 0.684970 | 0.662190 | 135168 → 235008 |
| normalization_T128_packedTrue_full | 1.915214 | 2.104027 | 201728 → 335360 |

[Before](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B007_mode_benchmark_before/benchmark.json), [after](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B007_mode_benchmark_after/benchmark.json).

### B049

| Workload | Before ms | After ms | Incremental bytes before → after |
| --- | ---: | ---: | ---: |
| liger_mean_kl_torch.float16_forward | 0.023491 | 0.023519 | 1536 → 1536 |
| liger_mean_kl_torch.float16_backward | 0.179927 | 0.193173 | 6291968 → 8389120 |
| liger_mean_kl_torch.float16_full | 0.283447 | 0.253339 | 6292480 → 8389632 |
| liger_mean_kl_torch.float32_forward | 0.026607 | 0.026595 | 1536 → 1536 |
| liger_mean_kl_torch.float32_backward | 0.188061 | 0.200780 | 8389120 → 8389120 |
| liger_mean_kl_torch.float32_full | 0.253603 | 0.321156 | 8389632 → 8389632 |
| torch_mean_kl_torch.float16_forward | 0.054108 | 0.054046 | 10485760 → 10485760 |
| torch_mean_kl_torch.float16_backward | 0.033992 | 0.034010 | 8389120 → 8389120 |
| torch_mean_kl_torch.float16_full | 0.084111 | 0.084295 | 10485760 → 10485760 |
| torch_mean_kl_torch.float32_forward | 0.058194 | 0.058113 | 12582912 → 12582912 |
| torch_mean_kl_torch.float32_backward | 0.033792 | 0.033847 | 8389120 → 8389120 |
| torch_mean_kl_torch.float32_full | 0.088251 | 0.088413 | 12582912 → 12582912 |

[Before](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B049_mode_benchmark_before/benchmark.json), [after](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B049_mode_benchmark_after/benchmark.json).

[Actual procedures and limits](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/measurement_code/README.md).
