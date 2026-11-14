# Bounded operator measurements

RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0. Three warm timing samples per workload in isolated before/after processes on the same device; setup is excluded. Medians are descriptive, not speedup estimates. Timings cover the Python operator call using CUDA events and can include dispatch gaps. Incremental PyTorch peak allocation is measured separately and excludes driver/compiler memory. No model training or end-to-end latency claim.

| Case / workload | Before ms | After ms | Incremental bytes before → after |
| --- | ---: | ---: | ---: |
| B007 / normalization_T1_packedFalse | 0.016908 | 0.023263 | 1,536 → 2,048 |
| B007 / normalization_T128_packedFalse | 0.025306 | 0.034516 | 98,304 → 197,632 |
| B007 / normalization_T128_packedTrue | 0.365197 | 0.420195 | 131,584 → 230,912 |
| B038 / dropout_backward_dense | 0.026170 | 0.026318 | 82,944 → 82,944 |
| B038 / dropout_backward_varlen | 0.026507 | 0.026205 | 82,944 → 82,944 |
| B047 / mean_l1 | 0.141348 | 0.138362 | 803,840 → 803,840 |
| B049 / mean_kl_torch.float16 | 0.297636 | 0.264479 | 2,098,176 → 4,195,328 |
| B049 / mean_kl_torch.float32 | 0.265826 | 0.271038 | 4,195,328 → 4,195,328 |
| B054 / masked_softmax_contiguous | 0.274097 | 0.273862 | 524,800 → 524,800 |
| B054 / masked_softmax_strided_mask | 0.260647 | 0.251002 | 524,800 → 524,800 |

FLA measurements are normalization-helper forward calls, not full attention. Its normalized Z storage changes even where the table only shows transient peak allocation. Trident before/after stride-mask timing compares an incorrect original result with the correction and must not imply equivalent work. Liger trades extra low-precision-teacher temporary storage for correctness. FlagGems measures the same-mask core, not forward RNG integration.

The first B038 no-dropout benchmark overlapped a separate test on GPU 2. The first Trident before/after pair used different physical GPUs. Both are retained in runs.json and excluded from this final comparison; the Trident pair was rerun serially on GPU 2. Other repositories were tested concurrently on different GPUs; these measurements do not establish system-wide isolation.
