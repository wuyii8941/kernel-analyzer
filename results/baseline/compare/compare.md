# 直接差分基线与工具的逐输出对照

## opinfo_loose（210 个用例，300 个输出）

| 工具分档 | 输出数 | CI 式差分失败 | 高精度差分 > 1e-5 |
|---|---:|---:|---:|
| none | 192 | 1 | 2 |
| not Triton | 40 | 0 | 0 |
| modified after Triton write | 31 | 0 | 0 |
| mixed (external re-entry) | 18 | 0 | 0 |
| unresolved | 12 | 4 | 12 |
| programs aborted | 4 | 0 | 0 |
| candidate | 2 | 2 | 2 |
| constant-level | 1 | 0 | 0 |

工具报警（候选、错误、特殊值类别不一致）14 个输出，其中 CI 式差分也失败 6 个；CI 式差分失败而工具未报警 1 个。

## opinfo_onesample（1439 个用例，1800 个输出）

| 工具分档 | 输出数 | CI 式差分失败 | 高精度差分 > 1e-5 |
|---|---:|---:|---:|
| none | 1129 | 0 | 6 |
| not Triton | 304 | 0 | 0 |
| constant-level | 226 | 0 | 1 |
| small | 46 | 0 | 0 |
| programs aborted | 36 | 2 | 3 |
| modified after Triton write | 27 | 0 | 0 |
| unresolved | 14 | 4 | 4 |
| candidate | 8 | 6 | 8 |
| error | 6 | 6 | 6 |
| mixed (external re-entry) | 4 | 0 | 0 |

工具报警（候选、错误、特殊值类别不一致）18 个输出，其中 CI 式差分也失败 16 个；CI 式差分失败而工具未报警 2 个。

## opinfo_screen（745 个用例，849 个输出）

| 工具分档 | 输出数 | CI 式差分失败 | 高精度差分 > 1e-5 |
|---|---:|---:|---:|
| none | 535 | 0 | 1 |
| not Triton | 210 | 0 | 1 |
| constant-level | 48 | 0 | 0 |
| programs aborted | 31 | 0 | 0 |
| modified after Triton write | 10 | 0 | 0 |
| compensated (rounding-dependent decision) | 6 | 0 | 0 |
| unresolved | 6 | 0 | 0 |
| small | 2 | 0 | 0 |
| mixed (external re-entry) | 1 | 0 | 0 |

工具报警（候选、错误、特殊值类别不一致）0 个输出，其中 CI 式差分也失败 0 个；CI 式差分失败而工具未报警 0 个。

## all（2149 个用例，2949 个输出）

| 工具分档 | 输出数 | CI 式差分失败 | 高精度差分 > 1e-5 |
|---|---:|---:|---:|
| none | 1856 | 1 | 9 |
| not Triton | 554 | 0 | 1 |
| constant-level | 275 | 0 | 1 |
| programs aborted | 71 | 2 | 3 |
| modified after Triton write | 68 | 0 | 0 |
| small | 48 | 0 | 0 |
| unresolved | 32 | 8 | 16 |
| mixed (external re-entry) | 23 | 0 | 0 |
| candidate | 10 | 8 | 10 |
| error | 6 | 6 | 6 |
| compensated (rounding-dependent decision) | 6 | 0 | 0 |

工具报警（候选、错误、特殊值类别不一致）32 个输出，其中 CI 式差分也失败 22 个；CI 式差分失败而工具未报警 3 个。

## CI 式差分失败而工具未报警的输出

| 集合 | 用例 | 输出 | 工具分档 | 工具 e_num 相对 RMS | 高精度相对 RMS | eager fp32 相对 RMS |
|---|---|---|---|---:|---:|---:|
| opinfo_onesample | oib_masked_prod_22 | d0 | programs aborted | None | 4.3867418253079197e-08 | 4.3867418253079197e-08 |
| opinfo_onesample | oib_masked_prod_27 | d0 | programs aborted | None | 0.0 | 0.0 |
| opinfo_loose | oib_logcumsumexp_0 | d0 | none | 0.0006861824179989464 | 0.0006777212378168249 | 0.0007237256615090643 |

## 工具报警而 CI 式差分通过的输出

| 集合 | 用例 | 输出 | 工具分档 | e_sem 相对 RMS | 特殊值 | 高精度相对 RMS | 梯度检查被关 |
|---|---|---|---|---:|---|---:|---|
| opinfo_onesample | oib_nn_functional_normalize_5 | d0 | candidate | 0.999999990918762 | (0, 0) | inf | False |
| opinfo_onesample | oib_nn_functional_rms_norm_2 | d0 | candidate | 1.000415288343336 | (0, 0) | 1.0 | False |
| opinfo_loose | oib_index_reduce_amax_0 | d0 | unresolved | None | (0, 9) | 0.0 | True |
| opinfo_loose | oib_index_reduce_amax_0 | d1 | unresolved | None | (0, 9) | 0.0 | True |
| opinfo_loose | oib_index_reduce_amax_1 | d1 | unresolved | None | (0, 9) | 0.0 | True |
| opinfo_loose | oib_index_reduce_amax_2 | d0 | unresolved | None | (0, 9) | 0.0 | True |
| opinfo_loose | oib_index_reduce_amax_2 | d1 | unresolved | None | (0, 9) | 0.0 | True |
| opinfo_loose | oib_index_reduce_amax_3 | d1 | unresolved | None | (0, 9) | 0.0 | True |
| opinfo_loose | oib_index_reduce_amin_1 | d1 | unresolved | None | (0, 9) | 0.0 | True |
| opinfo_loose | oib_index_reduce_amin_3 | d1 | unresolved | None | (0, 9) | 0.0 | True |

耗时（每用例，不含进程启动）：工具中位数 11.5 s，直接差分中位数 0.2 s。
