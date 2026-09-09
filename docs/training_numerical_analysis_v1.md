# Training Numerical Analysis v1

历史结果提示：本页使用额外 BF16 舍入的写入模拟，未通过与目标 AdamW 写入一致性的
验收。数值保留作历史记录，不作为当前真实参数写入证明。修正见
[readback-v2](training_numerical_analysis_v2.md)，新旧参数表示不可直接混比。

本页由 `scripts/build_training_numerical_analysis_report.py` 从机器记录生成。
它不替代原始结果，也不将重新分析写成未见确认。

| 案例 | 用途 | 测量状态 | 主要位置 | 写入差异 RMS | 相对缩放区间 | 等价性判断 |
|---|---|---|---|---:|---:|---|
| phi4_seq64_lmhead_dx | CONVENTIONAL_IMPLEMENTATION_FAMILY | VALID | FIXED_SUITE_PARAMETER_WRITE | 0.000% | — | EQUIVALENT |
| gemma4_text128_scan_0037 | FIRST_TRITON_FAMILY | ABSTAIN_RUNTIME_PACKAGE_INCOMPLETE_FOR_SHARED_RUNNER | NOT_CAPTURED | — | — | NOT_ASSESSED |
| llama32_text128_scan_0000 | SECOND_TRITON_FAMILY | ABSTAIN_RUNTIME_PACKAGE_INCOMPLETE_FOR_SHARED_RUNNER | NOT_CAPTURED | — | — | NOT_ASSESSED |
| liger_fused_ce_t128 | MECHANISM_AND_CONSEQUENCE_SUPPORT | VALID | FIXED_SUITE_PARAMETER_WRITE | 4.936% | [-0.236%, -0.125%] | NON_EQUIVALENT |
| deepseek8b_seq256_backward_1714_in_out_ptr0 | STATE_DEPENDENCE_PIPELINE_VALIDATION | VALID | FIXED_SUITE_PARAMETER_WRITE | 53.039% | [-16.760%, -12.152%] | NON_EQUIVALENT |
| deepseek8b_seq128_backward_1256_out_ptr0 | STATE_DEPENDENCE_PIPELINE_VALIDATION | VALID | FIXED_SUITE_PARAMETER_WRITE | 46.016% | [-12.079%, -9.338%] | NON_EQUIVALENT |

## 配对训练验证

新冻结的 Liger 数据流完成 4096 步；被测实现减参考实现的验证 loss 为 -0.00946。
两条历史数据流的对应差为 +0.02778、+0.02821；三条差的描述性区间为 [-0.03820, +0.06923]，跨过零。
这支持配对轨迹不同，不支持质量变化具有可重复方向。

完整数值与限制见同目录 `summary.json` 及各个重新计算结果。
