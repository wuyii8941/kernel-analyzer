# Gemma RMS 同精度归约顺序干预

本实验在真实 Gemma-4 E2B 编译训练图中，针对两个已接入的 RMS-normalization
forward endpoint，固定相同输入和模型状态，仅改变 FP32 reference 计算均方和时的
feature 顺序：`FP32_NATIVE` 与 `FP32_REVERSE_FEATURE_ORDER`。候选仍是实际生成的
Triton 计算；结果只覆盖声明的 32 个固定输入状态，不是总体保证。

运行入口：

```bash
PYTHONPATH=src:. python scripts/run_gemma_rms_order_intervention.py --steps 32 \
  --output results/property/numerical_coverage_v1/gemma_rms_forward_order_intervention_v1/trajectory32_isolated.json
```

## 结果

最终采用 endpoint-by-endpoint 隔离运行：两个 endpoint 共用一个 carrier，不能在同一
次观察中同时替换后再把 gradient 变化归因给其中一个。

| task | native candidate−reference endpoint RMS | reverse candidate−reference endpoint RMS | 直接 reverse−native endpoint RMS | 直接 reverse−native gradient RMS | 预测 |
|---|---:|---:|---:|---:|---|
| `forward:116:out_ptr0` | `1.4020e-5` | `1.5022e-5` | `5.3948e-6` | `1.8114e-4` | `CHANGED`，但无确认方向性 bias |
| `forward:200:out_ptr0` | `9.5368e-6` | `9.5368e-6` | `0` | `0` | `NO_MATERIAL_CHANGE` |

两个 endpoint 各自完成 32/32 状态，未发生编译、执行或非有限值错误。`forward:116`
的 native/reverse reference 直接差异很小且固定方向投影为零；`forward:200` 的
差异在隔离后逐位为零。

一次早期的“同时替换两个 endpoint”运行曾得到 `forward:200` 的较大差异；那是
`forward:116` 的替换沿真实图向下游传播造成的区域效应，不能归因给 `forward:200`
自身。该对照正是 endpoint 隔离的必要性。

## 解释边界

这是一个同精度来源干预的阴性结果：在本次 Gemma 输入状态下，改变 FP32 feature
归约顺序没有产生可确认的系统性 RMS bias。它还没有修改真实候选 Triton kernel
本身，因此不能单独声称已经证明候选内部唯一采用了哪一种硬件归约顺序。也没有进行
长程 optimizer 或 loss 实验，`training_outcome` 保持 `NOT_MEASURED`。

原始结果：
`results/property/numerical_coverage_v1/gemma_rms_forward_order_intervention_v1/trajectory32_isolated.json`。
