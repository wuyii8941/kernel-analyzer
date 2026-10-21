# FlagGems 5.0.2 slice_backward 忽略 grad_output 的 stride（路径 A2，2026-10-22）

## 现象

路径 A2 记录到三个 `aten.slice_backward.default` 调用（Qwen2.5-0.5B 反向，rotate-half 的切片梯度），grad_output 都是非连续视图：

| 工作负载 key | grad_output | stride | 与原生 PyTorch 一致 | 最大差 |
|---|---|---|---|---|
| qwen bf16 124 | [1,2,128,32] | [16384,64,128,1] | 否 | 7.7e-3 |
| qwen bf16 129 | [1,14,128,32] | [114688,64,896,1] | 否 | 3.0e-3 |
| qwen fp32 150 | [1,14,128,32] | [114688,8192,64,1] | 否 | 3.1e-3 |
| 连续输入对照 | [1,2,128,32] | 连续 | 是 | 0 |

## 原因

`flag_gems/ops/slice_backward.py` 中 `slice_backward_kernel` 按 `grad_output_ptr + offsets` 线性读取 grad_output。
包装函数没有先把 grad_output 变成连续的，所以非连续视图读到的是存储中其他元素的值。

## 在路径 A2 中的表现

- 测量工具判定输出依赖计划未声明的存储（视图之外的元素），拒绝绑定。
- 偏差 D-A2-8 对精确数据搬运加入黑盒对照，使这类调用可以按合同诊断。
- 当前合同 LAYOUT `real` 对含负零的数据报 invalid_input，而这些梯度中含 −0.0；负零的处理另行起草并复核。
- 上述与原生的对照只是辅助证据，判定依据仍是合同。

## 状态

候选上游 issue，只起草，由用户决定是否提交。
