# PyTorch Inductor 常用激活函数前向与完整输入 VJP 绑定

本轮绑定三个 PyTorch 2.10.0 公开调用：`torch.relu`、`torch.tanh` 和 `torch.sigmoid`。每个调用接收连续有限
float32 `[32,128]` 的 `x` 与同形状显式 upstream，并在同一个 `torch.compile` 区域中返回完整 `y` 和 `dx`。
tanh 与 sigmoid 的前向使用 MATHFN `real` profile，输入还受其 `|x| <= 100` 范围约束；ReLU 前向及三个 VJP
使用 ACT `torch` profile。

此前 ACT 只有 `aten.tanh_backward(g,y)` 与 `aten.sigmoid_backward(g,y)`，其中 `y` 是保存的前向输出。真实端到端
autograd 调用的外部输入是原始 `x`，因此不能把 `x` 错绑成 `y`。ACT 现保留原操作，并增加明确从原始输入出发的
`vjp.aten.tanh(g,x)` 与 `vjp.aten.sigmoid(g,x)`。二者分别由 PyTorch 2.10 官方 tanh、sigmoid/expit 前向定义求导，
再以保存的 PyTorch 反向分解交叉核对。独立测试用 dual number 对前向公式求导，并用高精度 Arb 验证严格包围；
ACT 检查结果为 37/37 个例子相容、20/20 个错误读法被拒绝。

`relu_compiled_observed_20261019.json`、`tanh_compiled_observed_20261019.json` 与
`sigmoid_compiled_observed_20261019.json` 显示三个调用各生成一个融合 Triton kernel，完整写出两个输出，输入保持
不变。ReLU 的设备检查显式包含 `-0`、`+0`，梯度在两者上均为0；tanh 生成程序调用 `libdevice.tanh`；sigmoid
生成程序调用 `tl.sigmoid`。这里与 eager 的比较只核对编译边界，不充当 ground truth。

三份正式报告分别位于：

- `.cache/spec_trial_20261019_relu/production_report_20261019.json`
- `.cache/spec_trial_20261019_tanh/production_report_20261019.json`
- `.cache/spec_trial_20261019_sigmoid/production_report_20261019.json`

每个输出均建立 98,304 个严格参考坐标，完整率和分辨率均为1；24次输入的重复启动逐位一致，语义缺失、绑定错误、
区间过宽和超预算计数均为0。程序参照与任务规格的差异 `Gπ-f` 没有任何坐标排除0。ReLU 的 `y`、`dx` 在三层比较
中平均和最大绝对差均为0；由于每个统计样本都相同，均值检验正确报告为退化，证据含义是本批坐标逐点相容，不能据此
声称对所有输入证明无 bias。

tanh 的设备到程序差异 `K-Gπ` 为：前向平均绝对值 `1.6894e-8`、最大值 `1.0252e-7`，VJP 平均绝对值
`2.3794e-8`、最大值 `3.9247e-7`。前向的固定方向 R1 为正，Holm 校正 p 值约 `0.0379`，对齐方向 R2、R3 也为正；
VJP 在本批固定方向和对齐方向均未确认稳定作用。其 `Gπ-f` 前向为0，VJP 平均绝对值约 `4.20e-17`。

sigmoid 的 `K-Gπ` 前向平均绝对值 `1.7820e-8`、最大值 `1.0964e-7`，VJP 平均绝对值 `9.5176e-9`、最大值
`2.3619e-7`。前向 R1 为负、R5 为正，R2、R3 为负；VJP 的 R2、R3 为正，固定方向未确认。其 `Gπ-f` 前向与
VJP 平均绝对值分别约 `3.00e-17` 和 `1.46e-17`。完整 `K-f` 保留相同的已确认方向，说明本批稳定作用来自实际
浮点程序，而不是任务公式和程序参照之间的语义错配。

这些绑定只覆盖声明的 PyTorch 版本、shape、dtype、连续布局、有限范围和抽样分布，不扩展到原地/out 调用、其他 dtype、
特殊值、广播、任意同名 Triton kernel 或 signed-zero 位模式。某一方向“未确认”也不表示该输出没有 bias。
