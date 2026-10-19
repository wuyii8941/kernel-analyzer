# PyTorch Inductor saved-output softmax backward 绑定

调用边界是 PyTorch 2.10.0 的 `aten._softmax_backward_data`：显式 float32 `[16,128]` upstream、显式保存的
softmax 输出、`dim=-1` 和 `input_dtype=f32`，返回完整同形 `dx`。这是一份真实的 saved-output backward 绑定，
不把它写成尚未审核的“公开 softmax 前向加反向”组合任务。

任务规格使用已准入的 SOFTMAX `real` profile。保留的 PyTorch 2.10 分解明确给出
`dx = y*g - y*sum(y*g, dim, keepdim)`，独立精确有理数稠密 Jacobian 检查覆盖不同轴、dtype 参数和错误输入。
输入生成器用 PyTorch softmax 构造形状真实、正且行和接近1的保存输出，但参考答案只把收到的 float32 `y`、`g`
视为精确输入并应用上述独立公式；生成器的 softmax 数值不参与 `dx` ground truth。

`compiled_observed_20261019.json` 确认实际编译生成一个 Triton reduction kernel，完整写出 `[16,128]` float32
梯度，upstream 与保存输出都未改变。与 eager ATen 结果的差值只用于检查调用边界，不作为规格。

正式报告 `.cache/spec_trial_20261019_softmax_backward/production_report_20261019.json` 使用8个开发输入和16个确认输入。
49,152个程序与任务参照坐标的完整率、分辨率均为1；重复启动逐位一致，没有绑定、竞争、特殊值、预算或包围失败。

设备结果与程序参照的平均绝对差约为 `2.01e-10`，最大约 `1.97e-8`。当前固定方向和对齐方向规则均未检出
稳定平均作用；最接近阈值的是一个对齐规则，Holm 校正 p 值约 `0.0529`。程序公式与任务公式相容，
没有严格区间排除零。这里的“未检出”不是无 bias 或等价声明。

绑定不扩展到 softmax 前向与 backward 的端到端组合、log-softmax backward、其他 shape、dtype、轴、
half-to-float、特殊值、全 mask 行、非连续布局或任意同名 kernel。
