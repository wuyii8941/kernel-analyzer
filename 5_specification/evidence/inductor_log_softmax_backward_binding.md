# PyTorch Inductor saved-output log-softmax backward 绑定

调用边界是 PyTorch 2.10.0 的 `aten._log_softmax_backward_data`：显式 float32 `[16,128]`
upstream、显式保存的 log-softmax 输出、`dim=-1` 和 `input_dtype=f32`，返回完整同形 `dx`。
这只是一份 saved-output backward 绑定，不把它写成尚未审核的 log-softmax 前向加反向组合任务。

任务规格使用已准入的 SOFTMAX `real` profile。保留的 PyTorch 2.10 分解明确给出
`dx = g - exp(y)*sum(g, dim, keepdim)`；独立的稠密 Jacobian 复算覆盖不同轴、dtype 参数、
标量、空张量和错误输入。输入生成器用 PyTorch log-softmax 构造满足行 `logsumexp` 接近零的保存输出，
但 ground truth 只把收到的 float32 `y`、`g` 当作精确输入并应用上述独立公式；生成器的结果不参与
`dx` 的求值。

`compiled_observed_20261019.json` 确认实际编译生成一个
`triton_per_fused__log_softmax_backward_data_0`，完整写出 `[16,128]` float32 梯度，upstream 与保存输出
均未改变。编译结果与 eager ATen 的最大差约为 `2.38e-7`；这个比较只检查调用边界，不作为规格。

正式报告 `.cache/spec_trial_20261019_log_softmax_backward/production_report_20261019.json` 使用8个开发输入
和16个确认输入。49,152个程序与任务参照坐标的完整率、分辨率均为1，没有语义缺失、绑定、特殊值、
预算或包围失败。程序公式与任务公式相容，没有严格区间排除零；二者平均绝对差约 `1.55e-17`，最大约
`4.44e-16`。

设备结果与程序参照的平均绝对差约为 `1.91e-8`，最大约 `4.87e-7`。当前固定方向和对齐方向规则均未
检出稳定平均作用，四项 Holm 校正 p 值均为1。这里的“未检出”不是无 bias 或等价声明。

绑定不扩展到 log-softmax 前向与 backward 的端到端组合、softmax backward、其他 shape、dtype、轴、
half-to-float、特殊值、全 mask 行、非连续布局或任意同名 kernel。
