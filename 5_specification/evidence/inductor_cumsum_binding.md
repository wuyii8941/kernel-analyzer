# PyTorch Inductor cumsum forward 与输入 VJP 绑定

调用边界是 PyTorch 2.10.0 的公开 `torch.cumsum`：连续有限 float32 `[16,128]` 输入，沿维度1计算
完整同形前向结果；再以显式同形 upstream 求完整输入 VJP。两个输入均属于任务前态并保持不变。

任务规格使用已准入的 REDUCE `real` profile。PyTorch 2.10 官方文档定义每个前缀为截至当前位置的累加和；
VJP 是 upstream 沿同一轴的反向累加和。合同用独立精确坐标循环和完整 Jacobian 复算前向及反向，另检查
轴、dtype、空张量和 upstream 形状。ground truth 按收到的 float32 输入求精确实数前缀，不采用 eager
或已编译结果。

`compiled_observed_20261019.json` 确认实际编译生成3个 Triton kernel：前向 scan、包含反向 scan 的融合
kernel，以及最终 flip 写回；生成代码明确含 `associative_scan`。前向和 VJP 均完整写出 `[16,128]`，
输入未改变。与 eager 的最大差均约为 `1.91e-6`；这个比较只用于核对调用边界。

正式报告 `.cache/spec_trial_20261019_cumsum/production_report_20261019.json` 使用8个开发输入和16个确认输入。
前向与 VJP 各49,152个程序参照坐标的完整率和分辨率均为1；没有语义缺失、绑定、特殊值、预算、前提或
包围失败，完整公开输出参与比较，重复执行逐位一致。程序公式与任务公式逐坐标相容，两项 Gπ−f 的平均和
最大绝对残差均为零。

前向 K−Gπ 的平均绝对差约 `4.27e-7`、最大约 `3.90e-6`；VJP 分别约为 `4.08e-7` 和 `4.07e-6`。
当前固定方向和对齐方向规则在两项输出上均未检出稳定平均作用。这里的“未检出”不是无 bias 或等价声明。

绑定不扩展到其他 shape、dtype、轴、显式 dtype 转换、空张量、特殊值、非连续布局、out 重载、其他 scan
操作或任意同名 kernel。
