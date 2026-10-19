# PyTorch Inductor mean-MSE 前向与完整双输入 VJP 绑定

本绑定覆盖 PyTorch 2.10.0 的公开调用
`torch.nn.functional.mse_loss(x, target, reduction="mean")`。`x` 和 `target` 都是连续、有限、同形状的
float32 `[32,128]` 张量；显式 upstream 是 float32 标量。一个 `torch.compile` 区域返回完整标量 loss、完整
`dx` 和完整 `dtarget`。公开调用没有传入 weight，三个张量输入均声明并核验为只读。

绑定使用 MSE `torch210_mean_f32_same_shape` profile。前向绑定到 `aten.mse_loss`；VJP 绑定到
`vjp.aten.mse_loss`，并分别将 `/out/0` 和 `/out/1` 映射到 `dx` 与 `dtarget`。这样不会把底层只返回 self
梯度的 `mse_loss_backward` 误当成公开调用的完整 VJP。

`evidence/inductor_mse_loss/compiled_observed_20261019_r2.json` 显示该区域生成一个融合的 Triton reduction
kernel，名称为 `triton_red_fused_mse_loss_mse_loss_backward_0`。它写出三个声明输出，shape 与 dtype 全部正确，
输入保持不变。与 eager 的最大差在本次边界检查中为0；这项检查只核对编译调用边界，不充当 ground truth。

第一次正式测量报告 `.cache/spec_trial_20261019_mse/production_report_20261019.json` 被准入层拒绝：早期调用边界
额外接收恒为 `None` 的 `weight`，绑定却没有消费该输入。修正方式是从调用边界删除这个并未传入公开函数的参数，
而不是让绑定层静默忽略它。失败报告保留，修正后的正式报告为
`.cache/spec_trial_20261019_mse/production_report_20261019_r2.json`。

修正报告使用8个开发输入和16个确认输入。24份绑定全部可用于 bias 测量；loss 的24个参考值，以及 `dx`、
`dtarget` 各98,304个坐标，完整率和分辨率均为1。全部重复启动逐位一致；语义缺失、绑定错误、区间过宽和超预算
计数均为0。三个输出的程序参照到任务规格差异 `Gπ-f` 都没有坐标排除0：loss 的平均绝对差约
`2.41e-16`，两个梯度均为0。

实际设备到程序参照的 `K-Gπ` 中，loss 平均绝对差约 `2.74e-7`、最大约 `6.31e-7`；`dx` 与 `dtarget`
平均绝对差均约 `1.88e-11`、最大约 `5.17e-10`。完整 `K-f` 与这些量级一致。本批次的固定方向与对齐方向均未
确认稳定作用；部分梯度 R3 因16个确认样本的投影分布偏斜而拒绝判断。这个结果不能解释成“没有 bias”，只能说明
当前样本和预先声明的统计规则没有确认稳定方向。

该绑定只覆盖声明的 PyTorch 版本、shape、dtype、连续布局、有限输入和抽样分布。它不扩展到广播、加权、其他
reduction、其他 dtype、特殊值、空张量或任意同名 Triton kernel。
