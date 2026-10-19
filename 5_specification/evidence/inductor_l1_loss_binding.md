# PyTorch Inductor mean-L1 前向与完整双输入 VJP 绑定

本绑定覆盖 PyTorch 2.10.0 的公开无权重调用
`torch.nn.functional.l1_loss(x, target, reduction="mean")`。`x` 和 `target` 是连续、有限、同形状的 float32
`[32,128]` 张量，显式 upstream 是 float32 标量。一个 `torch.compile` 区域返回完整标量 loss、完整 `dx`
和完整 `dtarget`；三个张量输入均声明并核验为只读。

绑定使用 L1 `torch210_mean_f32_same_shape` profile。前向绑定到 `aten.l1_loss`；VJP 绑定到
`vjp.aten.l1_loss`，分别将 `/out/0`、`/out/1` 映射到两个梯度。输入生成器在每次正式调用中固定包含差值
`[-1,0,1]`，使公开编译路径实际经过等值点，而不是只在合同样例里检查零子梯度。

`evidence/inductor_l1_loss/compiled_observed_20261019.json` 显示该区域生成一个融合 Triton reduction kernel：
`triton_red_fused_abs_div_expand_mean_mul_neg_sgn_sub_0`。它写出三个声明输出，shape 与 dtype 均正确，输入
保持不变。显式等值坐标在 `dx`、`dtarget` 上都为0。与 eager 的最大差为0，这只核对编译边界，不充当
ground truth。

正式报告 `.cache/spec_trial_20261019_l1/production_report_20261019.json` 使用8个开发输入和16个确认输入。
24份绑定全部可用于 bias 测量；loss 的24个参考值，以及 `dx`、`dtarget` 各98,304个坐标，完整率和分辨率
均为1。全部重复启动逐位一致；语义缺失、绑定错误、区间过宽和超预算计数均为0。三个输出的 `Gπ-f` 均为
精确0，没有坐标排除0。

loss 的设备到程序差异 `K-Gπ` 平均绝对值约 `7.86e-8`、最大约 `2.19e-7`，固定方向和对齐方向均未确认
稳定作用。两个梯度在 `K-Gπ`、`Gπ-f` 和完整 `K-f` 上逐点为0；统计层因此把它们标为退化、信息不足，
而没有把样本内相等升级成总体无 bias 结论。

该绑定只覆盖声明的 PyTorch 版本、shape、dtype、连续布局、有限输入和抽样分布。它不扩展到广播、权重、
其他 reduction、其他 dtype、复数、特殊值、空张量或任意同名 Triton kernel。
