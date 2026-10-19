# PyTorch Inductor RMSNorm 前向与完整 VJP 绑定

调用边界是 PyTorch 2.10.0 的公开 `torch.nn.functional.rms_norm`，形状为 float32 `[16,128]`，
`normalized_shape=[128]`，带完整 weight 和显式 upstream。`torch.func.grad_and_value` 在同一个
`torch.compile` 区域返回完整 `y`、`dx` 和 `dweight`；RMSNorm 没有 bias 参数，内部 rstd 也不是该公开调用的输出。

任务规格使用已准入的 NORM `torch_autograd` profile。它依据 PyTorch 2.10 的 RMSNorm 页面、ATen 分解和
独立微分公式，定义 `x / sqrt(mean(x^2) + eps)`、affine weight 以及对 x/weight 的完整 VJP。
已有 NORM 独立复核从前向公式直接微分，不调用待测的 Inductor kernel。

`compiled_observed_20261019.json` 和 `compiled_precision_observed_20261019.json` 确认实际编译生成两个 Triton kernel，
三个输出的 dtype 和 shape 完整，三个输入都未改变。生成代码明确在 float32 张量上加入 `1e-5` 并调用
`libdevice.rsqrt`。与 eager 调用的差值只用于检查编译边界，不作为 ground truth；正式判断使用独立数学规格。

正式报告 `.cache/spec_trial_20261019_rmsnorm/production_report_20261019.json` 使用8个开发输入和16个确认输入。
`y`、`dx` 各49,152个参照坐标，`dweight` 有3,072个，完整率和分辨率均为1。两份 TTIR 完整捕获，
每个输出都由 Triton 写入且未被后续修改；重复启动逐位一致，没有绑定、竞争、特殊值、预算或包围失败。

设备结果与程序参照的平均绝对差依次为 `y=3.62e-8`、`dx=3.81e-8`、`dweight=2.35e-7`，最大差分别为
`6.69e-7`、`7.62e-7`、`1.57e-6`。固定方向和对齐方向在本次确认集上都未达到显著；最接近阈值的
Holm 校正 p 值分别约为 `0.0748`、`0.1010`、`0.0892`。这表示当前规则未检出稳定设备数值 bias，
不表示残差为零，也不支持 RMSNorm 普遍无 bias。

程序公式与任务公式有更小但可分辨的差异：`y`、`dx`、`dweight` 的平均绝对差约为 `1.02e-13`、
`1.06e-13`、`4.09e-13`，对齐规则均检出负向作用；分别有49,152、49,082和3,055个严格区间排除零。
这里的任务把 Python 传入的 binary64 `eps=1e-5` 作为精确输入，而生成程序在 float32 张量上使用该常数并调用
`libdevice.rsqrt`。报告分别保留程序精度选择与设备执行差异，没有用 eager 容差将它们合并。

绑定不扩展到其他 shape、dtype、normalized dimensions、缺省 weight、`eps=None`、offset RMSNorm、
LayerNorm、GroupNorm 或任意同名 kernel。
