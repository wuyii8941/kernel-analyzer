# PyTorch Inductor LayerNorm 前向与完整 VJP 绑定

调用边界是 PyTorch 2.10.0 的公开 `torch.nn.functional.layer_norm`，形状为 float32 `[16,128]`，
`normalized_shape=[128]`，带完整 weight、bias 和显式 upstream。`torch.func.grad_and_value` 在同一个
`torch.compile` 区域返回完整 `y`、`dx`、`dweight` 和 `dbias`；内部 mean 与 rstd 不是该公开调用的输出。

任务规格使用已准入的 NORM `torch_autograd` profile。它依据 PyTorch 2.10 的 LayerNorm 页面、ATen 分解和
独立微分公式，定义 biased variance、eps 位于平方根内、affine 前向以及对 x/weight/bias 的完整 VJP。

`compiled_observed_20261019.json` 确认实际编译生成两个 Triton kernel，四个输出的 dtype 和 shape 完整，
四个输入都未改变。与 eager 调用的差值只用于检查编译边界，不作为 ground truth；正式 bias 判断只使用上述独立数学规格。

正式报告 `.cache/spec_trial_20261019_layernorm/production_report_20261019.json` 使用 8 个开发输入和 16 个确认输入。
`y`、`dx` 各49,152个参照坐标，`dweight`、`dbias` 各3,072个参照坐标，完整率和分辨率均为1。
两份 TTIR 完整捕获，每个输出都由 Triton 写入且未被后续修改，重复启动逐位一致，未发现竞争或特殊值类别矛盾。

设备结果与程序参照的平均绝对差依次为 `y=4.09e-8`、`dx=4.52e-8`、`dweight=2.43e-7`、
`dbias=1.62e-7`。对齐方向规则在 `y`、`dx`、`dweight` 上均检出正向平均作用；Holm 校正后 p 值分别不大于
`6.53e-4`、`1.54e-3`、`7.24e-3`。`dbias` 未检出。这是当前声明输入分布上的数值 bias，不是整个 LayerNorm 角色的普遍结论。

程序公式与任务公式也有更小的可分辨差异：`y`、`dx`、`dweight` 的平均绝对差约为 `1.04e-13`、`1.08e-13`、
`4.14e-13`，对齐规则检出负向作用；`dbias` 精确为零。TTIR 明确将公开的 binary64 `eps=1e-5` 改为
float32 常数 `9.99999974e-6`，并调用 `__nv_rsqrtf`。这些是被测程序中可观察的精度选择；数学任务仍按公开 double eps 的精确值定义。
报告分别保留这一小差异与设备执行差异，没有用 eager 容差将它们合并。

绑定不扩展到其他 shape、dtype、normalized dimensions、缺省 affine 参数、RMSNorm、GroupNorm 或任意同名 kernel。
