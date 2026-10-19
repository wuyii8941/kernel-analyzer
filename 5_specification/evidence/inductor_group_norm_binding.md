# PyTorch Inductor GroupNorm 前向与完整 VJP 绑定

调用边界是 PyTorch 2.10.0 的公开 `torch.nn.functional.group_norm`，输入为 float32 `[4,16,8,8]`，
`num_groups=4`，带完整 weight、bias 和显式 upstream。`torch.func.grad_and_value` 在同一个
`torch.compile` 区域返回完整 `y`、`dx`、`dweight` 和 `dbias`；内部 mean 与 rstd 不是该公开调用的输出。

任务规格使用已准入的 NORM `torch_autograd` profile。它依据 PyTorch 2.10 的 GroupNorm 页面、公开接口、
ATen 分解和独立微分公式，定义连续通道分组、biased variance、eps 位于平方根内、逐通道 affine 前向，
以及对 x/weight/bias 的完整 VJP。公开接口的单值输入拒绝也属于该 profile，但本调用不触及该边界。

`compiled_observed_20261019.json` 确认实际编译生成四个 Triton kernel，四个输出的 dtype 和 shape 完整，
四个输入都未改变。生成代码明确在 float32 统计量上加入 `1e-5` 并调用 `libdevice.rsqrt`。与 eager 调用的
差值只检查编译边界，不作为 ground truth；正式判断使用独立数学规格。

正式报告 `.cache/spec_trial_20261019_groupnorm/production_report_20261019.json` 使用8个开发输入和16个确认输入。
`y`、`dx` 各98,304个严格参照坐标，`dweight`、`dbias` 各384个，完整率和分辨率均为1。四份 TTIR 的
公开输出覆盖完整，重复启动逐位一致，没有绑定、竞争、特殊值、预算或包围失败。

设备结果与程序参照的平均绝对差依次为 `y=4.09e-8`、`dx=4.50e-8`、`dweight=1.15e-6`、
`dbias=9.35e-7`。对齐方向规则在 `y` 和 `dx` 上检出正向平均作用，Holm 校正后 p 值分别不大于
`1.24e-4` 和 `5.15e-4`；`dweight`、`dbias` 未检出。完整设备到任务比较在 `y`、`dx` 上得到同方向结论。
这是当前声明输入分布上的数值 bias，不是整个 GroupNorm 角色的普遍结论。

程序公式与任务公式另有更小的可分辨差异：`y`、`dx`、`dweight` 的平均绝对差约为 `1.04e-13`、
`1.06e-13`、`1.72e-12`，对齐规则均检出负向作用；`dbias` 精确为零。任务把 Python 传入的 binary64
`eps=1e-5` 作为精确输入，生成程序则在 float32 统计量上使用该常数并调用 `libdevice.rsqrt`。报告分别保留
这一程序精度选择与设备执行差异，没有用 eager 容差将它们合并。

绑定不扩展到其他 shape、dtype、组数、缺省 affine 参数、channels-last 或非连续布局、LayerNorm、RMSNorm、
native 多输出、其他公开拒绝边界或任意同名 kernel。
