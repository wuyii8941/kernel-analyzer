# PyTorch Inductor logsumexp forward 与输入 VJP 绑定

调用边界是 PyTorch 2.10.0 的公开 `torch.logsumexp`：连续有限 float32 `[16,128]` 输入，
沿末维归约且 `keepdim=False`，返回完整 `[16]` 前向结果；再以显式 float32 `[16]` upstream
求完整 `[16,128]` 输入 VJP。两个输入都属于任务前态且必须保持不变。

任务规格使用已准入的 REDUCE `real` profile。PyTorch 2.10 官方文档定义
`logsumexp(x)=log(sum(exp(x)))`；VJP 由这一函数的导数得到，即 upstream 乘相应归约切片的
softmax。合同另以高精度求值、精确等值切片和完整形状检查复算前向与 VJP。这里的 task ground truth
独立按收到的 float32 输入求值，不采用 eager 或已编译输出。

`compiled_observed_20261019.json` 确认实际编译生成一个融合 Triton kernel，同时完整写出 `[16]` 的
前向结果与 `[16,128]` 的输入梯度，输入没有改变。编译结果与 eager 的最大差分别约为 `9.54e-7`
和 `1.19e-7`；这两个数只用于核对调用边界，不作为规格。

正式报告 `.cache/spec_trial_20261019_logsumexp/production_report_20261019.json` 使用8个开发输入和16个
确认输入。前向384个坐标、VJP 49,152个坐标的程序参照完整率和分辨率均为1；没有语义缺失、绑定、
特殊值、预算、前提或包围失败，完整公开输出均参与比较，重复执行逐位一致。

程序公式与任务公式相容，两项输出均没有严格区间排除零。前向的设备到程序平均绝对差约
`1.34e-7`，最大约 `4.11e-7`；VJP 分别约为 `9.42e-10` 和 `1.99e-7`。当前固定方向与对齐方向规则
在两项输出上均未检出稳定平均作用；这里的“未检出”不是无 bias 或等价声明。

绑定不扩展到其他 shape、dtype、维度列表、`keepdim=True`、空归约、特殊值、非连续布局、out 重载、
其他 reduction 或任意同名 kernel。
