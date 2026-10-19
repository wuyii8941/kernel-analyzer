# PyTorch Inductor `linalg.vector_norm` 与输入 VJP 绑定

调用边界是 PyTorch 2.10.0 的公开 `torch.linalg.vector_norm`：连续有限 float32 `[16,128]` 输入，
固定 `ord=2`、`dim=1`、`keepdim=False`，返回完整 `[16]` 二范数；再以显式 float32 `[16]`
upstream 求完整 `[16,128]` 输入 VJP。声明的正态输入使每行范数严格为正；两个输入都是任务前态并保持
不变。

任务规格使用已准入的 REDUCE `real` profile。保留的 PyTorch 2.10 文档将公开 vector 2-norm 定义为
平方和的非负平方根，并说明 `dim=None` 时展平、给定维度时按这些维度归约；保留的同版本 schema 固定
`linalg_vector_norm` 的公开 functional 重载。合同只选择实数 f32/f64、二范数和省略 dtype 的调用，
用独立100位平方根计算复算前向，并从 `upstream * x / norm` 直接复算完整梯度。零范数反向没有唯一
实数导数，因此拒绝，而不把 PyTorch 的零梯度约定当成通用答案。ground truth 不使用 eager 或已编译
输出。

`compiled_formula_observed_20261019.json` 确认实际编译生成一个融合 Triton reduction kernel，同时
写出16个范数和2,048个输入梯度，输入未改变。前向使用 `tl.sqrt_rn`；反向除以该范数，并保留范数为
零时写零的实现分支。当前任务范围只使用正范数行，因此该分支不参与裁决。编译结果与 eager 的前向
最大差为零，梯度最大差约 `2.98e-8`；这组 eager 比较只核对调用边界。

正式报告 `.cache/spec_trial_20261019_vector_norm/production_report_20261019.json` 使用8个开发输入和
16个确认输入。前向384个坐标、VJP 49,152个坐标的程序参照完整率和分辨率均为1；没有语义缺失、
绑定、特殊值、预算、前提或包围失败，完整公开输出均参与比较，重复执行逐位一致。

前向和 VJP 的 `Gπ-f` 都与精确任务相容，没有坐标严格排除零。设备到程序的前向平均绝对差约
`5.98e-7`、最大约 `2.03e-6`；梯度平均绝对差约 `2.55e-9`、最大约 `9.06e-8`。现有固定方向和
对齐方向规则均未确认稳定平均作用；这不是无 bias 结论，只说明当前24个输入和现有投影没有达到检出
条件。

绑定不扩展到其他 shape、dtype、维度、order、`keepdim=True`、显式 dtype 转换、零范数切片、空
归约、特殊值、复数、非连续布局、已弃用 `torch.norm`、out 重载或任意同名 kernel。
