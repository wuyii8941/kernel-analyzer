# PyTorch Inductor exact-form GELU 前向与完整输入 VJP 绑定

调用边界是 PyTorch 2.10.0 的公开 `torch.nn.functional.gelu(..., approximate="none")`，输入和显式 upstream
均为连续 float32 `[32,128]`。`torch.func.grad_and_value` 在同一个 `torch.compile` 区域返回完整 `y` 和 `dx`。
绑定分别使用 ACT `torch` profile 的 `aten.gelu` 与 `aten.gelu_backward` 两个任务，避免把反向输出附会成前向合同的额外字段。

任务规格依据保留的 PyTorch 2.10 GELU 页面、functional 页面和 ATen 分解，前向为
`x (1 + erf(x/sqrt(2))) / 2`，VJP 为 upstream 乘以 `Phi(x) + x phi(x)`。ACT 的独立复核从前向公式直接微分，
并用高精度 Arb 值检查严格包围；它不调用待测的 Inductor kernel。

`compiled_formula_observed_20261019.json` 确认实际编译生成一个融合 Triton kernel，完整写出 `y` 和 `dx`，
两个输入均未改变。生成程序直接使用 `libdevice.erf`、`libdevice.exp`、binary 常数 `0.7071067811865476`
和 `0.3989422804014327`。与 eager 的差值只检查编译边界，不作为 ground truth。

第一次正式报告 `.cache/spec_trial_20261019_gelu_erf/production_report_20261019.json` 没有被接受：虽然全部任务坐标都有严格
参照，但程序参照直接计算负尾部的 `1 + erf(x)` 时发生区间抵消，前向205个、梯度75个坐标未达到 `1/8 ULP` 分辨率，
报告明确包含 `enclosure too wide`。没有通过缩小输入范围或放宽阈值来消除失败。

程序参照随后加入严格恒等式 `1 + erf(x) = erfc(-x)`。它只用于未经过舍入的 erf 结果直接与精确常数1相加的情况；
任何中间运算或显式舍入都会清除该条件。独立300位 MPFR 测试检查负尾部真值始终包含在新包围中，并确认新包围比先求
erf 再相加更窄。修复后的新报告
`.cache/spec_trial_20261019_gelu_erf/production_report_after_erfc_identity_20261019.json` 保留旧报告且使用相同24个输入：
`y`、`dx` 各98,304个坐标的完整率、分辨率均为1，全部失败分类为零；报告还记录该恒等式规则实际使用。

设备结果与程序参照的平均绝对差为 `y=2.76e-8`、`dx=1.91e-8`。前向固定方向 R1 检出负向平均作用，
Holm 校正 p 值约 `2.55e-4`；梯度对齐方向 R2、R3 检出负向平均作用，校正 p 值分别约 `3.51e-7` 和
`5.81e-3`。程序公式与任务公式的平均绝对差为 `y=2.44e-9`、`dx=1.15e-9`，前向固定方向为正，
梯度对齐方向为负。两层差异分别报告；完整设备到任务比较在前向固定方向和梯度对齐方向也检出非零作用。

绑定不扩展到 tanh 近似、其他 shape 或 dtype、非连续布局、特殊值、原地 GELU、其他梯度边界或任意同名 kernel。
