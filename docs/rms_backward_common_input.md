# RMSNorm backward：相同输入参考

本页说明新接入的数学计算与证据范围，不沿用 AOT 参考端点替换的数值结论。

## 实际计算

已核对的 Triton 模板读取两路上游 gradient、归一化权重、输入 activation、
保存的 inverse RMS，以及一个已有的 gradient 累加值。记它们为
\(p_0,p_1,w,x,r,a\)，每行宽度为 \(d\)。令 \(g=(p_0+p_1)w\)，
则实数运算对应：

\[
y=a+gr-\frac{x r^3}{d}\sum_j g_jx_j.
\]

当 \(r=(d^{-1}\sum_jx_j^2+\epsilon)^{-1/2}\) 时，后两项就是
RMSNorm 对输入的梯度；\(a\) 是额外累加路径。不能漏掉这条路径后仍称为相同计算。

源码检查逐项匹配加载地址、两次归约循环、掩码、运算和最终写入；当前模板只接受
正的行数，以及满足系数精确表示条件的二次幂宽度。不同模型名称不改变检查规则。
CPU 测试用独立 autograd 验证公式，也检查改变系数、符号、地址或掩码会被拒绝。

## 参考如何运行

参考读取调用前复制的六个输入，以显式 FP32 运算求值，再按 candidate 输出表示写回。
这是预先声明的参考实现，不是绝对真值。实际运行还检查输入输出布局、inverse RMS
的数据类型，以及输出与其他只读输入是否共享存储。缺少这些证据时停止测量。

共享的 local → gradient → 实际参数写入采集、原坐标能量和统计阈值不变。
新比较只替换一个声明的输出，不将参考图上游差异引入这个局部参考。

## 当前范围

自动检查 DeepSeek seq256 的25个 Triton 定义，其中1个符合这一模板。
已据此绑定原 normalization backward 位置并完成32状态测量。实际参数写入的
确认集合 RMS 为0.02572776%，整体缩放约 −0.0000035053%，在预声明1% RMS范围内。
它是非零差异，不是全空间 identity，也不证明未来训练等价。
原 AOT 端点替换的大效应不能直接当作此相同输入比较的结论。

进一步通过已有 AOT 参数追踪生成 DeepSeek seq256 的1,267个 backward 映射，
同一源码检查器自动绑定其中35个符合模板的位置，其余1,232个保留为需要其他参考。
原35位置整批进程中断，`deepseek256_rms_all35` 保留未完成记录，不作完成证据。
恢复执行仍使用原35位置清单，按5位置分批，不按数值结果挑选。
首批 `deepseek256_rms_batch000_retry1` 和其余30位置的
`deepseek256_rms_remaining_queue_retry1` 均已完成；完整35位置×32状态已通过复算。
这35个位置属于同一实现模板，不能写成35种独立机制或35个已完成结果。

源码公式正确与 bias 均值非零是两个结论：这里的推导和 autograd 检查证明前者，
不自动证明后者，更不证明持久偏差、loss 分叉或训练质量变差。

实现为 `rms_backward_reference.py`，统一入口仍使用 `row-reference` / `row-capture`，
家族选择参数为 `--family RMS_BACKWARD`。结果在
`results/property/numerical_coverage_v1/deepseek256_rms_common_input/`。

## 批量自动验收

`scripts/finalize_numerical_family.py --root <采集目录> --output <新报告路径>`
从冻结计划读取全部位置，不需要手工列出已成功案例。它核对计划和输入银行摘要、
保存的源码、逐位置的参数范围和调用位置、相同输入参考、状态顺序、三阶段原坐标
统计及重复执行一致性，然后调用原来的 `analyze_artifact` 复算。分析代码改变后
不会静默沿用旧协议；缺失位置保留为 `NOT_CAPTURED`，不计为阴性或已完成。

报告的 `measurement_complete` 只表示这批声明位置的记录与复算齐全，
不表示整个研究计划完成，也不替代独立 GPU 复现或 loss 实验。
当前单位置及恢复后的完整35位置已通过此验收。
首批确认集合 update RMS 为0.0110581%–0.0215383%，均在原1%范围内，
但每个位置的16个确认状态均有非零更新差异。这不是持续均值偏差或 loss 结论。
完整35位置的RMS范围为0.00565205%–0.0323419%，全部满足原1%条件；
完整清单核验见 `family_execution_audit_20260907_post_queues.json` 的对应家族。
添加 `--baselines-dir <新图表目录>` 后，只有整批核验通过才会调用已有程序生成
全部声明位置的三阶段 JSON、CSV 和 SVG；无需手工选择成功案例或重写统计公式。

## 跨模型接入检查

同一检查器扫描四个模型、三种序列长度的45份已保存源码，共957条 Triton 定义记录，
匹配6个定义：DeepSeek 和 Qwen 各在64、128、256长度下匹配一次。
这些不是957种独立实现，也不是6项已完成训练实验。Phi 与 Mamba 未匹配这一
严格模板，不表示它们没有 RMSNorm 或没有 bias，只表示当前参考不能直接绑定。
逐定义记录见 `results/property/numerical_coverage_v1/rms_four_model_source_grid_extracted.json`。

基于已有 Qwen seq64 的930条 backward 参数映射，自动绑定27个匹配位置；
其余903个明确保留为需要其他参考。全部27位置、每位置32状态的相同输入测量
已在 `qwen64_rms_all27` 完成并通过整批核验，复用同一公式、FP32参考和1%固定集合 RMS阈值。
它使用此前已核对的重新声明 release 元数据，不覆盖原损坏文件。
27个位置在三个阶段都有非零差异；确认集合实际 update RMS范围为
0.0000443918%–0.0588663%，全部在声明的1%范围内。不能把范围内解释成没有 bias，
也不能推广为未来状态或完整训练等价。核验及81行三阶段记录分别见
`qwen64_rms_all27/completion_verification.json` 和 `verified_baselines/`。
这支持同一实现模板跨模型自动接入，不将位置数当作机制数；DeepSeek35位置也已完成复算。
