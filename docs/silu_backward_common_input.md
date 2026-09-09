# SiLU backward：相同输入、单输出的自动接入

本页对应新的实现比较，不是 saved-P/SiLU 历史上的人工正负响应实验。
既不预设有持续 bias，也不预设 loss 会分叉。

## 数学计算与实际边界

对 gated MLP 的 `up * silu(gate)`，上游梯度记为 `dy`，gate 梯度的实数公式为

\[
dy\,up\,\sigma(gate)\,[1+gate(1-\sigma(gate))].
\]

`silu_backward_reference.py` 核对整个已保存函数体、指针参数次序、索引和两个
输出，且要求运行时实际源码一致。参考使用同一次调用前的三个输入，采用 FP32
sigmoid 和上述公式计算，最终转换为声明输出 dtype。公式已与独立 autograd
计算核对；这只证明数学语义对应，不证明实际浮点误差的均值非零。

当前比较只替换 `in_out_ptr0` 的 gate gradient，融合 kernel 的另一个输出
`out_ptr0` 保持原实现结果。这是单个输出的局部替换，不是整个融合 kernel
两个输出同时替换，更不是完整 forward/backward 实现比较。

## 自动化与执行范围

在 DeepSeek seq128 已有 backward 源码中，24个定义有1个符合严格模板，
对应已有参数映射中的36个位置。其余位置保留在未接入清单，不视为阴性。
计划由 `build_row_reduction_contracts.py --family SILU_BACKWARD` 自动生成，
没有按数值大小或模型名称修改统计公式。

每个声明参数为一个12288×4096的 gate projection 权重，内存成本显著大于
normalization 参数。因此使用 `partition_family_plan.py` 按原顺序拆为36个
单位置任务，保留完整源计划摘要；没有筛掉位置或缩小覆盖分母。
首项在 `deepseek128_silu_batch000` 执行32状态采集，其他35项已按固定顺序进入
两个不重叠自动队列；中断恢复后，完整36位置均已完成并复算通过。
队列逐项保存源码快照、复算结果并保留失败，
不需要逐例手改 runner，也不根据输出选择下一个位置。
采集包括三阶段原坐标统计、声明的逐元素 tolerance 和成本，共用现有分析代码。
执行完成、参数范围核验和结果复算之前，不计为成功测量。
完整清单审查为 `family_execution_audit_20260907_all_three_complete.json`：
36位置确认集合实际更新RMS范围0.0125187%–0.0289637%，均在原1%范围内。
这不是36种机制，也没有将这些结果升级为持久bias或loss证据。

首项现已完成并复算：确认16状态中 local、gradient、实际参数更新均非零。
对应 RMS 分别为0.00585894%、0.00360110%、0.0289637%；最后一项在原1%
固定集合范围内。局部逐元素对照有11/16状态未通过声明 tolerance，不将其
称作该阈值下的漏检。参考 `deepseek128_silu_batch000/verified_baselines/`。
参数范围仅为 `model.layers.35.mlp.gate_proj.weight`，不是整个模型。

此前大向量仅保存完整能量和压缩方向摘要，不能从摘要补造完整均值。因此新增
`run_full_mean_family_capture.py`，对同一首项重采，直接累加完整坐标的
calibration/confirmation 均值、逐状态正交 residual 均值及冻结方向上的投影。
只保存累计向量和标量，不必永久保存32个大向量。最终要求原始三阶段 JSON
逐字节不变，并核对每个状态的能量与原记录一致；未完成前只算验证中的扩展。
结果目录 `deepseek128_silu_full_mean_replay`，不增加独立案例数。
它是有限集合的描述性测量，不给总体 CI、不证明持续性，也不改变 Q 等价性规则。

### 完整坐标复测结果

中断后的有效复测在 `deepseek128_silu_full_mean_replay_retry1` 完成。
原始三阶段 JSON 与首项逐字节一致，完整向量累加的逐状态能量也与原记录一致。
对50,331,648维参数更新，确认集合平均向量幅度为正常更新RMS的0.00724093%，
而总差异RMS为0.0289637%。这两个数不是同一统计对象。

用前16状态的完整均值方向，在后16状态上计算投影：15个为0，另一个约为
−1.91×10⁻¹³（以正常更新RMS归一化）。因此本次没有观察到该方向在确认状态中
继续同向推动的证据。有限集合均值非零、16状态均有差异，都不能单独解释成
总体均值偏差或持久性。这也不证明差异一定独立、没有任何状态相关结构或长期安全。

完整 local、gradient、update 均值和逐状态投影保存在 `full_coordinate_means.json`；
三阶段对照表与图在 `full_coordinate_baselines/`。这些均值不再由 CountSketch
推定，原1%固定集合判断仍然只使用完整更新能量，未改阈值。

### 同方法的 Qwen 验证

在 Qwen seq64 已保存的23个 backward 定义中，1个满足同一 SiLU 模板，
自动绑定到28个 gate-gradient 位置。计划为 `qwen64_silu_all_bound.json`，
按原顺序分成7批、每批4位置；首批 `qwen64_silu_batch000` 验证通过后才
启动 `qwen64_silu_remaining_queue`。状态数、参考公式、tolerance和1%范围
不因模型改变。这是同一实现家族的新模型条件测量，不计作新机制或全部 kernel 支持。
目前28个位置×32状态全部完成并通过复算，确认集合实际更新RMS范围为
0.00163600%–0.0355857%，均满足原固定集合1%条件。完整清单审查见
`family_execution_audit_20260907_post_queues.json`；没有新增持续bias或loss结论。
原来需要其他参考的902个 backward 映射继续保留。

本轮固定分析 tolerance 为 rtol=1e-5、atol=1e-8，不冒充作者原测试标准；
参数更新的固定集合 RMS 范围仍为1%，未为新家族调整。
