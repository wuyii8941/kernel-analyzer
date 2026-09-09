# Softmax backward：相同输入参考

这是新接入的计算家族，不是已有 RMSNorm 位置的重复计数，也不是 saved-P
人工正负扰动实验。它使用同一源码检查、参数追踪、三阶段采集与统计分析流程。

## 数学对象

实际模板输入是上游 gradient \(g\)、调用前 scores \(z\)、保存的行最大值
\(m\) 和归一化分母 \(d\)，另有源码声明的缩放系数 \(c\)。计算为：

\[
p=\exp(z-m)/d,\qquad y=c\bigl(g\odot p-p\sum_j g_jp_j\bigr).
\]

当保存的量来自对应 softmax 时，这是其输入梯度再乘以声明缩放。
CPU 测试与独立 autograd 对照验证这个公式。它不证明浮点误差均值非零。

参考复用 candidate 调用前的四个输入，不重新计算另一个 forward 的 \(m,d\)。
FP32 参考采用显式乘法、求和和相减；candidate 中含有融合乘加，计算顺序差异
属于本次声明的实现比较。参考不是绝对真值，修改后不预先称为修复。
实际系数按原实现的 FP32 表示使用，最终结果转换为 candidate 的输出表示。

## 自动检查与范围

`softmax_backward_reference.py` 检查完整函数计算、加载地址、掩码、归约维度和
写入。当前只接受归约宽度为二次幂、保存的归一化量为 FP32、只读输入与输出不共享
存储的这一模板。缺输入、布局不符、无效分母或源码不符均拒绝测量。

通过 `row-reference --family SOFTMAX_BACKWARD` 选择家族，不依据模型名选择
统计公式。Qwen seq64 的930条 backward 参数映射中，28个位置匹配此参考，
其余902个需要其他参考。载体参数由已有数据流映射规则选择，不按误差大小挑选；
本次主要是各层 `q_norm.weight`，不能把其更新等价称为全模型更新等价。

整批28位置、每位置32状态的测量已完成，目录为
`results/property/numerical_coverage_v1/qwen64_softmax_all28/`。
采用与其他家族相同的实际参数写入、原坐标能量和固定集合1% RMS规则。
已通过 `family-report` 核验全部28位置，生成84行三阶段数据及对应图。
仍不宣称新增持续 bias 或 loss 分叉：这批是相同输入局部替换、每状态 AdamW
moments 从零开始的固定集合测量，既不是长程训练，也不是全模型参数范围。

27个位置的所测参数更新在全部32状态中逐位相同。另一个位置为
`backward:1227:in_out_ptr0`，参数是 `model.layers.6.self_attn.q_norm.weight`：
确认集合 update RMS为4.0569853%，超过冻结的1%范围，整体缩放为−0.08421944%。
但非零更新差异仅出现在 `qwen3-1p7b-0186` 一个状态的一个参数坐标上；
calibration 中没有可识别的更新平均方向。因此它是集合级差异超范围，
不是已经证明的可复现平均 bias，更不是持久偏差。不能因4.06%的数值较大而升级结论。

核验记录为 `completion_verification.json`。后续描述性重算增加了非零差异状态数
及最大单状态能量占比，保存在 `baselines_state_concentration/`，原图表保留不覆盖。
新增字段不参与原等价性判断，也不改变其阈值。训练 loss 未在本批测量。

对唯一非零更新位置，已另建 `qwen64_softmax_vector_replay`，重放同一32状态并
保留128维参数的原值、candidate/reference gradients、输入 moments和实际更新。
记录器返回原 optimizer 调用的结果，不改变参数计算；结束时核对向量重算能量和
原始统计一致。该复查针对已经看见的结果选择，不算新案例或未见确认。
复查已完成：原32状态数据文件的摘要逐字节一致，64次实际 AdamW 写入均由保存
的输入独立复算成功。唯一变化仍在状态0186、坐标89：candidate gradient为
\(+8.6613\times10^{-8}\)，reference为\(-1.5181\times10^{-7}\)，实际更新分别为
\(-8.9645\times10^{-5}\) 和 \(+9.3937\times10^{-5}\)。输入的两种 moments均为零。

在零 moments、首步、无 weight decay 的实数计算下，AdamW 更新为
\(-\eta g/(|g|+\epsilon)\)。这里接近零的梯度跨过了零点，产生相反方向的更新；
实际 FP32 参数写入由代码复算核对，而非仅用这条实数公式近似。
这解释了本次较大的更新差异，但不证明误差长期同向，不推断 warm-state 或 loss。
证据为 `qwen64_softmax_vector_replay/mechanism_verification.json`。

### 同一案例的逐元素 tolerance 复查

`qwen64_softmax_tolerance_replay` 在执行前声明 `rtol=1e-5, atol=1e-8`，
直接保存每个状态的逐元素比较，不从 RMS 推测 allclose。它是已有案例的复查，
不是新的未见确认；此阈值是分析对照，不是原 kernel 作者的正确性标准。
16个确认状态中3个没有通过，因此本案例不能用来证明“通过该 tolerance 后仍漏检”。
其原始三阶段结果与原28位置批次中对应文件逐字节一致，64次 AdamW 写入复算通过。
较大的更新差异仍只来自一个状态的一个坐标，不升级为持续 bias 或 loss 分叉。

此次采集耗时80.53秒，PyTorch记录的GPU峰值分配为10,080,805,376字节。
计时包含模型加载、编译、重放和记录，不代表训练吞吐或相对开销。
结果、复算和图表分别位于该目录的 `raw/`、`mechanism_verification.json`、
`verified_baselines/`。旧实验未记录的逐元素比较与成本不补造。

四模型三长度的957条源码定义记录中，这一严格模板匹配3个定义：DeepSeek
seq64，以及 Qwen seq64/seq128。其他配置需要不同模板或参考，不能因同为
softmax backward 就直接套用。扫描结果保存在 `softmax_four_model_source_grid.json`。

同一参考已用于 DeepSeek seq64 的自动绑定：1,231条 backward 参数映射中，
36个位置匹配这一模板，1,195个需要其他参考。36位置、每位置32状态的跨模型
批量测量已在 `deepseek64_softmax_all36` 完成，36位置均通过来源、状态顺序、
原坐标统计和共享分析核验，生成108条三阶段对照记录。
16个确认状态中，local差异非零36/36，参数gradient非零32/36，实际参数更新
非零1/36；最大更新RMS为0.00263426%，36项均在原1%固定集合范围内。
不要将其概括为“35个位置整个实验都零更新”：检查全部32状态时，31个位置
始终零更新，另外5个各有一个状态出现非零写入，其中4个发生在calibration集合。
这些结果不是所有训练状态的零偏差保证，也没有测量 loss。
采集耗时1023.25秒，PyTorch峰值分配39,062,478,336字节，不是训练吞吐结果。
核验来源为 `completion_verification.json`，图表在 `verified_baselines/`。
它不读取已有数值结果来挑选位置，仍是当前固定集合协议。
