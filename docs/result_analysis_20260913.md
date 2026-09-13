# 结果分析：数值、范围与修改归因

历史基线为 `03bc18a42b4dd6761b88319d5b47c7270bc0e86f`。本轮重新分析记录并进行
小型受控 GPU 对照，并完成同路径训练复现；没有改旧阈值或旧冻结源码，
没有独立重训全部历史实验。本轮预定的记录分析、针对性归因对照和验证已完成，
不代表所有科学问题已解决。

[最终机器核验](../results/property/result_analysis_v1/final_analysis.json)的9项检查全部通过。
它重新计算测量与训练记录，不以训练结果必须阳性作为完成条件。

## 最重要的结论

同路径关闭/开启补偿的8组训练均跑满1024步。开启补偿的验证loss在8/8组中更低，
平均改善 **0.0220243**，95%配对t区间 **[0.0138561, 0.0301924]**，
整个区间超过预声明0.01门槛。16个最终模型重新加载后，32个评估输入上的loss均
与原记录一致；8份开启补偿训练的全部1024步训练loss及最终参数哈希均复现历史结果。

这确认了当前设置下残差补偿本身的训练作用，不只是不同optimizer代码之间的差异。
数据流来自已见历史实验，因此属于归因重放，不是新的未见数据确认；配对t区间的
总体解释仍依赖数据流生成和分布假设。也不能单独归因为“消除平均bias”，因为补偿
同时可能改变误差波动。基线比较未建立超越allclose的新增检出优势，区域差异也不能
全部归到单个kernel。负结果与这些边界均保留。

## 记录复算与比较范围

[自动复算结果](../results/property/result_analysis_v1/evidence.json)覆盖31项、992个状态。
分析文件哈希、原始文件哈希、冻结源码和生产分析结果全部一致。这不是由保存的
统计量重建原向量或原 GPU 执行。29项非零、2项零差异；全部31项是参考计算图
端点替换，不能全部归因于已隔离的单个 kernel。

预定义分类19种，实际非空分类17种，不是19个独立 bias。本轮31项分属 layout 5、
elementwise 13、reduction 12、mixed 1。沿用历史问题组，不按模型位置新增根因数。

DeepSeek 历史同位置对照中，图端点替换写入 RMS 34.7162%，同输入 FP32 重算
0.0004777564%。输入来源与参考求值都可能不同，不能把差额全部归于上游。

## 方向变化与幅度变化

令 c=r+u，则 C/R=1+Q+2β，其中 C、R 分别为两种更新的能量。
29项非零结果中28项的 sqrt(C/R) 与1相差不到0.1%。负 β 因而不能自动解释为
整体更新缩小。本轮报告更新幅度比与整体内积归一化量，不新增事后判决阈值。

历史固定集合 gradient 对照：Phi 总差异0.485%、平均方向0.402%，均值能量占比
68.81%；Llama 对应记录1.552%、0.401%、6.69%。RMS与共同平均分量提供不同信息，
但这不是总体保证或训练质量排序。

## 同数据 baseline

[可用性盘点](../results/property/result_analysis_v1/baseline_availability.json)以case、
原始数据哈希、stage去重，得到376组case/capture：82组local allclose通过、145组
未通过、149组未记录。重复报告的保存量未发现冲突。没有独立重跑这376组。

82组通过者都有写入RMS，最大约0.003052845%。因此这批记录没有建立“allclose
放过巨大更新差异”的证据。它不是全部历史实验，不能反向证明allclose足够。
149组缺失值不补造，已有对照主要支持解释价值而非新增检测准确率。

## 补偿训练与新增短对照

旧8组训练复算 mean(default−compensated)=0.0248481，95% t区间
[0.0168563,0.0328399]。运行检查使用过第0组前8步；排除第0组的事后敏感性均值
0.02646，区间[0.01807,0.03485]，不重新称为前瞻确认。补偿版更慢、峰值显存更高。
记录复算不等于本轮重训，也不证明收益完全由补偿造成。

新增实验只用4条合成梯度序列（4096/32768维×2个seed，每条8步），不是自然训练
新案例。起始参数与梯度相同，参数递推更新。对比默认编译、默认非编译、补偿开关、
旧补偿和FP32 AdamW。所有缓存均放在仓库内。

[第一轮](../results/property/result_analysis_v1/compensation_control/result.json)：
开启补偿与旧补偿全部32步写入和两个moment逐位相同。关闭补偿与默认非编译的
moment逐位相同，但写入不同；与默认编译版写入的最大相对差异约0.05431%。

[第二轮](../results/property/result_analysis_v1/compensation_control_scalar_followup/result.json)
仅对齐TorchAO的CPU FP32 lr/step标量计算：关闭补偿与默认非编译的32步写入全部
逐位相同。同一标量路径第8步，关闭补偿的写入差异为FP32参考的2.02%–2.29%，
开启后约0.01175%–0.01306%。这支持补偿本身改善这些受控梯度的更新。

默认编译与非编译仍有独立求值差异。短对照不能证明默认编译训练的loss改善完全
来自补偿，因此追加下述同路径训练。两轮短对照协议均在测量前保存。各自source_snapshots与冻结源码哈希
一致；第一轮快照在后续修改后按原内容恢复并经哈希核验。

## 统计保证与计划状态

继续使用[统计合同](population_inference_contract.md)：固定集合Q不是随机总体保证；
有界均值检验需要预先能量界；history超界比例不等于平均方向范数；方向诊断不替代
全空间mean等价上界。本轮不增加或替换检验。

| 工作 | 当前状态 |
|---|---|
| 本轮31项、训练记录复算 | 完成；不是独立重训 |
| 幅度/方向分析、baseline去重和缺失盘点 | 完成描述性分析 |
| 算子问题归类 | 复用旧问题组，本轮不增加独立根因计数 |
| 补偿同路径短对照 | 已测，标量差异已隔离，编译差异保留 |
| 相同执行路径训练归因验证 | 完成16份训练与模型重评估；8/8改善，配对区间超过0.01 |
| 相同参考下新增检测能力分析 | 已完成现有记录核查；未建立新增检出的正面证据 |

不能把审计文件存在称作科学问题全部完成。同路径训练补上了具体修改的归因证据，
不继续扩大扫描，也不因现有baseline结果不利而更换阈值。

## 目录整理

旧 `/data1/tzh/kernel-analyzer-audit-20260911` 的10份JSON完整移动到
`results/archive/kernel-analyzer-audit-20260911`，未删除文件。空的
`/data1/tzh/kernel_analyzer_spool/mamba_seq128_reach`及父目录用rmdir删除，没有数据损失。
两份旧catalog复算目录（catalog-repro-20260910-v1、kernel-catalog-check.2YvuxU）
经cmp确认对应文件一致，均完整迁入results/archive保留；空的kernel-catalog-check.yNigaB
已删除。没有删除非空实验记录。

本轮结果在`results/property/result_analysis_v1`，编译和临时文件在
`.runtime/result_analysis_v1`（不版本化）。`/data1/tzh/cache`仍含旧缓存、checkpoint与
共享资源，尚未完成依赖核查，未批量删除。模型、环境及其他项目保持不动。

早期短对照测试：数值分析与旧验证器10项通过；补偿同路径CPU测试2项通过。
随后完成下述完整测试和训练。GPU实验使用空闲GPU 2、3，旧冻结实验源码不变。
新生成的开发实验不能替代独立训练确认。

## 后续完整核查（本轮持续更新）

376组baseline已进一步核对423份报告，原始文件哈希、原坐标统计、已记录tolerance
日志和完整均值侧记录均复算一致。227组保存的tolerance均为rtol=1e-5、atol=1e-8，
不是被测kernel作者统一认可的容忍规则。82组通过者更新RMS均低于1%；145组未
通过者144组低于1%、1组不低于1%。1%仅为本表的明确比较线，不把局部失败称为
错误报警，也不把该表当检测准确率。
见[原始证据核验](../results/property/result_analysis_v1/baseline_verification.json)。

有界统计验证复跑发现旧合成样本与当前actual-write接口不兼容，完整流程返回
NOT_ASSESSED。保留[失败记录](../results/property/result_analysis_v1/bounded_validation_replay.json)，
改为真实SGD参数写入生成受控数据后，[新复跑](../results/property/result_analysis_v1/bounded_validation_actual_readback.json)
通过；两个2000次边界检验均零错误等价。生产判定没有放宽。
[超界比例精确复跑](../results/property/result_analysis_v1/exceedance_validation_replay.json)
也通过，三个边界的精确错误等价概率为0.02815、0.03752、0.04954，均不超过0.05。

旧方向频率下尾标签不能将零解释为负方向；新解释接口已修正命名，不改变历史
正方向计算。完整测试1577通过、4跳过，新追加的归因与验证测试12通过。

## 同路径训练最终结果

[独立核验与逐组表](../results/property/result_analysis_v1/same_path_training/verification.json)
及[CSV](../results/property/result_analysis_v1/same_path_training/verification.csv)覆盖全部8组、
16份训练。两条件共享执行代码和残差存储，只切换读取残差的乘数0/1；编译默认实现
继续使用历史结果，不冒充本轮重训。

| 数据流 | 关闭补偿−开启补偿的验证loss |
|---|---:|
| 0 | +0.009291 |
| 1 | +0.033682 |
| 2 | +0.029230 |
| 3 | +0.011144 |
| 4 | +0.023298 |
| 5 | +0.034531 |
| 6 | +0.016791 |
| 7 | +0.018227 |

排除曾用于历史短检查的第0组后，事后敏感性均值0.0238433，区间
[0.0155463,0.0321404]；它不是重新预注册的主结果。

历史默认−补偿的均值0.0248481，可写成默认−同路径关闭的0.00282384，加上
同路径关闭−开启的0.0220243。逐组恒等式复算成立；前一项包含其他求值差异，
并非独立因果贡献，不能把两个均值的比例解释成普适归因百分比。

同路径关闭条件故意保留残差的存储和计算，不是性能优化版本；本轮并发训练吞吐
也不是公平的速度比较。历史成本结论另行保留。本轮不声称无成本改进、跨模型收益、
崩溃预测，或收益仅由均值bias变化造成。数学关系见[推导](attribution_derivation_20260913.md)。

复算入口（仓库根目录，输出使用新路径，避免覆盖历史）：

```bash
PYTHONPATH=src:. python scripts/verify_same_path_training_attribution.py --root results/property/result_analysis_v1/same_path_training --output results/property/result_analysis_v1/same_path_training/verification_recheck.json
PYTHONPATH=src:. python scripts/build_final_result_analysis.py --output results/property/result_analysis_v1/final_analysis_recheck.json
```
