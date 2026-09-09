# 新家族计算边界核对

本记录是保存源码的审核，不是 GPU 重放、bias 确认或训练后果报告。

## 新的待接入族：Top-k / 排序选择

`results/coverage/moe_full_invocation_inventory_strong.json.gz` 是 Granite MoE
一次真实 forward/backward 的历史 dispatcher 清单，非 Triton 运行清单。
其中包含 24 次 `aten.topk.default`、24 次 `aten.sort.default`。首个 router
Top-k 输入为 FP32 `[128,32]`，k=8、dim=1、largest=True、sorted=True；输出
包含 FP32 数值与 int64 专家索引，并记录 `TopkBackward0`。

`scripts/audit_selection_family.py` 自动提取真实 overload、全部调用参数及输出的
直接使用者，结果为 `results/property/numerical_coverage_v1/granite_selection_family_audit_v1.json`。
它只恢复已有观测，不证明当前模型运行、参数差异可达或新 bias。Top-k 与排序
按一个选择族处理，不把 24 层计为 24 族，也不与 softmax 或索引累加混称。

参考接入现已增加 `selection_reference.py` 与 `router_selection.py`：只替换
选定 router 内恰好一次 Top-k，数值和索引同时返回，后续分组与专家计算继续
使用原实现。稳定排序仅是声明的合法变体，不是真值或已验证修复。相等分数的
索引变化允许通过数学语义检查；NaN/Inf 暂不作判断。PyTorch 的
[Top-k 文档](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html)
也明确不保证相等分数索引的稳定性。

6 项单元测试通过；已安装 GraniteMoeTopKGating 的 CPU 合成输入接口检查
观察到一次选择调用、router 权重梯度存在、退出后 forward 恢复。
该检查的 forward 源码 SHA256 为
`a83a4201867ddc157237678fef2aa86fc3b5fe03f731626590a03c1bf50f0eb4`。
它不替代真实模型输入、共同状态和实际参数写入测量，不能计为新训练案例。

`scripts/run_granite_selection_probe.py` 已接入真实文本 full forward/backward，
原 Top-k 与稳定排序各重复两次，检查相同 scores、重复确定性和模型参数未变。
实际写入复用 `adamw_parameter_write`，原坐标量复用现有统计函数；完整输出、
索引、梯度和写入向量均保存供复算。首批为 layer 0、两个固定文本状态，
仅 router 权重可训练，AdamW 为冷启动 FP32 master 重放，不冒充 warm 或全模型
训练等价。Top-k 索引差异单独报告，不与浮点值误差拼接。
最初任务记录位于 `granite_selection_probe_launch_v1`，结果位于
`granite_selection_probe_v1`，均在 `results/property/numerical_coverage_v1/`。
这些作为失败与预检历史保留。

### Top-k 预检重复性复查

第一版两个文本状态都未通过重复性检查：同一实现重复运行的 scores、Top-k
数值和索引相同，但后续 gradient、write、loss 不同。因此第一版 loss 差异
不可归因于 Top-k，记录保留为 INVALID_COMPARISON。

第二版 `granite_selection_probe_v2` 明确为双方开启 deterministic algorithms，
设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，使用相同模型、文本、选择比较和参数范围。
两个状态均通过重复性检查；数值、router gradient、实际 FP32 master 写入及
loss 差异均为零。索引排列分别改变 64、57 个坐标，但将每个 token 的所选
索引排序后比较，两个状态的 128 个 token 都没有改变专家集合。这是合法的
相等分数排列差异，不是已发现的路由 bias。

两版均保留相应源码快照。该对照显示重复性问题在第二版设置下不再出现，
尚未定位第一版具体哪个 downstream kernel 导致分歧；不得据此宣称已证明
某个原子累加就是唯一根因。第二版也只支持两状态工程预检，不外推全部输入。

随后在相同冻结比较和参数范围下扩展为 24 个固定文本状态，结果位于
`granite_selection_suite24_v1`。保留张量经 `verify_selection_capture.py`
复算：24/24 比较有效；所有状态中 scores 与所选专家集合相同，局部数值、router
权重梯度、实际 FP32 master 参数写入和 loss 差异均为逐坐标零；相等分数下的
索引排列仍可不同。该结果作为常规 `TRANSFORMERS_EAGER_ATEN` candidate 的
固定集合三阶段测量进入 `operator_family_report_v7.json`，但 `equivalence_decision`
仍为 `NOT_ASSESSED`，执行身份也没有被独立工具再次证明，不能升级为总体等价或
训练质量结论。

接入前必须明确相等分数、排序顺序和 NaN 的语义；不能把相等分数的合法索引
差异直接判为数值错误。参考应同时检查数值与所选索引，并让后续训练自然消费
对应选择；只替换数值、不替换索引不等于完整 Top-k 比较。常规 candidate
可先验证统一接入，但不得由此声称已测到 Triton Top-k。

## Phi 的 max 名称不是浮点最大值家族证据

已读取 `phi4_seq64_r1/trace/model__1_forward_segment1_executed/output_code.py`
中 `triton_per_fused_add_gt_max_0` 的完整函数体。输入是 int64，输出是 bool；
计算是对 64 个整数取最大值、加 1、与 4096 比较。不能将它计为新发现的
浮点 max 偏差案例。它可能影响训练控制路径，因此也不能直接称其无关或安全。

新增 `scripts/audit_operator_candidate_types.py` 对已有源码清单批量检查实际
Triton signature，并核对源码哈希、重复定义和未知类型。输出只用于区分
待审核的浮点计算与整数/布尔控制计算，不凭名称或 dtype 自动判定算子族、
参考正确性、运行覆盖或 bias。家族数量仍以经过语义审核的归并为准。

## GELU 统一接入进展（2026-09-08）

`gemma_gelu_reach_v2/legacy/engineering_reach.json` 的一次状态预检已成功结束，
34 个冻结位置中 8 个在声明参数上记录了非零梯度差异。这不是持续 bias 或
实际 update 的结论；全部 34 个位置继续保留，不按预检差异挑选。

第一次 32 状态完整任务因旧输入文件仅有 26 个状态而退出，未产生完整测量。
失败记录及当时源码保存在 `gemma_gelu_capture_v1`，不删除或覆盖。
现已增加加载模型前的状态数量与 ID 唯一性检查，并通过相关测试。

从同一缓存文本数据另取 block 26 起的 32 个长度 128 文本片段，保存为
`gemma_gelu_fixed32_state_bank_v1.json`。原输入文件继续用于源码捕获身份校验；
新的状态文件单独进入协议哈希。`gemma_gelu_capture_launch_v2` 已启动全部
34 个位置的三阶段测量，并安排完成后自动复核。此处是固定文本集合，不因
文件中的 TRAJECTORY 标签或状态数而获得独立随机总体推断资格。

上述路径均位于 `results/property/numerical_coverage_v1/`。当前仍待完整结果，
没有预写阳性、等价性或训练质量结论。

## Phi embedding backward

来源：`results/coverage/runtime_releases/phi4_seq64_r1/trace/model__8_backward_segment2_executed/output_code.py`。

该文件三个名称含 embedding backward 的 Triton 定义分别执行清零、padding token 梯度屏蔽、类型转换。不能将它们当作三个 embedding 求和实现，也不能因它们没有 atomic 指令便认为完整 embedding backward 没有累加。接入时必须进一步覆盖调用区域中的实际聚合运算，参考语义应分别对应各输出。

另有一个语义独立的 **forward embedding lookup** 已完成接入。它从
`phi4_seq128_r1` 的真实 Triton 源码和参数路径自动绑定，冻结计划只有 1 个
符合完整公式与存储条件的位置；该位置完成 32-state local、embedding 权重
gradient 与实际参数写入测量，保存记录复算通过，三阶段均逐坐标为零。结果见
`phi128_embedding_lookup_summary_v1.json`。这只覆盖 forward lookup，不补齐
上段所述 embedding backward 聚合，也不外推其他实现或随机训练总体。

## Mamba 卷积前的梯度合成与 bias reduction

来源：`results/coverage/runtime_releases/mamba_seq64_r1/trace/model__1_backward_segment0_executed/output_code.py`，符号结尾 `squeeze_t_transpose_unsqueeze_view_78`。

源码先合成三个梯度来源，经 SiLU 导数计算后，将前 64 个位置的结果写回原输入缓冲区，额外三个位置置零，同时将未经过 BF16 输出舍入的 FP32 结果求和写入另一个输出。这不是单纯的卷积计算，也不是只有一个输出的 SiLU reference。

现有 observer 已在 kernel 调用前克隆输入，包括会原地写回的输入。因此无需重新设计输入采集。尚需独立验证两个输出的参考：原地梯度输出与 bias 求和输出；不能用已写回的 BF16 梯度求和代替源码内部 FP32 reduction。应保留二者不同的计算边界和参数绑定。

以上说明下一步接入必须按实际运算与输出处理，不能仅按 kernel 名称归类；没有新增阳性结论。
