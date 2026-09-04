# 当前边界与未来增强

> 当前主线已从单一短筛扩展为“形成分解 → 三阶段效应量 → 长程后果”。本页中的
> Direct Persistence Screen 只保留为低成本排序组件。

当前结果已经足以支持一个范围明确的结论：

> 在 moments 从零开始、随后正常更新的 AdamW 设置下，16/32 步有效更新方向性可以给实验排优先级；长期标签由 warm-state 4096 步复核决定。它不是安全分类器。

下面的工作不是当前稿件的必做项。只有要升级相应主张时才需要继续。

## 已经完成的 optimizer 证据

- Liger、Phi、Qwen 和一个 Gemma 状态反馈对照已经完成同状态 AdamW、每步重置 moments 和无状态 SGD 比较。
- Qwen 已经完成早期、中期、后期真实权重、输入和 moments 下的响应测量。
- 这些结果证明 optimizer 会改变梯度差异进入参数更新的方式，但没有证明 AdamW 是数值误差的统一来源。

因此，“optimizer 对照未做”已经不是当前缺口。Qwen 的 4096 步结果还表明，cold-start 32 步抑制不能外推成长期抵消；要分离 moments、参数状态、输入序列和 horizon 的单独作用，仍需专门干预。

## 只有更强主张才需要的工作

### 跨未见实现泛化

当前已经完成一个 16 项验证集合，覆盖 DeepSeek、Qwen、Phi 和 Mamba 的多个 training
位置，并使用一套 48 项整体校正规则。15 项得到有效测量，9 项确认 update 缩小，
修正后的总体 update 证书给出 3 项通过固定输入 update 等价、3 项总体 update 比例
超过范围；其中两个 Mamba seq128 案例的三个冻结随机摘要均为零，但不能称为逐位
相同。Mamba seq256 对无法复现冻结 backward 图的位置保留无法判断。
旧三方向证书中的 Mamba seq64 等价标签已经撤销。Gemma 4 另有两个历史冻结位置
完成统一方法接入，结果均为阴性。它们支持方法可以跨模型运行，但还不是严格的
`NEW_IMPL / NEW_MODEL` 前瞻发现集合，也不能据此报告通用 recall 或自然发生比例。

### 完整 tolerance 对比

若要声称优于所有常见 tolerance，需要在同一冻结样本中保存 candidate/repair 的原始输出、梯度和更新位模式，再统一比较 max error、relative L2、ULP、`rtol/atol`、输出 RMS、梯度 RMS 和更新 RMS。当前只能稳妥声称短程方向性优于同层更新 RMS。

### 严重度与真实训练后果

当前已补两类后果检查：一类在保存参数附近比较测得方向和同长度随机方向的 loss
敏感性；另一类对三项训练位置运行四组互不重叠的 32 步配对输入流。前者证明 Liger
和 Phi 的测得方向不是任意随机方向；后者证明三项轨迹和参数会稳定分开，但稳定窗口
的平均 loss gap 都跨零，而且分离主要由 feedback 维持。因此仍不能声称完整训练质量
稳定变好或变坏。若要升级该主张，顶层单位必须是不同初始化的完整训练 run。

### catch-and-fix

只有当前瞻池出现直接持续正例后，才能做“筛出候选、确认、修复、复测”的完整演示。本轮没有合法对象，因此状态是 `NOT_APPLICABLE`，不是实验失败。

### v4.1

`results/property/direct_persistence_v4_1/` 是身份字段更完整的新一轮冻结入口，当前为 `NOT_STARTED_NEW_FREEZE`。它没有产生新结果，当前稿件不要求运行。

## 已知运行问题

Qwen seq64 的旧 `capture.json` 曾被一个 PyTorch checkpoint 覆盖；相同冻结任务清单
在正确的 r2 运行记录下已完整重跑。Mamba seq64 和两个 seq128 位置已经完成，其中
seq128 的 AdamW update difference 在三个冻结全坐标摘要中均为零；由于没有保留
原向量，不能写成逐位相同。seq256 虽能匹配具体输出，但实际 backward 图哈希与冻结
记录不同，因此保留为无法判断。没有用别的案例替换。

## 写作时的最终边界

- 可以写：在当前统一 AdamW 回溯样本中，短程方向性比同层 RMS 更适合做 fail-closed 分诊；当前长期标签以 4096 步结果为准。
- 不可以写：已经得到跨所有算子、optimizer 和训练阶段的通用安全 Oracle。
- 不可以写：4096 步直接方向等于完整训练已经收敛到不同 loss。
