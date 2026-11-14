# 五项修复：独立审核后的提交方案

这是对第一版固定提交 `1d2d480` 的修订。**没有向上游发 issue、PR 或评论。** 当前路线是两份修复 PR、一份 FLA RFC、一份 FlagGems 协调／依赖增量，Trident 按本轮审核决定存档。

先看：[合并文字预览](PR_TEXT_PREVIEW.md)。[GPU 原始记录与完整边界说明](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/README.md) 已先行发布到固定提交 `3a4cbf1a0128dc615dbd6134768955d4aec58716`，各正文都引用具体公开文件，复制到上游后仍可访问。

| 项目 | 当前处理 | 审核文件 |
| --- | --- | --- |
| Liger KL | 局部修复可供提交；保留显存代价和全局验证边界 | [PR](B049/pr_body.md) / [issue](B049/issue_draft.md) / [补丁](B049/proposed.patch) |
| Attorch L1 | 修正公开命令并实跑 354 项，确实含六项新回归 | [PR](B047/pr_body.md) / [issue](B047/issue_draft.md) / [补丁](B047/proposed.patch) |
| FLA | 先 RFC，直接展示约 +38%/+36%/+15% helper 成本；状态／输出 dtype 决策后再完善相应断言 | [RFC](B007/rfc_body.md) / [未来实现 PR 备稿](B007/pr_body.md) / [候选补丁](B007/proposed.patch) |
| FlagGems | 不再原样提交 dense+varlen；dense 与开放 PR8969 重合，当前只保留基于其 head 的 varlen 增量和 NVIDIA 回归 | [协调评论备稿](B038/coordination_comment_draft.md) / [依赖 PR 备稿](B038/pr_body.md) / [增量补丁](B038/proposed.patch) |
| Trident | 保留有效修复和适配记录；不向归档原仓库发 PR，不随意选 fork | [存档说明](B054/pr_body.md) / [补丁](B054/proposed.patch) |

## 本轮实质修改

- 查重更正：#8969 于 11 月 12 日打开，已有 dense DQ/DKV 相同修复。第一版漏检。新补丁的生产代码仅改两个 varlen 函数，不把已有 dense 修复再次算成我们的独立提交。
- FlagGems 回归补强：固定同掩码、非零 Q/K/V、独立 FP64 autograd；PR head 原版 4 失败/12 通过，varlen 修复版 16 通过。只撤回 varlen DKV，旧 Q=0 用例全通过、新增两项明确在 dK 失败。dV 的既有概率转换使用单独、显式的误差预算；最初四个 BF16 dV 失败也保留。没有新增 dBias 已验证主张。
- Attorch 命令改成 `l1_zero_residual_gradient or (L1Loss and not SmoothL1Loss)`；实际 354 项通过，采集结果确认六项都在里面。
- FLA 新增加了一次完整 58 项通过记录以及 Based/ReBased 调用验证。原 OOM 和原因未明的 SIGSEGV 没有删除，也没有宣布原因已解决。chunk/fused-chunk 默认 normalize=true 的范围已更正。
- 所有上游正文都有固定提交下的具体证据链接；永久 Z 的类型存储、临时分配、helper 计时与模型整体表现分别表述。

## 提交顺序与记录纪律

Liger、Attorch 的当前正文可先审核；FLA 可并行讨论 RFC。FlagGems 先把协调备稿交你审查，再向 #8969 作者／维护者确认是拆共享前置修复，还是合入后追加 varlen；当前补丁不宣称可直接套未修复的 master。Trident 仅存档。

| 状态 | 本轮事实 |
| --- | --- |
| 本地复现并修复 | 五个案例的既有范围；本轮另补非零 dK 回归和调用验证 |
| 已由我们向上游提交 | 无新增 issue、PR、评论 |
| 维护者确认 | 无新增确认；#8969 的作者与既有 dense 工作单独列明，不计成我们的确认 |
| 已合入 | 无新增合入 |

旧审核包保留为历史；这份修订不改写原数据，也不把重合 dense 修复或归档仓库算成新的可提交成果。
