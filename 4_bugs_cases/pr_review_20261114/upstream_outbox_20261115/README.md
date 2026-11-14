# 四项修复 PR 与待人工发布的 issue

| 项目 | 已发布 PR | Issue |
| --- | --- | --- |
| Liger KL | [#1625](https://github.com/linkedin/Liger-Kernel/pull/1625)，正式 PR | [草稿](B049/issue_draft.md)，由用户校对后人工发布 |
| Attorch L1 | [#22](https://github.com/BobMcDear/attorch/pull/22)，正式 PR | [草稿](B047/issue_draft.md)，由用户校对后人工发布 |
| FLA | [#1736](https://github.com/fla-org/flash-linear-attention/pull/1736)，Draft | 已有 [RFC #1735](https://github.com/fla-org/flash-linear-attention/issues/1735)；[当前正文](B007/issue_draft.md) |
| FlagGems varlen | [#9116](https://github.com/flagos-ai/FlagGems/pull/9116)，Draft | [草稿](B038/issue_draft.md)，由用户校对后人工发布 |

Trident 原仓库已归档，继续保留补丁，不提交 PR。

三个新增 issue **均未发送**。每项草稿都有独立标题、内嵌复现代码、实际输出、环境和对应 PR；不需要访问本仓库即可理解或复现。用户发布 issue 后再关联编号。

已发布的 FLA issue 和 [FlagGems 协调评论](https://github.com/flagos-ai/FlagGems/pull/8969#issuecomment-6880991318) 已改成自包含说明，并回读确认移除了 kernel-analyzer 链接。四份 PR 正文也不含内部案例编号或审核档案链接。

四个 PR 均已核对作者、线上正文、分支 SHA、一个修复提交和恰好两个改动文件。FLA Draft 等待状态／输出 dtype 政策；FlagGems Draft 仅包含当前 master 上的 varlen 修复，不重复 #8969 的 dense 改动。

FLA 的初始标题、版权头与 lint 检查通过；FlagGems 的 triage 检查通过。其他检查尚无结果，不能计作 CI 全通过。没有提前计为维护者确认或合入。

[机器可读状态](SUBMISSION_STATUS.json) · [Liger PR 正文](B049/pr_body.md) · [Attorch PR 正文](B047/pr_body.md) · [FLA PR 正文](B007/pr_body.md) · [FlagGems PR 正文](B038/pr_body.md)
