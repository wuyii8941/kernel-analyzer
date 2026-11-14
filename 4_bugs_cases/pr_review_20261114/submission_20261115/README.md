# 已批准的上游提交：2026-11-15

本轮按用户对审核提交 `2b927c5` 的批准执行。状态仅记录已发生的动作，不提前计为维护者确认或合入。

| 项目 | 当前状态 | 下一步 |
| --- | --- | --- |
| Liger KL | 两文件修复已形成提交 `a4addeb`，尚未开 PR | 创建个人 fork 后推送并开修复 PR |
| Attorch L1 | 两文件修复已形成提交 `0c63e26`，尚未开 PR | 创建个人 fork 后推送并开修复 PR |
| FLA | [RFC #1735](https://github.com/fla-org/flash-linear-attention/issues/1735) 已发布，正文已回读核对 | 等状态／输出 dtype 政策决定，暂不开实现 PR |
| FlagGems | [#8969 协调评论](https://github.com/flagos-ai/FlagGems/pull/8969#issuecomment-6880991318) 已发布，正文已回读核对 | 等共同基线确定，暂不开实现 PR |
| Trident | 保留存档 | 不向归档原仓库提交 |

Liger 上游主分支已从 `8fbf19f` 前进到 `83746f4`；变化涉及 device-context 编译分支及其测试，KL 源码与原测试未变。已在新基线上补跑完整 KL 文件：248 项通过、零失败、零跳过；固定 Ruff 0.14.11 的 `make checkstyle` 也通过，501 个 Python 文件无格式改动。未恢复整库测试或训练任务。

Attorch 上游主分支仍为 `67c2772`，沿用已审核的正确选择命令及 354 项通过记录（包括六项新增回归）；参数化数量不等于 GPU 执行次数。

两份待发 PR 的标题、正文与可应用提交分别在 [B049/](B049/) 和 [B047/](B047/)。旧证据仍使用固定提交 `3a4cbf1` 的链接；新基线验证单独保留在 [B049/latest_head_validation/](B049/latest_head_validation/)，没有覆盖旧结果。

当前发布步骤缺少两个个人 fork：GitHub 连接可发 PR，但没有创建 fork 的接口；账号下的两仓库查询均返回 404。已请用户在 [Liger fork 页面](https://github.com/linkedin/Liger-Kernel/fork) 和 [Attorch fork 页面](https://github.com/BobMcDear/attorch/fork) 创建到 `wuyii8941` 账号。正文审核已经完成，不需要再次批准提交。

[机器可读状态](SUBMISSION_STATUS.json) · [已发布的 FLA 正文](B007/rfc_posted.md) · [已发布的 FlagGems 评论](B038/coordination_comment_posted.md)
