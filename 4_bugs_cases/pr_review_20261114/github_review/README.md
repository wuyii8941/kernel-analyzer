# 五项上游修复：提交方案与表述审核

五项补丁的新增数值回归均已在真实 RTX A6000 GPU 上验证通过。这里公开的是你自己的 `kernel-analyzer` 审核分支中的补丁与拟用文字；**没有向上游提交 issue、PR 或评论，也没有修改本仓库 main。**

先审表达可直接打开：[五份 PR 正文＋FLA RFC 合并预览](PR_TEXT_PREVIEW.md)。

“GPU 上修好”和“现在能向原仓库发 PR”是两个状态。前三项可以在你审阅后发 PR；FLA 先走精度政策 RFC；Trident 原仓库归档，当前没有可接收 PR 的原仓库入口。不能把五项统一写成“随时都能发”。

## 建议怎样提交

| 顺序 | 项目与目标分支 | 操作与拆分 | 当前状态 |
| --- | --- | --- | --- |
| 1 | Liger-Kernel `main` | 单独一份 KL backward dtype 修复 PR，含回归；不改 forward | GPU 全 KL 文件 248 通过，可供你审查后提交 |
| 2 | Attorch `main` | 单独一份 L1 零残差梯度 PR，含六项回归 | 六项回归及 L1 选择通过，可供你审查后提交 |
| 3 | FlagGems `master` | 单独一份共享 dropout 导数 PR，dense/varlen 四处保持一致 | 八项新增回归通过；已有 BF16 失败如实披露 |
| 4 | FLA `main` | 先发 [RFC](B007/rfc_body.md)，同一讨论说明 bug 与 dtype 政策；达成一致后再发实现 PR | GPU 修复有效；状态精度／显式 FP32 Z 的输出 dtype 决策待同意 |
| 5 | Trident `main` | 保存补丁及描述；仓库重新开放，或有明确维护目标后再迁移 | GPU 修复有效；原仓库归档，提交渠道阻塞 |

每项一个分支、一个 PR，避免把五个仓库混为一份修改。Liger、Attorch 不必为同一修复先重复创建 issue，PR 本身已包含触发条件和验证；issue_draft 作为需要时的替代文本。FlagGems 已有 #8748 时优先关联现有讨论；不自动关闭它，因为本补丁没有验证其整条 forward/backward RNG 集成。

建议 fork 分支：`fix/kl-div-student-gradient-dtype`、`fix/l1-zero-residual-gradient`、`fix/attention-dropout-softmax-derivative`、`fix/linear-attn-fp32-z-state`、`fix/masked-softmax-strides`。本轮没有创建这些 fork 或分支，也没有向维护者发送消息。

## 请审查的具体文件

| 项目 | PR 标题／正文 | Issue 备稿 | 补丁 |
| --- | --- | --- | --- |
| B049 Liger | [标题](B049/pr_title.txt) / [正文](B049/pr_body.md) | [issue_draft](B049/issue_draft.md) | [proposed.patch](B049/proposed.patch) |
| B047 Attorch | [标题](B047/pr_title.txt) / [正文](B047/pr_body.md) | [issue_draft](B047/issue_draft.md) | [proposed.patch](B047/proposed.patch) |
| B038 FlagGems | [标题](B038/pr_title.txt) / [正文](B038/pr_body.md) | [issue_draft](B038/issue_draft.md) | [proposed.patch](B038/proposed.patch) |
| B007 FLA | [标题](B007/pr_title.txt) / [正文](B007/pr_body.md) / [RFC](B007/rfc_body.md) | [issue_draft](B007/issue_draft.md) | [proposed.patch](B007/proposed.patch) |
| B054 Trident | [标题](B054/pr_title.txt) / [正文](B054/pr_body.md) | [issue_draft](B054/issue_draft.md) | [proposed.patch](B054/proposed.patch) |

审核重点：问题边界与预期行为是否准确；补丁是否只解决对应问题；“验证通过”的范围是否明确；FLA 的 dtype 政策和额外成本是否可接受；FlagGems 关联 issue 是否表述谨慎。英文正文可以直接用于之后的 PR，但当前仍是待你审阅的备稿。

## GPU 证据

| 项目 | 同一新增回归的原版结果 | 修改后 | 更广验证／限制 |
| --- | --- | --- | --- |
| B007 FLA | 2 失败、2 FP32-state 控制通过 | 4 通过 | 受影响选择 41 通过；完整文件首轮触及显存 cap，未记为完整通过 |
| B038 FlagGems | 4 失败、4 p=0 控制通过 | 8 通过 | 522 通过、300 跳过、1 已在原版复现的 BF16 失败 |
| B047 Attorch | 6 失败 | 6 通过 | L1 选择 348 通过；部分既有参数化提前返回，不等于 348 次 GPU 执行 |
| B049 Liger | 新增矩阵 22 失败、122 通过 | 新增 144 通过 | 当前完整 KL 文件 248 通过；仓库级固定版本 Ruff 检查通过 |
| B054 Trident | 安全子集 6 失败、2 通过 | 完整 masked-softmax 文件 43 通过 | 使用旧 Triton API 名称／签名适配；未在历史原生软件栈重跑 |

环境：PyTorch 2.10.0+cu128、Triton 3.6.0、NVIDIA RTX A6000。这里是实际设备执行结果，已替代旧稿“仅 CPU 验证”的状态。两份更新仓库在准备期间上游又前进；本包已将 FLA/Liger 补丁移至 [记录的新基线](UPSTREAM_BASES.json)，并在新版本上重复关键 GPU 检查。旧失败和旧结果保留，未改断言容差以消除失败。

详细证据：[全部运行摘要](evidence/runs.json)、各项 `validation.json` 和 JUnit、[验证边界与失败记录](VALIDATION_NOTES.md)、[有限性能／内存测量](PERFORMANCE.md)、[复现与应用补丁](REPRODUCE.md)。这批数据支持具体修复，不证明分析器比简单检查强，也不作训练质量改善或全硬件覆盖的主张。

## 审阅通过以后

1. 以本包记录 SHA 为基础，在各自 fork 创建上述单独分支，应用对应 `proposed.patch`；如果默认分支已变，再检查相关 diff 并重跑受影响测试。
2. 使用各自标题／正文；不把内部 B 编号写成上游问题名称，不加入研究宣传或“全部故障已修复”的概括。
3. 先发布 Liger、Attorch；FlagGems 关联 #8748 并披露已知既有失败；FLA 先发 RFC。Trident 保留为渠道阻塞，不发往随便挑选的陈旧 fork。
4. 发布操作由你审阅后决定。本审核分支不是上游提交，也不等于你已批准五份 PR。

FLA 的 RFC 顺序来自其 [CONTRIBUTING.md](https://github.com/fla-org/flash-linear-attention/blob/2705ab8ee9a9086bcfa6d079858e4281e01215a1/CONTRIBUTING.md) 对精度／算法变更的要求；这是项目流程，不是 GPU 未修复的代称。
