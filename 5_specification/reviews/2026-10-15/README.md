# 2026-10-15 合同预审（原样存档）

- **审阅对象**：规格包 kernel_spec_package_2026-10-15，审阅报告给出的输入 SHA-256 为 `a9a2a9123c2f39f5f76d1c82d61d70869bc0ec8236cce16d978f9556293d15a3`。
- **收到的材料**：`kernel_spec_contract_review_20261015.zip`，SHA-256 `7db9e9e4aaa750ffa22b29ab25143b69d67bd0be60c5d22b61856279926b89ee`，放在仓库根目录，未跟踪。

审阅决定：方法通过。三份合同按交付包补全后，可以作为实现依据；原检查器不能用于认证。

| 文件 | 内容 |
|---|---|
| REVIEW.md | 审阅报告：决定、阻塞项 F1–F5、来源核对、需纠正的口径 |
| CODEX_TASK.md | 实施任务书：P0 检查器、P1 三个适配器、P2 验收 |
| HANDOFF_README.md、HANDOFF_MANIFEST.json、HANDOFF_SHA256SUMS | 交付包的说明、计数与哈希清单（清单里的路径按交付包的原布局写） |
| evidence/original/ | 审阅方对原包的运行记录：原检查、探针、注入失败的复现 |
| evidence/handoff/ | 审阅方对交付包的运行记录：结构校验、校验器回归、例子对照（`run_checks.py` 用它比对） |
| scripts/ | 审阅方的复现脚本，按原包的目录布局写成，没有适配本目录 |
| input_package/ | 被审规格包的顶层文件：README、NEXT_STEPS、run_all.sh。run_all.sh 有 F2 问题，已由 `../../run_checks.py` 取代 |

交付包里的其余文件已放到规格目录的对应位置：
- 合同：`../../contracts/`；
- 来源登记：`../../SOURCES.json`；
- 工具与测试：`../../tools/`、`../../tests/`；
- 实施口径：`../../docs/CONTRACT_PROTOCOL.md`。
