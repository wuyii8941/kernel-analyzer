# 下一步只做一件事：交付修复版，再继续增量14

**给Codex：先读 `CODEX_NEXT_TASK.md`。本轮不做正式发布，也不补写历史盲测成绩。**

顺序：在真实checkout复现问题 → 修整数和区间 → 修保证/来源/统计准入 → 打通统一入口并锁住全部测量设置 → 回归与材料交付 → 再做增量14。

这份补充包增加的是可复核证据与执行标准，不是把旧的抽查包装成全面认证。

## 文件入口

- `CODEX_NEXT_TASK.md`：可以直接发给Codex的任务书。
- `RECHECK_REPORT.md`：重查了什么，发现了什么，以及对上一版的纠正。
- `WORKPACKAGES.md`：W0至W7逐项账，避免只修bug后就宣布全部完成。
- `MATERIALS_LEDGER.md`：全部待交材料、责任方、完成条件；旧答案和新验收分开。
- `review_ledger.json`：七项审计发现的证据等级和关闭条件。
- `recheck/repro/run_recheck.py`：新增整数边界、声明哈希、执行状态测试。
- `recheck/results/additional_checks.json`：全部1,748个整数输入及结果、4个声明设置对照、4个执行状态对照。
- `rerun/results/`：上一包两套脚本本轮重跑的输出。
- `previous_package/`：旧17文件审计包原样保留；这是历史版本，不覆盖。
- `basis/rc3/`：从已有rc3包复制的规范原件，标明实施目标；不是实施证明。
- `old_integrity.json`：旧包CRC和16条SHA-256检查结果。
- `SHA256SUMS`、`verify_package.py`：本补充包完整性检查。

**本包未完成：**真实checkout原生运行、完整解释器/GPU入口运行、421项独立逐条核验、29提交逐项审计、新封存盲测及新硬件验证。
