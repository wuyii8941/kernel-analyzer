# DSL v2@1aee15e 审计材料

先读 `AUDIT.md`，再将 `CODEX_ACTIONS.md` 交执行方。

这不是工具的新版本，不包含答案文件，不修改生产仓库或旧成绩。

快速运行：

```bash
python repro/run_probes.py
python repro/run_binding_probes.py
```

上面执行的是当前源码摘录与最小CPU夹具。没有GPU，没有整包生产导入，`_scaled_dot`默认使用独立精确小例子算术后端。见日志中的scope字段。

有实际checkout和完整依赖时：

```bash
python repro/run_probes.py --repo /path/to/kernel-analyzer --native
python repro/run_binding_probes.py --repo /path/to/kernel-analyzer
```

`--repo`比较可执行AST，防止摘录漂移。结果写入本审计目录，不写入目标仓库。
`--native`本次未执行。请勿将文档中开发结果的计数写成审阅方重跑通过。
