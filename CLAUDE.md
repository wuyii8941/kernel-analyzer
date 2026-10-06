# 工作约定

- 研究口径先读 `docs/stage_summary_20261002.md`；与它冲突的旧结论以它为准。
- 所有读写都限定在 `/data1/tzh/kernel-analyzer` 内。任何情况下不在 `/home` 盘存读写：
  不写 Claude 记忆目录，不用 `~` 下的缓存。
- 运行 Python/pip/Triton 前把缓存和临时目录指到仓库内：

  ```bash
  export HOME=/data1/tzh/kernel-analyzer/.cache XDG_CACHE_HOME=/data1/tzh/kernel-analyzer/.cache \
         TRITON_CACHE_DIR=/data1/tzh/kernel-analyzer/.cache/triton \
         PIP_CACHE_DIR=/data1/tzh/kernel-analyzer/.cache/pip TMPDIR=/data1/tzh/kernel-analyzer/.cache/tmp \
         PYTHONPATH=src
  ```

- 系统 `python` 是 2.7，不可用。主线环境是 `/data1/tzh/envs/ka_main`（torch 2.10.0+cu128、
  triton 3.6.0、gmpy2、python-flint）。`liger` 等其他环境保留作测试与旧结果复现，
  清单见 `docs/environments.md`。旧测试套件仍在 `liger` 环境中跑。
- 资源规则见 `docs/resource_policy.md`。
- 按 `docs/next_phase_plan_20261006.md`（A 发布收尾、B 重要性标定、C 定向搜索、D 写作；M1–M4 的记录见
  `docs/evaluation_plan_20261006.md`）推进，不在每一步之后征求确认；只有遇到无法自行解决的
  问题才停下来询问。成绩只按三种状态写（`docs/status_ledger_20261006.md`）。盲测 v1、v2 已揭盲，只作回归集。
- 自动参照 K_R 的代码在 `src/kernel_analyzer/reference_eval/`（解析、映射、区间求值、捕获、重放），
  说明与结果见 `docs/auto_reference_results_20261002.md`。需要 Liger/torchao/transformers 的捕获
  在 `liger` 环境跑（只依赖 `reference_eval.capture`），分析统一在 `ka_main` 跑。
- 映射表 `ttir_mapping.py` 只对 `results/reference_eval/ttir_op_registry.json` 记录的 Triton 3.6.0
  构建成立；换版本先重跑 `scripts/enumerate_ttir_registry.py` 与枚举测试。
- 核内定位用 `reference_eval/emulate.py`（逐位模拟 + 逐节点精确替换）；下降规则同样只对锁定的 3.6.0 /
  sm_86 成立，换版本或换卡要先在 `.cache/version_ptx/captures` 这类捕获上重跑 `verify` 全部逐位复现。
- 新的测量统一走 `scripts/run_reference_analysis.py`（捕获包 + 声明 → 参照、残差、三类、统计），声明放在
  `results/reference_eval/declarations/`；不要再为单个案例复制统计脚本。
