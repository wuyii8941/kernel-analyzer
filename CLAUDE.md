# 工作约定

- 当前版本、入口、状态与待决事项只看根目录的 `CURRENT.json`（唯一当前索引）。仓库分四部分：`1_experiments`（完整实验）、
  `2_tool`（现有工具）、`3_audits`（完整审计）、`4_bugs_cases`（已找 bug 与重要案例），每部分只保留最新、最全的版本；
  2026-10-09 整理前的内容从标签 `pre-reorg-20261009` 取回，不再放回工作树。新结果放进对应部分并替换旧版本，不新增并列的历史副本。
- 所有读写都限定在 `/data1/tzh/kernel-analyzer` 内。任何情况下不在 `/home` 盘存读写：不写 Claude 记忆目录，不用 `~` 下的缓存。
- 运行 Python / pip / Triton 前把缓存和临时目录指到仓库内：

  ```bash
  export HOME=/data1/tzh/kernel-analyzer/.cache XDG_CACHE_HOME=/data1/tzh/kernel-analyzer/.cache \
         TRITON_CACHE_DIR=/data1/tzh/kernel-analyzer/.cache/triton \
         PIP_CACHE_DIR=/data1/tzh/kernel-analyzer/.cache/pip TMPDIR=/data1/tzh/kernel-analyzer/.cache/tmp \
         PYTHONPATH=2_tool/src
  ```

- 系统 `python` 是 2.7，不可用。主线环境 `/data1/tzh/envs/ka_main`（torch 2.10.0+cu128、triton 3.6.0、gmpy2、python-flint）；
  官方主线 Triton（e50b186e）在 `/data1/tzh/envs/triton_main`。清单见 `2_tool/docs/environments.md`，资源规则见
  `2_tool/docs/resource_policy.md`。
- 工作规则：独立规格以 `1_experiments/specs/` 最新入库的审阅版本为准，执行方不改规格语义，规格问题交审阅方；冻结的协议、答案、
  原始结果与 tag 不覆盖、不重算后冒充原数据；不在每一步之后征求确认，只有遇到无法自行解决的问题才停下来询问；成绩只按三种状态写
  （成立 / 不成立 / 无法判断）；盲测 v1、v2 已揭盲，只作回归集；测试不得改写已跟踪文件（`2_tool/tests/conftest.py` 守卫），
  输出写到 tmp_path；上游 issue 只准备草稿，由用户本人提交；公开推送前先经用户确认。
- 自动参照 K_R 的代码在 `2_tool/src/kernel_analyzer/reference_eval/`（解析、映射、区间求值、捕获、核内定位）。映射表
  `ttir_mapping.py` 只对 `2_tool/src/kernel_analyzer/reference_eval/data/ttir_op_registry.json` 记录的 Triton 3.6.0 构建成立；
  换版本先重跑 `2_tool/scripts/enumerate_ttir_registry.py` 与枚举测试。核内定位（`reference_eval/emulate.py`）的下降规则同样只对
  锁定的 3.6.0 / sm_86 成立。
- 新的测量走统一入口 `2_tool/scripts/measure.py --declaration D.json`（`kernel_analyzer.measure`）；其他环境捕获的包走
  `2_tool/scripts/run_reference_analysis.py`。不要为单个案例复制统计脚本或 p 值、投影、来源判断。
- 外部审计（`3_audits/README.md`）中未关闭的发现限制工具结论的可信范围；修复先写修复前失败的测试，再修，再交对照。
