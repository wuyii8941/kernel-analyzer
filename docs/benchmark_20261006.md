# Benchmark（计划第 4 节）：单位格式与第一批内容

## 单位

一个测试单位 = **计算 / 编译配置 + 输入或状态分布 + 参照定义 + 测量点 + 待检验命题 + 独立答案依据**。不给 kernel
贴永久的"有 / 无 bias"标签：同一 kernel 换分布、换测量点答案就会变。清单是 JSON（`results/benchmark/*.json`），
由 `scripts/benchmark_build.py` 生成，`scripts/benchmark_score.py` 对照存档的工具报告逐条检查命题。

| 字段 | 内容 |
|---|---|
| `config` | 用例组、用例名、环境（如 vLLM 修复版的 `PYTHONPATH`）、运行时补丁（`KA_PRELOAD`） |
| `reference` | 模式与规格的等级：严格包围（`scripts/strict_specs.py`）或筛查级（fp64 eager + 2⁻⁴⁰ 预算） |
| `measurement` | 输出名 |
| `proposition` | 允许的 e_sem 分档、特殊值不一致的下限、e_num 的上下限 |
| `answer_basis` | 答案从哪来：问题报告的证据表、独立复现脚本、运行前提交的预测——不来自被测工具自己 |

## 三部分

| 部分 | 用途 | 现有内容 |
|---|---|---|
| 有解析答案的构造任务 | 校准误报、检出能力、区间处理 | `results/reference_eval/detector_calibration_v2.json`；求值器的 8 项反例测试与乘积扫描等单元测试（`tests/test_reference_eval_*.py`） |
| 真实库捕获 + 独立高精度答案 | 实际接入、数值参照、统计判断 | blind_test_v1（审阅方的 C++/MPFR 独立核验约 520 万坐标）；两条深案例的严格包围（B012：MPFR，B016：精确有理数） |
| 真实问题的修复前后 | 发现、定位、修复验证 | `results/benchmark/real_bugs_pre_post.json`：38 个单位（B012 7 个条件 × 前后；B013 3 个阳性 + 3 个对照 × 前后；B015；B016 两种机制 × 补丁前后 + 对照） |

## 第三部分的当前成绩

`python scripts/benchmark_build.py && python scripts/benchmark_score.py results/benchmark/real_bugs_pre_post.json`：
**38 / 38 通过**。说明两点：

- 第一版清单里 B013 的两个单位标错了阳性与对照（凭记忆写的：`ua_bidir_sw8_qpkv16` 其实是 BLOCK_Q = 1 的对照，
  `ua_perseq_causal_sw8` 才是阳性），评分脚本把它们判为失败；改为按 B013 报告的证据表取标签后通过。答案依据必须来自
  报告，不能来自记忆或工具输出。
- 这一部分的单位都是开发过程中用过的案例，是回归与修复验证的证据，**不是盲测成绩**；泛化成绩只来自另留的新组合
  （揭盲评分与外部接入试验）。

组织方式参考 FPBench / FPCore（表达式、精度、前置条件、规格、误差度量），不另造语言。
