# Kernel Analyzer

测量 Triton kernel 的数值实现差异有没有系统性（平均）作用：从 kernel 的 TTIR 自动生成实数语义 G 的严格包围，
再用固定的方向规则与端点保守统计判断 e_num = K − G 的平均作用是否非零。

**当前版本、入口、能力范围与状态只在 [`CURRENT.json`](CURRENT.json) 维护**（当前冻结版本 `general-v3.1`，工具 3.1）。
本页不复制结论；研究历史与早期成绩见 [docs/history/README_before_20261008.md](docs/history/README_before_20261008.md)，
文档导航见 [docs/README.md](docs/README.md)。

## 安装

需要 CUDA GPU、PyTorch 2.10 与 Triton 3.6.0（TTIR 映射锁定在这个构建上）。在仓库内安装：

```bash
pip install -e ".[statistics,reference]"
```

目前请在仓库内运行（editable 安装或 `PYTHONPATH=src`）：统一入口还依赖 `scripts/essential/contract_v3.py`，非 editable 的 wheel
里没有它（已登记为待修的打包问题）。

## 推荐用法：一份声明

```bash
python scripts/measure.py --declaration examples/general/bsr_softmax.json --out report.json
```

声明只写四样：合法调用、每个输入的来源、比较方式、预算。

```json
{"call": "bsr_ops.py:bsr_softmax",
 "inputs": {"x": {"state": "bsr_ops.py:block_sparse", "shape": [64, 64], "dtype": "float32"}, "block": {"const": 16}},
 "compare": {"mode": "A", "measure": ["values"]},
 "budget": {"cpu_seconds": 1200, "gpu_seconds": 1200, "case_timeout": 900, "max_units": 96}}
```

工具自动展开其余部分：两类方向规则（固定均值 R1 / R5，对齐 R2 / R3，类内 Holm）、32 个开发 + 64 个确认单位、版本与因素水平、
允许的参照类别。缺项时逐条报出，不猜输入域。报告给出每个输出的完整参照率（调用级与 kernel 级）、宽度相对 ulp 的分布、
平均作用的判定（「非零（方向）/ 未确认 / 无法判断」），以及失败的五类归因。「未确认」不等于「没有 bias」。

## 能做与不能做

- **能**：单个 Triton kernel、输出由 Triton 写出、操作在规则注册表内（[注册表报告](docs/general/rule_registry_report.md)）；
  模式 A 测数值差异的平均作用；有任务规格时模式 B 另报 e_sem = G − f。
- **不能**：块指针与 TMA 描述符、`tl.cat`、histogram、atomic_cas、未登记的 libdevice / inline asm（明确拒绝）；cuBLAS / ATen
  写出的输出（没有参照）；保存值、反向与优化器写入作为一个自动调用图。
- 检测能力有限且已校准：256 维、64 个确认单位时，坐标均值 0.02 个噪声标准差的作用约 60% 检出，0.005 检不出
  （[校准表](docs/general/calibration_table.md)）。

## 三种常见的「未建立」

| 原因 | 含义 | 怎么办 |
| --- | --- | --- |
| 语义缺失 | kernel 用到注册表外的操作或归约合并函数 | 看注册表报告；不会被猜测接受 |
| 绑定问题 | 输出由非 Triton 操作写出、配对不可证，或调用拒绝了声明的输入 | 检查声明的调用与输入域 |
| 包围太宽 / 统计不足 | 参照分辨不了声明的效应尺度，或样本撑不起均值判断（零方差、n < 16、偏度 > 2 且 n ≤ 64） | 增加单位或缩小输入尺度；结论记「无法判断」 |

## 测试

```bash
PYTHONPATH=src python -m pytest -q tests
```

测试不得改写任何已跟踪文件（`tests/conftest.py` 的会话守卫会使测试失败）；输出一律写到 `tmp_path`。
