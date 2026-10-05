# 工具改动（2026-10-06，计划 WP1 工程冻结的第一部分）

## 1. 一个引擎，两种模式

- `src/kernel_analyzer/check.py`：原 `scripts/tool_spec_check.py` 的引擎（捕获、组合参照、来源检查、逐视图坐标框架、
  特殊值、判定、报告）移入包内。`scripts/tool_spec_check.py` 只剩用例组的命令行。
- **模式 A**：用例不给规格（`spec` 返回 None，或绑定文件不定义 `spec`）时，只报 e_num = K − K_R，报告写
  `"mode": "A"`、`"task_semantics": "not checked (no specification)"`；特殊值类别在 K 与 K_R 之间比较
  （`k_vs_kr_class_mismatch`），逐列剖面改为 e_num（`num_profile_last_axis`）。
- **模式 B**：照旧报 e_num、e_sem 与 e_total（三者一起），报告多一个 `"mode": "B"` 字段，其余不变。
- 同一个调用在两种模式下的 e_num 判定相同（`examples/bindings/softmax.py` 与 `softmax_no_spec.py` 实测）。

## 2. 命令行

```bash
kernel-analyzer check BINDING.py --out DIR --seeds 96            # 模式 A / B 由绑定文件是否给 spec 决定
kernel-analyzer detect -- --binding b.py --units 128 --out r.json  # 冻结的盲测检测协议（scripts/run_detection.py）
kernel-analyzer replay OLD.json ... --binding BINDING.py --seeds N # 重跑并列出判定相关的差异
```

绑定文件定义 `make_inputs(seed)`、`run(inputs) -> {输出名: 张量}`，可选 `spec(inputs)`、`setup()`、`NAME`、
`IMPLEMENTATION`、`SPECIFICATION`；也可以直接定义 `CASES`（`Case` 对象列表）。示例在 `examples/bindings/`。
`detect` 保留盲测时冻结的协议实现（逐 launch、程序子集、K − RN(K_R) 两种定义），不并入新引擎，以免改变已提交的
判定矩阵。

## 3. 回归比较

- `kernel_analyzer.check.compare_reports`：两份报告的判定（每条规则与检测器）、参照三类比例、特殊值计数必须相同，
  相对大小在 1e-6 以内。
- **浮点原子操作**：报告的每个 launch 记 `float_atomics`（TTIR 含 `tt.atomic_rmw fadd`）。这种 kernel 的 K 取决于原子
  更新的运行时次序，同一份代码跑两次 e_num 就不同（`sc1_mean_pool_one_graph` 实测：旧代码两次运行之间 e_num 相对 RMS
  1.24e-7 → 1.17e-7，R5 判定翻转）；K_R 与 e_sem 不受影响。比较时这类 e_num 差异列为"容许"，不算回归。
- `scripts/run_regression_sample.sh CODE_ROOT OUT`：代表性样本（`scripts/data/regression_sample.txt`，15 个用例：
  FlexAttention、Inductor（B010）、FlashAttention、mamba、FLA、Triton 教程、vLLM（B013、LoRA 指针表、top-p）、Rprop
  补偿例、PyG scatter（B016）、OpInfo（B015）），6 个种子；`scripts/regression_compare.py OLD NEW`。
- **本次重构的结果**：旧代码（git worktree，提交 3de2e98）与新代码，15 个用例 0 处变化（其中 1 个仅有容许的原子次序
  差异）。报告在 `results/regression/pre_refactor/`、`post_refactor/`。

## 4. 其他

- 每个用例开始时 `torch._dynamo.reset()`（B017：同进程的用例会复用别的算子的图）。
- `tt.scan` 的其他组合器（乘积、max/min）在数值差异模式下按组合区域顺序折叠，实数下精确；新增
  `tl.cumprod` 的单元测试（包含零因子）。此前 masked.prod 反向的 36 个输出因此判为"program 中止"。
