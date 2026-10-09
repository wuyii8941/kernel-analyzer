# 外部审计（基线 1aee15e）修复版交付

审计材料：[received/dsl_v2_audit_recheck_1aee15e_20261009](../received/dsl_v2_audit_recheck_1aee15e_20261009/START_HERE.md)
（任务书 CODEX_NEXT_TASK.md）。本页按任务书第 6 节交付：每项的修复前失败测试、修复后测试、普通对照、实际调用路径、原始日志、
输入 / 预期 / 实际、SHA 与环境，另附三张账。结果只按三种状态写（成立 / 不成立 / 无法判断），证据级别分开写：

- **规则级**：单个求值函数在构造值上运行；
- **完整解释器级**：编译出的 kernel（`triton.compile`，不启动）在合成捕获上由完整求值器求值，CPU；
- **GPU 端到端**：真实 sm_86 设备上 `measure.run` → `check.run` → 报告。

这不是七项之外的全面认证；审计包列出的其余未完成项（W0–W7、外部材料）见第 4、5 节，仍然开放。

## 0. 基线、提交与环境

| 项 | 值 |
| --- | --- |
| 审计基线 | `1aee15e9df7a45f434699b7016056e6650fdae99`（增量 13 的结果提交） |
| 修复的起点 | `bb0a61065186bb4e2328811295edf681ce19602f`（= 1aee15e + 增量 14、15 + 仓库整理；审计包到达前已完成，F01 的 CAS 部分已在增量 15 修复） |
| 修复提交 | `5fe8e5627388ee770c4cbb85aeda00003eee68c0`（F01–F07 的代码与测试）；`6f0ef38`（修复中另发现的 D3–D5）；本页、证据与重算的开发结果在其后的提交 |
| 环境 | Python 3.11.16、NumPy 2.4.6、torch 2.10.0+cu128、Triton 3.6.0、mpmath 1.3.0；NVIDIA RTX A6000（sm_86）×4，驱动 535.129.03；`/data1/tzh/envs/ka_main`；Linux 5.4.0-42 |
| 原始日志 | [evidence/](evidence/)（文件清单见第 6 节） |

修复前的测试在 bb0a610 的独立工作树上运行（把修复提交的测试文件复制进去，源码不变）；修复后在修复提交上运行。

## 1. 账一：七项发现逐项处理表

| 编号 | 复现（bb0a610 上） | 修复 | 修复前失败 → 修复后 | 证据级别 | 状态 |
| --- | --- | --- | --- | --- | --- |
| F01 整数精确域 | 成立：审计原生复核 1,748 例中 150 例错（全在 64 位）；仓库测试 11 条失败 | `_int_op` / `_cmpi` 的无符号运算在 uint64 上算；有符号除法族按 uint64 绝对值算 | 11 → 0；复核 150 → 0 | 规则级 + 完整解释器级 | 已修复 |
| F02 scaled dot | 成立：区间操作数只取下端；scale 的条件丢失 | 传完整上下界；scale 条件进输出；`scaled_upcast_fp8` 保留上端；单侧 scale 的 `tt.dot_scaled` 可解码 | 4 → 0 | 规则级 + 完整解释器级 | 已修复（设备未验证） |
| F03 两序一致不等于全部串行化 | 成立：三 program 非交换锁记为完整点值 3；16 program 锁与通用 scan 记无条件完整 | 证明状态轴「complete_under_premise」；≤ 4 个 program 时求值全部 program 顺序 | 4 → 0 | 规则级 + 完整解释器级 | 已修复为条件性诊断；无条件结论所需的程序证书未实现（欠账） |
| F04 数值相等当作复制 | 成立：值相等 / 全零 / 未记录的复制都升级为调用级；另发现原子更新读取目标未记为读取 | 删除两条判据；只认声明输入自身的未改写存储；逐 seed；原子读取记录 | 4 → 0（GPU） | GPU 端到端 + 规则级 | 已修复（保守：不升级） |
| F05 Gluon 统一入口 | 成立：`measure.run` 报 `KeyError: 'ttir'` | `launch_ir` 一处选择 IR，覆盖 / 原子识别 / 求值共用，报告 `ir_kind` | 1 → 0（GPU） | GPU 端到端 | 已修复 |
| F06 声明摘要 | 成立：alpha、units、resolution、M 改变时摘要不变（审计 4/4）；alpha 只用于类内 Holm，逐规则检验固定 0.05；δ、重复次数无声明字段 | 摘要覆盖整个展开声明 + 源码哈希；alpha / δ / 重复次数 / M 传入 `check.run` | 10 → 0 | 规则级（入口函数） | 已修复；自适应精度未实现（W2 / W6 欠账） |
| F07 统计准入 | 成立：未知有效性不阻断；有一个浮点原子即可平均任何差异；三种竞争原因码未被识别 | 结构化原因表；未知有效性停统计；元素级浮点原子标记 | 7 → 0 | 规则级 + 完整解释器级 | 已修复（GPU 上不运行竞争 kernel） |

修复前 / 后的测试计数指 `2_tool/tests/test_audit_findings.py` 中对应各项的测试。全文件 101 条：bb0a610 上 43 失败、58 通过（七项 41 条，
另 2 条是下文 D3、D4；[evidence/tests_before.log](evidence/tests_before.log)），修复后 101 通过（[evidence/tests_after.log](evidence/tests_after.log)）。

### F01 整数精确域

- **调用路径**：TTIR / TTGIR 的 `arith.divui` / `remui` / `ceildivui` / `shrui` / `maxui` / `minui` / `divsi` / `remsi` / `ceildivsi` /
  `floordivsi` / `cmpi` → `KernelReferenceEvaluator` 的逐操作分派 → `ttir_eval._int_op` / `_cmpi`。值、下标、地址、条件、掩码都走这两个函数
  （指针偏移由整数运算得到），所以一处修复覆盖全部用途。
- **缺陷**：`_unsigned(x, 64)` 得到 uint64 后又 `.astype(np.int64)`，最高位重新变成符号；64 位逻辑右移用了算术右移；有符号除法用
  `np.abs` 与取负，INT64_MIN 在宿主上溢出。
- **输入 / 预期 / 实际（修复前）**：`maxui(0, 2^63)` 预期 2^63，实际 0；`divui(1, 2^63)` 预期 0，实际 2^64 − 1；`cmpi ugt 2^63, 0` 预期 1，实际 0；
  完整解释器：uint64 存储上 `2^63 // 3` 预期 3074457345618258602，实际 15372286728091293013。
- **修复后**：审计原生复核 `run_recheck.py --native --expect fixed` 退出 0，1,748 例 0 错（[evidence/recheck_after.json](evidence/recheck_after.json)，修复前
  [evidence/recheck_before.json](evidence/recheck_before.json)）；仓库测试对 8 / 16 / 32 / 64 位、全部无符号与有符号除法族、移位、比较做参数化，
  边界含 2^53 ± 1、2^62、2^63、2^64 − 1、INT_MIN、INT_MIN + 1；非法输入（除数 0、INT_MIN / −1、移位量 ≥ 位宽）必须记未建立。
- **普通对照**：−9…9 与 0…11 的全部操作、全部位宽不变（`test_small_integer_ops_are_unchanged`）。

### F02 scaled dot

- **调用路径**：`tt.dot_scaled` → `_op_dot_scaled` → `_scaled_dot`；`ttng.tc_gen5_mma_scaled` → `_scaled_dot`（两层共用）；
  `amdg.scaled_upcast_fp8` → `_scaled_upcast`。
- **输入 / 预期 / 实际**：审计探针 a ∈ [1, 2]（1×32）乘全 1、scale 2^0：预期包住 [32, 64]，修复前 [31.999…, 32.000…]，修复后
  [31.999…, 64.000…]；只有 scale 带条件标记：修复前输出无条件，修复后有条件（无条件 scale 的对照仍无条件）。
- **完整解释器级**：`a = where(x / 3 * 3 == x, 2, 1).to(bf16)` 在实数中恒为 2，但比较在 x / 3 非浮点数时无法判定，参照取路径并 [1, 2]；
  `dot_scaled(a, "bf16", b, scale, "e4m3")` 必须包住独立的有理数结果。修复前这类单侧 scale 直接中止（未建立，没有假完整）；为了
  在解释器级检验区间端点，修复同时让单侧 scale 可以解码。把修复后的解码接上 **基线的** `_scaled_dot` 再跑，元素状态为完整而精确值落在
  包围之外（例：元素 (0, 0) 精确 1927.5625，包围 [1191.53, 1191.53]）——这正是审计说的「丢掉信息后报完整」
  （[evidence/f02_baseline_function.log](evidence/f02_baseline_function.log)）。
- **独立计算而非两层互比**：测试与精确有理数比较。增量 13 的 5 个官方 scaled-MMA kernel 用修复后的代码重算，结果与原记录完全相同
  （输入是点值，旧缺陷不触发；[evidence/inc13_scaled_mma_after.json](evidence/inc13_scaled_mma_after.json)）。设备验证（sm_100）：0。

### F03 两序一致不等于全部串行化（scan 一并处理）

- **调用路径**：`evaluate` → `_run_programs`（program 顺序）→ `_cas_reverse_order`（更多顺序）→ `_mark_premises` → `KernelReference.element_classes`
  / `under_premise` / `complete` → `check.run`（`ok` 排除、`complete_under_premise_fraction`、`proof_status`）→ `measure.run_level`
  （`complete_under_premise_rate`、失败类「unproven premise (conditional diagnosis)」）；捕获插件（`main_capture_plugin`、`cross_level_plugin`、
  `tutorial_capture`）把这类元素单列为「complete under premise」。scan：`_generic_scan` → `_check_scan_bracketing` 把前提记到整次启动。
- **修复内容**：
  1. 证明状态分三档：有证明（complete）/ 依赖未证明前提（complete_under_premise）/ 未建立。只在证据上核对过的前提——CAS 的串行化一致、
     scan 的两种括号一致——下成立的元素不进无条件完整率，也不进统计；读取或更新这类元素的后续启动继承前提（启动级，保守）。
  2. CAS 启动不超过 4 个 program 时求值全部 program 顺序，超过时仍是正序与逆序；所有求值顺序都建立且一致才保留值。
  3. 其余 `assumed:` 注记（掩码 lane 零填充——由 PTX 核对、锁定 lowering 的归约树、官方 lowering 的有符号整数 dot、编译器屏障插入、
     已返回启动的 poll 进展）归为 lowering 事实 / 可信顺序目标，仍算有证明；这个分类交审阅方核定（第 5 节）。
- **输入 / 预期 / 实际**：三 program 锁（x ← x + 1、2x、x + 1，x = 0）：六个顺序给 {2, 3, 4}；修复前记完整点值 3，修复后未建立。
  审计的「比较两序」探针在生产函数上：修复前状态 0，修复后未建立。16 program 锁的累加：值仍为 16，类别为 complete_under_premise（修复前
  complete_composed）。通用 scan（cummax 组合）：complete_under_premise；普通 cumsum（实数加法结合律是定理）仍为 complete_composed（对照）。
- **欠账**：program 顺序的穷举不覆盖一个 program 多次进入临界区的交错，超过 4 个 program 只有两序证据；无条件结论需要程序级证书或完整允许
  结果关系，本批未实现。

### F04 调用级来源

- **调用路径**：`check.run`（每个 seed）→ `torch_intermediates(launches, seq, inp, digests_before, input_ptrs)` → 输出的
  `depends_on_non_triton_intermediates` → `measure.run_level` 的 `reference_scope` / `complete_rate_call_level`。
- **修复内容**：删除「每个元素位模式都在输入中」（`[copy of inputs]` 标签）与「全零 = 精确常数」；一个被读取的浮点缓冲只有是声明输入
  张量自身的存储、且字节等于该输入启动前的字节时才算输入（同字节的另一块存储不算）；其余非 Triton 上游一律保持 kernel 级。原来只在
  第一个 seed 上判定，改为每个 seed 判定并取并集。修复中另发现：原子更新、CAS、poll 读取目标的捕获初值时没有记为读取，ATen 填充的
  累加器因此从来不被记为上游；现在记录。
- **GPU 端到端（`measure.run`）**：值相等的计算缓冲（x ∈ [1, 2)，`x + 2^-25` 舍回 x）、下溢成全零的缓冲（实数值非零）、ATen 布局复制、
  `torch.zeros` 的原子累加器：修复前四例都报调用级（值相等与复制的上游被标「[copy of inputs]」，全零缓冲被当作精确常数，累加器没有被记为上游）；修复后四例都是 kernel 级。
  普通 classic 对照仍是调用级（[evidence/e2e/](evidence/e2e/)）。审计的绑定探针三例在生产函数上都为 kernel 级
  （[evidence/binding_probes_after.json](evidence/binding_probes_after.json)）。
- **欠账**：真复制的升级需要生产者记录与元素映射（例如 ATen dispatch 轨迹），本批不做；所以真复制与「算术后恰好相等」两例都保守地不升级。

### F05 Gluon 统一入口

- **调用路径**：`measure.run` → `run_level` → `check.run`：`launch_ir`（TTIR，没有时 TTGIR）→ `kernel_coverage(parse_ttir(text))`、
  `_float_atomics(text)`、`evaluate_sequence`（同一个 `launch_ir`）→ 报告 `launches[].ir_kind`、`ir_kinds`、`ir_coverage_complete`
  （`ttir_coverage_complete` 只保留 TTIR 启动的值，TTGIR 启动为 null，不改名掩盖）。
- **GPU 端到端**：同一个仿射 kernel 的 Gluon 写法（`gl.BlockedLayout`，sm_86）经 `measure.run`：修复前 `status: error`、
  `KeyError: 'ttir'`；修复后 `status: ok`、完整率 1.0、`ir_kinds: ["ttgir"]`。classic 与浮点原子用例同样走通。

### F06 声明摘要与设置传递

- **修复内容**：`declaration_sha256` = 整个展开声明（除自身）规范 JSON 的 SHA-256：采样划分与 seed、alpha、因素与水平、分辨率、M、δ、
  重复次数、规则类与多重性、各策略文字、版本（torch / Triton / 设备 / 工具 / Python）、call / 输入生成器 / 规格 / 误差预算源码的内容哈希。
  输入采样改用 `sampling_seed_sha256`（与旧摘要同式），改变统计设置不改变样本。新增声明字段 `equivalence.rel`（δ）与 `repeats`。
  alpha、δ、重复次数、M 传入 `check.run`：逐规则检验在声明的 alpha 上（原来固定 0.05，声明的 alpha 只进类内 Holm），没有 δ 时结果写明
  「不作等价陈述」。分辨率：报告 `resolution_met`；自适应提高精度未实现，未达标不会写成达标。`bounded_route` 文字改为实际行为。
- **测试**：alpha、units、resolution、M、δ、重复次数、seed 起点各改一处，摘要都变；同一声明摘要相同；call 源码改动使摘要改变；改 alpha
  样本不变；入口把 alpha / δ / 重复次数 / M 交给 `check.run`（替身记录参数）。审计复核的四组对照 4/4。
- **欠账**：M 是声明值，不是工具证明；逐规则有界结果不合成全族保证；默认检测器（`check.run` 记录中）保持冻结校准，不随声明的 alpha 变；
  自适应精度（W2 / W6）。

### F07 统计准入

- **调用路径**：求值器的原因码 → `check.run` 每个 seed 的 `reasons` → `execution_findings`（`RACE_MARKERS` / `UNKNOWN_VALIDITY_MARKERS`，每条
  注明生产者函数）→ `_execution_status` → `statistics: withheld | within-input mean | per launch` → 报告与 `measure` 的 `execution`。
- **修复内容**：未知有效性（线程映射不可得、进展未证明）停统计；重复启动的差异只有全部落在「最后一次写入是浮点原子」的元素上才按
  输入内平均（元素级标记跨启动保持，普通 store 清除）；竞争原因码补上原来子串检查漏掉的「cross-program write race」「cross-program race
  on load」「conflicting lanes in one store」。
- **完整解释器 → 准入**：同一 program 先 store 后由别的元素位置 load、无屏障：去掉 TTGIR（线程映射不可得）时求值器给「execution validity」，
  保留 TTGIR 时给「execution race」，两者都使统计 withheld。kernel 只在合成捕获上求值，不在 GPU 上运行。
- **普通对照**：重复启动逐位一致、无发现 → per launch；只有浮点原子元素不同 → within-input mean；审计四例 4/4。

### 修复中另发现的问题（不在七项之内）

| 编号 | 问题 | 发现途径 | 处理 | 测试 |
| --- | --- | --- | --- | --- |
| D1 | 原子更新、CAS、poll 读取目标的捕获初值时没有记为读取：ATen 填充的累加器从不被记为上游（F04 同类） | F04 端到端用例 | `_note_read` | `test_upstream_values_equal_to_inputs_are_not_promoted_to_call_level[atomic_row_sum]` |
| D2 | `_execution_status` 的子串判定漏掉三个竞争原因码（「cross-program write race」「cross-program race on load」「conflicting lanes in one store」） | 按生产者核对原因码（F07） | `RACE_MARKERS` 按生产者列出 | `test_every_race_producer_withholds_statistics` |
| D3 | 用例超时在准备阶段之前武装，打断 `torch._dynamo` 的首次导入，半初始化模块破坏同进程后续水平 | 预算耗尽的结构化结果测试（任务书第 5 节） | 先导入再武装 | `test_heavy_imports_happen_before_the_case_timeout_is_armed`（新解释器子进程） |
| D4 | 超时在声明的调用中触发时被包装成「调用拒绝了声明的输入」（绑定失败），不是「over budget」 | D3 修复后同一测试 | `_Timeout` 改为 `BaseException` | `test_budget_exhaustion_gives_a_structured_result` |
| D5 | 回归脚本与 v3.1 的比较序列化整条记录，之后新增的字段（嵌套在规则记录里的 `mde_approximate` 等）使比较必然不等 | 重跑 v1.1 回归 | 只比较 v3.1 写过的字段（递归）；作业里的比较字段事后由作业自身输出重算 | 回归作业 `v31_comparison_note` |

## 2. 账二：测试汇总

回归 profile：环境 `/data1/tzh/envs/ka_main`（Python 3.11.16、torch 2.10.0+cu128、Triton 3.6.0、mpmath、gmpy2、python-flint），命令
`python -m pytest -q -p no:cacheprovider -rfEs 2_tool/tests`，GPU RTX A6000（sm_86，`CUDA_VISIBLE_DEVICES=1`），代码为提交 `6f0ef38`
（脚本 [evidence/scripts/run_full.sh](evidence/scripts/run_full.sh)，同时写触发轨迹供 W1 重建）。会话守卫（`2_tool/tests/conftest.py`）检查测试不改写
已跟踪文件，本次未触发。

| 运行 | 通过 | 失败 | 跳过 | 收集失败 | 日志 |
| --- | --- | --- | --- | --- | --- |
| 全量回归，修复后（46 个测试文件） | 1,536 | 0 | 0 | 0 | [evidence/full_tests.log](evidence/full_tests.log)（457 s） |
| `test_audit_findings.py`，bb0a610 源码 + 本批测试文件 | 58 | 43 | 0 | 0 | [evidence/tests_before.log](evidence/tests_before.log) |
| `test_audit_findings.py`，修复后 | 101 | 0 | 0 | 0 | [evidence/tests_after.log](evidence/tests_after.log) |
| 参照：整理前、增量 13 记录的全量回归（审计引用） | 2,948 | 0 | — | 15 | 增量 13 记录 |

关于审计说的「15 个旧缺依赖文件」：它们是研究阶段脚本的测试（导入 transformers 失败），在仓库整理（bb0a610）中随被删除的研究脚本一起删除，
可从标签 `pre-reorg-20261009` 取回；不是为了变绿而删的工具测试。核对见 [evidence/removed_tests_check.txt](evidence/removed_tests_check.txt)：整理时
共删除 387 个测试文件，15 个缺依赖文件都在其中；387 个文件中导入现存工具模块（check、measure、reference_eval）的 0 个、引用保留脚本的 0 个、
调用工具 API 的 0 个。所以测试数从 2,948 变为 1,435（整理后）再到 1,536（本批新增 101 条）。现在的 profile 没有收集失败，也没有靠跳过变绿
（跳过数 0）。

本批改动的已有测试（每一处都是因为旧期望编码了被审计的行为，或测试工具需要）：

| 文件 | 改动 | 原因 |
| --- | --- | --- |
| `test_upstream_sources.py` | 「纯数据搬运标为 [copy of inputs]」改为「未记录的复制不升级为声明输入」 | F04：旧期望正是审计否定的判据 |
| `test_repeat_launches.py` | 合成的原子行带上元素级「浮点原子写入」标记 | F07：输入内平均现在需要元素级证据 |
| `test_signatures_structural.py` | 测试工具 `_run` 增加 `full=True`（返回 KernelReference 与存储标识），断言不变 | 新测试需要读取元素类别 |

注册表重建后计数不变（支持 426、拒绝 93、声明前提 4、未建立 1）；W1 用本次全量回归的触发轨迹重建：430 / 430 条支持与声明前提条目都有
三类测试与触发证据（[1_experiments/dsl_v2/w1_contracts/coverage.json](../../1_experiments/dsl_v2/w1_contracts/coverage.json)）。这是执行方证据，
不是审阅方的独立核验（M04）。


## 3. 账三（a）：提交逐项表与影响

审计包写「29 个提交」，未给出基点；dsl-v2 相对 `main`（`4156422`）到 1aee15e 共 32 个提交，全部列出：
[commit_table.md](commit_table.md)（[commit_table.json](commit_table.json)：完整 SHA、父提交、改动文件、触及的测试与结果记录、diff 中触及的
七项发现相关函数；生成脚本 [commit_table.py](commit_table.py)）。七项发现的相关代码由下列提交引入或修改：

| 发现 | 引入 / 修改的提交（1aee15e 之前） | 更早的来源 |
| --- | --- | --- |
| F01 | `23939a6`（增量 3：`_int_op`、`_op_atomic_cas`）、`94f530f`（增量 4：`_unsigned`） | `_cmpi` 与无符号除法路径在 general-v3.1 中已存在 |
| F02 | `23939a6`（增量 3：`_op_dot_scaled`）、`681beda`（增量 12：`_scaled_upcast`）、`d863999`（增量 13：`_scaled_dot` 抽出共用） | — |
| F03 | `954e9af`（增量 7：`_cas_reverse_order`）、`126c398`（增量 8：`_check_scan_bracketing`） | — |
| F04 | `6718f00`、`3d94b9e`、`94f530f`（`torch_intermediates` / `run_level` 修改） | `[copy of inputs]` 与全零豁免来自 3.1（general-v3.1 冻结版） |
| F05 | `3be713e`、`cba4ee7`、`fffc0a1`（`evaluate_sequence`；增量 10 接入 TTGIR 时 `check.run` 未改） | `check.run` 读 `asm["ttir"]` 来自 3.x |
| F06 | `3d94b9e`（`expand` 增加 M 等字段，摘要范围未扩） | 摘要式来自 general-v3.1 的统一入口 |
| F07 | `6718f00`（增量 1 C 部分：`_execution_status`） | — |

结果影响（只重算受影响的新分支开发结果；冻结记录不动）：

| 结果 | 位置 | 受哪项影响 | 处理 | 结论 |
| --- | --- | --- | --- | --- |
| general-v3.1 冻结版（`d6e177e`）的结果 | `1_experiments/general_round_v3_1/results` | F04：4 个结果文件中有输出因「[copy of inputs]」被记为调用级（`demo/bsr_dense_mm.json`、`demo/bsr_softmax.json`、`fr_modeB_v3_1/embedding.json`、`fr_modeB_v3_1/gather_layout.json`）；F01 的无符号路径与 F06 的摘要范围在该版本中已存在 | 冻结，不重算 | 这 4 个输出的「调用级完整」应读作 kernel 级；其余结论无法判断是否受 F01 影响（未重算） |
| 结构验收 v1.1-rc1 运行 r20261008T2030 | `1_experiments/structure_acceptance_v1_1` | 用 general-v3.1；归档中没有「[copy of inputs]」标签 | 冻结，不重算；计分仍等 M10 | F04 不经标签影响该运行；F01 是否触发无法判断 |
| 增量 13 scaled-MMA 两层对照 | `1_experiments/dsl_v2/captures/inc13_scaled_mma_synthetic.json` | F02 | 用修复后的代码重算 | 与原记录完全相同（点值输入不触发） |
| 规则注册表 | `2_tool/data/rule_registry.json` | F03（CAS、scan 的契约文字） | 重建 | 计数不变：支持 426、拒绝 93、声明前提 4、未建立 1 |
| W1 契约 | `1_experiments/dsl_v2/w1_contracts` | 新测试与契约文字 | 用修复提交上的全量回归与触发轨迹重建 | 430 / 430 不变（支持与声明前提条目都有三类测试与触发证据）；契约中的触发计数与两类前提的文字更新 |
| 官方测试广泛捕获（增量 15，12,766 次启动） | `1_experiments/dsl_v2/captures/inc15_broad.jsonl.gz` | F01（64 位无符号）、F03（CAS 竞争、通用 scan 的启动改记「完整（前提下）」） | 用修复后的代码重跑同样 6 个官方测试文件（`captures/auditfix_broad.jsonl.gz`，替换 `inc15_broad`；逐启动对照 `captures/auditfix_compare_broad_vs_inc15.json`） | 12,766 次启动逐一匹配；状态改变的只有 709 次，全部是「完整 → 完整（未证明前提下）」：通用 scan 703 次（`test_scan2d` 的 cumprod / cummax / get_first_element / roll / linear_recurrence 等组合区域，cumsum 是可识别的求和，不受影响），CAS 竞争 6 次（`test_atomic_cas`）。其余状态（部分 118、中止 2、集合目标 260）与增量 15 相同；无条件完整率 98.05% → 92.50%（含集合目标）。官方测试本身的 3 个失败（sm_86 共享存储不足等）与增量 15 的运行日志相同（[evidence/inc15_broad_run.log](evidence/inc15_broad_run.log)、[auditfix_broad_run.log](evidence/auditfix_broad_run.log)）。捕获记录只存状态，不存参照值，所以 F01 对值的影响由逐签名测试证实，不由捕获比较 |
| v1.1 程序回归（增量 1，reg20261009T1051） | `1_experiments/dsl_v2/regression_v11` | F04（范围）、F07（准入）、F03 | 用修复后的代码重跑 28 个常规程序（`regression_v11/reg20261009T2319`，替换 reg20261009T1051）；4 个竞争程序不在 GPU 上重跑，旧作业文件随附 | 28/28 与 reg20261009T1051 相同：观察、命中、K−G 判定、执行状态（4 个原子程序仍为输入内平均，且差异只在浮点原子元素上）；预计命中 27/28（未命中 prog_29 与原运行相同），安全标准违反 0 |
| 其他捕获：增量 14 原子、增量 10 Gluon test_core、增量 11 / 12 两层对照、增量 4 / 7 教程 | `1_experiments/dsl_v2/captures` | F03、F01 | 本批未重算 | 影响方向已知：含 CAS 竞争或通用 scan 的启动从「完整」改记「完整（前提下）」；增量 7 的教程 05 原记录已写明「CAS 顺序前提下完整」 |


## 4. 账三（b）：W0–W7 差距表

按审计包 WORKPACKAGES.md 的八个工作包；「名字登记」「测试触发」「独立核验」「设备验证」分开报告，本批改变的只在「本批」一栏。

| 工作包 | 名字登记（W0） | 测试触发（W1，执行方） | 独立核验（审阅方） | 设备验证 | 本批 | 仍缺 |
| --- | --- | --- | --- | --- | --- | --- |
| W0 官方基线 | 源展开 / 构建注册各 1,281 个名字，READY_FOR_MANUAL_REVIEW，verified 路线 0 | — | 未做（M03） | — | 无变化 | 四条来源链、属性 / 类型 / 效应、差集与路线的人工核账 |
| W1 规则契约 | — | 430 / 430（本批后重建；执行方证据，不是独立核验） | 未做（M04） | — | F01–F04 涉及的条目增加了反例测试；CAS 与通用 scan 两类声明前提的契约文字改为证明状态轴 | 逐条独立审核；组合传播 |
| W2 通用求值 | — | 同上 | 未做 | — | F01（整数）、F02（区间端点）修复 | 自适应提高精度、按预算终止与分辨率的实现；结合性证书 |
| W3 前后端 | classic / Gluon、NVIDIA / AMD TTGIR 名字 | 同上 | 未做 | sm_86：classic、Gluon、浮点原子经 `measure.run` 端到端（GPU 测试）；sm_90 / sm_100 / gfx942 / gfx950：0 | F05 修复 | 其余目标的真实设备运行（M11） |
| W4 运行时 | — | 同上 | 未做 | 竞争 kernel 不在 GPU 上运行 | F03（证明状态轴、≤ 4 program 全顺序）、F04（来源）、F07（准入）修复为条件性诊断 | 程序证书 / 完整允许结果关系；作用域、事件、异步完成 |
| W5 格式 / 外部 | — | 同上 | 未做 | 0 | F02 修复（scaled dot、scaled upcast） | 其余官方需求族的实现与测试欠账 |
| W6 统计 / 质量 | — | — | 未做 | — | F06（完整声明摘要；alpha、δ、重复次数、M 进入统计层）、F07 | M 的总体证据；族级控制；自适应精度；sensitivity 的事前功效表述（§5 第 3 项） |
| W7 验收 / 发布 | — | — | — | — | 无变化 | 冻结、外部保留集、独立 oracle / 生成器、新盲测（M06–M10） |


## 5. 交审阅方与仍开放的事项

1. F03 的前提分类：哪些 `assumed:` 注记算「可信顺序目标 / lowering 事实」（仍计无条件完整），哪些算「未证明前提」（单列）。本批只把 CAS
   串行化一致与 scan 括号一致列为未证明前提（`ttir_eval.UNPROVEN_PREMISE_MARKERS`）。
2. F04 的保守处理使真复制也不升级调用级；是否接受以 ATen dispatch 轨迹作为生产者证据，属于后续设计。
3. 审计 §5.3：`sensitivity.py` 的 guaranteed_detectable_effect 含本次观察的平均区间宽度，不能读成事前功效保证——本批未改，保留为 W6 欠账。
4. 外部材料 M03–M12 不因本批改变（[3_audits/README.md](../README.md) 第 3 节）。

## 6. 证据文件

| 文件 | 内容 | 级别 |
| --- | --- | --- |
| [evidence/environment.txt](evidence/environment.txt) | 基线 SHA、工作区差异、Python / NumPy / torch / Triton / 设备 / 驱动版本（修复后同一环境） | — |
| [evidence/tests_before.log](evidence/tests_before.log) | bb0a610 工作树 + 修复提交的测试文件：`test_audit_findings.py` 的修复前结果 | 全部级别 |
| [evidence/tests_after.log](evidence/tests_after.log) | 同一文件在修复后的结果 | 全部级别 |
| [evidence/full_tests.log](evidence/full_tests.log) | 修复后全量回归（`2_tool/tests`，`-rfEs`） | 全部级别 |
| [evidence/recheck_before.json](evidence/recheck_before.json) / [recheck_after.json](evidence/recheck_after.json) | 审计复核脚本 `run_recheck.py --native` 在 bb0a610 与修复后的原生运行（整数 1,748 例、声明摘要 4 组、执行状态 4 例） | 规则级 |
| [evidence/probes_after.json](evidence/probes_after.json) | 审计 `run_probes.py` 的四个探针在修复后生产函数上的结果（脚本 [scripts/probes_native_fixed.py](evidence/scripts/probes_native_fixed.py)：去掉只认基线摘录的 AST 闸门） | 规则级 |
| [evidence/binding_probes_after.json](evidence/binding_probes_after.json) | 审计 `run_binding_probes.py` 的三个来源例在修复后生产函数上的结果 | 规则级 |
| [evidence/f02_baseline_function.log](evidence/f02_baseline_function.log) | 把基线 `_scaled_dot` 接进修复后的树跑 F02 测试：区间端点缺陷在完整解释器级可见 | 完整解释器级 |
| [evidence/inc13_scaled_mma_after.json](evidence/inc13_scaled_mma_after.json) | 增量 13 的 5 个 scaled-MMA kernel 用修复后代码重算（与原记录相同） | 完整解释器级 |
| [evidence/e2e/before/](evidence/e2e/before/) / [e2e/after/](evidence/e2e/after/) | `measure.run` 报告：classic、Gluon、浮点原子、值相等上游、全零上游、真复制；修复前 / 后 | GPU 端到端 |
| [evidence/removed_tests_check.txt](evidence/removed_tests_check.txt) | 整理中随研究脚本删除的 387 个测试文件（含 15 个缺依赖文件）都不触及现存工具代码的核对 | — |
| [evidence/inc15_broad_run.log](evidence/inc15_broad_run.log) / [auditfix_broad_run.log](evidence/auditfix_broad_run.log) | 广泛捕获修复前后两次运行的官方测试结果行 | — |
| [evidence/regression_run.log](evidence/regression_run.log) | v1.1 回归重跑的逐程序退出记录 | GPU 端到端 |
| [evidence/scripts/](evidence/scripts/) | 生成以上证据的脚本（全量回归、重跑广泛捕获、重跑 v1.1 回归与事后重算 v3.1 比较、端到端报告） | — |
| [commit_table.md](commit_table.md) / [commit_table.json](commit_table.json) | 32 个提交逐项表 | — |

