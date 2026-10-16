# 规格：现状、已知问题与待决事项（2026-10-15）

## 1 已有的证据

| 证据 | 结果 | 出处 |
|---|---|---|
| 参照测试 | 222 项通过：关系实例 9、舍入 oracle 4、phase1 105、phase2 89、phase3 15 | `run_checks.py` |
| 派生记录复现 | 关系检查、目录计数、随机验证、phase1 独立复算四份记录，重算后与提交的一致 | `run_checks.py` |
| phase1 独立复算 | 交叉熵用 mpmath 300 位独立求值；池化与 index/scatter 用另写的蛮力实现；0 处不一致 | `specification_v2/algorithm_specs/phase1/README.md` |
| 关系检查 | 70 项（成立 55、反例 15），26 条关系各 200 组随机输入（含 4 条阴性对照） | `specification_v2/SPECIFICATION_v2.md` §5、§8 |
| ONNX Runtime 对照 | 31,133 次比较。定义相同的运算上没有发现规格错误；发现 ORT 自身问题 3 个（其中 softcap 与掩码的次序已经 onnx#7867 修复） | `evidence/ort_witness/REPORT.md`（本次未重跑） |
| v1 错误的修正 | 8 个角色已改，各有复核值。SGD 与 BN 的 v1 行为已重建，由参数含义推出的关系能抓到 | `SPECIFICATION_v2.md` §3；`evidence/historical_misreadings/` |
| 合同内容审阅 | CE、AdamW 对照 PyTorch 2.10 文档逐条核过；22 个例子直接调用参照，58 个标量通过 | `reviews/2026-10-15/REVIEW.md` |
| Liger CE 对照 | 400 组配置上与 CE 规格一致；全部忽略时约定不同：规格为无定义，Liger 返回 0 | `evidence/declared_semantics/` |

这些证据说明参照代码与它们各自引用的文档、与独立实现大体一致。它们不说明参照已可用于裁决（见 2.1）。

## 2 已知问题

### 2.1 合同层（2026-10-16 规格 v3 第 1 阶段之后）

10-15 审阅的 F1–F5 与 10-16 审阅第 5 项已处理（计划见 `PLAN_v3.md`）：

| | 处理 |
|---|---|
| 检查器 | 新的唯一检查器 `tools/check_contracts.py`：期望值用 AST 解析，不用 eval；精确值与严格包围按 PLAN_v3 §3 比较，不设隐藏容差；每个例子、读法、变异要求都参与结论；矛盾、结构错误、实现错误退出 1，未建立退出 2 |
| 注入失败 | `tests/test_checker.py` 的 35 项测试，含注入 A–I：错误期望值、不存在的来源、子进程 exit 9、倒置区间、缺输出 / 状态 / 行、错误读法、未审阅或未绑定或哈希过期的自动启用、只靠异常「杀死」的变异、隐藏容差。全部按预期失败 |
| 适配器 | `adapters/CE.py`、`ADAMW.py`、`RANKMEAN.py`：输入先校验（不截断计数、不丢多余项）；CE 返回 loss 与对显式上游的 VJP；ADAMW 读入前态、返回 theta / m / v / t；参照调用 v2.4 代码，误读由独立实现给出 |
| 结果 | 29 个例子全部一致（含 7 个非法输入 / 未支持例）；20 条变异要求全部在指定输出上被拒绝 |
| 来源 | 三个来源的原文快照存于 `sources/`，SHA-256 记在 `SOURCES.json`；作者已逐条对照，未发现与条款冲突 |
| 启用 | 唯一入口 `kernel_spec/registry.py`：只读 `contracts/*.yaml`；要求已审阅、`contracts/LOCK.json` 中的哈希与当前文件一致、来源有快照、需任务选择的 profile 已显式绑定。CE 的非任务 profile 与 ADAMW 现可启用；CE.R_A 与 RANKMEAN 需每次绑定 |

被取代的文件：`contracts/draft/` 的草案检查器与适配器保留作历史；`tools/validate_contracts.py`、`tools/run_review_checks.py`、
`tests/test_contract_validator.py` 已由上面的检查器与测试取代（其中的结构检查已移到 `kernel_spec/schema.py`，原测试逐项移植）。

### 2.2 来源

- **凭记忆写、原文未核对的条目**：
  - CosineAnnealingLR；
  - HF warmup + cosine；
  - GradScaler 的默认值与跳步规则；
  - checkpoint 的重算约定；
  - `torch.var` 的 correction；
  - `torch.matmul` 的广播；
  - Embedding 的 padding_idx 与 max_norm。

  清单在 `specification_v2/algorithm_specs/phase2/contract_v2_all_families.md` 末尾。
- **合同的两个 PyTorch 2.10 页面**：通过 web 读过，但没有留存原文快照与哈希（`SOURCES.json` 的 verification 字段）。
- **未进合同的 28 个角色**：来源栏多数只写「文档」，没有钉版本（参照文件开头有 URL 与抓取日期）。

### 2.3 记账

- **phase2 的哈希清单**：`specification_v2/algorithm_specs/phase2/SHA256SUMS` 没有随 v2.4 的修改更新，8 个文件校验失败。
- **顶层哈希清单**：`specification_v2/SHA256SUMS` 把自身列了进去，校验必然失败。
- 上面两份清单 `run_checks.py` 都不校验；要修，需要审阅方重新签发。
- **区间实现有三套**：`spec_cross_entropy.py` 内的 Decimal 区间、`rigorous.py`、`rigorous_mp.py`。审阅意见是先统一结果接口，不急于合并。
- **phase2 的 README**：仍是第二阶段 v2.0 的说明，文件内容已是 v2.4。

### 2.4 与 Triton 审查的衔接

- **审查入口不用合同**：`audit(spec=...)` 收一个返回 float64 的函数，用对齐度比较，没有用到合同、保证标签和三值比较。flash attention 案例的 spec 是 `fa_inputs.spec_attention`，这是用 torch 写的实现参照。
- **kernel 与合同怎么绑定，还没有方案**：
  - kernel 常融合多个运算，比如 flash attention 含 matmul、softmax、matmul；
  - kernel 的接口与合同不同，比如 Liger 的 fused linear CE 原地写梯度、不产出 logits；
  - kernel 有角色没定义的输出，比如 flash attention 的 M。
- **很多角色单个 kernel 绑定不到**：31 个角色里 7 个是训练程序级，6 个只在 torch.compile 下才成为 Triton kernel（`roles.md`）。
- **直接调用 phase1 CE 规格的地方**：`2_tool/scripts/essential/common.py` 和 `2_tool/src/kernel_analyzer/contract_v3.py`，路径已改到本目录。
- 审查接入规格的计划是 `2_tool/docs/audit_status.md` 不足 1 的第 2 项。

### 2.5 理论部分的撤回

- `THEORY.md` 中三处已撤回，反例见 `evidence/theory_checks/checks.py`：
  - 定理 4 的「五类互斥」；
  - 定理 5 第 2 条；
  - 定理 6 的「可见集合包含」。
- 不再追求的目标（`METHOD.md`）：唯一刻画、格子表、优越性、按 kernel 库列分母。
- `THEORY.md` 与 `SPECIFICATION_v2.md` 原样保留作记录，其中涉及上述内容的段落不再成立。

## 3 待决事项

| 事项 | 谁决定 | 出处 |
|---|---|---|
| 合同实施 P0–P2 由谁做（交付包写的是 Codex） | 你 | `reviews/2026-10-15/CODEX_TASK.md` |
| RANKMEAN 是否用有效 token 等权；CE 的加权 label smoothing 是否用 R_A | 任务负责人，按任务绑定 | 两份合同的 decisions |
| 范围：要审查哪些训练配置和 Triton kernel，由此决定哪些角色先迁到合同 | 你 | `reviews/2026-10-15/input_package/NEXT_STEPS.md` 第 1 步 |
| 其余 28 个角色的迁移顺序 | 你 | 同上，第 4–6 步 |
| 讨论中、未采纳的方向：算子语义层由 PyTorch 官方分解生成 f，标为实现参照；手写合同作为高价值算子的独立第二路线。可行性实验未做 | 你 | 本次讨论（见下） |

关于最后一项，torch 2.10（ka_main）中的实测：
- `torch._decomp.decomposition_table` 有 1124 条。
- 有分解的：log_softmax、nll_loss、layer_norm、rms_norm、softmax、silu、gelu、embedding、addmm、cumsum、var、SDPA。
- mm 这类本身是 core ATen 算子，语义要自己定。
- `_fused_adamw_` 没有分解。

## 4 版本沿革

| 时间 | 版本 | 内容 | 现在在哪 |
|---|---|---|---|
| 10-07 | 第一阶段 v0.4 | CE、池化、index/scatter、累积；12 条歧义经审阅决定 | `specification_v2/algorithm_specs/phase1`，与旧副本逐字相同 |
| 10-08 | 第二阶段 v2.0 | 第一档 15 个家族 | 旧副本；v2.4 的 phase2 是它的修订 |
| 至 10-15 | v2.1–v2.4 | 31 个角色、phase3、关系检查、THEORY、8 处修正 | `specification_v2/`。仓库里没有 v2.1–v2.3 的记录，v2.4 随 10-15 的规格包入库 |
| 10-15 | 合同层草案 | 三份试点合同、适配器、检查器 | `contracts/draft/` |
| 10-15 | 审阅与交付 | 方法通过；合同补全为 schema 1.1；检查器问题 F1–F5 | `contracts/`、`tools/`、`tests/`、`reviews/2026-10-15/` |

旧副本 `1_experiments/specs`（第一阶段 v0.4 加第二阶段 v2.0）已移入本目录并换成 v2.4，git 历史保留（`git log --follow`）。
- **与 v2.4 的差别**：只在 phase2 的 8 个文件，包括 SGD 的 maximize 次序、BN 计数器、注意力窗口、MoE 联合枚举、RMSprop 区间的平方、长度校验。
- **旧副本仍带着 v1 错误**：它们的修正在仓库外完成，此前没有回到仓库。
- **冻结记录里的引用**：`1_experiments/closure_round_v2`、`general_round_v3_1` 中的 `specs/phase2/... v0.1` 指那份旧副本，可用 `git show ff5e449:1_experiments/specs/phase2/<文件>` 取回。
