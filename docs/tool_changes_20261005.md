# 工具改动（2026-10-05）：对规格检查入口与参照求值器的四处精度修正

## 1. 对规格检查入口 `scripts/tool_spec_check.py`

把 blind_test_v2 阶段 2 的做法（e_sem = K_R − f 进入统一判定层）做成通用入口：

- 一个用例 = kernel 调用（在 `TritonLaunchRecorder` 下运行）+ 规格 f（被实现或被替换函数的文档语义，
  在同一输入上求值）；
- K_R 由 `evaluate_sequence` 对该调用的全部 launch 组合求值；e_num = K − K_R 与 e_sem = K_R − f
  都以有向区间进入 `analysis.assess_units`（R1/R2/R3/R5、端点保守推断、检测器 2.1）；种子默认
  0–31 开发、32–95 确认；
- 规格两种写法：严格的 float64 区间包围（RoPE 用例），或 float64 求值加声明界 2⁻⁴⁰·max|f|（筛查用；
  确认一个发现时再换严格包围）；
- 报告里写明：TTIR 覆盖、参照完整比例、未建立原因、中止的程序及原因、非 Triton 写出的输出、
  在最后一次 Triton 写入之后又被非 Triton 操作改过的输出（例如 FLA 反向的 `dk.add_(dk2)`），
  后两类不参与判断。

用例分组：`tool_spec_cases_{liger,flex,inductor,tridao,fla}.py`。

## 2. 参照求值器 `reference_eval/ttir_eval.py` 的四处修正

每处都有对应测试：修正前失败、修正后通过（`tests/test_reference_eval_ttir.py`，52 项全过）。

| 修正 | 起因 | 规则 | 测试 |
|---|---|---|---|
| 已建立的精确 0 吸收未定义整数（`muli`、`andi`） | FlexAttention 的块稀疏遍历：越界预取的 next_block（掩码读取、无 `other`）只以 `jump * needs_jump` 进入指针跳转，needs_jump = 0 时与它无关；原规则把未定义一路传到全部输出 | 任一操作数是已建立的 0 时结果为已建立的 0（整数 x·0 = x&0 = 0 对任何 x 成立；浮点不适用，0·NaN ≠ 0） | `masked_prefetch_jump` |
| 同一个 SSA 值的自比较只由 NaN 决定 | Inductor 的 `maximum` / `clamp_min` 写成 `where((a > b) \| (a != a), a, b)`；原规则把 `a != a` 的两侧当成区间里的两个独立数，判不出，路径合并使参照宽到 0.2（`F.normalize`） | 两侧是同一个值时：非 NaN 则 eq/ge/le 为真、ne/gt/lt 为假；NaN 时按有序/无序谓词 | `nan_propagating_clamp_div` |
| bool 存储可经 int8 指针读写 | Inductor 把 bool 输出转成 int8 写（交叉熵反向），原规则拒绝这种重解释 | i8 与 i1（torch.bool）互访；写入存 0/1，读出未决的 bool 记为未建立 | （由 Inductor 交叉熵反向用例覆盖） |
| 自定义组合函数的 scan | Inductor 的 logcumsumexp 用带组合区域的 `tt.scan`，原来只支持求和 | 按扫描顺序依次折叠组合区域；Triton 要求组合函数可结合，实数下与树形顺序结果相同，每个前缀的区间包住其实数值 | `log_cumsum_exp` |

另有一处正确性修正（不只是精度）：

- **中止程序之后的组合求值。** 一个程序中止时（例如遇到没有声明语义的内联汇编），原规则只把它中止前
  已写的元素标为未建立；它本该写、却没来得及写的元素保留写之前的参照值和"已建立"状态，下一个 launch
  读到时会当作有效值。FLA 的 gate kernel（`tt.elementwise_inline_asm`）就触发了这个问题：下游输出被标成
  "完整"，K_R 却错了（e_num 相对 RMS ≈ 1）。新规则：有程序中止时，凡是本次 launch 中实际字节变了、
  参照却没写的元素，一律标为未建立。测试 `test_values_changed_by_an_aborted_program_are_not_established_downstream`
  修正前失败、修正后通过（53 项全过）。此前的单 launch 结果不受影响；多 launch 且有中止程序的组合结果需重跑。

另有两处小修正：

- `arith.constant true/false` 在 MLIR 里不打印类型，解析器补成 i1（原来导致 IndexError）；
- `tt.dot` 不打印 `inputPrecision` 时默认是 IEEE（与 `emulate.py` 一致）；原来把它标成 tf32，只影响
  原因标签，不影响参照值。

## 3. 第二轮改动（同日）

| 改动 | 起因 | 内容 | 测试 |
|---|---|---|---|
| 掩码通道补 0 的声明假设 | flash-attn / mamba 归一化反向、FlexAttention 反向读取不带 `other` 的掩码通道，TTIR 里是未定义值 | `masked_fill_zero` 选项：只有在 `ptx_zero_fills` 核实该调用每个 launch 的 PTX 都先把目标寄存器置 0 后，入口才自动启用；报告记 `masked_lane_assumption` | `test_zero_fill_assumption_is_checked_on_ptx_and_defines_masked_lanes` |
| 内联 PTX 片段解释器 | FLA 的 softplus 用多条谓词化 PTX（`setp` / `mov` / `mul` / `ex2.approx` / `add` / `lg2.approx`） | `parse_ptx_program` 解析直线型 f32 子集；算术取精确实数，`.approx` 函数取精确函数（近似误差归 K − K_R），谓词执行写成 select；f32 立即数按 f32 舍入 | `test_straight_line_ptx_snippet_has_exact_lanewise_semantics` |
| 多操作数 scan | Inductor 的 cummax（值 + 下标） | 走通用折叠 | 由 Inductor 用例覆盖 |
| 依赖非 Triton 中间值的输出单列为"混合" | FLA 反向的 dg：先由 torch 对分块部分和归约，再进入后续 Triton kernel；组合求值把这个捕获值当成精确输入，dg 的 K_R 带上了上游 TF32 误差（e_sem 1.7e-3），而 dq/dk/dv/dbeta 是 1.3e-8 | 评估器记录每个 launch 实际读了哪些存储、哪些读到的是参照从未写过的初始值；入口按内容（不只按地址，地址会被缓存分配器复用）追踪来源，读到的初始值既不是用例输入、也不是全零常量时，记为非 Triton 中间值，下游输出的 e_sem 标为混合 | FLA 反向用例：dg 标为混合，其余 4 个输出不标 |
| 被改过、未被写、写它的程序中止的输出分开报告 | FLA 的 `dk.add_(dk2)`、libdevice `erfinv` 等 | 报告分别列出 `outputs_modified_after_last_triton_write`、`outputs_not_written_by_triton`、`outputs_whose_writing_programs_aborted` | — |

FLA 用例另把反向里的两次 torch 原地加（`dk.add_(dk2)`、`dg.add_(dg2)`）换成逐位相同的 Triton 加法 kernel
（与 Liger dW 累加的做法相同），使整个反向成为 Triton launch 链；dg 仍因上游的 torch 归约而为混合。

另一处正确性修正：**经未建立的地址写入。** Inductor 的 `max(dim)` 反向先由一个带下标的归约 kernel 算出下标
（其组合函数工具不认识，程序中止，下标被正确标为未建立），再按下标 scatter 梯度。原规则对地址未建立的通道
要么跳过写入，要么按一个无意义的偏移写，目标缓冲区里真正被写的位置保留之前已建立的 0，于是参照错得很大
却标为完整（e_num 相对 29）。新规则：一个 launch 里只要有通道经未建立的地址写某个缓冲区，launch 结束时该
缓冲区全部元素标为未建立（程序之间没有次序，之后的确定写入也不能恢复）。测试
`test_store_through_a_not_established_address_invalidates_the_target` 修正前失败、修正后通过（56 项全过）。
只影响含中止程序、未定义值或竞争且其结果被用作地址的用例；这些用例在正式结果中按新版本重跑。

评估器测试 56 项全过。

## 5. 第三轮改动：特殊值类别、区间证实的语义偏差

**特殊值类别比较。** 入口对每个输出比较 f、K_R、K 的特殊值类别（NaN / +inf / −inf / 有限），类别不一致的元素数
单独报告（`special_values`）。实际值从解码器的状态数组还原（解码器把特殊值存成 0 加状态码，之前直接比较会把
NaN 输出误判成"写后被改"）。

**区间证实（汇总层，只用于 e_sem）。** 统计规则（R1/R2/R3/R5）与默认检测器检验的是系统偏差：投影均值或方向。
一个算法错误若让偏差的符号随输入变化（例如注意力丢掉了几个 key，缺的是随机 V 的贡献），投影均值约为 0，规则
不会确认。但 e_sem 的区间很紧（K_R 与 f 都是精确或近精确的区间），只要某个坐标的区间 K_R − f 不含 0，
就已证明 kernel 的实数语义在那里与 f 不同（f 的声明误差界已在 f 的区间里），不需要统计。于是汇总时，
e_sem 只要"规则或检测器确认"或"有坐标被区间证实"，就算检出；量级仍按相对 RMS 分档（常数取整级 < 1e-7、
小、候选 ≥ 1e-5）。e_num 不用这条：舍入使 K ≠ K_R 在几乎所有坐标上成立。

对以往普查的影响（重算 `results/tool_spec/census` 的汇总）：20 个输出改档。17 个从"未检出"变为常数取整级或
"小"档（fp32 常数，如 FLA recurrent 的 log2(e)、优化器与归一化反向里的 fp32 系数，相对 1e-11–2e-7）；
2 个 Triton 教程反向输出（已知的 B011）与 1 个 `lgamma` 反向转为"混合"；1 个转为候选：
`ind_sort_values_bwd` 的 dx（相对 RMS 1.4e-3），只在约 157 万个坐标里的 2 个上被证实，是一对梯度互换——随机 fp32
输入里偶有相等值，`torch.sort` 默认不稳定，并列元素的次序本身未定义，属规格歧义，不是缺陷。vLLM
`unified_attention` 的 15 个用例里，B013 的三个用例（相对 RMS 0.905、0.115、0.216）都归入候选，其中两个只靠
区间证实检出；对照用例为 0。

## 6. 第四轮改动：总误差与"被舍入抵消"的语义偏差；视图级统计

**总误差 K − f。** 每个输出另记 e_num 与 e_sem 中点之和的相对 RMS（`total`）。K_R 按参照值（实数语义）判定离散
分支（`test_branch_is_decided_by_the_reference_value` 规定的设计），所以当程序的某个判断只靠运行时舍入才成立时，
K_R 与设备走不同分支：e_sem 与 e_num 都很大、符号相反，K 本身却等于 f。实例：同一进程里先后编译两个 Rprop 优化器，
Dynamo 把变化了的 `etaminus` 变成 f64 标量参数，而另一处仍是编译期写死的 f32 常数；kernel 里 `sign == etaminus`
比较的是 fp32(0.3) 与 truncf(0.3)，执行时相等，按实数语义不等（`opt_rprop_etas` 的 prev：e_sem 0.56、e_num 0.56、
总误差 0）。汇总把"e_sem 检出、相对 RMS ≥ 1e-5、但总误差 < 1e-5 且不到 e_sem 的十分之一"归为
"compensated (rounding-dependent decision)"，不作为候选。

**视图级统计。** 输出是共享缓冲区的视图时（AOTAutograd 把几个梯度作为一块缓冲区的视图返回），特殊值类别和参照
完整度只统计本视图的元素（之前按整块缓冲区统计，别的视图的元素在本视图的 f 里是空位，被误计为 NaN 不一致，
完整度也偏低）。0 维输出的剖面按单列处理。

## 4. 已知局限

- 掩码读取不给 `other` 的值在 TTIR 语义里是未定义的。FlexAttention 反向在 Q_LEN 不是块大小整数倍时，
  dv 依赖这样一个值（越界行的 LSE 进入 exp2 后与 dO 相乘），工具判为参照未建立。PTX 显示 Triton 3.6
  的 NVIDIA 后端在这些读取前把寄存器置 0，所以这台机器上结果正确；这是对实现行为的依赖，不是工具错误。
- deepseek_v4 这类含离散选择的模型，一 ulp 扰动的条件数底本身约为 1，普查判不了。
