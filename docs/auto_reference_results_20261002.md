# 自动参照 K_R：第 1–6 步实施结果

2026-10-02。对应 [阶段性总结](stage_summary_20261002.md) 第 11 节的六步。代码在
`src/kernel_analyzer/reference_eval/`，机器结果在 `results/reference_eval/`。

**当前定位**：具备通用解释核心、经多类真实 kernel 验证的自动参照工具，是原型里程碑。参照
由通用解释器按 TTIR 组合生成，不按算子名查手写公式；从捕获到 bias 判定的统一入口见文末。
审阅中发现的四处会破坏严格包围或分类的实现错误已修复并进入生产求值器的回归测试（见
「审阅后修复」）；「支持范围内始终给出严格包围」以这些回归测试和下列限制为前提。

## 版本锁定

| 项 | 值 |
|---|---|
| Triton | 3.6.0（wheel，无 git 提交号）；ka_main 的 libtriton sha256 `77cde538…`，liger 环境（py3.10 构建）`8a84317b…`，每个捕获包各自记录 |
| torch | 2.10.0+cu128 |
| 目标 | NVIDIA RTX A6000，sm_86，ptxas 12.8.93 |
| 默认编译选项 | `enable_fp_fusion=True`、`enable_reflect_ftz=True`、`default_dot_input_precision=tf32`，全表见 `results/reference_eval/version_lock.json` |
| 提取阶段 | 实际那次编译的 `asm['ttir']` |
| 参照模式 | 缺省数值差异模式；舍入核验模式作对照 |

## 第 1 步：参照求值器

`numbers.py`（精确有理数端点、MPFR 定向包围、IEEE 就近舍偶）、`program.py`、`evaluator.py`
（局部、组合、导数三种执行）。第 11 节的 8 项反例测试全部通过，另有随机抽样的包含性测试：
12 种初等函数在随机区间与内点上由 2048 位真值核对，exp→sum→divide 链的区间包住真值且宽度
小于 1e-60。测试：`tests/test_reference_eval_counterexamples.py`、`tests/test_reference_eval_enclosure.py`。

## 第 2 步：TTIR 解析与映射

**注册表**由锁定构建枚举：对 libtriton 中所有 `dialect.op` 字符串在加载 Triton 方言的上下文里做
generic 形式解析，MLIR 报 unregistered 的不是操作。结果 tt 50、arith 51、math 45、scf 12、
cf 4、gpu 66、ub 1，共 229 条（`results/reference_eval/ttir_op_registry.json`）。映射表
`ttir_mapping.py` 对每一条给出 SUPPORTED（内部参照运算与类别 A–I）或 REJECTED（理由）：
128 条支持、101 条拒绝，无遗漏、无多余条目，由枚举测试保证。拒绝的主要是块指针、TMA 描述符、
`tt.cat`（元素次序未声明）、`atomic_cas`、microscaling 与 gpu 方言中 TTIR 阶段不产生的操作。

**解析器** `ttir_parser.py` 按行解析 Triton 打印的 TTIR（区域栈、按词法作用域的值名、`%x:2`/`%x#1`
多结果、`scf.for/while` 头部声明的块参数）。语料 65 份 TTIR（28 个不同 kernel 名：自写练习、
Inductor、Liger、torchao）全部解析，指令处理覆盖全部完整。这是**静态覆盖**：语料采集没有保存操作数，
这 65 份没有逐一求值；实际求过值的 kernel 另列在「覆盖的两层」。

**求值**在 `ttir_eval.py`：float64 区间端点；加减乘除与开方用 TwoSum、Dekker TwoProduct 判定舍入
方向，精确结果宽度 0，否则只在正确一侧放宽 1 ulp；初等函数逐元素调用 MPFR 53 位定向舍入；
归约与点积用 γ_n·Σ|x| 先验界。参照内存、分支按参照值判定、未决 `scf.if` 两支取并集、跨 program
读写冲突判为参照未建立、atomic 返回值在被使用时判为参照未建立。每个输出元素归入完整组合参照、
条件局部参照、参照未建立三类。

**验证**（`results/reference_eval/validation/exercise_kernels.json`、`tests/test_reference_eval_ttir.py`）：

| 检查 | 结果 |
|---|---|
| 舍入核验模式，只含 IEEE 正确舍入运算的内核（缩放、四项顺序和、fp16/bf16/fp8 转换、存储读取链、位级、while 循环、带掩码复制） | 全部逐位复现 K |
| 同一 `p=a·b, y=p+c` 内核，`enable_fp_fusion=False` / `True` | 关闭时逐位复现，开启时出现差异：差异来自 fma 合并 |
| 数值差异模式参照包住独立精确值 | 四项和、ieee matmul 抽样元素（有理数）、exp-sum-divide（400 位 MPFR）全部包含 |
| 参照宽度与残差 | 同一 matmul：tf32 的 max\|K−K_R\| 0.022，ieee 5.1e-6，参照宽度 2.1e-13 |

舍入核验模式下近似指令（`ex2.approx`、近似除法与 rsqrt、内联 PTX）和 fma 合并处出现差异，属于
K 的实现选择；归约、扫描、dot 与 atomic fadd 的累加次序未声明，该模式下判为参照未建立。

## 第 3 步：捕获与绑定

`capture.py` 包装 `CompiledKernel.run` 返回的启动器，JIT 内核与 Inductor 内核（关闭其静态
CUDA 启动器后）都经过这里。每次启动保存 TTIR/TTGIR/LLIR/PTX、cubin 哈希、编译元数据、grid、
全部实参及 constexpr 标记、操作数在启动前后的存储副本（可只取声明的窗口），并把编译产物一起
存入捕获包；重放直接加载保存的二进制，不重新编译。

| 内核 | 重放两次 |
|---|---|
| softmax（同进程、从捕获包在新进程加载） | 逐位相同 |
| torchao AdamW8bit 三次 step、Liger 交叉熵 64 次与 RMSNorm 前后向、7 个 Inductor 内核 | 全部逐位相同 |
| 跨 program 的 `atomic_add` | 两次重放之间不同：并发次序是需要抽样的因素，报告据此标注 |

## 第 4 步：自动 K_R 对照手工参照（Liger FP32 dW 累加）

**位置**：Liger 0.7 fused linear CE 中 `grad_weight += torch.mm(grad_logits_chunk.t(), x_chunk).float()`。
该原位 FP32 加法换成执行同一运算的 Triton 内核 `accumulate`；每个状态都核对了 loss、隐藏层梯度
与权重梯度和未修改的 torch 路径逐位相等，即 K 不变。Qwen3-1.7B FP32，长度 64，64 个 chunk，
原实验的 32 状态库与抽样（种子 20260919，32 次开发、64 次确认，共 96 次抽取），original 与
reverse 两种次序。坐标集 D 为事前以种子 20261002 抽取的 256 行 lm_head（524,288 个坐标，
占 3.1 亿的 0.17%），每行一个 program，捕获只取 D 上的窗口。

**K_R**：64 次启动经同一参照内存串联的组合参照；每个 G_c 在两次启动之间由 cuBLAS 写入，作为外部
输入进入，累加器从未在启动之外被改写。**手工参照**：同一批 G_c 在 D 上的 FP64 顺序累加，误差界
γ₆₃·Σ|G_c|。

| | original | reverse |
|---|---:|---:|
| 抽取 × 坐标 | 50,331,648 | 50,331,648 |
| 完整组合参照 | 全部 | 全部 |
| 手工 FP64 落在参照区间（按手工误差界放宽）内 | 全部，0 个不一致 | 全部，0 个不一致 |
| 参照区间最大宽度 | 8.9e-16 | 6.1e-16 |
| K − K_R 区间含 0 的坐标 | 2 | 3 |
| K − K_R 正 / 负 | 25,181,625 / 25,150,021 | 25,169,204 / 25,162,441 |
| 每状态每次序的求值时间 | 34.8 s | 34.6 s |

两种次序的组合参照在全部抽取上相交（精确和与次序无关）。结论：自动 K_R 与手工参照在双方误差
预算内相容（不是逐位相同），可以进入第 5 步。结论范围：96 次是从 32 状态库有放回抽取（32 开发、
64 确认），不是 96 个独立状态；分析对象是换成逐位等价 Triton `accumulate` 之后的累加区域；坐标为
256 行、全体的 0.17%。机器结果：`results/reference_eval/liger_fp32_order_auto_reference.json`。

## 第 5 步：受控重放与端点保守检验

u = write(K) − write(RN32(K_R))，write 为原实验同一设置的一步真实 torch AdamW（lr 1e-4，
betas (0.9, 0.95)，eps 1e-8，零动量，单张量路径，GPU）在 D 上的参数写入。RN32(K_R) 在全部坐标上
无歧义（RN32(lo)=RN32(hi)），故 l=h。固定方向规则：32 次开发抽取的平均 u 归一化后冻结，在 64 次
确认抽取上打分；对齐规则：w = r/‖r‖，r 为参照写入。区间为 95% t 区间；六个事前比较做 Holm 校正。

| 比较 | 规则 | 平均投影 | 95% 区间 | 正/负 | Holm 后判定 |
|---|---|---:|---|---|---|
| original − reverse（原实验比较，限于 D） | 固定方向 | 8.76e-9 | [6.61e-9, 1.09e-8] | 47/17 | 检出平均方向作用（正） |
| original − K_R | 固定方向 | 8.52e-9 | [6.43e-9, 1.06e-8] | 43/21 | 检出平均方向作用（正） |
| reverse − K_R | 固定方向 | 6.89e-9 | [5.14e-9, 8.63e-9] | 51/13 | 检出平均方向作用（正） |
| reverse − K_R | 对齐 | −4.37e-11 | [−5.86e-11, −2.88e-11] | 19/45 | 检出对齐效应（负） |
| original − K_R | 对齐 | −1.93e-11 | [−4.28e-11, 4.07e-12] | 33/31 | 未确认 |
| original − reverse | 对齐 | 2.43e-11 | [−3.85e-12, 5.25e-11] | 39/25 | 未确认 |

**与已知结果对照**：原实验在全部坐标上得 original − reverse 固定方向平均投影 2.20e-7，95% 区间
[1.68e-7, 2.71e-7]，44/20。D 上同一比较方向相同、区间不含 0。量级比约 0.040，与 √(|D|/N)=0.041
相符，即这一作用在坐标上是弥散的。

**新信息**：以精确和为参照，两种次序沿**各自开发样本学得的方向**都有正的平均投影，即各自的误差
模式在确认样本上持续；两者不是沿同一个预先命名的方向。reverse 次序另有沿参照更新方向少推的对齐
效应，original 次序该效应未确认。u 是 write(K) − write(RN₃₂(K_R))，不是把无限精度梯度直接写入
optimizer；区间是 t 区间加 Holm 校正，不因参照严格就成为严格的总体保证。平均每个状态 D 上 original 有 4103 个坐标的写入与
参照写入不同，reverse 为 2191 个。

推断范围：声明的 32 状态库内有放回抽样；D 为 0.17% 的坐标；只回答存在性，不推出训练 loss 后果。

## 第 6 步：负对照与泛化

`results/reference_eval/step6_negative_controls.json`：

| 结构 | 结果 |
|---|---|
| 归约漏一项（`row_sum_drops_last`） | 无规格：K−K_R 只有舍入量级（最大 3.9e-6），缺项不可见，报告写明任务语义未检验。以源计算 `row_sum` 作 f：K_R − f 在 64/64 行检出，且区间恰好包含 −x_last |
| exp → sum → divide 三种写法 | 三者参照由同一求值器生成，无专属 reference，相对宽度 ≤ 2.3e-13。e·(1/s) 与 e/s 的参照相交；exp2(x·log₂e) 写法与 exp 写法的声明语义差 994/1000 为正：舍入后的 log₂e 常数略小（第 5 节 exp 下降例子的同一项） |
| 计算 → 保存 → 读取 → 继续计算 | 组合参照保留上游差异（1000/1000 完整）；把读取固定为捕获值后 1000/1000 降为条件局部参照，残差被抹掉 |

另测的陌生组合（均无专属 reference）：

| 内核 | 规模 | 结果 |
|---|---|---|
| Liger `liger_cross_entropy_kernel`（online softmax、vocab 循环、ignore_index 早退 `cf.cond_br`、原位写梯度） | 64 次启动，2,048,000 个 dlogits | 全部完整组合参照；手工 FP64 公式 (softmax−onehot)/N 全部落在区间内；K−K_R 正 1,515,774 / 负 500,180 / 含 0 32,046，40 s |
| torchao AdamW8bit 融合内核（428 条 op：反量化、动量与参数更新、块 absmax、二分比较重新量化） | 3 次 step × 256 program | 参照已建立的 8-bit 码在 step 1、2 全部与 K 相同；step 0 两路各有 1 个码不同（参照的实数比较有定论、FP32 比较落在另一侧）；exp_avg_sq 每行 1 个码（0.39%）、exp_avg 0.14–0.17% 的码因 x/absmax 的区间依赖问题判为参照未建立；参数写入残差 1e-8 量级 |
| 7 个 Inductor 内核（chunk 求和、交叉熵前后向、bf16 RMSNorm、AdamW） | 全量 | 覆盖完整、逐位重放、参照全部完整组合 |

交叉熵 dlogits 残差约 75% 为正，这是符号比例，不是自然训练总体的均值 bias，也不是训练损害。其来源
由下面的干预确定。

**exp / 除法干预**（`results/reference_eval/exp_intervention.json`）：同一批 logits 上替换 Liger
交叉熵 kernel 的实现，一次一处，残差统一相对原 kernel 的 K_R（实数 exp 与实数除法）：

| 变体 | 改动 | 正 / 负 | 正比例 | 平均残差 |
|---|---|---|---:|---:|
| original | 无（与未修改 kernel 逐位相同） | 1,515,774 / 500,180 | 0.752 | 1.13e-14 |
| exp2_rn | `tl.exp(x)` → `exp2(x·RN₃₂(log₂e))` | 与 original 完全相同 | 0.752 | 1.13e-14 |
| exp2_up | 常数取 RN₃₂(log₂e) 的上一个 FP32 数 | 1,279,930 / 736,030 | 0.635 | 5.10e-15 |
| libdevice | `libdevice.exp` | 1,542,686 / 473,259 | 0.765 | 1.05e-14 |
| div_rn | 两处除法（÷d、÷N）改为正确舍入 `div_rn` | 1,080,618 / 935,322 | 0.536 | 4.28e-15 |
| libdevice_div_rn | exp 与除法都替换 | 1,061,507 / 954,427 | 0.527 | 4.14e-15 |

结论：exp2_rn 与 original 逐位相同，证实 `math.exp` 下降为 ex2.approx(x·RN₃₂(log₂e))；但常数舍入只
解释偏正的一小部分（改常数后 0.752→0.635 且未翻转，换 libdevice 不减）。主要来源是近似除法：
`arith.divf` 下降为 `div.full.f32`，只把两处除法改成正确舍入，正比例降到 0.536、平均残差降到原来的
38%。此前「exp 常数是主因」的解释被这组干预否定。两处除法各自的份额与剩余 0.53 的来源（行和 d 的
FP32 累加等）尚未拆分。

**导数参照**：TTIR 层前向自动微分（区间切向量；缺规则的运算中止该 program，不静默置零）。
x³ 内核上 J(x)u 精确、宽度 0（含 x=0），伴随残差恰为 0，中心差分残差恒为 −h²u³v；softmax 内核的
切向量包住 300 位解析导数 p⊙(u−⟨p,u⟩)。测试：`tests/test_reference_eval_derivative.py`。

**成本**：Liger 位置 524,288 坐标 × 64 次启动约 35 s；交叉熵 2.05M 元素 40 s；torchao 全量 256
program 数秒。瓶颈是逐 program 的 Python 开销，与元素数近似线性。

## 审阅后修复

两份审阅指出、并经代码核对成立的问题，以及核对中另发现的同类问题。全部在生产求值器
`ttir_eval.py` 上修复，回归测试在 `tests/test_reference_eval_ttir.py`：

| 问题 | 后果 | 修复与回归测试 |
|---|---|---|
| `_int_to_float` 先转 float64 再判 \|x\| ≤ 2⁵³ | sitofp(2⁵³+1) 得零宽 [2⁵³, 2⁵³]，再减 2⁵³ 得 [0,0]，真值 1，且标为完整 | 在整数上判断，超出 2⁵³ 时按转换值与原整数的大小向正确一侧放 1 ulp；测 sitofp(2⁵³+1) − 2⁵³ 的区间含 1 |
| `copysign` / `signbit` 用 `<0`、`>=0` 判符号 | copysign(1, −0.0) 得 [1,1] | 零端点读 IEEE 符号位，两端符号可能不同则不判定（参照未建立）；NaN 的符号不跟踪，判未建立；测 copysign(1, ±0)、signbit(±0) |
| `_maybe_round` 无舍入模型时只写原因 | 舍入核验模式下 f8E4M3FN 等格式仍列为完整 | 同时置未建立；用手写 TTIR 测 truncf 到 f8E4M3FN |
| 控制依赖不传给写入 | pinned load 决定分支、分支内写常数，被标为完整 | `ProgramState` 维护控制条件栈：`scf.if` 区域入栈，`cf.cond_br` 与 while 条件之后持续生效，`scf.for` 的边界作用于循环体与结果；store、atomic 取并集；测三种结构（scf.if、早退 return、循环上界） |
| 中止的 program 未写出的输出 | 整个 kernel 中止时比较报告为空，看似干净 | 未写入元素单列 `not_written`，报告写明中止 program 数与原因 |
| `math.round` 用 floor(\|x\|+0.5) | 0.49999999999999994 与 \|x\| ≥ 2⁵² 处错一 | 用 \|x\| − floor(\|x\|) 的精确差判定 |
| `pow` 次正规结果的上界 | 上界可能低于真值 | 向上放宽 |
| `divsi` 的 INT_MIN / −1 | 未定义结果被当作值 | 判未建立 |

**对已报告结果的影响**：这些 kernel 中的整数转浮点都是 i32；copysign 未出现；无舍入模型的分支只在舍入
核验模式或声明量化下触发；pinned load 只用在第 6 步，那里原有的值走算术链。修复后重跑的覆盖报告与
第 6 步结果，和此前数字一致。第 6 步第三项新增分支版本：不固定时 16/16 完整，固定读取值后 16/16
降为条件局部。

另：sm_86 上 fp8e4b15 的转换在 TTIR 中是 rtz 转 f16 加打包内联 PTX（`packed_element=4`），映射表拒绝，
program 中止，输出不写入；这是正确行为，但意味着该格式目前不在支持范围内。

## 覆盖的两层

`results/reference_eval/layered_report.json`。

**静态覆盖**（解析 + 指令处理）：65 份 TTIR、28 个 kernel 名（练习 42 份、Inductor 18、Liger 4、
torchao 1），全部完整。

| 方言 | 操作实例 | 不同操作 |
|---|---:|---:|
| tt | 1,632 | 25 |
| arith | 1,330 | 25 |
| math | 44 | 12 |
| scf | 38 | 5 |
| gpu | 8 | 1 |
| cf | 2 | 1 |

**求值覆盖**（实际捕获并求值的 kernel，数值差异模式）：

| 来源 | kernel（说明） | 已写元素 | 完整 | 条件局部 | 特殊值 | 未建立 | 中止 program | 求值时间 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| exercise | `scale_masked`（scale_masked） | 1,000 | 1,000 | 0 | 0 | 0 | 0 | 0.022 |
| exercise | `sequential_sum4`（sequential_sum4） | 1,000 | 1,000 | 0 | 0 | 0 | 0 | 0.01 |
| exercise | `add_then_mul`（add_then_mul） | 1,000 | 1,000 | 0 | 0 | 0 | 0 | 0.009 |
| exercise | `softmax_rows`（softmax_rows） | 800 | 800 | 0 | 0 | 0 | 0 | 0.041 |
| exercise | `layernorm_rows`（layernorm_rows） | 800 | 800 | 0 | 0 | 0 | 0 | 0.046 |
| exercise | `matmul`（matmul_tf32） | 2,048 | 2,048 | 0 | 0 | 0 | 0 | 0.049 |
| exercise | `matmul`（matmul_ieee） | 2,048 | 2,048 | 0 | 0 | 0 | 0 | 0.033 |
| exercise | `atomic_accumulate`（atomic_unused） | 4 | 4 | 0 | 0 | 0 | 0 | 0.011 |
| exercise | `atomic_accumulate`（atomic_used） | 1,004 | 4 | 0 | 0 | 1,000 | 0 | 0.011 |
| exercise | `branch_on_scalar`（branch_on_scalar） | 1,000 | 1,000 | 0 | 0 | 0 | 0 | 0.01 |
| exercise | `conversions`（conversions） | 4,000 | 4,000 | 0 | 0 | 0 | 0 | 0.034 |
| exercise | `elementary`（elementary） | 1,000 | 1,000 | 0 | 0 | 0 | 0 | 0.402 |
| exercise | `nan_rules`（nan_rules） | 1,000 | 857 | 0 | 143 | 0 | 0 | 0.007 |
| exercise | `scan_and_argmax`（scan_and_argmax） | 201 | 201 | 0 | 0 | 0 | 0 | 0.003 |
| exercise | `bit_level`（bit_level） | 1,000 | 1,000 | 0 | 0 | 0 | 0 | 0.006 |
| exercise | `inline_asm`（inline_asm） | 1,000 | 1,000 | 0 | 0 | 0 | 0 | 0.026 |
| exercise | `while_loop`（while_loop） | 1,000 | 1,000 | 0 | 0 | 0 | 0 | 0.017 |
| exercise | `store_load_chain`（store_load_chain） | 2,000 | 2,000 | 0 | 0 | 0 | 0 | 0.008 |
| exercise | `store_load_chain`（store_load_chain, load pinned） | 2,000 | 1,000 | 1,000 | 0 | 0 | 0 | 0.008 |
| exercise | `exp_sum_divide`（exp_sum_divide） | 200 | 200 | 0 | 0 | 0 | 0 | 0.005 |
| exercise | `branch_after_rounding`（2**24 + 1 then branch） | 16 | 16 | 0 | 0 | 0 | 0 | 0.003 |
| exercise | `masked_copy`（masked load without other） | 228 | 200 | 0 | 0 | 28 | 0 | 0.002 |
| inductor | `triton_poi_fused_add_0`（chunk_sum） | 262,144 | 262,144 | 0 | 0 | 0 | 0 | 1.741 |
| inductor | `triton_per_fused__log_softmax_prepare…`（cross_entropy） | 128 | 128 | 0 | 0 | 0 | 0 | 1.033 |
| inductor | `triton_per_fused__log_softmax_nll_los…`（cross_entropy） | 2 | 2 | 0 | 0 | 0 | 0 | 0.009 |
| inductor | `triton_per_fused__log_softmax__log_so…`（cross_entropy） | 64,000 | 64,000 | 0 | 0 | 0 | 0 | 1.001 |
| inductor | `triton_per_fused_add_mean_mul_pow_rsq…`（rmsnorm_bf16） | 32,768 | 32,768 | 0 | 0 | 0 | 0 | 0.408 |
| inductor | `triton_poi_fused_add_addcdiv_copy__di…`（adamw） | 24,576 | 24,576 | 0 | 0 | 0 | 0 | 0.456 |
| inductor | `triton_poi_fused_add_copy__1`（adamw） | 1 | 1 | 0 | 0 | 0 | 0 | 0.002 |
| liger | `_rms_norm_forward_kernel` | 8,224 | 8,224 | 0 | 0 | 0 | 0 | 0.181 |
| liger | `_rms_norm_backward_kernel` | 29,696 | 29,696 | 0 | 0 | 0 | 0 | 0.473 |
| liger | `liger_cross_entropy_kernel`（64 launches summed） | 2,048,063 | 2,048,063 | 0 | 0 | 0 | 0 | 49.11 |
| torchao | `triton_per_fused__to_copy_abs_add_ama…`（step 0） | 788,480 | 787,459 | 0 | 0 | 1,021 | 0 | 13.439 |
| torchao | `triton_per_fused__to_copy_abs_add_ama…`（step 1） | 788,480 | 787,010 | 0 | 0 | 1,470 | 0 | 13.821 |
| torchao | `triton_per_fused__to_copy_abs_add_ama…`（step 2） | 788,480 | 787,088 | 0 | 0 | 1,392 | 0 | 13.956 |
| liger_order | `accumulate (Liger FP32 dW chunk accum…`（96 draws (32-state bank) x 2 orders x 64 launches on 524,288 coordinates） | 100,663,296 | 100,663,296 | 0 | 0 | 0 | 0 | 6662.7 |

**规则触发**（非零项；参照内存 = load 读到本启动内写入的参照值 / 外部输入重新进入；路径 = 由参照值决定的分支 / 两支并集 / select 并集通道 / 中止；可观测性 = 固定为捕获值的读取通道；atomic = 返回值未建立通道 / 次序无关折叠通道）：

| kernel（说明） | 参照内存 | 路径 | 可观测性 | atomic | 冲突/未定义 |
|---|---|---|---|---|---|
| `scale_masked`（scale_masked） | — | — | — | — | masked_load_undefined_lanes=24 |
| `sequential_sum4`（sequential_sum4） | — | — | — | — | masked_load_undefined_lanes=96 |
| `add_then_mul`（add_then_mul） | — | — | — | — | masked_load_undefined_lanes=72 |
| `layernorm_rows`（layernorm_rows） | — | — | — | — | masked_load_undefined_lanes=448 |
| `atomic_accumulate`（atomic_unused） | — | — | — | folded_order_free_lanes=1,000 | — |
| `atomic_accumulate`（atomic_used） | — | — | — | return_value_not_established_lanes=1,000 | — |
| `branch_on_scalar`（branch_on_scalar） | — | branch_decided_by_reference=4 | — | — | — |
| `conversions`（conversions） | — | — | — | — | masked_load_undefined_lanes=24 |
| `nan_rules`（nan_rules） | — | — | — | — | masked_load_undefined_lanes=24 |
| `bit_level`（bit_level） | — | — | — | — | masked_load_undefined_lanes=24 |
| `inline_asm`（inline_asm） | — | — | — | — | masked_load_undefined_lanes=24 |
| `store_load_chain`（store_load_chain） | load_reads_reference_value=1,000 | — | — | — | masked_load_undefined_lanes=48 |
| `store_load_chain`（store_load_chain, load p） | — | — | pinned_load_lanes=1,000 | — | masked_load_undefined_lanes=48 |
| `masked_copy`（masked load without othe） | — | — | — | — | masked_load_undefined_lanes=28 |
| `liger_cross_entropy_kernel`（64 launches summed） | — | branch_decided_by_reference=64 | — | — | — |
| `triton_per_fused__to_copy_abs…`（step 0） | — | select_union_lanes=1,021 | — | — | — |
| `triton_per_fused__to_copy_abs…`（step 1） | — | select_union_lanes=1,470 | — | — | — |
| `triton_per_fused__to_copy_abs…`（step 2） | — | select_union_lanes=1,392 | — | — | — |
| `accumulate (Liger FP32 dW chu…`（96 draws (32-state bank)） | load_reads_reference_value=33,030,144, external_reentry=63 | — | — | — | — |

未列出的行四条规则均未触发。Liger 累加的类别来自 96 次抽取 × 2 种次序的逐单元记录，规则次数来自重新捕获的一个单元。

## 统一入口

`kernel_analyzer.reference_eval.analysis`（CLI `scripts/run_reference_analysis.py`）。输入是捕获包和一份
声明：位置、被测输出缓冲、坐标集（program 选择）、参照模式、测量方式（输出本身或一步 AdamW 写入）、
比较集合（候选对 K_R、候选对候选）、方向规则、开发/确认划分、α、五项因素的处理方式。两段执行：
参照段逐单元串联启动得组合参照、K 与 K_R、三类与残差，单元之间可并行；统计段做受控重放
u = measure(K) − measure(RN(K_R))、按声明的方向规则投影、端点保守 t 检验与 Holm 校正，输出报告必填项。
绑定（kernel 在哪里、怎样捕获）仍由人工提供，统计层不再有案例专用脚本。

验收：用 `results/reference_eval/declarations/liger_fp32_order.json` 从重新捕获开始重跑 Liger 第 4、5 步
（`scripts/run_unified_liger_rerun.sh`），与原案例专用脚本的结果比对：6 项统计（3 种比较 × 2 种方向规则）的平均投影、
t 区间、正负计数与 Holm 后判定，以及两种次序的参照汇总（三类、残差正负、最大宽度）全部逐项相同
（`results/reference_eval/liger_fp32_order_unified_check.json`）。重跑从重新捕获开始，96 次抽取 × 2 种
次序全部完整组合参照，无中止、累加器未在启动外被改写、RN₃₂(K_R) 无歧义元素。

## 限制

- float64 端点：精确值需要超过 53 位时宽度不再为 0（仍包围）；区间算术的依赖问题使少量离散判定
  成为参照未建立（torchao 码 0.14–0.39%）。
- 逐 program 实例执行；只抽样部分实例时，跨实例冲突只在已求值实例之间检查。
- 窗口化捕获要求连续张量；窗口外的访问判为参照未建立。
- 内核内部中间值不可观测，局部参照的最小单位是一次内核启动；跨启动用组合参照。核内的节点级定位改由逐位模拟完成（`emulate.py`，见工具检验第 5 节），不依赖观测中间值。
- 舍入核验模式下未声明次序的累加判为参照未建立；块指针、TMA、`tt.cat`、`atomic_cas` 拒绝。
- 第 4、5 步把 Liger 的 torch 原位加法换成逐位等价的 Triton 内核，以便进入 TTIR 流程。
- pinned load 读的是启动结束后的捕获值，对启动内先读后写的地址不是读取时刻的值；它只作负对照，结果
  标为条件局部。
- 控制依赖对 `cf.cond_br` 与 while 条件采取保守处理：一旦由条件局部值决定，此后该 program 的写入都标
  条件局部。
- 每份报告的必填项（版本、五项因素、覆盖与三类、宽度、剔除比例、统计设置、f 的说明）写在对应
  JSON 中。

## 文件

| 文件 | 内容 |
|---|---|
| `src/kernel_analyzer/reference_eval/` | numbers、program、evaluator（第 1 步）；ttir_parser、ttir_mapping、intervals、ttir_eval、capture |
| `scripts/enumerate_ttir_registry.py`、`record_version_lock.py`、`build_ttir_corpus.py` | 注册表、版本锁定、语料 |
| `scripts/validate_ttir_reference.py`、`validate_inductor_reference.py` | 第 2 步验证 |
| `scripts/capture_liger_order_accumulation.py`、`analyze_liger_order_auto_reference.py`、`run_liger_auto_reference_pipeline.sh`、`stats_liger_order_auto_reference.py` | 第 4、5 步 |
| `scripts/run_step6_negative_controls.py`、`capture_liger_kernels.py`、`analyze_liger_kernels.py`、`capture_torchao_adamw8bit.py`、`analyze_torchao_adamw8bit.py` | 第 6 步与泛化 |
| `src/kernel_analyzer/reference_eval/analysis.py`、`scripts/run_reference_analysis.py`、`results/reference_eval/declarations/` | 统一入口：捕获包 + 声明 → 参照、残差、三类、统计 |
| `scripts/build_layered_coverage_report.py` | 静态 / 求值覆盖与规则触发表 |
| `scripts/capture_exp_intervention.py`、`analyze_exp_intervention.py` | 交叉熵 kernel 的 exp / 除法干预 |
| `scripts/run_unified_liger_rerun.sh` | 用统一入口从重新捕获开始重跑 Liger 第 5 步 |
| `results/reference_eval/` | 上述各步的机器结果 |
