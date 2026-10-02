# 自动参照 K_R：第 1–6 步实施结果

2026-10-02。对应 [阶段性总结](stage_summary_20261002.md) 第 11 节的六步。代码在
`src/kernel_analyzer/reference_eval/`，机器结果在 `results/reference_eval/`。

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
多结果、`scf.for/while` 头部声明的块参数）。语料 65 个内核（自写练习内核、Inductor 的 chunk 求和、
交叉熵前后向、RMSNorm、AdamW、Liger 交叉熵与 RMSNorm、torchao AdamW8bit）全部解析，指令处理覆盖
全部完整。

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
| 状态 × 坐标 | 50,331,648 | 50,331,648 |
| 完整组合参照 | 全部 | 全部 |
| 手工 FP64 落在参照区间（按手工误差界放宽）内 | 全部，0 个不一致 | 全部，0 个不一致 |
| 参照区间最大宽度 | 8.9e-16 | 6.1e-16 |
| K − K_R 区间含 0 的坐标 | 2 | 3 |
| K − K_R 正 / 负 | 25,181,625 / 25,150,021 | 25,169,204 / 25,162,441 |
| 每状态每次序的求值时间 | 34.8 s | 34.6 s |

两种次序的组合参照在全部状态上相交（精确和与次序无关）。结论：自动 K_R 与手工参照在区间内一致，
可以进入第 5 步。机器结果：`results/reference_eval/liger_fp32_order_auto_reference.json`。

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

**新信息**：以精确和为参照，两种次序各自都对写入有正的平均方向作用；reverse 次序另有沿参照更新
方向少推的对齐效应，original 次序该效应未确认。平均每个状态 D 上 original 有 4103 个坐标的写入与
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

交叉熵 dlogits 残差约 75% 为正。可对照的解释是 TTIR 中 `math.exp` 在下降时成为 `ex2.approx(x·c)`，
c=RN₃₂(log₂e) 略小于 log₂e，使尾部 exp 偏大；这对应 K−G_TTIR 中的 (G_PTX − G_TTIR) 一项。该解释
未经干预验证。

**导数参照**：TTIR 层前向自动微分（区间切向量；缺规则的运算中止该 program，不静默置零）。
x³ 内核上 J(x)u 精确、宽度 0（含 x=0），伴随残差恰为 0，中心差分残差恒为 −h²u³v；softmax 内核的
切向量包住 300 位解析导数 p⊙(u−⟨p,u⟩)。测试：`tests/test_reference_eval_derivative.py`。

**成本**：Liger 位置 524,288 坐标 × 64 次启动约 35 s；交叉熵 2.05M 元素 40 s；torchao 全量 256
program 数秒。瓶颈是逐 program 的 Python 开销，与元素数近似线性。

## 限制

- float64 端点：精确值需要超过 53 位时宽度不再为 0（仍包围）；区间算术的依赖问题使少量离散判定
  成为参照未建立（torchao 码 0.14–0.39%）。
- 逐 program 实例执行；只抽样部分实例时，跨实例冲突只在已求值实例之间检查。
- 窗口化捕获要求连续张量；窗口外的访问判为参照未建立。
- 内核内部中间值不可观测，局部参照的最小单位是一次内核启动；跨启动用组合参照。
- 舍入核验模式下未声明次序的累加判为参照未建立；块指针、TMA、`tt.cat`、`atomic_cas` 拒绝。
- 第 4、5 步把 Liger 的 torch 原位加法换成逐位等价的 Triton 内核，以便进入 TTIR 流程。
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
| `results/reference_eval/` | 上述各步的机器结果 |
