# 测过的算子：清点表（规格 v3 的分母，2026-10-16 冻结）

2026-10-19 新的[外部公开调用批次](external_batch/README.md)按真实任务分别尝试标准实现和独立数学参照：
attorch 五项开发调用已运行，并由实际缺口补齐原始输入 softmax VJP 与 SmoothL1/Huber 数学参照；
后两项已用原8次调用的完整保留结果复核，不算新验收。trident 四项独立验收调用因缺少
nvtx 暂未运行成功，全部保留。该批不是盲测，也不加入下文正式绑定覆盖数量。
后续显式参数组的84次调用复现了 SmoothL1(beta=0) 相等点的双侧梯度错误；不改规格再对
FlagGems 另一实现的两个前向入口验收48次，完整前向均可求参照，容差对照相符。
后者没有验收反向或替代 trident 的失败项；详情与原始结果仍在上述批次目录。
MSE/L1 的 none/sum 缺口随后通过扩展现有合同解决：24次完整开发调用及另一份 FlagGems MSE
实现的12次前向验收已保留。原 mean profile 与正式绑定的范围不变。

FlagGems 5.0.2 的原始 MSE 前向随后完成 `[7,41]`、none/sum/mean 的一份正式调用绑定。
三种 reduction 各24次测量均有完整调用级参照；内部零累加器的来源也得到核对。
mean 的程序系数产生可独立推导的固定比例差异；sum 的统计判断仍未建立。
见[绑定与推导](../evidence/flaggems_mse_binding.md)，不覆盖反向或其他形状。

公开算子调用使用 `call_binding.py` 做逐项对应，不再用算子名或一次数值相近来代替绑定。绑定计划必须列出每个合同输入、每个实际收到的输入、每个公开输出、全部张量实参的前后态，以及调用范围内的全部 Triton 启动；未进入合同的输入必须逐项说明为何与任务无关，任何已变化输入都必须对应合同声明的后态。结果分别记录候选合同、绑定状态、参照状态和 bias 状态。草案计划通过只能记为 `development_checked`；只有独立审阅计划的原文件保留在 `validated/bindings/`，且合同和参照也通过准入，才可交给后续 bias 实验。单次调用始终不能建立总体 bias。

当前 `call_bindings/` 有六十二份计划。PyTorch 2.10 Inductor 的末维 softmax、saved-output softmax backward、saved-output log-softmax backward、logsumexp forward+VJP、gather、cumsum forward+VJP、corrected variance forward+VJP、corrected standard deviation forward+VJP、ordered var_mean forward+combined VJP、public vector 2-norm forward+VJP、二维 mean CrossEntropy loss+VJP、mean MSE loss+完整 input/target VJP、mean L1 loss+完整 input/target VJP、LayerNorm forward+VJP、RMSNorm forward+VJP、GroupNorm forward+VJP、exact-form GELU forward+VJP、SiLU、ReLU/tanh/sigmoid forward+VJP，以及直接 Triton float32 matmul、causal GQA SDPA、embedding forward+dense VJP、supplied-table Llama RoPE、Mixtral top-2 routing+combination、BSR softmax values、deterministic top-k/top-p keep mask、Mamba selective state update、LoRA shrink、causal paged attention、packed varlen attention、PyTorch KL batchmean loss+input VJP、MXFP8 E5M2 FLOOR、grouped Conv2d、行求和、逐元素 exp、显式 bf16 转换、逐元素乘加、矩阵转置、大于零比较、行 index-select、无并列值行排序、max_pool2d、BatchNorm 推理、dropout backward、index_add、梯度值裁剪、SGD、Adam、RMSprop、RAdam、Adafactor、Rprop 和 torchao OptimState4bit state-copy、FlagGems MSE 前向与公开默认 RMSNorm 完整 VJP、Inductor 二维平均池化完整 VJP 已完成独立审核并在 `validated/` 保留当前计划和调用文件；FlagGems sort、Liger RMSNorm、Liger q+k RoPE 与 Liger 原地 CrossEntropy 仍为草案。四个外部库调用的本机 GPU 重跑都通过输入/输出/后态的开发核对；五十八个已审核调用另逐项对应其全部任务输入、公开输出和未修改输入状态。正式 bias 结论只允许使用五十八个已审核调用及其声明分布，不能扩展到同名 kernel 或整个角色。FlagGems public sort 的正式入口因九次启动间内部缓冲区的 producer 证据不完整而拒绝并保持草案，见 `evidence/flaggems_sort_binding_gap.md`。 MXFP8 的直接绑定在安全输入上逐项一致，在完整有限范围内检出363个有限值被写成正负无穷；证据见 `evidence/triton_mxfp8_e5m2_floor_binding.md`。KL batchmean 的24个 loss 和12,288个 input VJP 坐标也都建立完整参照；0.25公式注入使全部坐标排除零，证据见 `evidence/triton_kl_batchmean_binding.md`。MSE 的24个 loss 和 input/target VJP 各98,304个坐标均完整；程序公式与任务相容，当前规则未确认稳定方向，证据见 `evidence/inductor_mse_loss_binding.md`。L1 的24个 loss 和 input/target VJP 各98,304个坐标也均完整；两个梯度逐点相等，loss 的当前规则未确认稳定方向，证据见 `evidence/inductor_l1_loss_binding.md`。

torchao 0.16.0 OptimState4bit state-copy 将 canonical qmap 作为显式只读前态，并在三次 Triton 启动后核对完整 packed codes 和 block scales。24个输入的49,152个 code 坐标和768个 scale 坐标全部建立点值参照且残差为零。全零残差使统计规则不声称总体无偏；这不是不一致。该绑定只覆盖 signed float32[4096]、block size 128 的 state-copy，不覆盖完整 AdamW4bit。证据见 `evidence/opt4bit_state_copy_implementation.md`。

PyTorch 2.10 Inductor LayerNorm 绑定公开 float32 `[16,128]` affine 前向和完整 x/weight/bias VJP。两次 Triton 启动在24个输入上为 `y`、`dx` 各建立49,152个严格参照坐标，为 `dweight`、`dbias` 各建立3,072个。`y`、`dx`、`dweight` 的设备到程序对齐规则检出正向平均作用，`dbias` 未检出。结论只属于该 shape、dtype 和输入分布。证据见 `evidence/inductor_layer_norm_binding.md`。

PyTorch 2.10 Inductor RMSNorm 绑定公开 float32 `[16,128]` affine 前向和完整 x/weight VJP。两次 Triton 启动在24个输入上为 `y`、`dx` 各建立49,152个严格参照坐标，为 `dweight` 建立3,072个。设备到程序的现有规则均未检出稳定平均作用；程序到任务比较在三项输出上检出约 `1e-13` 至 `4e-13` 的更小负向差异。未检出不表示无 bias，结论只属于该调用、shape、dtype 和输入分布。证据见 `evidence/inductor_rms_norm_binding.md`。

PyTorch 2.10 Inductor GroupNorm 绑定公开 float32 `[4,16,8,8]`、4组、带 affine 参数的前向和完整 x/weight/bias VJP。四次 Triton 启动在24个输入上为 `y`、`dx` 各建立98,304个严格参照坐标，为 `dweight`、`dbias` 各建立384个。设备到程序的对齐规则在 `y` 和 `dx` 上检出正向平均作用，两项参数梯度未检出；程序到任务的更小差异另行报告。结论只属于该调用、shape、dtype、组数和输入分布。证据见 `evidence/inductor_group_norm_binding.md`。

PyTorch 2.10 Inductor exact-form GELU 绑定公开 float32 `[32,128]` 前向和完整输入 VJP。第一次正式报告因负尾部 `1+erf(x)` 的区间抵消拒绝205个前向和75个梯度坐标；通用程序参照改用严格等价的 `erfc(-x)` 后，同一批24个输入的 `y`、`dx` 各98,304个坐标全部达到既定分辨率。设备到程序在前向固定方向和梯度对齐方向检出负向平均作用，程序到任务的较小差异分开报告。证据见 `evidence/inductor_gelu_erf_binding.md`。

PyTorch 2.10 Inductor saved-output softmax backward 绑定显式 float32 `[16,128]` upstream、保存的概率输出和末维归约，返回完整 `dx`。一个 Triton reduction kernel 在24个输入上建立49,152个严格参照坐标，完整率和分辨率均为1；设备到程序平均绝对差约 `2.01e-10`，现有规则未检出稳定平均作用，程序公式与合同相容。未检出不表示无 bias，也不代表已经审核前向与反向的端到端组合。证据见 `evidence/inductor_softmax_backward_binding.md`。

PyTorch 2.10 Inductor saved-output log-softmax backward 绑定显式 float32 `[16,128]` upstream、保存的 log 概率输出和末维归约，返回完整 `dx`。一个 Triton reduction kernel 在24个输入上建立49,152个严格参照坐标，完整率和分辨率均为1；设备到程序平均绝对差约 `1.91e-8`，现有规则未检出稳定平均作用，程序公式与合同相容。生成保存输出的 log-softmax 不参与 ground truth；范围不扩展到前向与反向的端到端组合。证据见 `evidence/inductor_log_softmax_backward_binding.md`。

PyTorch 2.10 Inductor logsumexp 绑定公开 float32 `[16,128]` 末维前向和完整输入 VJP。一个融合 Triton kernel 在24个输入上为前向建立384个、为 VJP 建立49,152个严格参照坐标，完整率和分辨率均为1；程序公式与任务公式相容。设备到程序平均绝对差分别约为 `1.34e-7` 和 `9.42e-10`，现有规则均未检出稳定平均作用。证据见 `evidence/inductor_logsumexp_binding.md`。

PyTorch 2.10 Inductor gather 绑定公开 float32 `[32,64]` 输入、int64 `[32,24]` 索引和 `dim=1`。一个 Triton kernel 在24个输入的18,432个输出坐标上得到精确 K−Gπ=0、Gπ−f=0 和 K−f=0。零残差使统计样本退化并保留 `statistics insufficient`，不能由这批一致结果声称总体无 bias。证据见 `evidence/inductor_gather_binding.md`。

PyTorch 2.10 Inductor cumsum 绑定公开 float32 `[16,128]` 前向和完整输入 VJP。实际路径包含3个 Triton kernel、前向 scan、反向 scan 和 flip 写回。24个输入为两项输出各建立49,152个严格参照坐标，程序公式与任务公式逐点相容；设备到程序平均绝对差分别约 `4.27e-7` 和 `4.08e-7`，当前规则均未检出稳定平均作用。证据见 `evidence/inductor_cumsum_binding.md`。

PyTorch 2.10 Inductor corrected variance 绑定公开 float32 `[16,128]`、`dim=1`、`correction=1` 的前向和完整输入 VJP。一个融合 Triton kernel 在24个输入上建立384个前向和49,152个梯度严格参照坐标。前向程序公式与任务公式相容；梯度程序把精确 `2/127` 生成为 float32 系数，全部49,152个 Gπ−f 坐标排除零，平均绝对差约 `7.55e-11`，对齐规则检出正向平均作用。设备到程序差更大，当前规则未确认稳定平均作用。证据见 `evidence/inductor_var_binding.md`。

PyTorch 2.10 Inductor corrected standard deviation 绑定同形、同轴和同 correction 的公开前向与完整输入 VJP。一个融合 Triton kernel 在24个输入上建立384个前向和49,152个梯度严格参照坐标。前向程序公式与任务公式相容；设备到程序的平均绝对差约 `5.32e-8`，固定与对齐规则检出正向平均作用。梯度程序的 float32 系数使全部 Gπ−f 坐标排除零，平均绝对差约 `1.82e-11`；R5 检出负向、R2/R3 检出正向平均作用。证据见 `evidence/inductor_std_binding.md`。

PyTorch 2.10 Inductor ordered var_mean 绑定同形、同轴和同 correction 的完整有序前向对，以及同时使用方差和均值 upstream 的完整输入 VJP。一个融合 Triton kernel 在24个输入上分别建立384个方差、384个均值和49,152个梯度严格参照坐标。均值的程序公式与精确任务公式一致；方差也没有坐标严格排除零。组合梯度中，方差分支的 float32 `2/127` 使全部 Gπ−f 坐标排除零，平均绝对差约 `6.99e-11`，R2/R3 检出正向平均作用；均值分支的 `1/128` 可精确表示。证据见 `evidence/inductor_var_mean_binding.md`。

PyTorch 2.10 Inductor public vector 2-norm 绑定公开 float32 `[16,128]`、`ord=2`、`dim=1` 的前向和完整输入 VJP。一个融合 Triton kernel 在24个输入上建立384个前向和49,152个梯度严格参照坐标，完整率和分辨率均为1。前向与 VJP 的程序公式都与任务公式相容；设备到程序的平均绝对差分别约 `5.98e-7` 和 `2.55e-9`，现有规则均未确认稳定平均作用。零范数 VJP 因没有唯一实数导数而不在绑定范围。证据见 `evidence/inductor_vector_norm_binding.md`。

PyTorch 2.10 Inductor ReLU、tanh、sigmoid 分别绑定公开 float32 `[32,128]` 前向和完整输入 VJP。端到端 tanh/sigmoid VJP 明确从原始 `x` 求导，与接收保存输出 `y` 的 ATen backward 操作分开。三个调用各生成一个融合 Triton kernel；每个输出的98,304个坐标均完整并达到分辨率，Gπ−f 无坐标排除零。ReLU 三层差异逐点为零；tanh 前向检出正向设备作用；sigmoid 前向及对齐的梯度方向检出设备作用。证据见 `evidence/inductor_common_activations_binding.md`。

Inductor CE 候选先在隔离的临时复核副本中跑过24个独立输入，只验证正式测量链：基线 loss 与 gradient 均未检出稳定 bias；保持相同输入并只在编译区加入 `+0.25` 后，loss 检出约 `-0.25000004`，gradient 结果不变。随后按当前 PyTorch 2.10 的实际 ATen 图审核并保留当前调用与计划。正式运行的三个 TTIR 启动均完整覆盖，24个 loss 参考值和4416个梯度坐标全部建立；Gπ−f 没有坐标排除零，K−Gπ 与完整 K−f 均未检出稳定均值。记录在 `.cache/spec_trial_20261018_diverse/inductor_ce_controlled_20261018/`，审核说明见 `evidence/inductor_cross_entropy_binding.md`。

SOFTMAX 合同和适配器完成独立来源、公式、输入域和完整输出复核后，其 `real` profile 已通过规格参照准入。新的 Inductor softmax 候选同样用24个独立输入跑通隔离测量：基线未检出稳定 bias；在同一输入序列中只把编译公式改为 `softmax(x)+0.01` 后，K−Gπ 仍未检出稳定差异，而 Gπ−f 与完整 K−f 均检出非零。记录在 `.cache/spec_trial_20261018_diverse/inductor_softmax_controlled_20261018/`。这说明工具能把注入项归到任务公式而不是数值执行；该隔离试验本身不批准候选，正式审核结果见下一段。

该 softmax 调用随后完成正式绑定审核。当前 PyTorch 2.10 的 `make_fx` 图把公开调用明确展开为 `_softmax(x,-1,false)`；计划覆盖全部输入、完整返回值和未修改输入状态。隐藏浮点及整数张量的反例均在统计前拒绝。保留当前计划和调用文件后，正式声明连续运行两次，两份报告除时间外在24个绑定、14208个参考坐标、统计、来源和失败分类上完全一致。Gπ−f 未检出任务差；K−Gπ 的固定方向 R1 检出约 `-6.58e-9` 的小数值偏差。证据见 `evidence/inductor_softmax_binding.md`；其他计划只有经过各自审核才能升级。

ACT 的两个 profile 完成独立复核后，Inductor SiLU 用24个相同分布、相同 seed 的输入跑通隔离测量。基线的 Gπ−f 未检出；`silu(x)+0.01` 的 Gπ−f 与 K−f 均检出非零。随后按当前 PyTorch 2.10 的实际 `aten.silu` 图审核并保留当前调用与计划。正式入口建立全部16,728个参考坐标，没有失败分类；Gπ−f 没有坐标排除零，K−Gπ 在 R1、R2 检出约 `1e-8` 投影量级的负向数值差异。记录在 `.cache/spec_trial_20261018_diverse/inductor_silu_controlled_20261018/`，审核说明见 `evidence/inductor_silu_binding.md`。

MATMUL 的有限浮点默认重载完成独立复核后，直接 Triton float32 matmul 用24个固定输入跑通隔离测量。基线的 Gπ−f 未检出；只在累加结果加入 `0.01` 后，Gπ−f 在6144个输出坐标上全部排除零并检出固定均值非零，K−Gπ 则仍未检出稳定均值。随后依据 Triton 3.6 `tl.dot` 定义和实际索引审核并保留当前调用与计划。正式入口建立全部6144个参考坐标，输入来源和完整写回均闭合，没有失败分类；Gπ−f 没有坐标排除零，当前 K−Gπ 与完整 K−f 投影均未检出稳定均值。审核说明见 `evidence/triton_matmul_binding.md`。范围限于16×32乘32×16的连续 float32 输入和明确的 ieee dot 选择。

CONV 的批量非转置 1-D/2-D 有限浮点 profile 完成来源、schema、域和实现复核后，直接 Triton grouped Conv2d 完成正式绑定。计划覆盖三个张量、六个任务参数、全部前态和2×6×8×5输出。正式入口的24个调用均捕获一个完整 TTIR 启动，11,520个参考坐标全部达到声明分辨率，没有失败分类；Gπ−f 没有坐标排除零，K−Gπ 与完整 K−f 在R1、R2、R3、R5上均未确认稳定均值。只在写回增加 `0.01` 的受控版本使11,520个任务残差全部排除零，R1和R5检出非零，而K−Gπ仍未确认。正式运行约257秒，说明扩大卷积形状前还需优化严格参照的批处理，但没有通过删减输出降低要求。审核说明见 `evidence/triton_conv2d_binding.md`。

REDUCE 的公共有限浮点归约通过规格参照准入后，直接 Triton float32 行求和完成正式绑定。计划固定 `[32,256]` 连续输入、沿第1维求和、完整 `[32]` 输出和未修改输入状态；24个调用的768个坐标均建立严格参照，所有 TTIR 程序与写回完整覆盖。正常实现的 Gπ−f 没有坐标排除零，设备与任务值的最大绝对差约为 `4.86e-6`，当前统计规则未确认稳定 bias。只在 `tl.sum` 结果上加入 `0.01` 后，768个任务残差全部排除零且平均残差约为 `0.01`，K−Gπ 仍未检出稳定差异。审核说明见 `evidence/triton_row_sum_binding.md`。范围只限于声明的形状、dtype、轴和实现。

MATHFN 的有限公共逐元素函数通过规格参照准入后，直接 Triton float32 exp 完成正式绑定。计划固定 `[1024]` 连续输入、完整同形输出和未修改输入状态；24个调用的24,576个坐标均建立严格参照，一次启动中的四个 program 及全部写回完整覆盖。正常实现的 Gπ−f 为零；设备到程序和完整设备到任务比较在声明的正态输入上检出很小的负向数值偏差，R1、R2 的投影均值约为 `-1.21e-7`，完整 K−f 平均绝对残差约为 `6.57e-8`。在 `tl.exp` 后加入 `0.01` 后，24,576个任务残差全部排除零，平均绝对任务残差约为 `0.01`。审核说明见 `evidence/triton_exp_binding.md`。这一区分保留了任务公式差和设备近似误差各自的含义。

CAST 的有限浮点 RTNE profile 通过规格参照准入后，直接 Triton float32 到 bfloat16 转换完成正式绑定。计划固定 `[1024]` 连续输入、显式 `fp_downcast_rounding="rtne"`、完整同形输出和未修改输入状态。24个调用的24,576个坐标全部建立并达到声明分辨率；程序值、任务值和设备值逐坐标相同。由于这里的任务本身就是明确舍入的转换，计划显式选择 `rounding_check` 程序参照，最终报告记录该选择。临时版本在转换前加入 `0.25` 后，K−Gπ 仍为零，Gπ−f 与完整 K−f 的全部坐标排除零，平均绝对差约为 `0.249998`。审核说明见 `evidence/triton_cast_bf16_binding.md`。该结果不覆盖其他 dtype、隐式舍入、特殊值、融合转换或内存选项。

ARITH 的有限同 dtype 浮点 profile 通过规格参照准入后，直接 Triton float32 逐元素乘加完成正式绑定。计划固定三个 `[1024]` 连续输入、完整同形输出和三个未修改输入状态。24个调用的24,576个坐标全部建立严格参照；正常实现的 Gπ−f 全部为零，K−Gπ 的平均绝对差约 `2.32e-8`、最大约 `4.13e-7`。这把乘加是否在设备上采用单次舍入造成的普通数值差留在程序层。临时版本在结果加入 `0.25` 后，K−Gπ 仍在普通浮点误差量级，全部 Gπ−f 坐标排除零且平均绝对差恰为 `0.25`，完整 K−f 的规则确认非零效应。审核说明见 `evidence/triton_multiply_add_binding.md`。该结果不覆盖广播、其他 dtype、形状或同名实现。

LAYOUT 的有限浮点逻辑值与形状 profile 通过规格参照准入后，直接 Triton float32 矩阵转置完成正式绑定。计划固定一个连续 `[32,64]` 输入、完整 `[64,32]` 输出和未修改输入状态。24个调用的49,152个坐标全部建立严格参照；正常实现的 K−Gπ、Gπ−f 与完整 K−f 逐点为零。统计规则因全零样本没有宣称总体等价。临时版本在每个输出加入 `0.25` 后，全部 Gπ−f 坐标排除零且平均绝对差为 `0.25`，语义方向规则与完整输出规则确认非零效应。审核说明见 `evidence/triton_transpose_binding.md`。该结果不覆盖非连续输入、storage/stride/alias、其他 dtype、形状或同名实现。

LOGIC 的两个物理布尔结果 profile 通过规格参照准入后，直接 Triton float32 大于零比较完成正式绑定。24个调用的24,576个坐标全部建立精确布尔参照，正常实现零误判。临时版本把实现阈值改为0.25后检出2,474个假阴性；工具同时保留 K−Gπ、Gπ−f 与完整 K−f，并单独报告不会因符号抵消的布尔误判计数。审核说明见 `evidence/triton_greater_zero_binding.md`。

INDEX 的只读物理 lookup profile 通过规格参照准入后，直接 Triton float32 行 index-select 完成正式绑定。24个调用的49,152个输出坐标全部建立严格参照，正常实现的 K−Gπ、Gπ−f 与完整 K−f 逐点为零。临时版本在每个查得值后加入0.25，全部 Gπ−f 坐标排除零，完整输出规则确认非零效应。审核说明见 `evidence/triton_index_select_binding.md`。

SORT 的物理有限、无并列值 profile 通过规格参照准入后，直接 Triton float32 行排序完成正式绑定。24个调用的24,576个排序值和24,576个原始索引都得到完整精确 K−f=0。临时版本只给排序值加入0.25，完整任务比较得到约0.25的平均绝对残差并确认非零效应，索引仍全部正确。程序参照目前不认识 Triton bitonic sort 的归约组合器，因此本例的 K−Gπ 明确保留为未建立。审核说明见 `evidence/triton_sort_unique_binding.md`。

POOL 的有限二维 max_pool2d 前向值 profile 通过规格参照准入后，直接 Triton float32 2×2、stride 2 池化完成正式绑定。24个调用的9,216个完整输出坐标在 K−Gπ、Gπ−f 与 K−f 上都精确为零。临时版本在每个窗口最大值后加入0.25，全部 Gπ−f 坐标排除零，完整输出规则确认非零平均效应。审核说明见 `evidence/triton_max_pool2d_binding.md`。

BN 的有限 BatchNorm 推理前向值 profile 通过规格参照准入后，直接 Triton float32 调用完成正式绑定。24个调用的24,576个任务坐标都达到声明分辨率，正常程序公式没有 Gπ−f 坐标排除零；设备到程序平均绝对差约3.46e-8。临时版本在输出加入0.25后，全部 Gπ−f 坐标排除零且完整输出规则确认非零平均效应，同时 K−Gπ 仍在普通 float32 量级。审核说明见 `evidence/triton_batch_norm_eval_binding.md`。

RANDOM 的显式 mask dropout backward profile 通过规格参照准入后，直接 Triton float32 调用完成正式绑定。24个调用的24,576个程序语义坐标与任务公式精确一致；设备到程序平均绝对差约1.61e-8。临时版本在输出加入0.25后，全部 Gπ−f 坐标排除零，语义与完整输出规则均确认非零平均效应，同时 K−Gπ 保持普通 float32 量级。审核说明见 `evidence/triton_dropout_backward_binding.md`。随机训练前向没有唯一值参照，仍不作语义裁决。

INDEXREDUCE 的函数式 index_add 前向 profile 通过规格参照准入后，固定偶数行 index、alpha 0.5 的直接 Triton float32 调用完成正式绑定。24个调用的98,304个程序语义坐标与任务公式精确一致；设备到程序平均绝对差约9.04e-9。临时版本在输出加入0.25后，全部 Gπ−f 坐标排除零，语义与完整输出规则均确认非零平均效应，同时 K−Gπ 保持普通 float32 量级。审核说明见 `evidence/triton_index_add_binding.md`。

CLIP 的有限单梯度 `clip_grad_value_` profile 通过规格参照准入后，直接 Triton float32 原地写回调用完成正式绑定。绑定从完整调用前态构造任务输入，并把同一梯度存储的完整调用后态作为输出。24个调用的98,304个坐标在 K−Gπ、Gπ−f 和 K−f 上都精确为零。临时版本在裁剪后加入0.25，全部 Gπ−f 坐标排除零，方向规则与完整输出规则均检出非零效应。审核说明见 `evidence/triton_clip_grad_value_binding.md`。

SGD 的单参数完整状态 profile 通过规格参照准入后，直接 Triton float32 warm-buffer Nesterov 调用完成正式绑定。参数、梯度和旧 momentum buffer 的完整前态进入任务输入，参数和 buffer 的完整写回同时作为输出。24个调用中，两项输出各98,304个 Gπ−f 坐标都精确为零；设备舍入在 buffer 上留下约2.92e-8的平均绝对 K−f，并被数值规则检出。临时版本只给 buffer 写回加入0.25，全部 buffer Gπ−f 坐标精确偏移0.25，而参数的 Gπ−f 仍为零。审核说明见 `evidence/triton_sgd_binding.md`。

ADAM 的单参数完整状态 profile 通过规格参照准入后，直接 Triton float32 warm-state coupled Adam/AMSGrad 调用完成正式绑定。参数、梯度、两个矩、AMSGrad 最大矩和 step 的完整前态进入任务输入，参数及全部四项优化器状态同时作为输出。24个调用中，四个向量输出各98,304个 Gπ−f 坐标和24个 step 坐标均与任务公式相容；设备执行的平均绝对 K−f 分别约为 theta 1.75e-8、m 1.63e-8、v 2.87e-8、vmax 1.07e-8，step 精确。临时版本只给 vmax 写回加入0.25，全部 vmax Gπ−f 坐标排除零，其他四项仍为零。审核说明见 `evidence/triton_adam_binding.md`。

RMSPROP 的单参数完整状态 profile 通过规格参照准入后，直接 Triton float32 warm-state centered RMSprop 调用完成正式绑定。参数、梯度、square average、centered average、momentum buffer 和 step 的完整前态进入任务输入，参数及全部四项优化器状态同时作为输出。24个调用中，四个向量输出各98,304个 Gπ−f 坐标和24个 step 坐标均与任务公式相容；设备执行的平均绝对 K−f 分别约为 theta 1.80e-8、v 3.40e-8、gave 6.60e-9、b 3.14e-8，step 精确。临时版本只给 b 写回加入0.25，全部 b Gπ−f 坐标排除零，其他四项仍为零。审核说明见 `evidence/triton_rmsprop_binding.md`。

RADAM 的 PyTorch 2.10 单参数完整状态 profile 通过规格参照准入后，直接 Triton float32 warm-state rectified RAdam 调用完成正式绑定。参数、梯度、两个矩和 step 的完整前态进入任务输入，参数及三项优化器状态同时作为输出。第一次绑定把 rectification factor 预先化成 host float，Gπ−f 检出 theta 的公式差，因此没有作为正常基线接受；当前调用从 beta2 在程序图中计算完整标量公式。修正后24个调用中，三个向量输出各98,304个 Gπ−f 坐标和24个 step 坐标均与任务公式相容；设备执行的平均绝对 K−f 分别约为 theta 1.72e-8、m 1.63e-8、v 2.58e-8，step 精确。临时版本只给 m 写回加入0.25，全部 m Gπ−f 坐标排除零，其他三项仍为零。审核说明见 `evidence/triton_radam_binding.md`。

ADAFACTOR 的 PyTorch 2.10 向量与末两维分解 profile 通过规格参照准入后，直接 Triton float32 32×32 warm-state factored 调用完成正式绑定。一次启动完整更新参数、row、col 与 step。修正首次编译暴露的归约轴广播错误后，24个调用中 theta 的24,576个、row/col 各768个 Gπ−f 坐标和24个 step 坐标均与任务公式相容；设备执行的平均绝对 K−f 分别约为 theta 1.87e-8、row 2.71e-8、col 2.74e-8，step 精确。临时版本只给 row 写回加入0.25，全部768个 row 坐标排除零且两类规则均确认非零，theta、col 与 step 不误报。审核说明见 `evidence/triton_adafactor_binding.md`。

RPROP 的 PyTorch 2.10 单参数完整状态 profile 通过规格参照准入后，直接 Triton float32 length-4096 warm-state maximize 调用完成正式绑定。参数、梯度、previous gradient、step size 和 step 的完整前态进入任务输入，参数及三项优化器状态同时作为输出。24个调用中，三个向量输出各98,304个 Gπ−f 坐标和24个 step 坐标都与任务公式相容；设备执行的平均绝对 K−f 约为 theta 1.02e-8、eta 3.15e-9，prev 和 step 精确。临时版本只给 eta 写回加入0.25，全部98,304个 eta 坐标排除零，theta、prev 与 step 仍为零。审核说明见 `evidence/triton_rprop_binding.md`。

ATTN 的 PyTorch 2.10 defined-row 公共前向 profile 通过规格参照准入后，直接 Triton float32 `[2,4,16,16]` q、`[2,2,32,16]` k/v causal GQA 调用完成正式绑定。24个调用的49,152个输出坐标全部建立完整 call-level 参照；Gπ−f 没有坐标排除零，设备到程序平均绝对差约3.01e-8，对齐规则检出小的负向数值偏差。临时版本只给最终输出加入0.25，全部49,152个 Gπ−f 坐标排除零，两类语义规则均确认非零。审核说明见 `evidence/triton_sdpa_binding.md`。

EMB 的 PyTorch 2.10 dense functional embedding profile 通过规格参照准入后，直接 Triton float32 `[64,16]` weight、`[128]` indices 与 `[128,16]` upstream 的 forward+dense VJP 调用完成正式绑定。最终声明分布让每个词表行恰好出现两次；24个调用的49,152个 forward 坐标与24,576个 dweight 坐标全部建立完整 call-level 参照，Gπ−f 没有坐标排除零。临时版本只给非 padding 梯度行加入0.25，24,192个 Gπ−f 坐标排除零，两类语义规则均确认非零。审核说明见 `evidence/triton_embedding_binding.md`。

ROPE 的活动合同现在只覆盖 Transformers 4.57.3 Llama supplied-table rotate-half q/k 前向与完整 q/k VJP；旧合同混入的表生成、partial/interleaved 和其他模型任务已移回草案。直接 Triton float32 两次启动的正式基线中，q_out/dq 各49,152个、k_out/dk 各24,576个程序坐标全部建立并与任务公式逐点一致；设备到程序平均绝对差约1.6e-8。只给 dq/dk 写回加0.25后，全部73,728个梯度 Gπ−f 坐标排除零，两个前向结果保持零任务差。审核说明见 `evidence/rope_implementation.md` 与 `evidence/triton_llama_rope_binding.md`。该绑定不扩展到其他 RoPE 布局、表生成器、模型族、形状或 dtype。

MOE 的活动合同只覆盖 Mixtral 无并列 top-2 路由和 supplied expert outputs 的加权组合；原合同混入的 Switch capacity、raw gates、auxiliary loss 和其他路由任务已移回草案。直接 Triton float32 正式基线中，24,576个 Boolean 选择位零误判，24,576个权重与98,304个组合输出坐标均获得完整、达到分辨率的任务参照。设备到程序平均绝对残差约为权重4.89e-9、组合结果3.06e-8；相对精确实数任务保留测得的微小 float32 数值效应。只给组合结果写回加0.25后，全部98,304个 Gπ−f 坐标排除零，而选择和权重保持正常。程序分析仍把 Triton 3.6 reduction lowering 次序列为显式前提，整体语义状态因此保持未完全建立。审核说明见 `evidence/moe_implementation.md` 与 `evidence/triton_mixtral_top2_binding.md`。该绑定不扩展到其他形状、dtype、tie、capacity、auxiliary loss、expert MLP 或分布式 dispatch。

BSR 的活动合同只覆盖 PyTorch 2.10 二维 canonical BSR softmax value core；原合同混入的 BSR 矩阵乘和 sampled_addmm 已移回草案。固定 `[128,64]` 矩阵、`[16,16]` block 和每个 block row 两个指定 block 的直接 Triton float32 core 与本机官方 Triton 调用逐位一致。正式基线的98,304个坐标全部建立并达到分辨率，程序公式与精确实数任务相容；设备到程序平均绝对残差约2.00e-9，并保留测得的微小 float32 数值效应。只给写回加0.25后，全部98,304个 Gπ−f 坐标排除零且完整比较检出。审核说明见 `evidence/bsr_softmax_implementation.md` 与 `evidence/triton_bsr_softmax_values_binding.md`。该绑定不扩展到其他 pattern、形状、dtype、batch/hybrid、矩阵乘、sampled_addmm 或梯度。

SAMPLING 的活动合同只覆盖 vLLM 0.16.0 确定性 top-k 后 top-p 的完整 Boolean keep mask；原草案混入的 masked logits、概率和随机抽样已移回草案，并列值明确拒绝。直接 Triton float32 正式基线的98,304个坐标全部匹配，重复执行逐位一致；第一次使用普通正态输入时出现物理 float32 并列并被正确拒绝，随后改用经过24个种子核验的无并列输入生成器。每行翻转 token 0 的临时版本恰产生1,536个错位，程序公式和完整输出比较均检出非零作用。审核说明见 `evidence/sampling_filter_implementation.md` 与 `evidence/triton_topk_topp_keep_binding.md`。该绑定不扩展到其他形状、dtype、tie policy、masked logits、概率、随机抽样或替代 backend。

SSU 的活动合同只覆盖 mamba-ssm 2.3.2.post1 的 float32 单步选择性状态更新，并要求该版本实际可启动的 D 与 dt_bias 均存在。审核发现旧合同混入论文 ZOH 公式、把 threshold-20 softplus 误写成处处相同的精确函数，还没有完整 state 比较；同时复现了官方 wrapper 在 D 或 dt_bias 为空时启动前报错。当前合同用18个例子、11个错误公式、7组768位独立逐元素复算、公开 PyTorch 公式及6组官方 Triton/参考调用核对。直接 Triton 基线建立24,576个 y 和196,608个完整 state 参照；state 检出约1.01e-8平均绝对 float32 残差及小的正向平均作用。只给最终 state 写回增加0.25后，全部196,608个 state 坐标排除正确参考，y 保持正常。审核说明见 `evidence/ssu_implementation.md` 与 `evidence/triton_mamba_ssu_binding.md`。

LORA 的活动合同只覆盖安装的 vLLM 0.16.0 LoRA A 投影。合同显式接收 token 映射和完整 float32 输出前态：存在活动 token 时清零完整输出并写回各 slice，`-1` 行保持零；全部 token 都为 `-1` 时官方 wrapper 提前返回并保留旧输出。16个例子、5个错误实现、12组独立精确循环与 PyTorch 公式以及7组官方 Triton 调用共同核对这些语义。直接 Triton float16 基线的24,576个输出坐标全部建立严格参照，程序公式与任务相容，设备执行平均绝对差约2.31e-8。临时版本给全部输出加0.25后，每个坐标都排除正确参考且方向规则检出非零作用。审核说明见 `evidence/lora_shrink_implementation.md` 与 `evidence/triton_lora_shrink_binding.md`。

PAGEDATTN 的活动合同只覆盖安装的 vLLM 0.16.0 causal、未量化、二维输出路径。旧草案接受该版本明确拒绝的双向调用，且没有物理 dtype 和完整输出检查，现已归档。18个例子、8个错误公式、7组768位独立复算、3组 dense PyTorch 公式和5组官方 vLLM GPU 调用核对分页、bottom-right 位置、左窗口、GQA、scale 与 softcap。直接 Triton 基线的7,680个坐标全部建立严格参照，程序公式与任务相容；设备执行平均绝对差约1.28e-8，并在对齐方向上检出小的负向数值作用。临时给全部输出加0.25后，所有 Gπ−f 坐标排除零。审核说明见 `evidence/paged_attention_implementation.md` 与 `evidence/triton_paged_attention_binding.md`。

PACK 的活动合同现在只覆盖 FlashAttention v2.8.3 非分页 varlen 前向的完整输出；旧草案把文档 mask、position_ids 与 attention 输出混成一个任务，现已归档。两个 profile 分别固定 bottom-right causal 与 bidirectional 语义，并核对累计长度、max_seqlen、连续 GQA、scale、f16/bf16 物理输入和完整输出。15个例子、8个错误公式、10组768位独立复算、3组 dense PyTorch 公式及4组已安装 vLLM vendored wrapper GPU 调用通过。直接 Triton f16 基线的7,680个坐标全部建立严格参照，程序公式与任务相容，完整设备到任务平均绝对差约4.02e-5；临时给全部输出加0.25后，所有 Gπ−f 坐标排除零且两类语义规则均检出。审核说明见 `evidence/pack_implementation.md` 与 `evidence/triton_pack_varlen_binding.md`。

多启动区域的可重复结构检查入口：`python 5_specification/corpus/check_region.py --capture-dir CAPTURE_MODEL_CONFIG_DIR --output-code WRAPPER.py --first FIRST_INDEX --last LAST_INDEX --out NEW.json`。
增加 `--check-device` 会用保存的二进制各重放两次，并以固定规则仿真完整网格、核对全部写入元素；任何不一致或不完整检查都返回非零。新捕获的 `region_device_check.json` 中，启动 12–16 的 913 个程序、1026 个写入元素全部逐位一致，两次重放均与捕获字节一致。该证据限于捕获输入；IR 写入事件和设备最终字节不能替代设备内存访问轨迹，也不自动升级语义认证。
它重新读取原始捕获、调用栈及编译来源，检查连续存储、参数、完整编译图归属、wrapper 返回位置及未返回临时存储；不读取旧报告的通过字段作为依据。
manifest 必须明确记录同一次 recorder 和保留存储生命周期；旧记录缺失时拒绝，不自动补全。入口也按编译 origin 绑定区域外部输入并提取完整任务图；需要读写前态的区域不能隐去前态后继续。退出码 0 表示请求的结构检查或所选参照求值完成，2 表示查询未建立/不可用，1 表示实现异常；已有输出禁止覆盖，语义判定不因退出 0 而升级。
2026-10-18 的新本地 GPU 捕获 `.cache/spec_region_entry/` 保存 26 个 kernel。启动 12–16 的新入口结果在 `region_outputs_check.json`：5 次启动、7 个存储、8 条连续状态连接；4 个计算输出、3 个透传返回位置、2 个区域未观测返回、1 个已核对 wrapper 生命周期的临时存储。数学参照和语义判定仍未建立。对应的旧捕获因缺 recorder 声明被拒绝，见 `.cache/region_entry_old_capture.json`。

`--evaluate-task` 只比较设备结果与捕获的编译图公式；编译图来自被审程序的编译链，不是独立任务规格 f，报告以 `captured_graph_complete` 标识，不能据此作语义裁决。增加 `--softmax-inputs LOGITS_NODE LABELS_NODE` 可显式选择已有的 softmax/mean-NLL 独立参照。只有提取后的全部任务节点及四个输出公式一致、转换前提成立时才求值；不同合法任务不被判错。`--max-unique` 明确指定不同指数值的求值预算，默认200000。对上述新捕获，`mm` 与 `constant_pad_nd` 两个输入的24节点任务已提取：默认预算结果保留在 `region_task_reference_default_budget.json`，退出2；显式指定1000000后的完整参照在 `region_task_reference_complete.json`，处理6434816个 logits、294973个不同指数值，返回全部行最大值、log mass、mean loss 和 count=127。设备访问范围、完整 kernel 绑定认证及 G 与 f 的语义比较仍未建立。

再增加 `--compare-g` 可连续求值整段 G，并对全部计算输出记录精确有理数差区间；`--max-programs` 默认 2048，预算不足不抽样替代。完整网格、无中断/未证前提、无设备中间值重置、版本计数支持的状态衔接均为必要条件。新捕获结果 `region_composed_comparison.json` 完整执行 913 个程序实例、8 次衔接、无外部重置：128 个最大值及 count 精确相等，loss 和 128 个 log-mass 的 G-f 区间包含零，最大绝对值上界分别约 6.95e-15、2.15e-15。包含零不证明恒等；该结果是选定任务上的条件性比较，设备效果与完整绑定认证仍未建立。旧捕获的程序实例数与结果不得移用于此次新捕获。

由 `build_inventory.py` 从各实验的记录生成，冻结之后只能加行、不能删行（`../docs/PLAN_v3.md` §1）。

| 文件 | 内容 |
|---|---|
| `instances.jsonl.gz` | 每个测过的实例一行：实验、类别、算子种类、算子、参数、当时用的 f、当时记录的规格状态、怎样重建输入 |
| `tested_operators.json` | 按实验与种类的计数、three_way 的对应情况、读过的记录 |

冻结后的 `instances.jsonl.gz` 作为只追加记录保留；其身份由文件路径、冻结日期和对应生成过程说明。

运行 FlagGems 合同开发查询时使用 `python 5_specification/corpus/run_corpus.py --out NEW.jsonl`。
输出文件必须不存在，已有记录不会被覆盖。脚本保留每次求值的结果类别和草案标签，执行异常或参照实现错误导致非零退出。
输入是按调用配置重新生成的，并非已核对一致的捕获输入；因此每条记录的绑定和语义判定仍为 `not_established`。
脚本正常退出只说明记录过程完成，不能把 `invalid_input`、`domain_undefined` 或未建立的查询计入可靠参照覆盖。

本质错误轮的开发查询使用 `python 5_specification/corpus/run_essential.py --out-dir NEW`，目录必须不存在。
生成失败和实现错误按条件、种子保留并导致非零退出；空查询集合不能通过。每个输出保留合同标签，重新生成的输入不等于已完成捕获绑定。
softmax 前向到反向、批量 matmul 到偏置相加均在 GRAPH 内组合求值，不再以区间中点或 NumPy 浮点中间值代替参照值。

静态 Inductor 候选匹配使用 `python 5_specification/corpus/bind_inductor.py --out NEW.json.gz`。
输出必须是新文件；空清单不能通过，未匹配项仍逐项保存。候选匹配状态、完整绑定状态和参照状态分开记录；
即使图操作都受支持，也不升级绑定或语义判定。冻结的 `bindings/three_way.json.gz` 不由此命令覆盖。

2026-10-18 重新核对 2120 个 three_way 实例（新结果 `.cache/spec_static_candidates_current.json.gz`，未覆盖历史文件）：
1542 项找到唯一来源且图操作受支持，25 项找到唯一来源但含未支持的原地写操作，552 项有多个来源，1 项无匹配。
源码完全相同或模板匹配均不能在多个来源中静默选第一项。上述仍只是候选结果；完整绑定均未建立，参照均未求值。
本轮直接比较源码文本，不依赖源码摘要判断相等。

动态复核使用 `python 5_specification/corpus/dynamic_three_way.py --bindings CANDIDATES.json.gz --out NEW.jsonl`。
候选文件必须显式指定；抽样保留所有逐位一致的实例，包括无匹配和多来源项，不按存在 output_code 过滤。
2026-10-18 新抽样结果 `.cache/spec_dynamic_candidates_current.jsonl` 保留 100 项：62 项缺已执行证据、30 项来源有歧义、
7 项没有对应的捕获来源记录、1 项无候选。全部尚未建立完整绑定，参照未求值。抽样范围已纠正，不能与先前只从有 output_code 的项中抽样所得比例直接比较。

新的模型记录可用 `prov_model.py MODEL NEW_PROVENANCE.jsonl --config bf16 --capture-dir NEW_CAPTURE_DIR`
在同一进程获取编译来源和第二次训练步骤的实际启动；输出路径必须不存在，大小超限项保留在 manifest 中。
静态匹配的 `--units`、`--capture-root`、`--code-dir` 和动态检查的 `--capture-root`、`--provenance-root`
用于指定本次记录。新捕获没有自动获得 K−G 逐位验证；只有显式使用 `--binding-only --include-unverified`
才能在未完成该验证时检查绑定结构，报告仍保留 `verify: not_run`。

2026-10-18 新设备记录 `.cache/spec_capture_current/`：OPT-350M 两层、128 token、bf16，
26 个实际执行的 kernel 全部保存；同一进程有 47 条编译来源。候选唯一 13 项、来源有歧义 13 项。
编译缓存复用可令捕获的源码位置保留旧路径；只在完整源码文本相同的情况下关联新路径，并保留全部调用来源。
26 项绑定检查中，7 项通过输入、输出、状态的结构检查，其中 5 项全网格写入检查超预算，2 项完成 IR 写入范围检查。
另外 19 项因来源歧义、输出形状或缺失图输入被拒绝。全部仍未建立正式语义认证，未合并进历史覆盖率。

随后在 `.cache/spec_capture_callsite/` 生成同条件的新捕获，记录启动时的 Python 文件、行号和函数名
（不保存局部变量；旧捕获的该字段仍为缺失）。静态匹配只在记录位置对应候选文件中该 launcher 的实际
`.run(...)` 调用时消除歧义，原始全部来源仍保留在 `source_candidates`。
这次 26 项均找到唯一候选，其中 13 项由实际调用位置区分；1 项仍含未支持的 `aten.index_put_`。
动态检查仍只有 7 项通过结构检查：另 13 项的来源记录混合了两次编译、调用数不对应，6 项仍缺形状或图输入映射。
不得为通过检查而静默合并重复编译来源；还需明确记录来源所属的生成图文件。

`.cache/spec_capture_graph/` 的后续新记录补齐了这项证据：通过生成 kernel 与返回图模块的同一个
`GraphLowering` 对象，47 条编译来源均记录所属的 `output_code`。动态检查先按已选图文件筛选，
再核对 kernel 源码和调用参数，不以重复内容去重来凑齐调用数。26 项中 15 项通过结构检查，
11 项仍缺输入、输出或中间分量映射；其中涉及 `prepare_softmax_online` 的 3 项缺少明确的输出分量对应。
15 项结构通过项中仍仅 2 项完成全网格 IR 写入检查，13 项超预算。该结果仍是绑定开发证据，不是语义正确性认证。

这两项（整数 add、constant_pad_nd）后续通过各两次设备逐位重放及固定规则的完整网格仿真，
记录为 `small_device_verification_complete.jsonl`。只在新的 `candidates_verified_small.json.gz`
中将它们的 K−G 核对状态更新为 `bit_identical`；26 项原始候选记录不覆盖。
动态比较的 `--full-output-limit N` 核对每个输出的全部坐标，超过 N 则拒绝，不退回抽样；
区间内相容不等于精确相等，设备与参照不同也不直接标为语义错误。
`small_full_reference.jsonl` 中，两项在 eager、inductor_default 下的全部 128、129 个输出均精确相等；
该记录仍保留 `semantic_verdict: not_established`，因为不能用实例通过替代合同审阅及完整认证。
单次调用的完整输出比较只写为 `inside_declared_result` 或 `outside_declared_result`；后者可能只是有限精度舍入，
不能写成语义矛盾。允许集合和离散输出只检查整体成员关系，不构造没有数学含义的坐标平均差。

后续 `binding_check_views_checked.jsonl` 核对原地复用的前态视图：同时要求来源记录、
生成代码中的单次 `reinterpret_tensor`、前态布局声明及捕获布局相符，按前态自己的 stride 读取。
目前仅支持零偏移、无间隙的单层视图；元素数相等本身不能建立对应。
这使结构检查通过数从 15 增至 18 / 26；另 8 项仍缺输出分量或输入对应。
18 项中仍只有 2 项完成全网格 IR 写入检查，16 项超预算，所有项均保留未建立语义认证的状态。

`prov_model.py` 现在另存同次编译的完整 FX 图（`*.graphs/graph_*.json`），由同一个
`GraphLowering` 对象关联 kernel、图模块和节点记录。保留节点顺序、参数表达式、依赖、tuple 分量
及各张量布局；缺失或不支持的元数据明确保留，不执行表达式，也不把编译器记录当独立参照。
`.cache/spec_capture_full_graph/` 的真实运行得到 26 份设备捕获、47 条 kernel 来源、9 份完整图，
所有图文件与对应来源记录一致，无捕获跳过。
其中 `prepare_softmax_online` 的单个数学节点被拆为分片最大值、最大值合并、分片指数和等多次启动；
两个数学输出的形状相同，必须按明确的 tuple 分量及整个计算区域建立对应，不能按形状猜测。
`binding_complete_graph_validated.jsonl` 已用这些完整图补齐局部片段遗漏的定义，核对图模块归属、
节点表达式及依赖，并将已捕获的中间输入保留为计算边界。结构通过仍为 18 / 26；
部分拒绝现在明确落在 tuple 值与实际缓冲区的对应，而不是缺失计算定义。
完整图没有提供的缓冲区分量关系仍不能靠同形状或已有数值推断。

`binding_scratch.jsonl` 进一步区分缓冲区复用与真实的前态输入。仅当 TTIR 中的指针流
只通过明确支持的地址变换到达 store、不发生读取或逃逸，且无来源节点的临时占位符未被数学图使用时，
才移除该占位符；全部输出和存储变化检查保留。两项 LayerNorm 融合实例因此完成结构绑定，
通过数为 20 / 26。指针读取、原子操作、调用或不支持的指针变换不会获得此项放行。
此结果仍不是完整设备—参照比较或语义认证。

多次启动的区域检查从捕获衔接开始。`region_storage.check_sequence` 要求同次 recorder、
捕获期间保留存储对象、连续启动编号及完整字节记录；按存储对象核对前次后态与后次前态，
拒绝仅靠相同地址建立对应。`softmax_region_storage_validated.json` 中编号 12–16 的五次启动
有 8 条字节一致的衔接，尚未认证中间所有设备工作或区域的数学输出，绑定状态仍为未建立。
旧捕获的存储保留条件由生成代码复核并记入该结果；新捕获清单直接记录这两个条件。

该区域随后由完整图和 wrapper 返回顺序对应到四个输出；`softmax_region_outputs.json`
保留未观察到的 wrapper 返回值和未返回的写入，未自动忽略任何一项。临时存储 `buf2 → buf4`
的局部分配、使用、删除关系另记在 `softmax_region_temporaries.json`，不替代设备内存效果检查。
`softmax_region_task.json` 以明确来源 `mm`、`constant_pad_nd` 为输入边界，包含 24 个运算节点。
`softmax_region_reference_budgeted.json` 对全部四个输出保留结果：有效标签计数在两种读法下
均完整精确相等，其他三个输出均超过每次 10,000 次未缓存标量求值的预算，状态为未建立。
`lazy.evaluation_budget` 计入依赖节点和嵌套预算，限制的是求值次数，不是墙钟时间或任意精度内存。
整个区域仍未获得语义认证。

`softmax_region_reference_complete.json` 后续完成了所选有限输入实数公式的全部求值。
独立求值器 `softmax_reference.py` 将每行相同的精确输入值按整数次数合并：
`sum(exp(x_i-m)) = sum_v count(v)*exp(v-m)`，以 Arb 严格区间计算，端点按精确有理数保存。
这不改变数学目标，也不把显示小数送回计算。当前图的 BF16→FP32 转换与 128 行内的计数转换均精确；
公式对应来自该图明确的 slice、ignore_index=-100、softmax、gather、求和和均值定义，未自动推广到其他图。
此次 6,434,816 个输入只需 294,433 个不同值的指数项，显式上限为 1,000,000，求值约 2.3 秒。
全部 258 个输出坐标均比较：128 个最大值与计数精确相等，损失与 128 个 log-mass 均在严格参照区间外。
这是设备值与所选实数公式的差异，包含浮点效应，不能直接用作 G−f 的语义错误结论；
区域的设备对应和内存效果认证仍未完成。

`softmax_region_device_summary.json` 汇总后续设备证据：五次启动各重放两次，全部存储字节一致；
固定规则、不根据输出调整的完整网格仿真覆盖 1,026 个写入元素，全部逐位一致。
`softmax_region_ir_writes.jsonl` 的通用 rounding-check 写入检查仅两项通过，另三项拒绝；
诊断记录确认原因均为浮点求和归约顺序未声明，不能把硬件仿真通过直接当作这个检查也已通过。
设备逐位重放、固定规则仿真、严格实数参照和内存效果检查仍分别记录，不自动合并为语义认证。

`softmax_region_emulated_writes.jsonl` 在同一次固定规则仿真中同时核对输出和写入事件。
五次启动均完整覆盖网格、所有捕获存储及声明输出范围，1,026 个写入元素逐位一致。
该检查包含未改变数值的写入，不以字节差分代替写入事件；它不覆盖设备实际访问追踪，
也不改变上述通用 rounding-check 的拒绝结果。共享的写入结果检查现在还拒绝错误网格坐标、
重复程序和遗漏的捕获存储。

`softmax_region_composed_g.json` 用 numerical-difference 模式共享参照内存，完整执行五次启动，
不将中间设备值重新作为输入。所有启动无中止、无附带原因；8 次状态传递实际均有字节与版本计数一致的证据，
无外部状态重置。`softmax_region_three_way_summary.json` 汇总全部 258 个输出坐标的严格差区间：
最大值与计数的 G−f 精确为零；损失和 log-mass 的区间均包含零，绝对值上界分别约为
7.45e-15、2.07e-15。相比之下，本例 K−f 约在 1e-7 量级。
这些是固定输入、所选实数任务下的区间界，包含零不等于证明恒等或普遍不存在 bias，合同状态不自动升级。

独立参照的受控入口为 `softmax_reference.evaluate_region`：先按完整表达式核对全部四个输出、
标签切片、忽略值、归约轴及均值分母，再确认输入的 FP32 转换精确，最后才求值。
节点改名或输出重排不影响对应；不同合法公式返回未建立，不判成错误。
`softmax_region_formula_checked.json` 记录了通过该入口重新计算的结果，严格端点与此前完整参照一致。

## 计数

共 5218 个实例。

| 类别 | 实例 | 内容 |
|---|---|---|
| T（torch 算子与组合） | 5083 | 本质错误轮第一阶段 1724（交叉熵 274、融合线性交叉熵 72、池化 776、index / scatter 602）与 s3 补充 342；第二阶段 2b 的 15 个家族 391；通用轮 18（15 个回归案例与 3 个稀疏演示）；容差研究 21；flash attention 7；three_way 2120；组合研究的 Liger / FlagGems 语料 428 次调用与 TF32 修改 1；`4_bugs_cases` 中的 23 个缺陷 |
| I（出题方定义的程序） | 107 | 盲测 v1 37、v2 38；结构验收 v1.1 32 |
| P（任务就是程序） | 28 | dsl_v2 的官方测试（12,766 次启动，一行）；组合研究的原语与随机程序（两行）；第一批验收 24 项；B024 |

## 已知要在第 4 阶段补的

- **three_way 的对应**：
  - 2120 个单元中，1728 个按 kernel 源码与 Inductor 生成代码里的计算图片段精确对上；逐点、归约、持久归约三类 kernel 中有 1573/1574 个对上。
  - 其余 391 个都是模板 kernel（`triton_tem_`），是自动调优时试跑的候选配置，最终的生成代码里没有它们。第 4 阶段先去掉块大小等配置常数再按源码对应；还对不上的，开溯源选项重新编译。
  - 另有 1 个逐点 kernel 的源文件已不在缓存里。
- **融合线性交叉熵**：72 个条件当时只在汇总里有计数，没有逐条的状态记录。条件定义在 `pre-reorg-20261009:scripts/essential/conditions.py`。
- **当时的「无 f」「契约外」「规格拒绝」**：
  - `status_then` 一栏照录。例如矩阵乘的批量广播被拒绝、空维归约、含 −∞ 的行、Adafactor、clip_grad_value_、checkpoint 与训练程序没有 f。
  - 这些正是 V3 要清零的对象。
