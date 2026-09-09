# 逐通道一维卷积：接入审查

这是此前十族目录之外的具体接入，不是新增 bias 或训练后果结论。
补查历史记录发现 Gemma 视觉卷积已经做过 forward/backward gradient 测量，
因此它不是仓库首次出现的卷积族。Mamba 的逐通道 Conv1d 与 Gemma 的视觉
patch convolution 可区分为实现子类，但按卷积族合并计数。
同一计算的不同模型、序列长度和调用位置只计一个家族。

## 已核对的真实计算

Mamba seq64/128/256 的 forward segment0 各有 24 个
`extern_kernels.convolution` 调用，共 72 个位置。其设置为 stride=1、padding=3、
dilation=1、groups=1536、非转置、无内部 bias，权重预期为 [1536,1,4]。
`depthwise_conv1d_source.py` 检查调用表达式，运行时仍需检查真实张量与调用身份。

名字含 convolution 的相邻 Triton kernel 不能都称为卷积：
`...split_transpose_1` 实际进行布局转换；`...split_transpose_2` 将 bias 加到
BF16 卷积输出后再次写入 BF16。卷积本体属于外部库，整体是混合计算路径。

## 参考与训练边界

`depthwise_conv1d_reference.py` 逐卷积核位置显式计算逐通道 cross-correlation，
不调用 candidate 的卷积 API；FP32 累加后转换回输入精度。这是声明参考，不是
数学实数真值。它直接读取现有外部调用 observer 的调用前快照，统计公式不改变。

现有 seq64 task 清单中的 `forward:3:output_0` 属于内部实现缓冲区，
没有独立 exact AOT endpoint；清单记录它由 `forward:4:in_out_ptr0` 的闭合
语义输出覆盖。因此不能把源代码检查通过等同于完整训练接入。下一步应核对
卷积加 bias 边界及参数路径，或补内部输出的明确绑定；不能通过名称/shape 猜测。
完整区域替换和只替换卷积输出必须使用不同的比较声明。

当前完成参考与源代码检查，尚未完成该族的真实三阶段采集、bias 确认或 loss 验证。
原始来源位于 `results/coverage/runtime_releases/mamba_seq{64,128,256}_r1/trace/`。

## 接入进展（2026-09-08）

`mamba64_convolution_bound_v1.json` 已保存 24 个卷积输出边界的参数绑定，
参数路径经下游独立 bias 输出连接；这不把内部卷积缓冲区改称 exact AOT 输出。
相关记录位于 `results/property/numerical_coverage_v1/`，不替代全量参数映射。

独立扩展 `scripts/convolution_capture_observer.py` 复用现有采集接口，不修改
统计公式。调用前保存输入，从实际执行栈核对源码调用，缺少声明绑定时拒绝执行。
`test_convolution_capture_observer.py` 连同卷积参考和源码测试共 15 项通过。
这里的执行位置测试使用文件中真实调用的 CPU 测试程序，不是 GPU 卷积训练结果，
也不证明实际库内部使用了哪个设备 kernel。真实三阶段采集仍待完成。

## 历史卷积记录，不能漏算

`results/property/tcmp_allop_v1/candidate/gemma3_vision_convolution_population.json`
及 `gemma3_vision_convolution_backward_population.json` 各保存 8 个状态、
完整 gradient Gram 与所测参数范围。旧名称 population 和旧 sign-flip 输出
不自动赋予新协议的总体保证。本次只核对记录，不重新运行，也不由它们推出
实际 optimizer update 等价或训练质量结论。

## 一状态接入预检

`mamba64_convolution_reach_v1/legacy/engineering_reach.json` 记录 24 个位置，
每个位置的 local changed-coordinate count、local error energy 与所选参数的
gradient error energy 均为零，进程正常退出。这里的 `carrier_reached=false`
表示没有观察到该数值替换带来的梯度差异，不代表运行失败，也不能证明参数路径
不存在。预检未给出完整状态集合的 actual-update 判断。

源码已保存于该目录的 `source_snapshot.json`。随后对相同 24 个位置、相同参考
启动 32 状态采集，记录目录为 `mamba64_convolution_capture_launch_v1`。
该启动不是完成证据；完整三阶段与统计结果须另行核验。
# 完整测量补充（2026-09-08）

`results/property/numerical_coverage_v1/mamba64_convolution_capture_v1/completion_verification.json`
现已记录 24/24 个位置通过保存记录一致性检查，每个位置包含 32 个状态。
逐文件重新核对 SHA256 后汇总原坐标统计：LOCAL、PARAMETER_GRADIENT、
PARAMETER_WRITE 各有 768 条观测，所有 effect energy 与非零差异坐标数均为零；
另存的 ADAMW_UPDATE 统计也为零。

该结果只覆盖声明的 Mamba seq64、逐通道卷积输出替换、参考实现和参数范围。
它不是新增 bias，不是所有卷积实现的等价保证，也不是独立重复 GPU 复现。
本次协议未预声明等价性 margin，因此正式 equivalence_decision 保持
NOT_ASSESSED，不在看到零差异后回填阈值。历史卷积结果与这次测量均保留。
