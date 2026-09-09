# 选定时间步的 gradient × SiLU activation

这是正在接入的 Mamba 计算，不是已经完成的训练实验。
它与已测的 gated-SiLU backward 不同：源码计算的是
`gradient[channel, time] * silu(activation[time, channel])`，
不是对 activation 求 SiLU 导数。

`selected_silu_product_reference.py` 核对完整函数体、输入地址、时间索引、
掩码和输出写入。只有两个输入选择同一时间步、尺寸和布局匹配、没有输出别名时，
才允许使用相同输入的 FP32 参考。公式检查不证明均值bias非零。
CPU测试独立对乘法权重求导，并拒绝改变时间索引、算术、掩码或写入地址的源码。

四模型源码清单中的此类定义数量较大，很多只是不同时间步或序列长度的重复生成。
新增检查器的匹配数量不能称为新增机制数量，也不能直接计作支持训练测量。
仍需通过已有参数映射绑定位置、验证实际执行，再调用共同的三阶段统计代码。

扫描由 `scripts/scan_selected_silu_product.py` 读取旧清单绑定的全部源码执行，
包括没有 Triton 定义的源码文件；它不读取数值结果。初版扫描只有42份非空源码，
其输出保留为部分记录；v2补齐原45份源码分母，不覆盖旧文件。

当前运行中的其它家族保留冻结代码。这个独立新增模块尚未接入正在运行的队列，
不改变它们的阈值、输出或参考实现。源码匹配、CPU公式测试和真实训练验证分开验收。

## 参数映射缺口与接入方式

Mamba seq64 的64个此类定义在训练图中共有1536个调用输出。旧的逐AOT端点
映射未收录这些位置：它们是编译器生成的内部临时结果，没有一一对应的AOT tensor。
直接套用旧绑定程序得到0个位置，不是这些计算没有参数作用，也不是零bias。
空计划保留在 `family_plans/mamba64_selected_silu_all_bound.json`。

`bind_internal_reference_cases.py` 复用已有的下游计算区域和参数映射，按固定的
AOT距离、参数名和任务ID选择声明测量参数；不读取数值结果。新计划
`family_plans/mamba64_selected_silu_internal_bound.json` 绑定1536个待测位置。
它保留内部输出的 `exact_aot_endpoint_id=null`，不伪造对应的AOT节点。
内部临时结果仍可用其本身的调用前输入计算参考，再让真实backward继续运行。
静态选择仅提供待测参数；是否实际可达、替换是否成功仍必须由运行验证。

新增家族通过9项CPU公式与拒绝错误输入测试。完整源码扫描v3含45份源码，
匹配448条此类定义；合并原四个家族后，957条定义中463条有严格源码参考，
494条仍需其他参考。见 `common_input_source_coverage_with_selected_silu_v3.json`。
463不是已完成训练测量数量，1536也不是新增机制数量。

## 实际指针布局检查

原始生成调用中，gradient布局为`(1536,64)`、stride`(64,1)`，
activation布局为`(1,1536,64)`、stride`(98304,1,1536)`。
后者是稠密转置，直接`reshape(-1)`会按逻辑坐标重排，不能代表kernel的线性指针读取。
新参考先验证存储稠密且不重叠，再按物理存储顺序读取；带空洞或重叠的输入仍拒绝。
新增转置布局测试后，参考模块10项测试通过。

`mamba64_selected_silu_preflight_retry1` 已运行一个原始保存kernel的三组局部输入；
`mamba64_selected_silu_preflight_layout` 又使用上述真实转置布局复查。
两次的普通、宽范围activation和零gradient输入均与声明参考逐位一致，输入未被改写。
这只验证该kernel的局部执行，不是自然训练bias的阴性结果，也没有loss结论。
首次预检查因旧版`grid`导入接口不可用失败，日志保留；修正使用生成代码中原有的
调用签名，不改变kernel源码。

新的运行参考记录为`mamba64_selected_silu_contracts_v2.json`，对应完整待测计划
`family_plans/mamba64_selected_silu_internal_bound_v2.json`；按原顺序每4个位置分批，
共384批，不依据数值结果选择。真实训练运行与逐批复算仍待执行；
其它仍在运行的采集依赖保持冻结，不因新增参考而更改旧协议。

旧SiLU队列完成后，共享采集入口已接入`source_reference_registry.py`。
源码发现与运行使用同一份家族注册表，不需要为不同模型另写判断逻辑。
原四个家族的变体、后缀和统计阈值不变；注册表源码也加入新的采集快照。
注册表调整后的完整CPU测试为710项通过、1项跳过。

最终执行清单为`family_plans/mamba64_selected_silu_internal_execution_plan_v1.json`，
仍为原1536位置，额外保留完整来源位置数；对应分批目录为
`family_plans/mamba64_selected_silu_execution_batches_v1`。
首批`mamba64_selected_silu_batch000`已开始真实训练采集。
其余383批分为四条互不重叠的队列，必须在首批退出成功且复算通过后才启动。
统一清单审查入口读取`mamba_selected_silu_execution_manifest.json`；
首批与队列的启动记录都不当作完成记录，尚未采集的位置保留在1536分母中。

另外，Mamba seq128和seq256的静态参数映射已分别扩展为6679、12824个
backward位置，与旧清单重叠的73、77项全部一致。这不是这两个长度的运行验证。

这里声明的candidate是保存的PyTorch/Inductor展开式Mamba计算，
不是`mamba-ssm`或`causal-conv1d`扩展的快速计算路径。环境中缺少这些扩展时，
Transformers会打印使用顺序实现的提示；必须继续核对实际生成源码与冻结清单，
不能因此宣称已经测量快速扩展，也不应临时安装扩展后沿用旧candidate身份。
