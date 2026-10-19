# 保留 vLLM 分片合并函数的设备核对

`observed.json` 是本机 RTX A6000 的两组调用记录。`run.py` 从保留的
`VLLM_UA.py.gz` 和 `VLLM_HELPERS.py.gz` 原样提取 `reduce_segments`、
`cdiv_fn`、`find_seq_idx`，放入临时模块后调用；未运行完整 vLLM launcher
或 attention producer。临时模块直接使用安装的 torch/triton，提供未使用的
FP8 默认值；调用明确选择 `USE_FP8=False`，因此不核对量化输出。

任务采用 K14_COLL 的 `scaled_numerator_out`。两组序列长度分别为
`[5,0,3]`、`[1,9]`，query offsets 为 `[0,2,2,3]`、`[0,1,3]`；
空序列不发起 query。每组 grid=(3,2)，四个分片槽，TILE_SIZE=2，
HEAD_SIZE=HEAD_SIZE_PADDED=4，BLOCK_Q=1，num_warps=4。
source 的动态分片选择包含一、二、三个有效分片，未用输入槽填 NaN。
输出是 float32，strides=(12,4,1)，storage_offset=1；每组完整保存38个存储位置。

每个 token/head 的有效分片具有相同的有限 shift，因此指数缩放严格为1。
期望输出由收到的分片 numerator 精确求和后除以收到的 mass 精确求和得到，
结果均可用 float32 精确表示。这些输入在设备执行前确定；独立核对器是在
设备实验后编写的，不是盲测。它不导入 `kernel_spec` 或 GPU 库，从实际输入位模式
复算全部结果，并核对任务记录与实际输入、重复执行及完整输出存储。

两组每次共48个输出坐标，分别用两种不同值初始化输出存储；所有输出精确相等。
每组14个非输出位置保持初值，全部输入缓冲区的前后位模式相同。记录缺失、
错误任务输入、输入改写和伪造相符标记会被核对器拒绝；若差异如实记录，
核对器保留差异数量，而非把它改成通过。

```bash
/data1/tzh/envs/liger/bin/python 5_specification/evidence/partial_merge/verify.py
# 复现需要可访问本机 GPU，并设置仓库缓存目录；--out 必须指向新文件。
/data1/tzh/envs/liger/bin/python 5_specification/evidence/partial_merge/run.py --out .cache/partial_merge_new.json
```

这提供固定调用上的输入、输出地址和精确结果证据，不认证任意长度、非等 shift、
全空 key 行、log2、FP8、其他布局或完整 attention。分片数据在此人工构造，
没有证明真实 producer 的键集合恰好形成所需分区，也没有对写集合作全域证明。
`semantic_verdict` 保持 `not_established`，
不增加可信 kernel 覆盖计数。保存内容未覆盖先前 `.cache/partial_merge_device.json` 的首次试验。

## 实际调用的 TTIR

`observed_with_ttir.json` 是后续调用返回的编译 TTIR、签名及编译常量，原记录未覆盖。
两份记录的实际输入和设备输出相同。`evaluate_ttir.py` 将记录中的缓冲区放入互不重叠的
模拟地址范围，按签名绑定全部参数，用现有 DSL 的 `numerical_difference` 模式计算。
该映射只适用于此函数的参数相对寻址，不证明任意原始指针或别名关系。

最新结果是 `ttir_evaluation_with_reads.json`：每次完整执行6个程序实例，没有中止；
每次24个输出地址恰好匹配任务，没有其他缓冲区写入。三种分片输入的实际读取坐标也与
任务列出的全部分片对应，输出初值没有被读取。每个输出的 Gπ 包围均包含任务值，
但宽度不为零，因此记为 `enclosed_not_equal_proved`，不声称 Gπ=f。
没有增加容差，也没有为通过这些样例修改 DSL 的数值规则。
不支持的 IR 指令会中止求值并保留未建立状态，不从未写出的存储初值推导矛盾。

```bash
/data1/tzh/envs/liger/bin/python 5_specification/evidence/partial_merge/evaluate_ttir.py --out .cache/partial_merge_ttir_new.json
```

## 真实 producer 与 merge 的固定调用

`run_producer.py` 原样提取保留源码中的 JIT 函数，连接
`kernel_unified_attention(IS_3D=True)` 和 `reduce_segments`。使用33个query、
33个key、单head、32个坐标，causal、无窗口/softcap/sink/量化；Q=K=0，
V[j,c]=2j+c-5。逻辑块按物理块 `[2,0,1]` 放置，未使用的KV尾部填NaN。
BLOCK_Q=BLOCK_M=32、TILE_SIZE=16、四个分片槽，包含被读取的空分片和完全未写入的分片槽。
设备记录及两个函数的实际编译TTIR在 `producer_observed.json`，不覆盖先前试验。

`verify_producer.py` 仅用标准库，从实际输入恢复逻辑序列，逐query列出key分区，
独立核对完整分片状态，再把分片并集与完整因果attention比较。每次有51个非空分片，
32个未进入循环的空分片（M=-inf,L=1,U=0），16个整行掩码空分片（M=0,L=0,U=0）。
第四个槽未写入且不参与merge。零质量必须由分片含义确认，不能仅检查L=0。
所有非空分片的mass和完整numerator均等于独立求和；每个query的key恰好出现一次。
测试另从相同实际输入调用当前PAGEDATTN参照，并在空键证明之后把两种表示转换为K14空状态，
两份参照的全部输出均与独立定义一致。

设备最终1056个坐标中，每次有122个K-f差异（120正、2负），最大绝对值1/262144；
两次不同初值下输出位模式相同。全部134个非输出位置保持初值，producer没有提前改写最终输出，
输入及merge读取的分片前后不变。这将该样例的差异定位到merge阶段。完整链的后续Gπ
求值见下节；不能把K-f改称Gπ-f，也没有采样分布支持统计bias结论。
`producer_independent.json` 保留完整分区和实际差异，核对器不会把非零差异变成通过。

```bash
/data1/tzh/envs/liger/bin/python 5_specification/evidence/partial_merge/verify_producer.py --out .cache/producer_audit_new.json
```

## 完整链的参照状态传递

最新原始记录为 `producer_with_arguments.json`，对应完整链结果为
`producer_composed_with_arguments.json`。启动前按函数签名绑定实际传入参数并保留默认值，
逐项记录标量类型和值以及 tensor 数据指针；求值前核对完整参数名集合、任务选择和实际
存储对应。错误 scale、类型、指针或缺失记录均拒绝，旧记录不自动获得这项证据。
此前的布局记录 `producer_with_layouts.json` 与结果 `producer_composed_with_layouts.json`
保持原样。四个时点均直接读取实际 tensor 的
storage/data 地址、存储字节数、shape、stride、dtype 与 device。
重定位前核对这些布局与该固定任务相符、不同缓冲区存储范围不重叠，且跨启动未变；
偏移、stride、长度、别名、时点缺失或布局变化都会拒绝。旧记录没有这些观察值，
仍明确标为仅依据脚本声明，不补造实际地址证据。新 GPU 调用及组合求值保留了此前的
全部输出和差异结论；新旧记录分别保存，不覆盖。

这些核对补充的是本次调用的输入/状态/输出对应证据。绑定是否完整与 G 是否等于 f
是两个问题：不能要求先证明 G=f 才允许绑定，也不能因记录了实际布局就宣布完整绑定
认证。这里仍依赖已记录的执行范围、固定任务选择和参数相对寻址；完整 vLLM API、
其他输入或任意外部写入不在结论范围内。

`producer_with_versions.json` 另存实际调用及各缓冲区在两个launch前后的版本计数，
不覆盖原记录。`evaluate_producer.py` 使用现有 `evaluate_sequence`，让merge读取
producer算出的参照状态。该阶段结果是 `producer_composed_exact_values.json`，
保存的有限设备值、任务值和已建立的区间端点均为精确有理数字符串。
较早 `producer_composed.json` 的设备值是显示小数字符串，原比较使用精确接收值，
但重新核查数值应使用新记录或原始位模式。

两次调用各有8个producer实例和33个merge实例，全部完成。producer只写三份分片状态，
merge只写最终output；1056个输出都得到完整组合参照。实际字节及版本计数共同支持状态
连续性，未发生中间状态重新载入。版本计数只记录ATen可见修改，已记录Triton写回由
对应launch求值处理；这不是检测任意外部原始指针写入的证明。

每次122个K值严格位于Gπ包围之外，为这组输入上的非零数值差异提供见证。
其余934个K值位于包围中，不据此证明K=Gπ。全部1056个f值位于Gπ包围中，
宽度非零，不证明Gπ=f，也不能排除更小的语义差异。四项专项测试覆盖完整状态传递、
同字节但版本变化的拒绝、缺版本记录时不伪造计数，以及producer中止后不产生差异结论。
所有结果保持 `semantic_verdict: not_established`。

```bash
/data1/tzh/envs/liger/bin/python 5_specification/evidence/partial_merge/evaluate_producer.py --out .cache/producer_composed_new.json
```
