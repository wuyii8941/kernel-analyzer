# 外部公开调用：两种参照路线的开发与验收

本批由真实公开接口出发，先复用已有规格；只对实际发现的缺口补充参照。
`selection_20261019.json` 在运行前固定五个 attorch 开发调用和四个 trident 验收调用。
两库此前均有静态调查，attorch L1 的相等点问题也已被怀疑；因此不是盲测或首次接触命中率。
trident 是本次修改未使用的另一组实现，编译、导入失败仍留在分母中。

## 任务依据与结果

attorch 的公开损失模块继承 PyTorch 对应类，模块文档写明同形输入和 reduction，上游测试核对
前向及 input/target 两个梯度。Softmax 公开类明确仅支持末维。trident 的公开函数和测试分别
对应 torch.sum、torch.mean、torch.relu、torch.softmax。原仓库版本、路径和具体资料在选择文件中。
标准实现固定为安装的 PyTorch 2.10.0+cu128，CUDA float32、无 autocast。

每个可运行调用包含两个形状 `[4,32]`、`[3,17]` 和两组输入。样本含相等、正负差及分段边界；
它们是有意选择的诊断输入，不是训练输入分布。检查完整前向和全部承诺的梯度，并确认数据输入未改动。

| 调用 | 标准实现路线 | 独立数学路线 | 当前缺口 |
|---|---|---|---|
| attorch MSE mean | 四组完整输出逐点一致 | 复用 MSE 的前向和两侧 VJP | 尚未正式建立调用级 Gπ 与绑定 |
| attorch L1 mean | 前向一致；两侧梯度在相等点不一致 | 复用 L1，精确零子梯度独立确认该差异 | 统计 bias 和正式绑定未建立 |
| attorch SmoothL1 beta=0.5 | 四组完整输出逐点一致 | 首轮缺失；新增 HUBER 后已复核保留结果 | 正式绑定及统计 bias 未建立 |
| attorch Huber delta=0.5 | 四组完整输出逐点一致 | 同上，完整前向和两侧梯度均可求值 | 同上 |
| attorch Softmax | 前向与非均匀 upstream 的 VJP 均通过声明容差 | 复用 SOFTMAX；补上原始输入 VJP 后两项均可求值 | 正式绑定及统计 bias 未建立 |
| trident sum、mean、ReLU、Softmax | 四个调用均因缺少 nvtx 导入失败 | 未在这四个调用上运行验证 | 环境依赖；不能计为通过，也不能归为算子错误 |

L1 的最小保留见证：`[4,32]` 的第一个坐标 x=target=0、upstream=1 时，公开 PyTorch 与
精确 L1 规格都给 dx=dtarget=0，attorch 给 dx=1/128、dtarget=-1/128。
四组输入共13个相等坐标，每侧梯度各13处与标准实现不一致。源码的 `where(0 <= diff,1,-1)`
解释了该已测差异；不需要为此新增规格。这个结论限于保留版本和任务，尚未量化训练影响。

## 实际驱动的完善

开发第一版只给 softmax 各输出相同的 upstream，主要检查了真实梯度为零的特殊方向。
验收前保留第一版结果，并在 `selection_20261019_v2.json` 明确改用非均匀 upstream；候选名单、
形状和种子不变。softmax 前向相对标准实现最大绝对差约2.98e-8，VJP 最大约2.98e-8。
这里的容差对照不是严格相等或无 bias 结论。

原 SOFTMAX backward 规格接收保存输出，不能直接代表公开调用从 logits 开始的完整 VJP。
现新增派生操作 `vjp.aten.softmax(g,x,dim)`，从 x 独立求出概率包围，再用完整 Jacobian 公式
计算梯度，不将候选概率当成正确前向。原数学合同和数值部件均复用；依据、独立 Arb 复算和拒绝
用例见 `../../evidence/softmax_implementation.md` 及 `../../tests/test_softmax_public_vjp.py`。

SmoothL1 与 Huber 的真实调用暴露了数学参照缺口。依据保留的 PyTorch 2.10 官方模块文档、
wrapper 和公式实现，新增共用 HUBER 合同：明确 beta/delta、三种 reduction、beta=0 的 L1
约定及完整 input/target VJP。依据与独立核验见 [实现说明](../../evidence/huber_implementation.md)。
新增参照不是由 attorch 的输出反推，现有运行入口已接入；原选择文件和旧结果保留不改。

`results/loss_math_replay_v1.json.gz` 用新参照复核原开发组的8次调用，没有重新运行 GPU，也不是新验收。
每种损失各720个输出及梯度坐标：522个等于精确数学值，198个存在精确非零残差。
PyTorch 和 attorch 的残差完全相同；非零部分均来自 `[3,17]` 形状，含不能在二进制有限精度中
精确表示的除以51。这些记录区分标准实现相符与数学相等，未据此宣称语义错误或统计 bias。

trident 的 nvtx 在其 requirements.txt 中明确声明。通过指定 localhost:27891 访问 PyPI 时，
代理返回 Connection refused；未使用其他网络路线，未用空模块替代依赖，未更换四个失败候选。
环境恢复后用同一选择和运行器在新目录重跑；如果修改了基于验收反馈的工具逻辑，这组材料即成为
开发证据，后续还需要另一组验收。

## 参数对应与另一实现的补充验收

原运行器只落实了首批固定的 mean 和0.5参数。后续改为候选、标准实现和数学查询共同读取显式
`parameters`；未认识的参数拒绝，未认识的操作不再自动走 softmax。当时 MSE/L1 非 mean 请求仍保持
原规格的 unsupported，不能把请求偷偷改为 mean。后续显式扩展见下段；原 mean profile 继续拒绝非 mean 请求。

`selection_loss_parameters_20261019.json` 在运行前选定21种真实 attorch 配置：SmoothL1 的
beta=0、0.1、0.5、2，以及 Huber 的 delta=0.1、0.5、2，各含 none/sum/mean。
`parameters` 是传给构造器的关键字参数；entry 中的字典仅展示同一组参数，不是位置实参。
84次完整前向与双侧 VJP 均运行并获得数学参照。正阈值的72次调用与 PyTorch 在声明容差内相符。
beta=0 的12次调用前向相符，但每侧梯度在 none/sum/mean 下分别有12、13、13个不一致坐标。
例如 x=target=0、sum、upstream=1：attorch 给 dx=1、dtarget=-1，PyTorch 与已选 L1
子梯度规格均给0。none 的零 upstream 会掩盖一个相等坐标，因此必须保留完整 upstream。
这一问题由已读源码推测后复现，属于开发发现，不是未知实现的盲测发现。

随后在规格不变的情况下，增加另一实现的补充验收。`selection_flaggems_forward_20261019.json`
固定 FlagGems 5.0.2 experimental SmoothL1/Huber 的原始前向函数；这些入口没有提供完整公开
autograd，因此任务明确只承诺前向结果与输入不变，不删掉某个失败的反向结果来扩大成功数。
选择 `[2,19]`、`[5,33]` 与新种子，以及 beta=0/1.5、delta=0.125/1.5 和三种 reduction。
原模块按文件直接加载，代码不改写；CUDA 输入保证走原 Triton 路径。包元数据、许可证及完整
原模块文本一并保留。该包此前用于其他任务，这两份接口也在选任务时读过，不声称首次接触盲测。

48次前向调用、1656个输出坐标均完成标准实现和数学参照比较，全部输入保持不变。
相对 PyTorch 仅 SmoothL1(beta=1.5, none) 有36个不相等坐标，最大绝对差约5.96e-8；
全部在声明容差内。精确数学比较另存，不能从容差通过推出无 bias。这里没有修改规格来匹配
FlagGems 结果，也没有新增合同或正式绑定。原 trident 四项导入失败仍保留；本次没有验收另一
实现的反向、所有通用算子或真实训练分布。

结果为 `results/loss_parameters_v1.json.gz` 与 `results/flaggems_forward_acceptance_v1.json.gz`。
这一阶段51项相关专项测试通过，随后继续处理真实接口承诺但数学参照缺失的 reduction。

## MSE 与 L1 的完整 reduction

attorch 的 MSE/L1 接口与测试明确支持 none/sum/mean。依据已保留的 PyTorch 2.10 模块公式、
wrapper 以及 refs `_apply_loss_reduction`，两份现有合同各增加一个全 reduction profile。
三种计算复用同一精确求值代码；原 mean profile 及其已绑定调用的范围不变。`none` 保留完整
输出并要求同形 upstream，`sum` 不除元素数，所有反向都返回 input 和 target 两侧梯度。
L1 文档的一句“sum 除以N”与同页公式冲突，明确公式、后续说明和 refs helper 共同确定 sum
不除N；没有根据 attorch 的结果选择公式。详情见已有 MSE/L1 实现核验说明的扩展部分。

`selection_mse_l1_reductions_20261019.json` 先选定两种损失×三种 reduction，再运行24次
完整前向与双侧 VJP。MSE 的12次调用全部与 PyTorch 逐点一致；L1 前向一致，每侧梯度在
none/sum/mean 下分别有12、13、13个相等点坐标不一致，均被已有零子梯度约定识别。
所有请求均获得完整数学参照，没有把正确 mean 答案用于 sum/none。

保持规格不变，`selection_flaggems_mse_20261019.json` 使用另一份 MSE 前向实现、新形状
`[3,23]`/`[7,41]` 和新种子，完成12次验收。none 的712个坐标和 sum 的4个坐标均与精确
数学值相等；mean 的4个坐标保留非零残差，相对 PyTorch 有2个不相等、最大约4.77e-7，
均在声明容差内。全部输入保持不变。这一验收只覆盖前向，不是另一实现的反向验收或统计 bias 结论。

完整记录是 `results/mse_l1_reductions_v1.json.gz` 和 `results/flaggems_mse_acceptance_v1.json.gz`。
没有新增合同或正式绑定；新 profile 的可用性与真实 kernel 的完整绑定继续分开报告。
扩展后的完整必需检查通过，包括882项注册入口等单元测试及78项相关 pytest 检查；
原 MSE/L1 mean 调用的2项工具绑定检查通过。合同检查包含39个样例及16个被拒绝的错误公式。

## 后续完整调用绑定

上述 FlagGems MSE 随后对 `[7,41]`、三种 reduction 完成了正式前向绑定，新增一份调用计划，
而不是把本批所有通过项自动升级。每种 reduction 各24个输入都建立了完整调用级参照。
sum/mean 的内部累加器由通用 producer 检查确认在本调用内初始化为零；没有把其观测值作为
额外任务输入。mean 检出的程序系数差可独立解释为 `Gπ−f=(45/2^32)*f`；sum 的统计判断
因既定分布检查失败而保留“无法判断”。完整来源、范围、测量报告和推导见
[FlagGems MSE 绑定核验](../../evidence/flaggems_mse_binding.md)。本节不改变之前记录的开发/验收身份。

## 公开 top-k、cumsum 与索引调用

`selection_common_20261019_v1.json` 在执行前固定 FlagGems 5.0.2 的公开调用、两种形状和两个种子，
复用 SORT、REDUCE 和 INDEX 参照，对照 PyTorch 2.10.0+cu128 原生 CUDA。24次调用包含
top-k 的最大/最小选择、cumsum、gather，以及并列和 `sorted=false` 两类拒绝范围。
首次运行保留在 `results/common_before_keywords_20261019_v1.json.gz`：12次完成两种参照比较，
4次 gather 暴露 `index=` 被内部 `idx` 名称拒绝，8次 top-k 按既有规格拒绝。进程非零退出。

根据已归档的官方 schema 和 API 文档补齐参数对应后，同一批复测为16次完整输出精确一致、
8次继续拒绝；记录为 `results/common_after_keywords_20261019_v1.json.gz`。数学公式未修改。
并列和未排序 top-k 的设备结果与 PyTorch 的索引/顺序不同，不能由逐点不等直接推出错误；
当前规格继续不对此裁决。原批 gather 已成为开发案例，不能继续作为未知实现的验收成绩。

随后固定 `selection_index_acceptance_20261019_v1.json`，使用此前本批未执行的 FlagGems
index-select 实现，以及新形状 `[7,43]`/`[9,29]`、新种子、不同索引维度和全 keyword 调用。
8次 index-select 加4次新范围 gather，共12次完整输出同时与标准实现和数学参照精确相等。
记录为 `results/index_acceptance_20261019_v1.json.gz`。它是分开的实现/调用验收，不是外部盲测；
尤其4次 gather 只检验新调用范围。全部调用保持输入不变，包含重复索引；没有为验收结果修改参照。

每个记录包保存所用运行器、选择、原始库源码、版本、输入和完整输出。所有样本实际捕获到 Triton
启动，单独追踪前后的输出与输入一致，启动名称和 grid 一致。追踪也确认公开 dispatcher 入口可以
隐藏内部 ATen 操作；见 [排序拒绝原因](../../evidence/flaggems_sort_binding_gap.md)。这批比较不自动
增加正式绑定或 bias 覆盖。完整必需规格检查通过（含889项 unittest 和78项外部调用 pytest），
工具来源/绑定回归102项及16个 subtest 通过。

运行器为 `run_common_acceptance.py --selection <选择文件> --out <不存在的目录>`。使用主环境，
将 `2_tool/src`、`5_specification` 和 `.cache/pydeps/` 下 `flag_gems_5_0_2`、`pyyaml_6_0_2`、
`sqlalchemy_2_0_48` 加入 `PYTHONPATH`，`LD_LIBRARY_PATH` 指向主环境的 `lib`；
`FLAGGEMS_CACHE_DIR` 指向仓库 `.cache/flaggems`，其余缓存设置同下。该入口在执行错误、
预期内参照缺失或任何所选精确比较失败时返回非零；声明过的规格外拒绝仍完整留在报告中。

## 公开 embedding 前向与完整权重梯度

真实 FlagGems 5.0.2 `F.embedding(indices, weight)` 与 PyTorch 输出一致，但 INDEX 把第一个参数
当成 weight 而拒绝。已按归档的 PyTorch 2.10 API 和 functional 源码修正公开入口：indices 在前、
weight 在后，padding 默认 None；native `aten.embedding` 仍是 weight 在前、indices 在后、默认 -1。
`max_norm` 会修改 weight，不能作为只读 lookup 悄悄放行。INDEX 的默认前向复用原数学公式；
非默认 padding 和频率缩放下的完整前向／梯度复用 EMB，不新造一份公式。原始失败保存在
`results/embedding_public_before_fix_v1.json.gz`。

`selection_embedding_20261019_v1.json` 在批次执行前固定两种实现、输入形状、种子和六种参数组合。
开发使用 FlagGems `[11,7]` 权重、`[2,5]` 重复索引；另一批使用 TorchInductor `[13,17]` 权重、
`[3,5]` 索引和不同种子。组合覆盖 padding=None/0/-1，各自带或不带频率缩放；完整观察 `out`、
`dweight`、dtype、shape 和输入状态。TorchInductor 与标准实现共享框架来源，不能当作独立来源；
两者另与已有精确数学参照比较。这是分开的实现验收，不是第三方盲测。

| 批次 | 完整调用 | 前向精确等于数学值 | 梯度精确等于数学值 | 梯度与 PyTorch 最大差 |
|---|---:|---:|---:|---:|
| FlagGems 开发 | 12 | 840/840 坐标 | 895/924 坐标 | 2.98e-8 |
| TorchInductor 验收重跑 | 12 | 3060/3060 坐标 | 2561/2652 坐标 | 5.96e-8 |

其余梯度坐标保留精确有理残差，不能由它们直接推出 API 错误或统计 bias。全部标准实现比较满足
事先记录的容差，输入均未改变；默认任务中 INDEX 前向与 EMB 前向也相等。前向、反向均实际捕获到
Triton kernel；频率缩放额外的计数和归约启动没有丢弃。

首次 TorchInductor 批次的12项捕获失败全部保留：运行器先预热，缓存了不经过记录器的 host launcher。
生成源码确认使用 Triton，修正为先启用记录基础设施、在不记录启动的上下文中预热后，同一批捕获成功。
记录器关闭 static CUDA host launcher，这项干预已写入报告，没有切换数值精度。首次与重跑的输入完全
相同，11/12份完整输出相等；`padding_last_frequency`、seed 20261127 的5个梯度坐标不同，最大
1/33554432。对此固定做了20次完全不启用记录器的重复执行，也得到两种结果。它支持存在重复执行差异，
并不证明这次差异的唯一原因，也不认证捕获完全不扰动执行。

保留文件为 `results/embedding_development_v1.json.gz`、`embedding_acceptance_capture_failure_v1.json.gz`、
`embedding_acceptance_v2.json.gz` 和 `embedding_repeat_without_capture_v1.json.gz`（均在 `results/`）。
含全部输入、输出、数学值、残差、运行器、库源码或生成代码和捕获到的 TTIR。复现使用
`run_embedding.py --split development|acceptance --out <不存在的目录>`，环境同上；acceptance 另将
`TORCHINDUCTOR_CACHE_DIR` 指向仓库缓存。没有新增正式绑定。修复后的完整必需规格检查通过，含894项
unittest及78项外部调用 pytest；针对性检查17项、87个 subtest 通过。

## max_norm 的完整状态与版本对照

`selection_embedding_state_20261019_v1.json` 固定另一项真实任务：公开 embedding 的 max_norm 更新、
查表和完整 dense weight VJP。比较对象是明确版本的 PyTorch 2.10.0+cu128 原生 CUDA，实现必须观察
全部 `out`、`dweight` 和 `weight_after`，而不是只比较查到的行。共享执行器按任务声明允许权重更新，
仍拒绝索引或 upstream 的值被修改；无 max_norm 的旧任务继续要求权重值不变。缺失或截短后态不能通过。

官方 functional 源码明确：先以 no-grad 更新权重，再查表；即使某行是 padding，只要出现在 indices 中，
也参与更新，之后才按 padding 约定屏蔽梯度。检查包含重复索引、padding=None/0/-1、频率缩放、
一范数和二范数、零上界，以及无需更新的输入。未访问行必须保持原值，输出必须取自权重后态。

开发为 FlagGems `[11,7]` 权重、`[2,5]` 索引；分开的 TorchInductor 验收改用 `[13,17]` 权重、
`[3,5]` 索引和新种子，每批12次完整调用。开发的840个输出坐标、924个后态坐标，验收的3060个输出
坐标、2652个后态坐标均与标准实现精确相等。梯度分别有8/924和22/2652个坐标不相等，最大差分别
约2.98e-8和5.96e-8，全部处于预先规定的比较容差内。所有未访问行检查和输出／后态对应检查通过。

**这里使用版本化标准实现，没有新增严格数学 ground truth。** 归档 CPU 源码含
`max_norm/(norm+1e-7)`，但当前缺少对应 CUDA 源码证据，不能直接宣称严格数值语义相同。
编译后的实际调用仍包含 `aten.embedding_renorm`，FlagGems 也保留原生更新路径；两批状态一致不构成
该更新公式的独立验证。24次查询均保留 EMB 对 max_norm 的 `unsupported`，不增加合同、profile 或正式
绑定的覆盖计数，也不进行 Gπ−f 或统计 bias 裁决。Triton 捕获覆盖查表及梯度，未解释原生 ATen 更新。

完整记录为 `results/embedding_state_development_v1.json.gz` 和
`results/embedding_state_acceptance_v1.json.gz`，包含版本、完整前后态、梯度、比较、运行源码和生成代码。
运行器为 `run_embedding_state.py --split development|acceptance --out <不存在的目录>`，环境同上一节。
状态观察和相关回归29项、46个 subtest 通过；完整必需规格检查通过，含899项 unittest 和78项外部调用
pytest。正式数学参照的缺口仍明确保留，已有只读 embedding 合同的使用条件没有被放宽。

## 矩阵调用：复用公式、补齐参数、保留真实错误

`selection_matrix_20261019_v1.json` 选择公开 linear、addmm、bmm：包括向量和标量 bias、
无 bias 的批量 linear、alpha/beta 缩放、非连续输入，以及 beta=0 时含 NaN/Inf 的加项。
每次观察完整前向、全部输入 VJP 和输入值不变；无 bias 的梯度必须显式为 null。
依据是归档的 PyTorch 2.10 默认 ATen schema、linear/addmm 官方定义和原有 MATMUL.real。
真实参数名此前被内部 a/b/c 命名拒绝；逐项对照来源后补齐六个矩阵接口的显式对应，公式未改，
重复传参和不支持的重载继续拒绝。审核记录见 `../../evidence/matmul_implementation.md`。

开发采用 FlagGems 5.0.2、M/K/N=3/7/5、两个种子；分开的验收采用 TorchInductor 2.10、
M/K/N=4/11/6 和另外两个种子，各16次完整调用。它们是不同实现的有限验收；开发前已经发现
beta=0 的一个反例，不称为独立盲测。两批的数学查询均可取得完整参照。

| 实际路径 | 所选调用 | 与版本化标准实现相容 | 明确结果差异 | 执行/覆盖不足 |
| --- | ---: | ---: | ---: | ---: |
| FlagGems 开发 | 16 | 10 | 4 | 2：scalar bias 报 IndexError |
| Inductor 默认路径 | 16 | 16（只作数值对照） | 0 | 16：矩阵乘走原生库，未满足 Triton 验收要求 |
| Inductor 显式 Triton 矩阵后端 | 16 | 12 | 4 | 0 |

“相容”使用预先规定的 rtol=1e-5、atol=1e-6；精确数学差值另存，不用容差认证数学相等。
显式 Triton 后端使用 `max_autotune=True, max_autotune_gemm_backends='TRITON'`，生成代码及
前后向启动均保留。它是另一种编译配置，不能覆盖默认路径的结果。初次原生路径记录完整保留。

两套 Triton 实现均在 beta=0、加项为 NaN/Inf 时错误传播 NaN：开发4次共60个输出坐标、
验收4次共96个输出坐标；原生 PyTorch 和既有规格都要求忽略加项并得到有限值。
FlagGems 源码仍计算 `bias * beta`；Inductor 的 `addmm_epilogue` 和生成代码同样保留
读取加项并乘零。另用不导入分析工具、不启用记录器的独立程序复现：A 为4×11全一、B为11×6全一，
NaN/+Inf/−Inf 加项均应输出全11，编译后仍为 NaN；有限加项3的对照正常。所有输入梯度符合手算。
这证实所测版本/配置的接口语义错误，不声称它在最新版本仍存在，也不声称首次发现。

scalar bias 的两次开发失败也保留：官方 linear 明确允许零维 bias，但 FlagGems addmm 包装
在广播前读取 `bias.stride(0)` 而报错。其余有限差异集中在非二进制精确的 alpha=.1、beta=.3：
开发7个前向坐标与标准实现不同，最大约2.98e-8；Triton 验收18个前向、5个梯度坐标不同，
最大分别约2.98e-8、5.96e-8。它们不被自动归为接口错误或统计 bias。

记录在 `results/matrix_development_v1.json.gz`、`matrix_acceptance_native_path_v1.json.gz`、
`matrix_acceptance_triton_v2.json.gz` 和 `matrix_beta0_uncaptured_v1.json.gz`，含输入、完整输出、
数学值、残差、错误、运行器、相关库源码和生成代码。复现用
`run_matrix.py --split development|acceptance --out <新目录>`；Triton 验收加 `--triton-gemm`，
并将 `TORCHINDUCTOR_CACHE_DIR` 设为仓库内新目录。开发需要保留的 FlagGems 5.0.2 依赖及库路径。
完整必需规格检查通过（905项 unittest、78项外部调用 pytest）；参数和观察检查23项及78个 subtest
通过。没有增加合同、profile 或正式绑定计数；这些调用级结果不自动成为 Gπ−f 或统计 bias 裁决。

## 归一化调用：默认参数、完整梯度与另一套实现验收

`selection_norm_20261019_v1.json` 固定公开 LayerNorm、RMSNorm、GroupNorm 的8种调用，各用两个种子。
覆盖多维 normalized_shape、可选 weight/bias、常数输入、非连续布局和 RMSNorm 的 eps=None。
开发使用 FlagGems 5.0.2；验收使用 PyTorch 2.10 TorchInductor，并改变 batch、归一化尺寸和种子。
任务依据来自已归档的 PyTorch 2.10 文档及当前安装的公开包装器源码。

原有 `NORM.torch_autograd` 已足够，32次数学查询均得到完整参照。新增运行器 `run_norm.py`
把实际浮点输入、维度、分组和显式 upstream 对应到既有公式；公开 RMSNorm 的 eps=None 依据实际
输入 dtype 解析为 float32 的 2^-23。候选调用仍收到 None，不把解析后的值偷偷传给候选。
观察全部前向、所有存在参数的 VJP、可选参数的缺席，以及输入值不变；不把内部 saved statistics
当作公开输出。本批限定非空有限 float32，不从无法保留 shape 的空列表猜测任务。

| 批次 | 所选调用 | 完成完整 Triton 前后向 | 容差相容 | 其余结果 |
| --- | ---: | ---: | ---: | --- |
| FlagGems 开发 | 16 | 12 | 9 | 2次默认 epsilon 错误；1次近零梯度超容差；4次执行错误 |
| TorchInductor 验收 | 16 | 16 | 16 | 最大标准实现差约4.77e-7，严格数学残差另存 |

FlagGems 的三个具体问题均有公开任务依据：

- 在 `use_gems` 替换公开 `F.rms_norm(..., eps=None)` 的调用中，kernel 实收 eps=1e-5，
  与公开 float32 默认值2^-23不同。另做未启用记录器的对照：原生 None 与原生显式2^-23结果相等，
  FlagGems None 与双方显式1e-5结果相等，随后单独捕获到实际标量1e-5。小输入的首个输出由
  原生约0.52523变为约0.26740。结论限定于替换公开接口的任务；直接调用 FlagGems 自有函数时，
  默认值应根据该函数接口另外选择。
- 多维 weight=[5,7] 的反向返回平铺的[35]，autograd 拒绝；源码的 dw 最后执行 reshape(-1)。
  另一批 Inductor weight=[3,11] 保留完整梯度形状并通过对照。
- 公开接口允许 weight=None，FlagGems 对 None 调用 contiguous 而报错。另一批无 weight 调用正常。

常数 LayerNorm 的一个梯度坐标，数学值和原生结果均为0，FlagGems 得到约−1.76706e-6，
超过预设绝对容差。这里保留精确残差和失败，不仅凭这一点宣布公式错误、统计 bias 或训练影响。

初次开发运行误将 ATen dispatcher 名称传入 `use_gems(include=...)`；该接口要求 Python 函数名称，
导致10次 LayerNorm/GroupNorm 调用未启用 Triton。捕获准入检查拒绝了这些调用。原始16次记录保留于
`results/norm_development_registration_failure_v1.json.gz`，修正注册选择后的结果另存
`norm_development_v2.json.gz`；验收和默认 epsilon 复现分别为 `norm_acceptance_v1.json.gz`、
`norm_rms_defaults_v1.json.gz`。全部位于 `results/`，含输入、输出、数学包围、残差、异常、运行源码、
相关库源码和生成代码。记录器在预热前启用，预热启动不计入验收；捕获时的 host launcher 配置干预
与前面批次相同，并已记录。

复现用 `run_norm.py --split development|acceptance --out <新目录>`，依赖与矩阵批次相同；
验收另设仓库内 `TORCHINDUCTOR_CACHE_DIR`。针对性检查16项及5个 subtest 通过，完整必需规格检查
通过（909项 unittest、78项外部调用 pytest）。本轮没有修改 NORM 公式或增加合同、profile、正式绑定。
这是不同实现和输入的有限验收，不把开发案例计作盲测或用相容率代替泛化证明。

## 池化：完整公开输出与版本化参照

`selection_pool_20261019_v1.json` 在执行前固定10种公开调用：最大池化的普通、重叠相等值、
dilation、padding/ceil mode、无 batch 输入，以及平均池化的默认参数、ceil 尾部、排除 padding、
自定义除数和无 batch 输入。开发为 FlagGems 5.0.2；验收为 TorchInductor，使用不同尺寸和种子，
各20次调用。输入和 upstream 均为有限 float32，观察完整 values、最大池化 int64 indices、输入 VJP，
并检查输入值未变。最大值和索引要求精确相等，平均值和梯度保留完整差异及声明的数值容差。

任务明确选择与 PyTorch 2.10.0+cu128 原生 CUDA 公开接口一致。依据为已保存的官方
`T_MAXPOOL2D`、`T_AVGPOOL2D`、`T_F_AVG_POOL2D` 文档和本地对应版本 functional 源码。
本轮通过 localhost:27891 获取资料仍被拒绝，未绕过代理。相等最大值的索引及梯度选择是该版本
对照的内容，不推导为所有合法最大池化的唯一数学答案。

原 `POOL.finite_max2d_values` 直接复用，数学参照只核对其范围内的前向值，不把对 values 的投影
冒充完整带索引/梯度合同。开发和验收各8次可求出数学值，其余各12次明确保留 unsupported。
原生标准的16次前向数学对照全部精确相等；候选成功返回的14次、共1024坐标也精确相等。
平均池化、索引及梯度本轮有版本化标准对照，仍没有新增已审核独立数学规格。

| 批次 | 完整 Triton 前后向且相容 | 完整执行但不相容 | 仅部分 Triton 执行 | 执行失败 |
| --- | ---: | ---: | ---: | ---: |
| FlagGems 开发20次 | 6 | 2 | 8 | 4 |
| TorchInductor 验收20次 | 16 | 2 | 2 | 0 |

两组不相容均来自 `avg_pool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True,
count_include_pad=True)` 的尾部窗口。保留的未启用记录器复现 `probe_pool_tail.py` 使用4×4全1输入，
upstream 只选择最后一个输出：

| 实现 | 末尾输出 | 尾部四个输入各自的梯度 | 扰动末尾输入得到的前向差商 |
| --- | ---: | ---: | ---: |
| 原生 CPU / CUDA | 1 | 1/4 | 1/4 |
| FlagGems | 约4/9 | 约1/9 | 约1/9 |
| TorchInductor | 1 | 约1/9 | 1/4 |

差商用精确可表示的0.5步长。它是这个线性平均池化见证的交叉核对，不是用一般有限差分认证任意
非线性算子。FlagGems 前后向源码在 count_include_pad=true 时使用完整 kernel_h*kernel_w；
Inductor 前向按尾部有效区域缩小除数，反向 lowering 却在同一选项下使用 kernel_size 的乘积。
这解释了具体版本相对原生任务的差异，以及 Inductor 前后向之间的不一致。没有据此报告统计 bias。

部分执行不能从相容率中消失：FlagGems 最大池化8次只捕获到前向；实际 dispatcher 的反向是
`aten.max_pool2d_with_indices_backward.default`，其注册表却列 `max_pool2d_backward`。
Inductor 的2次 dilation 最大池化反向，由已保留 lowering 源码明确走原生 fallback。
这些调用完整值/索引/梯度均相容，但不能算作 Triton 反向验证。FlagGems 的4次无 batch 调用因
实现按4维解包3维输入失败；原生任务和另一组验收均接受该公开输入形状。

第一版验收最后4次被共用 Python 函数的重编译次数上限挡住。原始结果保留为
`results/pool_acceptance_compiler_limit_v1.json.gz`；改为按任务重置 Dynamo 缓存后，固定20次调用
重新执行到 `pool_acceptance_v2.json.gz`，未修改任务、输入或数值编译选项。
另保存 `pool_development_v1.json.gz`、`pool_tail_probe_v1.json.gz`、`pool_validation_v1.json.gz`，
含完整输入输出、源码、官方文档、生成 TTIR、错误、日志和检查。针对性检查10项及21个 subtest 通过。

复现入口为 `run_pool.py --split development|acceptance --out <新目录>` 和
`probe_pool_tail.py --out <新文件.json>`，环境与 NORM 批次相同。任一不相容、执行失败或 Triton
前后向不完整均使批次进程非零退出。没有增加合同、profile 或正式绑定数量；这轮完善的是完整任务
观察和明确版本的参照路线。Inductor 与原生 PyTorch 共享框架代码；不同实现/尺寸/种子的验收有用，
但不是独立来源的数学证明，也不是未见算子家族的泛化证明。

## 平均池化参照补齐后的复核与新参数验收

上节的数学参照缺口现通过已有 POOL 合同的 `finite_avg2d_autograd` 补齐，未增加合同数量。
它要求调用方明确采用“计入显式 padding、排除 ceil 越过显式 padding 的部分、override 优先”的任务，
返回完整精确前向和输入 VJP。没有把旧草案中的最大池化索引、NaN 或其他维度一起放行。
来源、线性映射推导和64组独立矩阵/转置复算见 [实现复核](../../evidence/pool_implementation.md)。

`results/pool_average_math_replay_v1.json.gz` 复核原20次平均池化调用，全部能得到完整数学参照；
其中2次原候选执行失败仍没有输出。它是旧记录复核，不是新验收。数学实数值与浮点结果之间的
非零残差逐坐标保留；例如除以3或5的普通舍入也会产生残差，不能把所有非零残差都判成实现错误。

公式和实现复核完成后，`selection_pool_average_20261019_v1.json` 固定了另一组8次 Inductor 调用：
不同尺寸、种子、矩形窗口、padding、非连续输入、负 override、无 batch 及单元素参数形式。
全部8次都得到前向和梯度的严格数学参照；6次完成 Triton 前后向并与原生实现相容，
标准差异最大约1.19e-7；2次 `kernel_size=[3]` 的反向编译失败，保留原始异常。

关闭记录器后的独立对照确认：原生 `[3]` 和 `[3,3]` 的前向/梯度精确相同，Inductor `[3]`
反向失败而 `[3,3]` 正常。对应版本 lowering 直接断言 kernel_size/stride 长度为2，未展开这个公开
允许的单元素形式。没有把验收调用改成 `[3,3]` 后计为通过。证据保存在
`results/pool_average_acceptance_v1.json.gz` 与 `results/pool_average_validation_v1.json.gz`。

完整必需检查通过，含927项 unittest、外部调用检查、合同样例和误读检查。当前共61份合同、
118个 profile，其中56个具备声明范围内的参照准入条件；正式调用绑定数量未变。
新8次是已见实现家族上的新参数验收，不能宣传为未见算子家族的泛化证明。
复现新验收使用 `run_pool.py --selection selection_pool_average_20261019_v1.json --split acceptance
--out <新目录>`（selection 路径须相对当前目录正确）；旧记录复核入口为 `replay_pool_average.py`。

## 最大池化：唯一最大值下的完整数学参照

先前批次实际观察了 max_pool2d 的 values、int64 indices 和输入 VJP，但数学参照只能回答 values。
现在新增的 `POOL.torch210_unique_max2d_autograd` 选择 PyTorch 2.10 公开任务，并只在每个有效窗口
有唯一最大值时返回完整三项结果。值和梯度由独立坐标公式给出：窗口输出是唯一最大输入，VJP 将
upstream 累加到所有选择该输入的窗口。`h*W+w` 索引是显式版本任务约定，并经 CPU/CUDA 公共调用
交叉核对。官方资料未充分规定并列时的统一选路，因此并列窗口保留 unsupported，不从候选实现猜测。

`selection_pool_max_unique_20261019_v1.json` 在 GPU 执行前固定了开发和验收两组不同尺寸、种子及
参数。CPU 先对全部16次公开调用交叉核对独立规格，values、indices 和 dx 均逐点精确相等。

| 固定集合 | 完整数学参照 | 候选结果相符 | Triton 前后向完整 | 保留的限制 |
| --- | ---: | ---: | ---: | --- |
| FlagGems 5.0.2 开发8次 | 8 | 6 | 0 | 6次只有 Triton 前向；2次 rank-3 输入因候选按4维解包而失败 |
| TorchInductor 验收8次 | 8 | 8 | 6 | 2次 dilated+ceil 调用只有 Triton 前向，反向不计入 Triton 覆盖 |

成功返回的14次调用与原生 PyTorch 及独立数学参照在全部 values、indices 和 dx 上精确相等。
这说明参照能力已经覆盖这些公开任务，不表示候选实现的所有阶段都有 Triton 覆盖，也不证明所有
max_pool2d 变体无 bias。第一版开发记录把缺少反向 Triton 启动笼统记成执行错误；原始记录保留，
修正后的脚本把“数值已比较”和“阶段覆盖不完整”分开报告，没有修改冻结输入。

证据为 `results/pool_max_unique_development_original_v1.json.gz`、
`pool_max_unique_development_v2.json.gz`、`pool_max_unique_acceptance_v1.json.gz` 和
`pool_max_unique_validation_v1.json.gz`。实现与公式复核见 [POOL 实现复核](../../evidence/pool_implementation.md)。
复现使用 `run_pool.py --selection selection_pool_max_unique_20261019_v1.json --split
development|acceptance --out <新目录>`；部分 Triton 阶段、执行失败或数值不相容都会使进程非零退出。

## 真实卷积调用：复用已有公式，补齐公共接口对应

`selection_conv_20261019_v1.json` 在执行前固定10类有限 float32 训练调用，每类两组输入：
conv1d/conv2d、分组、depthwise、dilation、channels-last、无 bias、无 batch，以及
`same`/`valid` padding。开发实现为 FlagGems 5.0.2；验收改用 PyTorch 2.10.0+cu128
Inductor、不同尺寸和种子。这是不同卷积实现的验收，两家库此前均已用于其他算子，不能称为盲测。

已有 `CONV.real` 包含完整前向及 input/weight/bias VJP，本轮没有新增公式、合同或 profile。
`run_conv.py` 只补公共调用到已有规格的对应：无 batch 输入增加一个大小为1的 batch，结果和
输入梯度再去掉该维；`valid` 对应零 padding；`same` 根据固定版本的官方
`torch/nn/modules/conv.py`，每维总 padding 为 `dilation*(kernel_size-1)`，左侧取下整，
其余放右侧。参照显式补零后调用原规格，再把输入梯度裁回原坐标。若补零写作线性映射 P，
这里执行的就是 `y=CONV(Px,w,b)` 与 `dx=Pᵀ dx_padded`；没有从被测输出推导公式。
候选仍接收原始公共参数，不替它改 stride、布局或 padding。不存在的 bias 梯度保留为 null，
其余输出和梯度必须完整。来源为已保留的 PyTorch 2.10 Conv1d/Conv2d 文档和安装源码；
本轮 localhost:27891 拒绝连接，没有改用其他网络路线。

公共接口对应和原卷积独立检查共15项测试通过，包含手算的非对称 padding、完整梯度、
遗漏梯度及错误 upstream 的拒绝。20项开发条件另在 CPU 上完整求出参照，已保留的 GPU
输入、upstream 和标准输出均逐项对上；这次复算不算新的候选执行或验收。

| 固定集合 | 数学参照 | 候选执行结果 |
|---|---|---|
| FlagGems 开发20项 | 20项完整可用 | 6项完整结果相符；12项接口或编译拒绝；1项超预算；1项因前项超时未执行 |
| Inductor Triton 模板验收20项 | 20项完整可用 | 10项完整结果相符；10项无适用模板 |
| 同一验收条件的默认后端对照 | 20项完整可用 | 20项均运行且结果相符；其中4项没有 Triton 启动，不计为新增验收 |

相符的开发调用共502个输出及梯度坐标，Triton 模板验收共5652个坐标，均等于精确数学值。
默认后端对照的8322个坐标也相等。输入选用小的二进制有理数，便于检查任务对应与公式，
这不是浮点精度压力测试，更不是无 bias 的证明。

实际发现的限制分别保留，不合并成“缺少 ground truth”：

- FlagGems 的 padding overload 接收列表形式的默认 stride，`same` 分支却检查
  `stride == 1`，合法调用因此被拒绝。无 batch 输入也被按带 batch 的维度检查。
  `probe_conv_public.py` 在无捕获工具时复现，三个最小对照的原生 CPU/CUDA 完整结果一致。
- channels-last 分组卷积在反向的 `input_c_stride[None]` 处发生编译异常。
  `probe_conv_channels_last.py` 独立复现同一失败；前向288个坐标与数学规格相等。
  这说明所选版本和调用不能完成反向，未据此宣称存在统计 bias。
- Inductor 的显式 Triton 模板选择不能执行本批1D和非单位 dilation 条件；默认后端
  对照能完成它们。原验收拒绝记录不替换，也不把原生 CUDA 结果计为 Triton 卷积覆盖。
- depthwise 的限时调用在反向 Triton `make_llir` 编译阶段未完成，执行栈和退出124均保留。
  原始顺序开发批次未预先设超时；观察到耗时后设900秒上限，在933秒停止，已完成8项。
  剩余类型分别运行，开始前各设240秒限制。这项运行方式变化属于开发记录，不能包装成盲测协议。

捕获只记录启动及输出存储与参数的对应，不能由参数名或一次 Triton 启动推出完整计算覆盖。
FlagGems 的 bias 梯度包含原生归约，Inductor 的 input/weight 梯度包含原生 convolution
backward；即使最终 crop 是 Triton，也不能隐藏它依赖的原生计算。此批没有建立完整 Gπ
或正式调用绑定，没有进行统计 bias 裁决；合同、profile 和绑定准入数量均不变。

原始记录在 `results/conv_*.json.gz`，逐项执行统计及日志为 `conv_validation_v1.json.gz`。
`run_conv.py --split development|acceptance --out <新目录>` 复现固定批次；`--case <原始ID>`
可单独运行耗时类型，`--diagnostic-after 120` 只输出执行栈；实际运行期限由外层 timeout 控制。
`--compiler-defaults` 单独生成默认后端对照，`prepare_conv_references.py` 只做开发条件的 CPU
参照复算。所有结果另存新文件；原来的通过、拒绝、超时和未执行项均保留。

## 保留材料与复现

`results/development_v1.json.gz`、`results/development_v2.json.gz`、`results/acceptance_v2.json.gz`
保留选择、版本、输入、完整设备输出、标准实现输出、数学参照的精确表示、逐坐标残差区间、
拒绝和错误。它们不覆盖旧记录。可执行运行器为 `run.py`，每次运行另存其源码，输出目录必须不存在。
后续数学复核使用 `replay_losses.py --source results/development_v2.json.gz --out <新文件.json.gz>`；
相对路径按当前工作目录解析，输出文件必须不存在。

```bash
env CUDA_VISIBLE_DEVICES=3 \
  XDG_CACHE_HOME="$PWD/.cache" TRITON_CACHE_DIR="$PWD/.cache/triton" TMPDIR="$PWD/.cache/tmp" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/5_specification:$PWD/.cache/pydeps/pyyaml_6_0_2" \
  /data1/tzh/envs/ka_main/bin/python 5_specification/corpus/external_batch/run.py \
  --split development --out .cache/external_batch_new/development
```

验收改为 `--split acceptance` 并指定另一个新目录。无需修改或安装两个原仓库，运行器核对选择的
源码版本并从对应目录导入公开模块。数学参照经 registry 正常准入，不能绕过实现审核。

参数组使用 `run.py --selection .../selection_loss_parameters_20261019.json --split development --out <新目录>`。
补充前向验收使用 `run_forward_acceptance.py --selection .../selection_flaggems_forward_20261019.json --out <新目录>`，
环境变量与上述 CUDA 命令相同。两个入口均保留运行源码和选定任务；输出目录必须不存在。

运行器完成所有所选调用并写出记录，不以进程退出0代表候选正确；须读取每项 status 和参照结果。
严格包围之外只报告相对数学目标的逐点差异，常规舍入也会产生这种差异。包围包含一个候选点不证明
等式。所有调用的 `formal_binding`、`bias_verdict` 保持 `not_established`，不新增生产覆盖计数。

## 公开 scaled dot-product attention：从完整训练任务补齐参照

这轮先读取真实公开调用，再决定参照范围。FlagGems 5.0.2 暴露
`scaled_dot_product_attention(query,key,value,attn_mask,dropout_p,is_causal,scale,enable_gqa)`，其 autograd
边界承诺前向输出以及 q/k/v 三份梯度。原 ATTN 合同只有 PyTorch 2.10 公开前向；旧草案又把公开
SDPA、FlashAttention、FlexAttention、LSE、window 和 softcap 混在一起。此次没有启用该草案，
而是在已经审阅的公开前向上增加一个独立 VJP profile，要求显式完整 upstream，并完整返回
`out/dq/dk/dv`。公式直接由同一 softmax 前向作链式求导，GQA 对共享 key/value head 汇总梯度。

独立的768位 softmax 实现和 PyTorch 2.10 强制 math backend 交叉核对因果、布尔 mask、GQA、rank-3
广播及 f16/bf16/f32/f64；遗漏任一梯度、增加 LSE 或 upstream 形状错误都会拒绝。外部运行器还修正了
一个由真实半精度调用暴露的问题：数学残差比较原来只接受 float32，现在按合同记录的物理 dtype
检查，不能因为参照支持 f16 就把 f16 候选误报成类型错误。候选的数值验收阈值与数学精确参照继续
分开；官方教程自己的 `atol=1e-2, rtol=0` 只作为该教程测试范围的复现，不表示精确语义相等。

执行前固定的 FlagGems 开发集包含普通矩形、因果矩形、布尔 mask、加性 mask 和 GQA，各两组种子。
十次调用都有完整数学参照，也都返回四项候选结果；没有因失败而从分母移除。结果如下：

| 实现与范围 | 完整数学参照 | 执行结果 |
|---|---:|---|
| FlagGems 5.0.2 开发10项 | 10 | 10项完成；6项普通/因果/GQA前向接近，10项均有至少一份梯度不相容，4项掩码前向也不相容 |
| 官方 Triton 3.6 教程，原 float32 选择4项 | 4 | 4项编译拒绝；原记录保留 |
| 教程测试形状但仍为 float32 的补充4项 | 4 | 4项因教程内部 fp16/fp32 dot 类型不一致而拒绝 |
| 教程实际测试的 float16 causal D64，2项 | 2 | 2项完成，四项结果均在教程源码的 `1e-2` 范围内，最大差 `9.77e-4` |
| 同一接口的 float16 causal D128，2项 | 2 | 2项因 A6000 共享内存上限拒绝，未改配置后冒充通过 |

无记录器的方形 FlagGems 复核进一步排除了捕获和矩形形状的影响。单头因果控制的最大差为：
前向 `5.96e-8`、dq `6.24e-5`、dk `2.26e-5`、dv `3.98e-4`；非因果 dq/dv 分别达到
`6.47e-2/5.74e-2`；布尔 mask 前向 `1.55e-1`；加性 mask 前向 `6.72e-3`；GQA dq
`9.25e-2`。对应 FlagGems 源码显示：布尔 `True` 被转为 `-1e6`，与公开任务的 keep 语义相反；
非因果路径把 qk 转为 base-2 单位后直接加未换底的 additive mask；autograd 反向把 mask 丢为
`None`；反向 kernel 不接收 causal 模式而固定走因果分块。GQA 的大 dq 已复现，但本轮没有为它
猜测尚未独立确认的源码原因。

`results/attention_development_v1.json.gz`、`attention_tutorial_original_v1.json.gz`、
`attention_tutorial_supported_f32_v1.json.gz`、`attention_tutorial_fp16_v2.json.gz` 和
`attention_flaggems_probe_v1.json.gz` 保留输入、四项完整结果、版本源码、错误和逐项比较；
`attention_validation_v1.json.gz` 是便于审核的摘要。选择文件与运行器均在本目录。所有报告继续写明
`formal_binding=not_established`、`bias_verdict=not_established`：本轮完善的是一个常见训练任务的
可靠参照和拒绝边界，没有把开发集差异直接升级为统计 bias，也没有新增正式 kernel 绑定。
完整必需检查通过，含932项 unittest 及外部调用检查。当前共61份合同、119个 profile，其中57个
具备声明范围内的参照准入条件；正式调用绑定数量不变。

## 二维 nearest 插值：从真实调用补齐版本坐标和完整 VJP

FlagGems 5.0.2 的公开 `F.interpolate(..., mode='nearest', size=...)` 会进入 Triton
`upsample_nearest2d_kernel`，但原目录没有能同时回答完整前向和输入 VJP 的合同。新增
`INTERP.torch210_nearest2d_size_autograd` 只选择 PyTorch 2.10、rank-4、显式正 size、legacy nearest
这一项任务。归档的官方 functional 源码确定公开参数和 ATen 对应，官方 decomposition 明确坐标为
`floor(output_index*input_size/output_size)`。`nearest-exact` 使用另一套合法约定，因此不被这份规格判错，
本 profile 每次使用还必须显式声明所选版本坐标。

前向是完整坐标 gather；输入 VJP 由其转置独立得到，把读取同一输入位置的全部 upstream 累加。
8个合同样例和独立整数循环覆盖上下采样、矩形尺寸及重复梯度贡献，拒绝 nearest-exact、round、交换
高宽比例、覆盖梯度和遗漏梯度五种错误。四种浮点 dtype 的 CPU 公共前后向均与规格精确一致。
GPU 前另冻结两组不同实现、形状、seed、size 和布局；16次 CPU 复核全部精确通过。

| 固定集合 | 完整数学参照 | 精确相符 | Triton 范围 | 保留结果 |
|---|---:|---:|---|---|
| FlagGems 5.0.2 开发8次 | 8 | 6 | 只替换前向，8次均捕获 | 两个非连续输入前向分别错150/162、144/162坐标 |
| TorchInductor 验收8次 | 8 | 8 | 8次均捕获前向和反向 | 无拒绝；包含另一组非连续输入 |

FlagGems 的输入 stride 为 `(105,35,1,5)`，捕获源码却按连续 NCHW 线性地址计算输入 offset，未读取
stride；这与差异位置相符。该包没有注册 nearest 的 Triton backward，所以完整公开调用中精确相符的
dx 来自 PyTorch 原生 backward，只能作为调用上下文，不能算成 FlagGems 反向实现的成绩。脚本最初把
“没有 Triton backward”记成阶段缺失，原报告保留；修正版按候选实际承诺只要求前向，并把两个数值
不一致明确标为 `compared_mismatch`。规格及冻结输入没有修改。

保留材料为 `results/interpolate_nearest_development_original_v1.json.gz`、
`interpolate_nearest_development_v2.json.gz`、`interpolate_nearest_acceptance_v1.json.gz` 和
`interpolate_nearest_validation_v1.json.gz`。选择为 `selection_interpolate_nearest_20261019_v1.json`，
运行器为 `run_interpolate.py --split development|acceptance --device cpu|cuda --out <新目录>`。
本轮增加的是一个有版本边界的可靠参照和开发/验收证据，没有建立正式调用绑定，也没有据此判断
统计 bias 或声称对其他插值方式泛化。
完整必需检查随后通过，包括942项注册与单元测试、全部合同检查和外部调用报告检查。当前共有
62份合同、121个 profile，其中59个具备声明范围内的参照准入条件；正式调用绑定数量未变。

## 公开 argmax：已有规格足够，只补调用对应

真实 FlagGems `torch.argmax` 调用进入 Triton，但现有 `REDUCE.real` 已根据 PyTorch 2.10 官方定义实现
dim、keepdim、全张量归约、int64 完整输出和并列时第一个最大值索引。这一轮没有新增公式、合同或
profile，只把公开 `torch.argmax(input, dim, keepdim)` 明确对应到同参数的 `aten.argmax` 查询。

`selection_argmax_20261019_v1.json` 在 GPU 前固定 FlagGems 与 TorchInductor 两组不同形状、种子、
归约维、keepdim、并列输入和非连续输入。20个 CPU 公开调用先与 REDUCE 逐项交叉核对，全部精确一致。

| 固定集合 | 数学参照 | 精确相符 | 保留结果 |
|---|---:|---:|---|
| FlagGems 5.0.2 开发10次 | 10 | 6 | 2次中间维调用编译失败；2次非连续全张量调用返回错误索引 |
| TorchInductor 验收10次 | 10 | 10 | 全部捕获 Triton，含未见维度、并列和非连续布局 |

FlagGems 中间维见证的逻辑形状是 `[3,5,7]`，官方实现和规格均返回每个位置的首个并列索引1；候选
在 `tl.arange(0,TILE_N)` 处因范围5不是2的幂而编译失败。两个非连续见证的 stride 为 `(35,1,5)`；
公开接口和规格返回逻辑展平索引32、77，候选返回物理连续读取对应的24、71。候选源码在 `dim=None`
分支直接按 `inp+offset` 读取，未先连续化或使用 stride；分维分支则显式调用 `inp.contiguous()`。
这些是开发集中的接口/布局错误，不是统计 bias 结论。

完整记录为 `results/argmax_development_v1.json.gz`、`argmax_acceptance_v1.json.gz` 和
`argmax_validation_v1.json.gz`；运行器为 `run_argmax.py --split development|acceptance --device cpu|cuda
--out <新目录>`。所有记录保持 `formal_binding` 与 `bias_verdict` 为 `not_established`，合同/profile和
正式调用绑定数量不变。

## `masked_fill`：从真实训练调用补齐广播和输入梯度

`selection_masked_fill_20261019_v1.json` 固定 PyTorch 2.10 的公开、非原地
`Tensor.masked_fill(mask, Python scalar)`，观察完整前向、输入 VJP、输入不变及函数式输出连续性。
原 `ARITH.real` 已能回答前向值，但没有完整训练接口。新增 profile 直接复用选择公式，并补上普通
广播、广播后梯度求和、公开重载选择和完整输出要求。20 次 CPU 调用逐坐标等于独立整数索引／精确
有理数实现及 PyTorch 2.10 autograd。

FlagGems 5.0.2 开发批次 10 次中有4次完整通过；4次合法的输入扩展或共同广播被其较窄的
mask-to-input 检查拒绝；2次非连续输入的值和梯度完全正确，但输出仍非连续，与版本化函数式参照的
新连续张量承诺不同。另一实现 TorchInductor 的冻结验收 10/10 通过，含未见形状、四维输入、另一
广播方向、全真掩码和非连续输入，前后向均捕获 Triton。记录为
`results/masked_fill_{development,acceptance,validation}_v1.json.gz`。这些结果不自动建立正式 kernel
绑定或统计 bias 结论。

## `constant pad`：已有前向不足以裁决公开训练调用

真实 `torch.nn.functional.pad(input, pad, mode='constant', value=...)` 调用会进入 Triton，但原
`LAYOUT.real` 只有 `aten.constant_pad_nd` 的逻辑前向。新增
`LAYOUT.torch210_functional_constant_pad_autograd` 选择 PyTorch 2.10 的完整公开参数、前向和输入
VJP。官方文档与保留的官方 reference 实现确定 pair 顺序、默认零、负 padding 先裁剪以及过度裁剪的
拒绝顺序；输入 VJP 由这份逐坐标 copy 的转置独立得到。四种浮点 dtype 的20组前后向均与独立循环及
PyTorch 2.10 CPU autograd 精确一致，39个合同样例相容，15个定向错误实现全部被拒绝。

GPU 前固定两组不同实现、形状、seed、pad、填充值和布局。20次 CPU 公共调用先全部交叉核对通过。

| 固定集合 | 完整数学参照 | 完整调用相符 | 保留结果 |
|---|---:|---:|---|
| FlagGems 5.0.2 开发10次 | 10 | 0 | 10次前向、776个坐标精确且均捕获 Triton；10次都缺输入梯度 |
| TorchInductor 陌生验收10次 | 10 | 10 | 1,444个前向和1,052个梯度坐标精确；每次均捕获前后向 Triton |

FlagGems 的公开 `aten::pad` 替换没有注册可用的 autograd 路径。第一次运行保留10次
“differentiated input was not used”失败；第二次用相同冻结输入允许记录缺失梯度，从而证明问题位于
完整训练接口而不是前向 padding 公式。缺少 `dx` 的调用仍标为 `compared_partial_task`，没有因前向
正确而升级为通过。TorchInductor 验收使用未参与开发的 rank、尺寸、三轴 pad、默认填充值和非连续
输入，10/10 完成。

记录为 `results/constant_pad_development_original_v1.json.gz`、
`constant_pad_development_v2.json.gz`、`constant_pad_acceptance_v1.json.gz` 与
`constant_pad_validation_v1.json.gz`。选择文件为 `selection_constant_pad_20261019_v1.json`，运行器为
`run_constant_pad.py --split development|acceptance --device cpu|cuda --out <新目录>`。本轮没有建立正式
kernel 绑定或统计 bias 结论；非 constant 模式、存储/别名效果和其他梯度继续拒绝。该轮完整必需检查
通过，其中注册与单元测试950项；当时共62份合同、123个 profile，61个具备声明范围内的参照准入条件。

## `torch.where`：复用选择公式，补齐两个广播分支的梯度

真实 FlagGems 5.0.2 `torch.where(condition,input,other)` 调用进入 `aten.where.self` Triton 实现。
`ARITH.real` 已有官方前向选择公式，所以没有重写参照；新增
`ARITH.torch210_where_tensor_autograd` 只补公共 Tensor/Tensor 重载、完整 `out`、`grad_input` 和
`grad_other`。官方 PyTorch 2.10 定义确定三方普通广播；两个 VJP 由逐坐标选择的转置得到，分别在
condition 真／假位置接收 upstream，再各自归约回原输入形状。四种 dtype 的16组独立坐标循环和
PyTorch CPU autograd 精确一致，ARITH 的44个样例相容、20个定向错误实现全部被拒绝。

GPU 前固定不同实现、shape、seed、condition 和布局，20次 CPU 公共调用先全部交叉核对通过。

| 固定集合 | 完整数学参照 | 完整调用相符 | 精确坐标 |
|---|---:|---:|---:|
| FlagGems 5.0.2 开发10次 | 10 | 10 | out 612；input 梯度416；other 梯度40 |
| TorchInductor 陌生验收10次 | 10 | 10 | out 1,434；input 梯度846；other 梯度410 |

两组每次均捕获前向与反向 Triton，且没有缺失任务输出。开发集覆盖两个数据分支分别广播、全真
condition 和非连续 input；验收集改用未见 rank、尺寸、广播方向、全假 condition 及两个非连续数据
输入。记录为 `results/where_{development,acceptance,validation}_v1.json.gz`，选择为
`selection_where_20261019_v1.json`，运行器为
`run_where.py --split development|acceptance --device cpu|cuda --out <新目录>`。这些有限相符结果不
建立正式 kernel 绑定，也不证明所有输入上没有 bias；scalar、out、单参数重载和 condition 梯度继续拒绝。
完整必需检查通过，其中注册与单元测试954项；该轮结束时共62份合同、124个 profile，62个具备声明范围内的
参照准入条件。

## `torch.cat`：已有逻辑前向不足以裁决完整输入梯度

真实 FlagGems 5.0.2 `aten.cat` 调用使用最多四个输入一组的 Triton copy kernel。原
`LAYOUT.real` 已有正确的逻辑前向，因此本轮没有重写拼接公式；新增
`LAYOUT.torch210_cat_autograd` 只补公共 Tensor-list 重载、完整 upstream，以及按原输入顺序返回的
全部 VJP。官方 PyTorch 2.10 定义确定输入顺序、维度、形状约束和特殊的 `(0,)` 空输入；每个 VJP
由逐坐标 copy 的转置独立得到，并与保留的官方 `cat_tensors_backward` 选择交叉核对。四种 dtype 的
24组独立坐标/autograd 对照全部精确，LAYOUT 的44个例子相容、18个定向错误实现全部被拒绝。

GPU 前固定两种实现、输入个数、shape、dim、空输入位置、布局和 seed。20次 CPU 公共调用先全部
交叉核对通过。

| 固定集合 | 完整数学参照 | 完整调用相符 | 精确坐标 |
|---|---:|---:|---:|
| FlagGems 5.0.2 开发10次 | 10 | 10 | out 440；全部输入梯度440 |
| TorchInductor 陌生验收10次 | 10 | 10 | out 1,052；全部输入梯度1,052 |

FlagGems 开发集覆盖两至五个输入、五输入跨两个 launch group、文档允许的 `(0,)` 输入和非连续输入；
每次都捕获真实 `cat` Triton 前向。验收集改用未见 rank、维度、六输入分组、零宽轴、另一空输入位置
和非连续组合；每次也捕获前向 Triton。两组都完整返回每个输入 VJP。`cat` 的 VJP 是 upstream 的
切片视图，不需要单独的算术 Triton kernel，不能把没有 backward launch 解释为梯度缺失。

记录为 `results/cat_{development,acceptance,validation}_v1.json.gz`，选择为
`selection_cat_20261019_v1.json`，运行器为
`run_cat.py --split development|acceptance --device cpu|cuda --out <新目录>`。这些有限相符结果不建立
正式 kernel 绑定，也不证明所有输入上没有 bias；out、named-dim、混合 dtype、特殊值、scalar、存储
和别名语义继续拒绝。
完整必需检查通过，其中注册与单元测试958项；当前共62份合同、125个 profile，63个具备声明范围内的参照准入条件。

## `torch.clamp`：版本化边界梯度与调用历史相关的编译复用错误

真实公开调用选择 PyTorch 2.10 函数式 `torch.clamp(input,min,max)` 的标量 bound 重载，观察完整前向、
输入 VJP 和输入值不变。原 `ARITH.real` 已有 lower-then-upper 的前向公式，但不能裁决训练调用，也没有
固定不可导边界的梯度选择。保留的 PyTorch 2.10 `derivatives.yaml` 明确说明 min/max 处导数未定义，
同时明确选择子梯度1；因此新增的 `ARITH.torch210_clamp_scalar_autograd` 是版本化 PyTorch 任务，不把
这个选择说成通用数学导数。独立逐坐标公式与公开 CPU autograd 在四种 dtype、24组 shape/bound 组合
上完全一致；`min > max` 得到全 max 与全零 VJP，`min == max` 只在相等坐标传递 upstream。

GPU 前固定两种实现、shape、bound、seed、精确边界位置和布局，两批20次 CPU 调用先全部交叉核对通过。

| 固定集合 | 完整数学参照 | 完整调用相符 | 保留结果 |
|---|---:|---:|---|
| FlagGems 5.0.2 开发10次 | 10 | 10 | out 690/690、dx 690/690 精确；每次捕获选定的 Triton 前向 |
| TorchInductor 陌生验收10次 | 10 | 8 | dx 1,476/1,476 精确；两个 `-0.75,1.25` 前向分别错110/115、112/115坐标 |

验收失败发生在同一进程先运行 `min=max=0.25` 后。两个失败前向几乎都返回0.25，但梯度仍按本次
`[-0.75,1.25]` 的边界正确选择。首次失败报告原样保留。随后用一个不导入分析工具和记录器的最小程序
复核：全新进程单独编译 `[-0.75,1.25]` 时115个前向和梯度坐标全部正确；先编译 `[0.25,0.25]`，
重置后再编译目标调用时，前向仍错110个坐标而梯度完全正确。失败生成代码在前向中把0.25写死为两个
clipping bound，梯度 mask 却读取本次运行的 scalar bound。因此本轮没有改规格迁就候选，而是保留为
PyTorch 2.10 该编译调用上下文中的参数复用错误。它尚未升级为统计 bias 或正式 kernel 绑定结论。

记录为 `results/clamp_development_v1.json.gz`、`clamp_acceptance_v1.json.gz`、
`clamp_compile_fractional_fresh_v1.json.gz`、`clamp_compile_sequence_failure_v1.json.gz` 和
`clamp_validation_v1.json.gz`。选择文件为 `selection_clamp_20261019_v1.json`；运行器为
`run_clamp.py --split development|acceptance --device cpu|cuda --out <新目录>`，最小复现为
`run_clamp_compile_diagnostic.py --scenario fractional_only|equal_then_fractional --out <文件>`。
完整必需检查通过，其中注册与单元测试962项；该轮结束时共62份合同、126个 profile，64个具备声明范围内的
参照准入条件。tensor bound、原地/out 重载、特殊值、bound 梯度、存储和别名继续拒绝。

## `torch.maximum/minimum`：复用前向，补齐相等坐标和双侧广播 VJP

真实 FlagGems 5.0.2 公开调用会分别进入 Triton `maximum_kernel` 与 `minimum_kernel`。原
`ARITH.real` 已定义有限值前向，因此没有另写一套前向公式；新增 profile 只补 PyTorch 2.10 的
Tensor/Tensor 调用、完整 upstream 和两个输入 VJP。保留的官方 `derivatives.yaml` 明确：严格胜出的
operand 接收全部 upstream，两输入相等时各接收一半；普通广播随后要求两个梯度分别归约回各自源形状。
独立参照使用自己的输出坐标到源坐标映射和精确有理数累加，不调用适配器的 extrema 或归约 helper。

四种 dtype、两种操作和三组 shape 共24个独立 subtest 与 PyTorch 2.10 CPU autograd 完全一致。
58个合同样例相容，28种定向错误全部被拒绝。GPU 前另固定开发与验收的 operation、shape、layout、
seed 和强制 equality 坐标；20次 CPU 公开调用先全部交叉核对通过。

| 固定集合 | 完整数学参照 | 完整调用相符 | 精确坐标 |
|---|---:|---:|---:|
| FlagGems 5.0.2 开发10次 | 10 | 10 | out 822；input 梯度502；other 梯度402 |
| TorchInductor 陌生验收10次 | 10 | 10 | out 1,712；input 梯度748；other 梯度532 |

开发批覆盖同形、两向广播、三维广播和双侧非连续输入；每次捕获所选 FlagGems Triton 前向。该路径
没有 FlagGems Triton backward，精确双侧 VJP 只作为完整公开调用上下文记录。验收批改用未见 rank、
尺寸、广播方向和布局，并加入整张量相等的强见证；所有 equality 坐标都正确分配 `upstream/2`，每次
均捕获 TorchInductor 前向及反向 Triton。

记录为 `results/extrema_{development,acceptance,validation}_v1.json.gz`，选择文件为
`selection_extrema_20261019_v1.json`，运行器为
`run_extrema.py --split development|acceptance --device cpu|cuda --out <新目录>`。这些有限相符结果不
建立正式 kernel 绑定，也不证明所有输入上没有 bias；scalar、out、reduction、fmax/fmin、混合 dtype、
特殊值、存储和别名继续拒绝。

完整必需检查通过，其中注册与单元测试966项；当前共62份合同、127个 profile，65个具备声明范围内的
参照准入条件。
