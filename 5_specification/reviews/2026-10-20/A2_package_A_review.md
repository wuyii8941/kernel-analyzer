# 路径 A 收口 A 包（NORM / EMB / NLL）独立复核（2026-10-21）

复核人：Claude（独立复核方，未参与起草）。我只新增了两个文件：本文件和 `tests/test_norm_emb_nll_a2_review.py`。没有修改任何草稿或共享代码。

## 复核对象

以下是当前工作树中的文件及其 sha256。

| 文件 | sha256 |
|---|---|
| contracts/draft/NORM_path_a_A2_20261020.yaml | fa43a297229cdc14d6911750f0dc3d596454ab858402d3912ded9f941f927422 |
| contracts/draft/NORM_adapter_path_a_A2_20261020.py | e1366624b4fd836caf0938005779eabdc9273bed1b80cd6f95d520ed9ad4a30c |
| contracts/draft/EMB_path_a_A2_20261020.yaml | c8f35b2536d76d8012a8ed7b25bbad252362cfd6af7117e45dcb30c6b68af847 |
| contracts/draft/EMB_adapter_path_a_A2_20261020.py | cfe06af0189bc916a6a309b782dec507f91e5717d6cbfc7931bd65a763054996 |
| contracts/draft/NLL_path_a_A2_20261020.yaml | bcbc3efe8167d78d690a8577a0af6a6f132e46b7b722962885e5b5b525f205f0 |
| contracts/draft/NLL_adapter_path_a_A2_20261020.py | bd4df2fb4908f16c1af1c87eebf8ddbc7c3b272f264231bb67d4ee4cb1422685 |
| kernel_spec/norm_emb_nll_aten.py（共用辅助，三份合同共用） | 2866f5b725b5040e8784bae76d396f4d03f631f2e0f407f8c9d3ec102a2a6e2a |

暂存目录：`.cache/spec_a2_review/A/{contracts,adapters}`，内容与草稿逐字节相同（有测试核对）。

## 结论

| 合同 | 结论 |
|---|---|
| NORM | **CONFIRMED**：上表中的 NORM 两个文件连同辅助模块可以提升，可放入 validated/。附一条不阻塞的说明，见 3.2 |
| EMB | **ISSUES**（1 条，只涉及文字；数值语义无误，记录到的调用不受影响） |
| NLL | **CONFIRMED**：上表中的 NLL 两个文件连同辅助模块可以提升，可放入 validated/ |

## 1 来源核对

我打开了快照逐项对照，条目内容都与原文一致：

- `NEN_LAYER_NORM_CPP`
  - STATS：第 110–112 行分配 `at::empty({M})`，dtype 为 `param_scalar_type`；`stat_shape` 由外层维加 1 组成。
  - RMS_EPS：`rms_norm_composite` 中 `eps.value_or(numeric_limits<float>::epsilon())`。
  - BWD：与原文一致。
- `NEN_REFS.NATIVE_LN`：第 3448 行，只有在 cpu/mtia 上才把 mean/rstd 转成输入 dtype；mean/rstd 保留 keepdim 形状。
- `NEN_DECOMP`
  - LN_BWD、NLL_FWD、NLL_BWD：与原文一致。
  - RMS_FUSED：第 1776–1780 行，float32 计算类型取 `finfo(float32).eps`，否则取 `finfo(float64).eps`。
  - EMB_DENSE_BWD：用 `indices == padding_idx` 直接比较。
- `NEN_LOSSNLL_CPP`
  - FWD：reduction none 且 2D 时 `total_weight` 保持为 0；空 target 时 mean 写 NaN；`total_weight` 是计数或权重和。
  - BWD：使用传入的 `total_weight`。
  - 补充：1D self 配 reduction none 时，原生代码走标量分支，返回 (−w·x, w)。辅助模块的结果与此相同。
- `NEN_DERIVATIVES`
  - LN、NLL：前向返回的 mean/rstd 和 total_weight 原样传给反向。
  - EMB：与原文一致。
- `TS_EMBEDDING_CPP.BACKWARD_DISPATCH`：第 90、155 行做字面比较；sparse 分派与条目一致。
- `TS_NATIVE_FUNCTIONS.NORM_EMB_NLL_SCHEMAS`、`NEN_REDUCTION_PY`、`NEN_FUNCTIONAL`、`T_NLLLOSS`、`T_NLL_LOSS_F`：均与条目一致。

PyTorch 2.10 eager 复查（CPU/CUDA）：

- native_layer_norm 的统计量：CPU 上 bf16 输入得到 bf16 [1,4,1]；CUDA 上 bf16 输入得到 f32，f64 输入得到 f64。
- rms_norm：bf16、eps=None、x=1e-3 时输出 0.9453，对应的 eps 是 2^-23。
- nll 1D none：输出 (7.5, 3)。
- 权重和为 0 的 mean：输出 −inf，total_weight 为 0。
- embedding_backward padding −1：最后一行有梯度。

以上都与草稿相符。

## 2 重点决定的判断

- **NM.ATEN_BWD_SUPPLIED / NL.BWD_SUPPLIED**（反向使用传入的统计量 / total_weight）：成立。
  - schema 把这些量作为参数传入，derivatives.yaml 传入的是前向的返回值，分解和原生代码都读这些参数。
  - layer_norm 反向的 schema 没有 eps，所以“从输入重算”在 layer_norm 上本来就无法定义。
  - NLL 把“按 target 重算”保留为诊断读法 NL.R1，做法诚实。
  - 我的测试独立推导了 N=2 时 layer-norm 的精确 Jacobian（e·s^{-3/2}/2）。用精确统计量时，草稿结果与它一致。
- **NM.ATEN_STATS**（统计量的形状）：
  - 前向参照把规范形状固定为 [外层…, 1]。`_check_aten` 要求形状完全相等（NORM 适配器第 129 行）。
  - 因此 FlagGems 返回 [128] 而不是 [1,128,1] **会被判为不符**，不会被掩盖。复核测试 `test_layer_norm_check_output_stats_shape_and_dtype` 验证了 [M] 形状被拒绝。
  - 只有反向的**输入**可以是任意含 M 个值的形状。这是被调用方的读取规则，原生 kernel 本来就按 M 个值读取，所以不掩盖缺陷。
- **NM.ATEN_DTYPES**（统计量的 dtype）：
  - 返回的统计量接受输入 dtype 或 f32。两种后端（CPU 给输入 dtype，CUDA 给 f32）都有源码依据，文档没有规定。
  - 数值仍按精确统计量比较，bf16 舍入造成的偏差照样可见。所以这条容差不掩盖值的缺陷。
  - 录到的 FlagGems 统计量是 bf16，这一项被接受；但它的形状 [128] 仍然不符。
- **NM.ATEN_RMS_EPS_NONE**（eps=None）：
  - f32/f64 下文档值与实现值相同，分别为 2^-23 和 2^-52。
  - f16/bf16 下文档写的是 finfo(x.dtype).eps，原生代码和分解用的是 f32 epsilon，两者确实矛盾。判为 unsupported 是保守而诚实的处理；本次没有录到 f16/bf16 的 eps=None 调用。
  - D5 的读法（eps=1e-5）与参照的包围不相交，复核测试已验证。
- **EM.ATEN_PADDING**（padding_idx 字面比较）：成立。ATen 层做字面比较，−1 不对应任何行；函数层在调用 ATen 之前已经做过归一化。
- **NL.FORWARD**（加权 mean）：按 NLLLoss 页面的公式 Σl/Σw_t。函数页“除以元素数”的说法被同页“忽略的 target 不计入平均”和权重说明否定，取舍合理。
- **NL.EMPTY_MEAN**：0/0 或 x/0 判为 domain_undefined，对设备输出不作要求，合理。
- **NL.BWD_EMPTY**：只有需要做除法的位置才会无定义；全部忽略时结果为全零，与原生代码一致。

## 3 问题

### 3.1 EMB（ISSUES）

1. `contracts/draft/EMB_path_a_A2_20261020.yaml:252-254`：合同级 `domain` 仍是旧 profile 的写法，与新 profile `aten_dense_backward` 的决定冲突。
   - `valid` 写“int64 in-range indices”“valid optional padding_idx”“nonempty positive dimensions”。
   - `invalid_input` 写“out-of-range … padding_idx”。
   - `unsupported` 写“non-int64 indices, empty tensors”。
   - 而 EM.ATEN_PADDING 与 EM.ATEN_BACKWARD 明确规定：任意整数 padding_idx 都按字面比较（−1 或越界值都不屏蔽任何行，结果为 value）；接受 int32；空 indices 得到全零。
   - 按 domain 文本，录到的 padding_idx −1 调用会被归为 invalid_input，与适配器给出的 value 相矛盾。
   - 证据：复核测试 `test_embedding_backward_refusals` 显示，适配器对 padding_idx=5（越界）、int32 indices、空 indices 都给出 value。
   - 修法（纯文字）：把这三项限定为“profile torch210_dense_real”，并为 aten_dense_backward 补上对应的 valid / invalid_input 条目。

### 3.2 NORM（不阻塞，只作说明）

- `contracts/draft/NORM_adapter_path_a_A2_20261020.py:133`：对 f64 输入也接受 f32 统计量。两个后端都不会这样输出，决定 NM.ATEN_DTYPES 给出的理由也只针对低精度输入。
- 由于值仍按精确统计量比较，这条容差不会掩盖值的错误；决定文本也如实写了“input dtype or float32”。
- 以后修订时可以收窄为：只对 f16/bf16 输入额外接受 f32。

## 4 测试

新增 `tests/test_norm_emb_nll_a2_review.py`，共 30 项。期望值用 Fraction 和整数平方根包围独立计算，不调用辅助模块或适配器的计算。比较方式：两边都是有理数时要求精确相等；否则要求包围相交，且参照包围宽度小于 2^-100。覆盖内容：

- 录到结构的小规模版本：
  - native_layer_norm：f32/bf16、eps 1e-12 和 1e-5，含一个常数行。
  - native_layer_norm_backward：bf16、扁平 [M] 统计量、所给统计量不是输入的真实统计量、各种 mask。
  - rms_norm：D5 结构。
  - embedding_backward：padding −1/0/正值、freq、int32。
  - nll forward/backward：三种 reduction × 有无权重，ignore −100。
- 边界与拒绝类别：
  - 被开方数 ≤0、形状不符、inf 值、统计量个数不符、权重缺失但 mask 为 true。
  - sparse=True。
  - 全部忽略、权重和为 0、空 batch、1D 的 none、ignore_index 等于类别下标、越界 target、reduction 3、i32 target。
  - 反向里 self 含 NaN、total_weight 为 0。
  - 诊断读法 NL.R1。
- 旧 profile 在随机输入上与 validated/ 中的适配器逐字段一致：NORM 的 torch / torch_autograd / rms_offset（layer、rms、group），EMB 的 torch210_dense_real。旧 profile 对 aten 调用返回 unsupported。

命令与结果：

```
cd 5_specification   # 已按 BRIEF 设置 HOME/TMPDIR 等环境变量
/data1/tzh/envs/hf_ga_5190/bin/python -m pytest -q -p no:cacheprovider tests/test_norm_emb_nll_a2_review.py  # 30 passed
/data1/tzh/envs/hf_ga_5190/bin/python -m pytest -q -p no:cacheprovider tests/test_norm_emb_nll_aten.py       # 15 passed
/data1/tzh/envs/liger/bin/python tools/check_contracts.py --contracts ../.cache/spec_a2_review/A/contracts \
    --adapters ../.cache/spec_a2_review/A/adapters --only NORM EMB NLL
# exit 0；NORM 34/34 例子、21/21 变异；EMB 26/26、11/11；NLL 18/18、读法 2/2、变异 11/11
```
