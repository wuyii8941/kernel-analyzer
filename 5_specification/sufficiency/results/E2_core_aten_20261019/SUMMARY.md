# E2 外部算子集：PyTorch 2.10.0 Core ATen（2026-10-19）

方案：`docs/SUFFICIENCY_PROTOCOL.md` 第 4 节；代码：冻结点 F（7115198）的 `sufficiency/core_aten.py`；
原始结果：本目录 `inventory.json`（清单与 schema）、`core_aten.json`（逐 overload 的对应、探测结果和说明）。

## 1 清单

锁定 torch 2.10.0+cu128 中带 `torch.Tag.core` 的条目：**192 个 overload，163 个基本名**。

## 2 名称对应

在锁定版本的 Core ATen 清单中，**110/192 个 overload（89/163 个基本名）存在合同名称对应**；完整签名与适用范围另行核验。
110 个都指向已准入（validated）的合同。这个数字可能低估（别名未识别），也可能高估（只支持部分 dtype、参数或反向），
因此不称为可信覆盖率的下界。

| 类别 | overload | 说明 |
|---|---|---|
| 直接对应 `aten.<name>` | 106 | |
| 登记别名 | 4 | `avg_pool2d`、`avg_pool2d_backward`、`max_pool2d_with_indices`、`max_pool2d_with_indices_backward` → POOL（公开函数分派到该 overload） |
| 结构／存储语义承担 | 21 | `empty`、`empty_strided`、`as_strided`、`copy`、`fill`、`full`、`full_like`、`scalar_tensor`、`resize_`、`sym_*`（5）、`_local_scalar_dense`、`arange`、`diagonal`、`repeat`、`flip`、`masked_scatter`、`nonzero`。不做算术不等于不影响答案（读哪块存储、未初始化内容、决定分母的形状值），**是否已被可靠处理未在本评价中核验** |
| 随机流 | 4 | `rand`、`randn`、`randperm`、`native_dropout`。RANDOM 只准入以抽样结果为条件的边界情形，随机前向不准入 |
| 无对应 | 57 | 见第 4 节 |

## 3 适用范围探测（PyTorch OpInfo 样例）

对有名称对应的 110 个 overload，用 PyTorch 自己的 OpInfo 样例（CPU；f32、bf16、f64、i64；每个 overload 每个 dtype 至多
50 次调用）在分派记录下取得实际调用，原样交给 `registry.query`（只用已准入合同，取第一个不需要任务选择的 profile）。
共 11,728 次查询。

| 输入 dtype | 查询 | value / domain_undefined | 比例 |
|---|---|---|---|
| f32 | 3,188 | 2,432 | 76.3% |
| bf16 | 2,920 | 2,218 | 76.0% |
| f64 | 3,188 | 2,452 | 76.9% |
| i64 | 2,432 | 548 | 22.5% |

按 overload：

| 结果 | overload |
|---|---|
| 全部样例调用都有定义 | 9 |
| 部分样例调用有定义 | 79 |
| 没有一次样例调用有定义 | 12 |
| OpInfo 没有产生这个 overload 的调用 | 10（`avg_pool2d_backward`、`bitwise_and.Scalar`、`bitwise_or.Scalar`、`convolution_backward`、`div.Scalar_mode`、`embedding_dense_backward`、`native_group_norm_backward`、`native_layer_norm_backward`、`upsample_nearest2d.vec`、`var.dim`） |

未取得参照的原因（按查询次数）：

- **unsupported 2,886 次**，几乎都是有意的准入边界：整数 dtype 在 LOGIC / LAYOUT / ARITH / REDUCE / MATHFN / MATMUL 的有限浮点
  profile 之外（i64 上的大部分拒绝）；MATHFN 只准入绝对值不超过 100；以及零维排序、多维 truth reduction、转置卷积、
  `var` 的 correction 只选 0 或 1、Tensor 形式的 clamp 边界、标量 index_select / gather 等未声明的重载。
- **合同接口不是 ATen 调用 625 次**：NORM（`native_layer_norm`、`native_group_norm`）、EMB（`embedding`）、POOL 的四个别名。
  合同的数学已写，但需要从 ATen 调用到合同输入的对应，属于「需要人工绑定」一类。
- **输入无法用实数编码 419 次**：含 NaN 的张量（177）、`layout` / `memory_format` / `device` 参数（218）、复数（24）。
- `invalid_input` 103、`not_established` 45 次。
- 「没有一次有定义」的 12 个中，`isnan` 是因为按登记规则取了 LOGIC 的第一个 profile `real`，而 `isnan` 在另一个
  profile `ieee_values` 中；其余为上面的接口对应缺口或未声明重载（`any.dims`、`bitwise_not`、`bitwise_or.Tensor`
  只收布尔、`clamp.Tensor`、`_to_copy` 的 layout 参数）。

数值复核（诊断，不计入相容）：88 个 overload 共有 value 与 float64 PyTorch 结果比较。超过 1e-9 相对差的只有 `cumsum`、
`prod`、`_log_softmax` 的部分样例；逐一复查均来自带 `dtype=` 参数的样例，复核脚本把 float64 重算仍放在该低精度 dtype 中，
是复核脚本的缺陷，不是参照与 PyTorch 的分歧。去掉这一类后没有分歧。

## 4 没有对应的 57 个 overload

| 族 | overload |
|---|---|
| 逐元素数学 | `acos`、`acosh`、`asin`、`asinh`、`atan`、`atan2`（2）、`atanh`、`cosh`、`sinh`、`tan`、`log2`、`log10`、`pow`（3）、`round`、`trunc`、`fmod`（2）、`remainder`（2）、`isinf` |
| 激活 | `elu`、`hardtanh`、`leaky_relu` |
| 逻辑位运算 | `bitwise_xor`（2）、`logical_xor` |
| 池化 / 填充 / 插值 | `_adaptive_avg_pool2d`（含反向）、`_adaptive_avg_pool3d`、`adaptive_avg_pool1d`、`avg_pool1d`、`avg_pool3d`、`max_pool3d_with_indices`、`reflection_pad1d/2d/3d`、`replication_pad2d/3d`、`upsample_bilinear2d`、`grid_sampler_2d`、`col2im` |
| 索引写入 | `index.Tensor`、`index_put`、`scatter`（2）、`scatter_add`、`scatter_reduce` |
| 归一化 | `_native_batch_norm_legit`（2）、`_native_batch_norm_legit_no_training`（BN 合同覆盖公开 `batch_norm` 推理，不同签名与输出） |
| 其他 | `_cdist_forward`、`_pdist_forward`、`_embedding_bag`、`_fft_r2c` |

## 5 结论的写法

- 可以写：「锁定的 Core ATen 192 个 overload 中，110 个有合同名称对应；在 PyTorch 自己的 OpInfo 浮点样例上，约 76%
  的实际调用能取得参照，整数输入约 22%；拒绝主要来自有意的准入边界和合同接口对应缺口。」
- 不能写：「覆盖了 Core ATen 的 57%」或「名称对应即有完整 ground truth」。结构／存储语义的 21 个条目是否被可靠处理，
  本评价没有核验。
