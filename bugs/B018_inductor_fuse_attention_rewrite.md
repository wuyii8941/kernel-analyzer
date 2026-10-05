# B018：Inductor 把手写注意力改写成 SDPA 时改变了语义（缩放常数、填充值、非 0/1 掩码、全掩码行）

日期：2026-10-06。对象：PyTorch 2.10.0（`ka_main`）Inductor 的注意力融合（`torch/_inductor/fx_passes/fuse_attention.py`，
默认开启）；main 见"状态"。类：④（编译后的语义与 eager 不符），静默错误。

## 现象（CUDA，`bugs/repro/B018_*`，输出 `results/tool_spec/final/fuse_attention/B018_repro_output_torch2.10.txt`）

| 写法 | 改写 | 编译与 eager 的最大差 |
|---|---|---|
| `softmax(q @ k^T / c + mask) @ v`，c = √head_dim（对照） | 是 | 1e-6 |
| 同上，c = √d_model = 16 / 温度 4.0 / 1.0 | 是 | **1.19 / 1.7 / 3.0** |
| DistilBERT 式 `masked_fill(mask == 0, -inf)`（对照） | 是 | 7e-7 |
| 同上，填充值 0.0 / 1.0 | 是 | **0.55 / 1.01** |
| 同上，掩码为段 id（1/2/3，0 为填充） / 软掩码（0/0.5/1） | 是 | **1.83 / 1.64** |
| 同上，整条序列都是填充，填充值 -inf | 是 | eager NaN，编译后 0 |
| 同上，整条序列都是填充，填充值 `finfo.min`（HF DistilBERT 的写法） | 是 | **0.70**（eager 为均匀平均，编译后 0） |

## 机制

**A. 模式里的常数是通配符（2.10）。** 模式用示例输入追踪后序列化，常数（如 `math.sqrt(query.size(-1))`、
`masked_fill` 的填充值）不参与匹配；替换函数却按模式的常数改写：模式 5 的替换不传 `scale`（SDPA 默认 1/√E），
模式 15/17/20 的替换用布尔掩码（等价于 -inf）。所以除以任何常数、用任何填充值的写法都被改写成同一个结果。受影响的是常数写死在模式里的 7 个模式：5、6（缩放 `math.sqrt(query.size(-1))`）、15、17、20（填充 -inf）、18、19（GPT-2 式：缩放 `value.size(-1) ** 0.5`、填充 `finfo.min`）；缩放作为匹配参数的模式（1、2：`.div(c)`、`* c`）把实际的 c 传给 SDPA，实测正确。
main 上 #195383（2026-09-21 合入）改为用匹配到的输入重新追踪模式来核实匹配，按其描述应能拒绝常数不同的匹配——
nightly 实测待补；2.10 发布版受影响。

**B. 模式 15/17/20 的替换语义与模式不一致（main 未变）。** 模式在 `attn_mask == 0` 处屏蔽，替换却在
`attn_mask == 1` 处保留（`_sfdp_replacement_15`：`(attn_mask == 1).view(...).expand(...)` 作为 SDPA 的布尔掩码）。
两者只在掩码取值为 0/1 时等价：段 id、软掩码这类其他取值被当成屏蔽。全掩码行上，SDPA 的约定输出 0，而模式（eager）
给出 NaN（填充 -inf）或对 v 的均匀平均（有限填充值）。`_sfdp_params_check` 只对加性掩码检查 dtype，对这三个模式的
掩码取值与 dtype 都不检查。这部分不是匹配问题，重新追踪核实不了它。

## 发现过程

不是工具检出：SDPA 改写后的计算落在 cuDNN / FlashAttention kernel 上，不是 Triton，工具捕获不到。按计划第 6 节
"先写假设"：Inductor 的模式改写会把用户代码换成另一个实现，若匹配条件比语义宽，改写就不保语义。读
`fuse_attention.py` 的匹配检查与替换函数，看到模式 15 的 `== 0` 与替换的 `== 1`，再用直接差分探针确认；探查常数时
发现 A（缩放常数与填充值都不参与匹配）。

## 影响

- A（2.10）：任何手写的带加性掩码的注意力，只要缩放不是 1/√head_dim——温度系数、按 d_model 缩放、不缩放——编译后
  静默地改成 1/√head_dim，输出 O(1) 级错误。DistilBERT 式写法用任何填充值都被改成 -inf 屏蔽。
- B（main 仍有）：DistilBERT 式写法遇到非 0/1 掩码（打包序列的段 id、软掩码）时屏蔽错 key；批里有整条填充序列时，
  输出与 eager 不同（`finfo.min` 填充时 eager 为均匀平均，编译后为 0）。

## 状态

检索：#195320（permute 被通配，#195383 修复）、#100452、PR #195907、torch-spyre#4526 均是相邻问题，未见 B 的报告。
修法（B）：替换里用 `attn_mask != 0` 作保留掩码（与模式的 `== 0` 屏蔽严格互补），并只在填充值为 -inf 时改写，或对
有限填充值保留模式在全掩码行上的结果。上游草稿 `bugs/upstream_drafts/B018_*.md`（只含 B；A 待 nightly 核实），由用户提交。
