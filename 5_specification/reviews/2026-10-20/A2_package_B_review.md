# 路径 A 收口 B 包（ATTN 融合 SDPA ATen 接口，dropout 0）独立复核

复核日期：2026-10-21。复核人：独立复核方（未参与起草）。

## 对象（sha256）

| 文件 | sha256 |
|---|---|
| `contracts/draft/ATTN_path_a_A2_20261020.yaml` | `825e09d6c0b0c41389984f23c603a0dcdfcea18542352461c798b423a166d85d` |
| `contracts/draft/ATTN_adapter_path_a_A2_20261020.py` | `ac49c95852442aa5e2a09f9792492a27f506fb03486142a4d93130e86af2dd5f` |
| `kernel_spec/attn_aten_semantics.py`（参照） | `711876d48b61461506bb20a133cf554d5232e73a1ab5dde09cdc5754e9018cc6` |
| `kernel_spec/attn_aten_indep.py`（只供变异） | `addf060b5a0b560740d89bd8d04c018ba041ecb35a19d7dcf04f96b801d9772f` |
| 对照：`contracts/ATTN.yaml` / `adapters/ATTN.py`（与 `validated/` 相同） | `2184efa0…2a9a` / `47bdb0f2…bf0` |
| 复核测试 `tests/test_attn_aten_a2_review.py` | `15c1454aa05b048372d1d5b5028ac071a80f2402a8da57846235da0853f1368c` |

## 结论：ISSUES（1 条，次要；不影响所记录的调用）

数值语义、各项决定和公开 profile 都成立。只有第 1 条需要修改：在 `attn_aten_semantics.py` 加一行拒绝，YAML 加一句话。
改完后只需复核这一处。

## 问题

1. **flash 两个 overload 接受 Ev ≠ E，与所引来源矛盾。**
   - 位置：`kernel_spec/attn_aten_semantics.py:163-170` 的 `parse_qkv` 只检查了 key 与 query 的 head dim；`:364` 的
     `output_layout` 把 flash 输出写成 `(B,Hq,L,Ev)`；YAML `:197-206` 的 AT.ATEN_DOMAIN 没有提这个限制。
   - 依据：AT.ATEN_AUX 自己引用的 `TS_META_SDPA.FLASH_FWD` 写的是 `attention = torch.empty_like(query)`，也就是输出形状为
     `(B,Hq,L,E)`。PyTorch 2.10 CUDA 实测：forward 报 "v must have shape (batch_size, seqlen_k, num_heads_k, head_size_og)"，
     backward 同样拒绝。草稿对这类调用给出数值，所写形状也与 meta 不一致；作者测试 `test_flash_forward_matches_float64`
     还用了 E=4、Ev=3。
   - 修法：flash 两个 overload 在 Ev ≠ E 时判 unsupported（或 invalid_input），在 AT.ATEN_DOMAIN 写明。efficient backward
     本来就支持 Ev ≠ E（实测 efficient forward 可以），不用改。
   - 复核测试：`test_flash_value_head_dim_must_equal_query_head_dim` 标为 `xfail(strict=True)`。修好以后它会变成 XPASS 而报错，
     届时去掉标记。

观察（不阻塞）：
- flash CUDA op 还要求 head dim 是 8 的倍数（"ensured by padding"，由公开调度层补齐），草稿的例子都用 E=1。这属于后端限制，
  不改变语义。按文档把 f 定义在更大的结构上可以接受，但 AT.ATEN_DOMAIN 用"CUDA op 的 dtype"作为拒绝 f32 的理由，
  却没有提 head dim。建议在文字里说明，只按 dtype 限制。
- efficient backward 在 L 不是 32 的倍数时也接受未补齐的宽度 L，而 PyTorch 会拒绝（"LSE is not correctly aligned"）。
  语义上仍然清楚：只读前 L 项。所记录的调用 L=128，两种宽度相同。
- efficient 内核非方阵 causal 用左上对齐，与公开文档一致，本来可以定义；草稿一律判 unsupported。这样更保守，可以接受。

## 1 来源核对（逐条对照快照）

- `TS_NATIVE_FUNCTIONS.SDPA_FUSED_SCHEMAS`：三个 schema、返回名称、`nondeterministic_seeded` 标签（flash fwd、eff bwd 有，
  flash bwd 没有）与快照第 15130–15177 行一致。
- `TS_AUTOGRAD_DERIVATIVES.SDPA_FUSED`：第 2882–2888 行，backward 接收 forward 自己的 output 与 logsumexp。一致。
- `TS_META_SDPA`：FLASH_FWD（lse f32 (B,H,L)；返回 None、None、L、S、seed (2,) uint64、offset ()、空 debug mask）、
  FLASH_BWD、EFF_FWD_LSE（ceil(M/32)*32）、EFF_BWD 都与快照一致。问题 1 源于 FLASH_FWD 的 `empty_like(query)`。
- `FA2_PAPER`：LSE（L = m + log ℓ，正文省略缩放与掩码）、BWD（Algorithm 2 第 4、11–16 行）、GQA（"sum … across heads that were
  implicitly duplicated"）、CHAIN（ds = (diag p − pp^T) dp）都与快照一致。
- `T_RANDOMNESS.NO_GUARANTEE`、`TS_ATTN_CU.FLASH_RETURN`（第 827 行）、`TS_ATTN_CU.EFF_CAUSAL`（第 1041 行 CausalFromTopLeft；
  第 1023–1029 行 batch > 65535 时要求 dropout 0）、`TS_MEMEFF_FWD_H.DROPOUT_RNG`（curand_init 第 692–696 行、kAlignLSE=32、
  1/(1−p)）都一致。

## 2 决定

- **AT.ATEN_SAVED 成立。** 设 L_i = log Σ_j e^{s_ij}、O_i = Σ_j P_ij v_j，则 D_i = dO_i·O_i = Σ_k P_ik dP_ik，
  于是 P_ij(dP_ij − D_i) = [(diag p − pp^T) dp]_j，正好是 softmax 的 VJP。因为 S = scale·QK^T，链式法则给 dq、dk 各补一个 scale
  因子。草稿公式与此一致。复核测试用三种方法核对：sympy 符号恒等（n=3）；mpmath 60 位下 FA2 与显式 Jacobian 的差小于 1e-50；
  适配器在给定（不精确）O、L 下与自写的 FA2 一致，`recompute_saved` 读法与自写的 Jacobian VJP 一致。选择"使用给定值"有
  schema、derivatives.yaml 和 FA2 作依据，备选读法另列为诊断读法，处理得当。
- **AT.ATEN_HEADS 成立。** 映射 h // (Hq/Hkv) 与 T_SDPA.GQA（repeat_interleave）和 FA_IFACE.GQA 的例子一致，dk/dv 按组求和。
  测试中的 round-robin 映射会落在包围之外，说明测试能区分两种映射。
- **AT.ATEN_CAUSAL 成立。** L=S 时左上和右下对齐重合；L≠S 时 flash 内核用右下对齐，与公开文档冲突，判 unsupported 是如实处理。
- **AT.ATEN_RNG 如实，不是回避。** 已登记的来源里只有 CUDA 源码（reference_program）定义了 (seed, offset) 到掩码的映射，
  T_SDPA、T_DROPOUT_F、T_RANDOMNESS 都没有。已验证的 RANDOM 合同同样不对随机流建模。RANDOM 草稿 RN.CONDITION 那种
  "以观测到的随机数为条件"的做法在这里也行不通，因为 efficient backward 在内部重新生成掩码，ATen 边界上看不到。
  建议在 statement 里补上这条备选及其不可行的原因，不阻塞。后果：bert-tiny#73、gpt2#87（dropout 0.1）判
  unsupported，note 里带有 AT.ATEN_RNG。
- **AT.ATEN_AUX 成立。** 比较位置 0、1，以及整数位置 4、5（max_q/max_k）；不比较 2、3、6、7、8。efficient backward 的第 3 位
  为 None，harness 用 probe 得到的 none_positions 处理（instantiate.py:296-350），与 `ATEN_RETURNS` 一致。注意 rule v2 不测量整数返回值。
- **AT.ATEN_DOMAIN 基本成立。** 宽度为 32 倍数、padding 只接受 +inf：NaN 与 0 的 padding 都判 unsupported（已测）。例外见问题 1。

## 3 独立实现与测试

`tests/test_attn_aten_a2_review.py`：自写 mpmath（60 位）与 Fraction 实现，前向依据 SDPA 页面，后向依据 FA2 Algorithm 2 和显式
Jacobian，没有导入 attn_aten_semantics、attn_aten_indep 或适配器的计算函数。草稿对暂存于 `.cache/spec_a2_review/B/` 的副本，
并用 sha 绑定所 import 的 kernel_spec 模块。覆盖内容：
- flash 前向 B=1、Hq=4、Hkv=2、L=S=5、D=3，causal 与否 × scale null/显式；非方阵非 causal；
- flash 后向用给定值（causal 与否），lse+1/2 时三个梯度都乘以 exp(−1/2)；q=0 和 k=0 两种 Fraction 精确用例；
  check_output 的完整输出、错值、错 dtype、缺位；
- efficient backward 的 lse 宽度 5 与 32；
- 按记录结构缩小：qwen 14/2 头、causal、scale 0.125、seed i64[2]；bert/gpt2 f32，dropout 0.1 判 unsupported，dropout 0 时数值正确；
- 28 类拒绝；
- 公开 profile 与已验证适配器对 42 个随机输入（含掩码、GQA、广播、rank 3/4、拒绝）逐端点相同。

测试修正：协调方报告的 `test_fa2_equals_exact_vjp_when_supplied_values_are_exact` 失败是测试本身的 bug，与草稿无关。
辅助函数 `M()` 对 mpf 调用了 `Fraction()`。已修正，并给每个测试固定 `mpmath.workdps(60)`，以免其他模块改动全局精度。

## 4 结果

- `pytest -q 5_specification/tests/test_attn_aten_a2_review.py`：48 passed，1 xfailed（问题 1）。
- 作者测试 `test_attn_aten_interface.py` 与本文件、`test_attention_vjp.py`、`test_attention_independent.py` 合跑：85 passed，1 xfailed。
- `-k "attn or attention"`：142 passed，1 xfailed。全量 `5_specification/tests`：1774 passed，1 xfailed，1 failed。
  失败的是 `test_norm_emb_nll_a2_review.py::test_review_stage_matches_drafts`（其他包，NORM 草稿在复核后被改过），与本包无关。
- `tools/check_contracts.py --contracts .cache/spec_a2_review/B/contracts --adapters .cache/spec_a2_review/B/adapters --only ATTN`：
  退出 0；examples 40/40，readings 2/2，mutations 18/18，consistent。
- 草稿与顶层合同的差异只有新增内容：公开 profile、决定、例子、适配器的公开路径都没有变（另经随机对照确认）。

## 5 修订后复核（2026-10-21，第二轮）

### 对象（sha256）

| 文件 | sha256 |
|---|---|
| `contracts/draft/ATTN_path_a_A2_20261020.yaml` | `82983298ae09d1af2bdabd5ad319fa3256e74b0126aeb048292e1723e78d100a` |
| `contracts/draft/ATTN_adapter_path_a_A2_20261020.py` | `5eefa125b2a7003e454c624da4dde3b78af2315a694a13602bd79439e97ea807` |
| `kernel_spec/attn_aten_semantics.py` | `2320f593c1a16a46537f24f6e879524bdb48e59a5b957778d6260c5cc2903e00` |
| `kernel_spec/attn_aten_indep.py`（未变） | `addf060b5a0b560740d89bd8d04c018ba041ecb35a19d7dcf04f96b801d9772f` |
| 复核测试 `tests/test_attn_aten_a2_review.py` | `6e882090744437e57db7fcab7bb39b5d0cc459c3847a3942b57fb24aa303f1e8` |

`.cache/spec_a2_review/B/` 已经刷新，暂存副本的 sha 与上表一致。顶层与 `validated/` 下的 ATTN 两个文件未变。

### 结论：CONFIRMED

上表四个文件可以晋升，复核确认后可以放入 `validated/`。晋升时把状态字段改为 reviewed / validated，YAML 的 sha 会随之变化，
但语义内容就是本轮复核的版本。

### 核对内容

- **问题 1 已修。** `attn_aten_semantics.py:208-213` 新增 `_flash_value_dim`，`parse_flash`（`:219-220`）和
  `parse_flash_bwd`（`:263-264`）在 Ev ≠ E 时判 invalid_input；efficient backward 不调用它。开关 `allow_value_dim` 只给变异
  AT.M19 用（适配器 `:421`），参照路径不会打开。AT.ATEN_DOMAIN 和 AT.C18 已经写明这一点。我改为非 xfail 测试，flash 前向与后向
  都判 invalid_input，AT.M19 给出数值（见证成立）。另加 efficient backward 的 Ev ≠ E 测试（宽度 3 与 32），结果与自写 FA2 一致。
- **AT.E24 手算成立。** E=4，默认 scale = 1/2，分数为 (2, 0)，p = (e²/(1+e²), 1/(1+e²))。v 的两行是 [1,1,1,1] 和 0，
  所以 out = e²/(1+e²)·[1,1,1,1]，lse = log(1+e²)，max_q=1，max_k=2。与例子相同。
- **AT.E42 手算成立。** s = ½·(1+1) = 1，给定 L=1，所以 P=1；D = 5·1 = 5，dP = 5·3 = 15，dS = 10。
  dq = ½·10·[1,1] = [5,5]，dk = ½·10·[1,1] = [5,5]，dv = 1·5 = 5，第 4 位为 null。与例子相同。这个例子同时用到了
  未补齐宽度（L=1）。
- **AT.E41：** E=2、Ev=1 判 invalid_input，正确。
- **选择 (b) 可以接受。** 保留未补齐宽度 L：前 L 项就是有定义的行值，32 对齐是内核的存储要求，不是任务语义。这与 (a)
  "域只按 dtype 限制"的原则一致，备选读法也已写明。所记录的调用 L=128，两种宽度相同，不受影响。
- **AT.ATEN_RNG 新增备选 (d)，成立。** backward 既不接收掩码也不返回掩码。文字上的小瑕疵（不阻塞）："neither the forward nor
  the backward overload returns … the mask"说得稍宽：flash forward 在 return_debug_mask=True 且 p>0 时会返回受 dropout 影响的
  debug_attn_mask。不过 profile 已拒绝 return_debug_mask，backward 也拿不到它，结论不变。
- 文字小瑕疵（不阻塞）：domain 的汇总列表 `aten_profile_valid` / `aten_profile_invalid_input`（YAML `:569-579`）没有列出
  "flash Ev = E"。规范性的 AT.ATEN_DOMAIN 和 AT.C18 已经写了，建议顺手补进列表。

### 结果

- `pytest -q 5_specification/tests/test_attn_aten_a2_review.py`：51 passed。
- 作者测试 `test_attn_aten_interface.py`：16 passed。`-k "attn or attention"`：146 passed，148 subtests passed。
- `tools/check_contracts.py --contracts .cache/spec_a2_review/B/contracts --adapters .cache/spec_a2_review/B/adapters --only ATTN`：
  退出 0；examples 42/42，readings 2/2，mutations 19/19，consistent。
