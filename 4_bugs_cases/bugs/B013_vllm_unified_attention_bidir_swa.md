# B013：vLLM `unified_attention`（TRITON_ATTN）在双向滑窗注意力里丢掉窗口右侧的 key

日期：2026-10-05。对象：vLLM main（2026-10-05，51eeb0c）`vllm/v1/attention/ops/triton_unified_attention.py`；
v0.24.0 起的各发布版（含今天发布的 v0.31.0；v0.23.0 的发布分支在引入之前切出，不含）代码相同。环境：ka_main（torch 2.10.0、Triton 3.6.0），RTX A6000，
vLLM 的 Triton 模块经 `.cache/pylibs/vllm_shim` 原样导入（`pre-reorg-20261009:scripts/build_vllm_shim.py`）。类：④（掩码与规格不符）。

## 机制

`kernel_unified_attention` 在启用滑窗时，除了对分数 S 做逐元素掩码（`compute_kv_seq_mask`，按每一行 query 的位置），
还把窗口外的 V 行置零，以免被回收的 KV 块里的 NaN 经 `0·NaN` 污染输出：

```python
if SLIDING_WINDOW:
    qpos_lo = q_block_local_idx * BLOCK_Q
    dist = context_len + qpos_lo - seq_offset[:, None]
    if USE_PER_SEQ_CAUSAL:
        sw_mask_v = tl.where(is_causal_seq, dist < SLIDING_WINDOW,
                             (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW))
    elif USE_CAUSAL:
        sw_mask_v = dist < SLIDING_WINDOW
    else:
        sw_mask_v = (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW)
    V = tl.where(sw_mask_v, V, 0.0)
```

这里的 `dist` 是相对于**块内第一个 query**（`qpos_lo`）算的，而一个块含 `BLOCK_Q = 16 / num_queries_per_kv` 个连续
query。

- 左边界（`dist < W`）对全块成立：落在第一个 query 窗口左侧之外的 key，也在块内每一行的窗口之外。
- 右边界（`dist > -W`）对全块不成立：块内第 d 行的窗口右端是 `qpos_lo + d + W − 1`，位置在
  `[qpos_lo + W, qpos_lo + W + d − 1]` 的 key 仍在它的窗口里，但它们的 V 已被置零。分数掩码正确，于是 softmax
  分母里有这些 key 的权重，分子里没有它们的贡献，输出偏小、方向错误。

因果注意力只有左边界，不受影响。触发条件：非因果（`causal=False`，或按序列的 causal 张量里为 False 的序列），
启用滑窗，`BLOCK_Q > 1`（`num_queries_per_kv < 16`），且同一序列内双向 query 段长于窗口（否则右侧不存在距离 ≥ W 的 key）。

引入：#45163（[Model] Add DiffusionGemma Support，2026-06-12）把原来只有左边界的 V 掩码改成了上面的三分支形式。

## 证据

1. 工具（`pre-reorg-20261009:results/tool_spec/final/vllm/`，96 个种子，0–31 开发、32–95 确认；参照完整度 1.00）：

   | 用例 | e_sem 相对 RMS | 区间不含 0 的坐标比例（最大区间宽度） | e_num 相对 RMS |
   |---|---|---|---|
   | `ua_bidir_sw8_mha`（MHA，窗口 (7,7)） | 0.905 | 0.87（7.7e-12） | 2.5e-4 |
   | `ua_bidir_sw24_gqa4`（GQA 8/2，窗口 (23,23)） | 0.115 | 0.26（6.7e-12） | 2.5e-4 |
   | `ua_perseq_causal_sw8`（按序列 causal = [False, True]） | 0.216 | 0.44（9.1e-12） | 2.3e-4 |
   | 对照 `ua_bidir_sw8_qpkv16`（BLOCK_Q = 1） | 2.2e-16 | 0 | 2.5e-4 |
   | 对照 `ua_causal_sw24`、`ua_bidir`（无窗口）及其余 10 个配置 | ≤ 3e-16（`ua_causal_gqa4_d80` 为 9e-9：fp32 的 80^-0.5） | 0 | 2–3e-4 |

   e_num 只是 fp16 舍入，说明 kernel 忠实地执行了它自己的、错误的掩码。MHA 用例被 R5 与默认检测器（vector_mean）
   检出；另两个用例的偏差方向随输入变化，统计规则不确认，而区间证实（e_sem 区间不含 0，详见
   `pre-reorg-20261009:docs/tool_changes_20261005.md` 第 5 节）直接给出偏差的证明，三者都归入"候选"。
2. 复现 `4_bugs_cases/bugs/repro/B013_repro_vllm_unified_attention_bidir_swa.py`（输出
   `pre-reorg-20261009:results/tool_spec/final/vllm/B013_repro_output.txt`）。用的正是 vLLM PR #51257 为这个组合新加、尚未合入的测试
   配置（head 128、block 16、bf16、窗口 64、序列 (129, 256)、(5, 64)、(65, 128)）。按序列相对误差（bf16 噪声约 2e-3）：

   | 头数 | causal | 序列 0（129 个 query） | 序列 2（65 个 query） |
   |---|---|---|---|
   | 4/4 | [False, True, False] | **0.153** | **0.033** |
   | 8/2 | [False, True, False] | **0.078** | **0.0088** |
   | 16/1 | [False, True, False] | 0.0021 | 0.0021 |
   | 4/4，`causal=False`（bool），序列 (200, 70) | — | **0.204** | **0.131** |

   PR #51257 的 `assert_close(atol=rtol=1e-2)` 在 Triton 路径上会失败；该 PR 的作者只在 CPU 上核对了参照公式，
   GPU 结果尚未贴出。
3. 修法验证（`pre-reorg-20261009:results/tool_spec/final/vllm/B013_repro_output_fixed.txt`）：把 V 掩码恢复成只有左边界
   （`sw_mask_v = dist < SLIDING_WINDOW`，即 #45163 之前的形式），全部 12 个配置回到 2e-3。右侧不需要置零：
   位置 ≥ seq_len 的槽位已由 `tile_mask` 的掩码读取补 0，窗口右侧的 key 都是本步写入的有效数据。

## 影响

- 后端声明同时支持非因果（`supports_non_causal`）与滑窗（`supports_sliding_window`），两者组合时输出错误，且不报错。
- 目前已知的使用者：DiffusionGemma（去噪阶段用按序列 causal 张量，Gemma4 的滑窗层）；DFlash 草稿模型（非因果）。
  按默认配置（canvas 256、滑窗 512/1024；DFlash 块约 16），双向段短于窗口，所以现有模型的默认配置**不触发**；
  canvas 长于窗口、窗口较小的模型，或任何走 TRITON_ATTN 的非因果滑窗用法会触发。属于潜在缺陷，而不是已在线上
  大面积出错。
- 编码器类型（ENCODER_ONLY，如 ModernBERT）走的是 `context_attention_fwd`，不经过这段代码。
- 检索 vLLM issue / PR（unified_attention 非因果滑窗、sw_mask_v、per_seq_causal sliding）：没有报告；#51257 的测试会
  暴露它，但 PR 描述认为"today's main computes this correctly on both backends"。

## 相关：ROCm 的 `prefix_prefill` 非因果滑窗只有左边界

`vllm/v1/attention/ops/prefix_prefill.py`（`context_attention_fwd`，ROCM_ATTN 等后端；`CAUSAL=False` 用于 DFlash 的
双向草稿块）在 query 自注意力部分的滑窗掩码是 `offs_m - k < SLIDING_WINDOW`，非因果时右侧没有限制，而
`unified_attention`（`compute_kv_seq_mask`）与 FlashAttention 后端（窗口对称化为 (w, w)）都是 |q − k| < W。
同样只在双向段长于窗口时有差别，与 B013 同属"非因果 + 滑窗"这一少有人用的组合；未单独复现，记为观察。
