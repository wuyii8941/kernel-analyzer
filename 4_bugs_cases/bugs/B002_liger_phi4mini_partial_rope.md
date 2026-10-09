# 发现：Liger 已发布版本把 Phi-4-mini 的 RoPE 算错（④ 算法与规格不符）

日期：2026-10-04。环境：`/data1/tzh/envs/latest_train`（torch 2.10.0+cu128、transformers 5.18.0、
liger-kernel 0.8.4 = PyPI 最新发布）与 `liger`（transformers 4.57.3、liger-kernel 0.7.0）。

## 现象

Phi-4-mini-instruct 的 `model_type` 是 `phi3`，`head_dim = 128`，`partial_rotary_factor = 0.75`，
即只旋转前 `rotary_dim = 96` 维，按 (i, i+48) 配对，后 32 维原样通过（HF `modeling_phi3.apply_rotary_pos_emb`）。

`apply_liger_kernel_to_phi3` 默认 `rope=True`，把 `apply_rotary_pos_emb` 换成 `liger_rotary_pos_emb`。
0.8.4 及以前的 Triton RoPE kernel 假定 `rotary_dim == head_dim`：按 (i, i+64) 配对、旋转整个 head，
cos/sin 只有 96 宽却按 64+64 读取。没有报错，也没有警告。

| 配置 | wikitext-103 验证集损失（32768 token，bf16） |
|---|---|
| HF 原版 | 2.700 |
| + Liger 0.8.4 RoPE（`model=` 实例补丁，HF Trainer `use_liger_kernel=True` 走的路径） | **12.928** |
| + Liger 0.7.0 RoPE（transformers 4.57.3） | **12.928** |
| + Liger main（含 PR #1451） | 2.700 |

单层 q 的 RoPE 输出相对差 0.84；本应原样通过的后 32 维相对差 0.91。

## 来源与状态

- 普查 `pre-reorg-20261009:scripts/census_liger_integration.py`（tiny 随机模型、fp32、对照未打补丁的同一实例、
  一 ulp 参数扰动给出条件数底）在 `phi3 / partial_rotary_0.75 / rope` 上报出梯度相对差 3.8e-2，
  是底的 6.9 万倍；真实 checkpoint 复核见 `pre-reorg-20261009:scripts/liger_rope_checkpoint.py`，
  结果 `pre-reorg-20261009:results/census/phi4mini_liger_rope_{0.7.0,0.8.4,main}.json`。
- Liger 已知自己的 RoPE kernel 不支持部分旋转，对 GLM-4、Qwen3-Next、Qwen3.5 抛
  `NotImplementedError`，Nemotron 不打 RoPE 补丁；唯独 `phi3` 没有检查。
- main 上的 PR #1451（2026-10-02 合并）把 kernel 改成支持 `rotary_dim < head_dim`，顺带修好了这个问题。
  但 PR 把它当作新功能（列出 Qwen3-Next、GLM-4、GPT-NeoX、Phi-2、StableLM），没有提到 phi3 / Phi-4-mini
  在已发布版本里一直被静默算错，也没有发补丁版本。检索 Liger 的 issue，没有找到相关报告。
- 受影响范围：实测 0.7.0 与 0.8.4；#1451 之前 kernel 一直假定全旋转，推断 transformers 4.49
  （Phi3 加入 partial_rotary_factor）以来的发布版本都受影响（未逐一实测）。
  `model_type == "phi3"` 且 `partial_rotary_factor < 1` 的 checkpoint：Phi-4-mini-instruct（实测）、
  同架构的 Phi-4-mini-reasoning（未实测）。入口：`apply_liger_kernel_to_phi3`，包括 HF Trainer / TRL SFT 的
  `use_liger_kernel=True`、`AutoLigerKernelForCausalLM`。

## 可提给上游的内容（由用户决定是否提）

1. 说明已发布版本对 Phi-4-mini 静默算错，附上面的复现与数字，请求发一个含 #1451 的补丁版本。
2. 补一条 phi3 + `partial_rotary_factor=0.75` 的回归测试（mini model 收敛测试目前用的是全旋转配置）。
