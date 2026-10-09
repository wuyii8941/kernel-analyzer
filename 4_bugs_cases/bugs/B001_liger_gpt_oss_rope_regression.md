# 发现：Liger main 的部分 RoPE 改动把 GPT-OSS 的 RoPE 算错（④，未发布的回归）

日期：2026-10-05。环境：`/data1/tzh/envs/latest_train`（torch 2.10.0+cu128、transformers 5.18.0）；
Liger 发布版 0.8.4（PyPI）与 Liger main（2026-10-04 下载，rope.py 最新提交 b297821787 = PR #1451，
2026-10-02 合并；10-05 重新下载核对，main 未变）。

## 机制

HF GPT-OSS 的 rotary embedding 返回宽度为 `head_dim/2` 的 cos/sin（`emb = freqs`，不复制成两份），
`_apply_rotary_emb` 旋转 (x[:d/2], x[d/2:])。

- Liger 0.8.4 的 kernel 假定 `rotary_dim == head_dim`，从 cos/sin 读前 d/2 个，恰好与 GPT-OSS 的半宽
  cos/sin 一致，结果正确。
- PR #1451 为支持部分旋转，按 `rotary_dim = min(cos.shape[-1], head_dim)` 推断旋转宽度。GPT-OSS 的
  cos 宽 d/2，于是被当成"只旋转前一半"：前 d/2 维按 (i, i+d/4) 配对旋转，后 d/2 维原样通过。没有报错。

`apply_liger_kernel_to_gpt_oss` 默认 `rope=True`，所以一旦发版，HF Trainer / TRL 的
`use_liger_kernel=True` 微调 gpt-oss 都会走到这里。

## 证据

1. 普查 `pre-reorg-20261009:scripts/census_liger_integration.py`：`gpt_oss / rope` 在 0.8.4 上梯度相对差 7e-7（底的 1 倍），
   在 main 上 1.4（底的 180 万倍）。结果 `pre-reorg-20261009:results/census/liger_integration_{latest,main}.jsonl`。
2. 最小复现 `pre-reorg-20261009:scripts/repro_liger_gpt_oss_rope.py`（GptOssConfig 默认值，head_dim 64，cos 宽 32）：

   | | q 相对差 | dq 相对差 |
   |---|---|---|
   | 0.8.4 | 3.3e-8 | 4.0e-8 |
   | main | 0.970 | 0.970 |

3. Liger 自己的 fp32 收敛测试 `test/convergence/fp32/test_mini_models.py::test_mini_model[mini_gpt_oss-...]`
   在 main 上失败；只把 `pre-reorg-20261009:src/liger_kernel/ops/rope.py` 换回 0.8.4 的版本、其余保持 main，同一测试通过。
4. 真实 checkpoint（gpt-oss-20b）验证集损失：见下表（待补）。

## 检索

Liger 仓库 2026-08 以来提到 gpt_oss 的 issue/PR 为 0；10-03 的 #1508（A100 上的收敛失配）列出的是
Qwen3-MoE、Llama4、Qwen3.5-MoE，没有 GPT-OSS。

## 可提给上游的内容（由用户决定是否提）

1. 在下一次发版前报告：#1451 用 `cos.shape[-1]` 推断 rotary_dim，对 cos/sin 本来就是半宽的模型
   （GPT-OSS）会误判成部分旋转。附最小复现与收敛测试失败。
2. 修法的方向：rotary_dim 不能只从 cos 宽度推断，要区分"半宽、不复制"的约定（GPT-OSS）与
   "复制成两份、宽度 = rotary_dim"的约定（Llama/Phi3），例如由调用方显式传入，或对 gpt_oss
   单独包一层把 cos/sin 复制成全宽。
