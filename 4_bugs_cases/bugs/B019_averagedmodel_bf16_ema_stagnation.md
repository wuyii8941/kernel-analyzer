# B019：`AveragedModel` 对 bf16 模型做 EMA 时停滞，且不接受 fp32 的平均副本

日期：2026-10-07。对象：PyTorch 2.10.0（`ka_main`）与 nightly 2.15.0.dev20260907+cu126，`torch.optim.swa_utils`。

## 现象

`AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))` 对纯 bf16 模型（参数为 bf16，大模型微调常见）做权重 EMA：
平均副本由 `deepcopy` 得到，也是 bf16；每次更新的增量 (1 − decay)(p − ema) 小于 ema 在 bf16 中的半个 ulp（约 2⁻⁹·|ema|），
就近舍入后丢失，EMA 几乎冻结在开始平均时的权重上。想改用 fp32 的平均副本，`update_parameters` 按 dtype 分组时直接报错
（2.10：`Tensors of the same index must be on the same device and the same dtype ...`；nightly：`expected dtype float for end
but got dtype c10::BFloat16`）。文档没有提到 dtype。timm 的 `ModelEmaV3`（deepcopy）与 diffusers 的 `EMAModel`（clone）同样
继承模型的 dtype，不强制 fp32。

## 证据

- 最小复现 `4_bugs_cases/bugs/repro/B019_repro_averagedmodel_bf16_ema.py`（输出 `pre-reorg-20261009:results/directed_search/B019_repro_output.txt`）：模型权重
  漂移后再做 1000 次更新，fp32 EMA 与当前权重的相对距离从 0.268 降到 0.098（缩小 63.4%，按 1 − 0.999¹⁰⁰⁰ 应为 63.2%），bf16
  的 `AveragedModel` 只降到 0.262（缩小 2%）；fp32 平均副本被拒绝；按参数的 `avg_fn`（在 fp32 中 lerp）可以绕过。2.10 与 nightly 结果相同。
- 训练层面（`pre-reorg-20261009:scripts/importance/ema_deep.py`，`pre-reorg-20261009:results/directed_search/ema_deep_s*.json`）：4 层小模型纯 bf16 训练 2,000 步，
  同一次训练内三份 EMA 配对，种子 0–3：

  | 种子 | 训练后的模型 | `AveragedModel`（bf16） | fp32 EMA 参照 | 修正（fp32 副本 + fp32 `avg_fn`） |
  |---|---|---|---|---|
  | 0 | 4.891 | **9.096** | 5.164 | 5.164 |
  | 1 | 4.881 | **9.025** | 5.148 | 5.148 |
  | 2 | 4.906 | **9.010** | 5.171 | 5.171 |
  | 3 | 4.895 | **8.969** | 5.165 | 5.165 |

  bf16 平均模型的验证 loss 比 fp32 EMA 高 3.86（种子波动 σ = 0.0145 的约 270 倍，接近初始化附近的 loss）；修正后与参照逐位相同。
- 发现过程：阶段 C 的影子状态单步测量（`pre-reorg-20261009:docs/directed_search_results_20261007.md`）中，bf16 EMA 的单步作用 γ̂ = −0.51、b̂ = 0.50，
  是 δ（0.05）的十倍以上，随后按预注册的深案例步骤复现。

## 上游状态

检索（2026-10-07，GitHub API）：pytorch/pytorch 中没有关于 `AveragedModel` / `get_ema_multi_avg_fn` 与 bf16 精度的 issue；
`swa_utils.py` 的文档没有 dtype 说明。原理（低精度 EMA 会停滞）在社区里是常识，但 PyTorch 的接口既不提示，也不允许 fp32 的
平均副本。草稿：`4_bugs_cases/bugs/upstream_drafts/B019_averagedmodel_bf16_ema.md`（由用户决定是否提交）。

## 修法方向

允许 `AveragedModel` 的平均副本与模型 dtype 不同（在平均副本的 dtype 中做 lerp），或提供 `dtype=` 参数；至少在文档中说明
低精度模型的 EMA 应保存在 fp32。
