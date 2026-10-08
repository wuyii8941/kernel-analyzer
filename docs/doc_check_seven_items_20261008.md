# 七项「文档待抓取核对」：锁定版本文档原文 × 规格 docstring 差异报告（2026-10-08，交审阅方；规格未改）

文档原文取自锁定版本已安装包的 docstring（torch 2.10.0+cu128、transformers 4.57.3；在线文档由这些 docstring 渲染），逐字保存在
`results/closure/doc_check/*.txt`（`meta.json` 记版本）。HF 的 `lr_lambda` 在文档里只有文字描述，另附 4.57.3 源码中的函数作对照
（`2b_hf_cosine_lr_lambda_source.txt`），源码只作对照，不作为语义来源。按任务书第 9 节：**docstring 与原文不符的条款停止裁决，交审阅方**
（下表「处理」一栏）。

| # | 条目 | 原文要点（锁定版本） | 规格 docstring / 实现 | 差异 | 处理 |
|---|---|---|---|---|---|
| 1 | CosineAnnealingLR | 「The learning rate is updated **recursively** using η_{t+1} = η_min + (η_t − η_min)·(1 + cos((T_cur+1)π/T_max)) / (1 + cos(T_cur π/T_max))」；「This implements a **recursive approximation of the closed-form schedule** proposed in SGDR」（闭式随后给出）；「without restarts, so T_cur = t and increases monotonically」 | `spec_training_program.cosine_annealing` 取闭式；docstring 称文档说递推「在 lr 被调度器之外修改时才与闭式不同」（标「doc text not re-fetched」）；SCH-A1 据此把闭式作为规格 | (a) 2.10 原文没有「只在外部修改 lr 时不同」这句话，改为「递推是闭式的近似」，并以递推为更新式；(b) 原文的递推在 T_cur = T_max 时分母 1 + cos π = 0，T_max 之后的值原文没有给出（闭式继续按余弦回升）。T_cur < T_max 时两者在实数上逐项相同 | **SCH-A1 中 T_cur ≥ T_max 的部分停止裁决**；T_cur < T_max 不受影响（实数上相同）。请审阅方确定 T_max 及以后按哪一式 |
| 2 | HF get_cosine_schedule_with_warmup | 文字：warmup 期间从 0 线性升到初始 lr，之后按余弦从初始 lr 降到 0；num_cycles 默认 0.5；未写定义域 | 规格 `hf_cosine_with_warmup`：step < warmup 时 step / max(1, warmup)；之后 progress = (step − warmup)/(total − warmup)，max(0, ·)；total ≤ warmup 拒绝（契约 v2：total > warmup）。源码对照：progress 的分母是 **max(1, total − warmup)**，另有 min_lr_rate（默认 0） | 原文未规定 total ≤ warmup；源码对 total = warmup 有定义（分母取 1），规格与契约拒绝。不是原文冲突，是规格把定义域取得比实现窄；min_lr_rate = 0 时公式一致 | 不停止；契约 v2 的「total > warmup」条款请审阅方确认（2b 的 steps = 1 条件因此记「不合法」） |
| 3 | GradScaler | 默认 init_scale 2¹⁶、growth_factor 2.0、backoff_factor 0.5、growth_interval 2000；`step` 先 `unscale_` 并检查 inf/NaN，有则跳过 `optimizer.step()`；`update` 在有跳步时乘 backoff，在连续 growth_interval 次未跳步后乘 growth；警告：scale 不保证 > 1 | `ScalerState` / `scaler_step` 默认值与规则相同；跳步时 tracker 归零（对应原文「consecutive」）；规格把 update 合并进每步 | 无冲突。规格的合并写法等价于「每次迭代 step 后调用一次 update」，这是原文的推荐用法；原文的 `enabled=False`、手动 `new_scale` 规格未建模 | 不停止 |
| 4 | torch.utils.checkpoint | 反向时重新调用 function 重算未保存的张量；警告：反向时的调用若与前向不同（如全局变量），可能不等价、静默错误；`preserve_rng_state` 默认保存并恢复 RNG；`determinism_check="default"` 比较重算张量的元数据；可重入与不可重入的差异列表 | `prop_checkpoint_equivalence`：前提 dropout 为零、同输入同参数，实数上前向与梯度相同 | 无冲突。规格的前提比原文窄（原文在 dropout 下也通过保存 RNG 求等价），属范围选择 | 不停止 |
| 5 | torch.var 的 correction | σ² = (1 / **max(0, N − δN)**) Σ (x_i − x̄)²；correction 默认 1（Bessel） | `spec_base_ops.rvar`：N − correction ≤ 0 时 SpecNotEstablished（BASE-A1「无定义」）；docstring 说「torch 在某些版本记录 NaN/inf」并标「not re-fetched」 | 2.10 原文用 max(0, N − δN) 写出了这种情形的结果：分母为 0，Σ = 0 时为 0/0，Σ > 0 时为 Σ/0——原文没有把它写成「无定义」，也没有点名 NaN/inf | **BASE-A1 的 var / std 部分停止裁决**；请审阅方确定按「无定义（记约定）」还是按原文公式的扩展实数结果（NaN / +inf）对照 |
| 6 | torch.matmul 的广播 | 1-D 与 2-D 的提升规则；N > 2 时批维按广播语义广播（例：(j×1×n×m) 与 (k×m×p) 得 (j×k×n×p)） | `spec_base_ops` 只实现 2-D 与同批 3-D（BASE-A4）；docstring 对规则的描述与原文一致；契约 v2 把批维广播放在「合法域之外 → 拒绝」 | 原文中广播是**合法**输入；规格不覆盖它。契约把它记成「不合法」与原文不符——应是「规格未覆盖」而不是「输入不合法」 | 不涉及规格公式；**契约条款请审阅方改标**（2b 的 batch-broadcast 3 个条件目前记 D，结论只作「规格拒绝，不计对错」） |
| 7 | Embedding / EmbeddingBag | padding_idx：不贡献梯度、不更新；新建时默认全零；EmbeddingBag 中「excluded from the reduction」。max_norm：「each embedding vector with norm larger than max_norm is renormalized to have norm max_norm」，Embedding 的 forward 原地修改 weight。scale_grad_by_freq：按 mini-batch 中的词频倒数缩放梯度。空 bag 返回零；per_sample_weights「only supported mode is sum」；include_last_offset 末 offset 为输入长度 | E-D1、E-D3、E-D4、空 bag、per_sample_weights、include_last_offset 与原文一致。E-D2 写成「对**被查到的行**原地重归一化」；EMB-A1 说实现除以 (norm + c) | (a) 原文没有把 max_norm 的作用范围限定在被查到的行（「each embedding vector」字面上包括全部行）；(b) 原文没有提到常数 c；(c) padding 行「不计入 mean 的计数」是由「excluded from the reduction」推出，原文没有直说 | **E-D2 的作用范围停止裁决**（2b 的事后性质「未被引用的行不变」依赖它，暂不作结论）；EMB-A1 与 (c) 不停止，请审阅方确认 |

## 停止裁决的条款汇总
- SCH-A1（CosineAnnealingLR，T_cur ≥ T_max 的部分）；
- BASE-A1（var / std 在 N − correction ≤ 0 时）；
- E-D2（max_norm 的作用范围）。

这些条款涉及的条件在四组对照中照常运行并记录数据，结论栏标「条款待审阅」。另有两项契约标注请审阅方确认：HF 调度的 total > warmup，
matmul 批维广播的「不合法」应为「规格未覆盖」。
