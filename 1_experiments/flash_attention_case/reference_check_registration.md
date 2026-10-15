# 与参照程序的比较：事前登记（2026-10-15，实现之前）

审查入口 `audit(spec=...)` 的语义比较改为四处（讨论记录：问题 B「kernel 除去舍入以外，算的和参照程序是不是同一个函数」）：

1. 逐元素判断，报告超出阈值的元素个数与所在 program；
2. 参照在某些元素上不是有限值时，记为「参照无定义」，计数后排除，容差检查也排除；
3. kernel 输出不是有限值、参照是有限值时，计入不一致；
4. 报告写明参照程序是什么，用词是「参照程序」，不写「规格」。

## 判定规则（实现按此写，不在看到结果后调整）

- 比较 G 与 S：G 是声明程序的精确值（区间中点，半宽 h_i），S 是参照程序在 float64 下的值；只在已求值的元素上比较。
- 尺度：scale_i = max(|S_i|, rms(S))，rms 取全部参与比较的元素。
- 阈值：θ = max(1e-6, 输出格式的单位舍入)，与现有对齐度判断的阈值相同。
- 元素超出：|G_i − S_i| − h_i > θ · scale_i。
- S_i 不是有限值：参照无定义，不比较。
- kernel 输出不是有限值：
  - S_i 有限且 |S_i| 不超过输出格式的最大有限值，计入不一致；
  - |S_i| 超出输出格式的范围，记为「参照超出输出格式的范围」，不判不一致。
- 结论：
  - 不一致：有元素超出；或 |⟨G − S, S⟩ / ⟨S, S⟩| > θ；或有上一条计入不一致的非有限输出；
  - 无法判断：没有参与比较的元素；
  - 其余为未发现差异。

## flash attention 案例的验收

参照程序（torch，float64，在 kernel 实际收到的输入上）：
- O：`F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)`，即 kernel 替换掉的 PyTorch 运算；
- M：`logsumexp(q kᵀ · sm_scale（causal 时上三角屏蔽）, -1) / ln 2`。教程的 kernel 按以 2 为底存 M，这是教程声明的约定。

| 案例 | O | M |
|---|---|---|
| tutorial_fp16、tutorial_fp16_causal、planted_tf32、planted_l_bf16、planted_acc_fp16 | 未发现差异 | 未发现差异 |
| planted_scale | 不一致 | 不一致 |
| planted_mask_diag（新增：causal 对角块的屏蔽 `>=` 改成 `>`，每个 query 看不到自己） | 不一致 | 不一致 |

另外两项核对：
- 输入、kernel 与 program 抽样都不变，K − G 部分不应受影响：O 的实测对齐与 `results/` 中已有记录的相对差 < 1e-9
  （planted_mask_diag 无旧记录，不适用）；
- 两种 O 参照互相核对：`scaled_dot_product_attention` 与 `fa_inputs.spec_attention` 的最大绝对差 < 1e-12。

## 单元测试（先写，修复前应失败）

- relu kernel 写成 leaky（负数乘 0.01），参照为 relu：误差全在 S = 0 的元素上，对齐度恰为 0。应判不一致，超出的元素个数等于
  负数输入的个数。
- 参照在部分元素上为 NaN：这些元素计为参照无定义；其余元素判未发现差异；宽容差检查通过。
- kernel 在部分元素上输出 NaN，参照有限：判不一致。

结果逐项按成立 / 不成立 / 无法判断报告。
