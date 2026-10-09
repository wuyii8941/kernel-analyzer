# B020：`scatter_reduce` / `index_reduce` 的 amax、amin 在 `include_self=False` 时，反向把被排除的 self 计入并列分摊

日期：2026-10-07。对象：PyTorch 2.10.0（`ka_main`）eager CPU 与 CUDA、Inductor；nightly 2.15.0.dev20261005（CPU）与
2.15.0.dev20260907（CUDA）eager 与 Inductor；main（2026-10-07 的 `FunctionsManual.cpp`）代码相同。类：反向公式与前向语义不符
（共有错误：eager 与 Inductor 一致，二者都偏离规格）。

## 现象

```python
s   = torch.tensor([3., 5.], requires_grad=True)      # s[0] 被排除（include_self=False），恰好等于结果
src = torch.tensor([3., 1.], requires_grad=True)      # 位置 0 收到 3（唯一最大）与 1
s.scatter_reduce(0, torch.tensor([0, 0]), src, "amax", include_self=False).backward(torch.ones(2))
src.grad   # tensor([0.5, 0.])   应为 [1., 0.]
s.grad     # tensor([0., 1.])    正确
```

`include_self=False` 时 out[0] = max(src[0], src[1]) = src[0]（唯一最大者），与 s[0] 无关，所以 ∂out[0]/∂src[0] = 1，
∂out[0]/∂s[0] = 0。这里函数在该点可微（参与归约的值中没有并列），与并列时如何分摊梯度的约定无关。PyTorch 给 src[0] 0.5：
被排除的 s[0] 因为数值恰好等于结果，被算进了并列个数，分走一半，而它自己又被清零——梯度总量从 1 变成 0.5。amin 对称。
`index_reduce` 相同。

## 机制

`torch/csrc/autograd/FunctionsManual.cpp`，`scatter_reduce_backward` 与 `index_reduce_backward` 的 amax/amin 分支（v2.10.0 第
7177 / 7273 行；main 第 7750 / 7846 行）：

```cpp
Tensor self_is_result = (self == result).to(self.scalar_type());     // 不看 include_self
Tensor src_is_result = (src == value).to(self.scalar_type());
Tensor N_to_distribute = self_is_result.scatter_add(dim, index, src_is_result);
Tensor grad_distributed = grad / N_to_distribute;
grad_self = (self == result) * grad_distributed;
grad_src = (src == value) * grad_distributed.gather(dim, index);
...
if (!include_self) {
  grad_self = grad_self.scatter(dim, index, 0);                      // 事后才把 self 清零，src 已经被多除
}
```

同一函数里 `mean` 分支正确处理了 include_self（`N = include_self ? ones_like(grad) : zeros_like(grad)`）。修法：在
include_self=False 时，被写入的位置上 `self_is_result` 取 0，再计算 `N_to_distribute`。

## 证据

- 最小复现与不使用工具代码的独立复算：`4_bugs_cases/bugs/repro/B020_repro_amax_include_self_false_grad.py`（输出
  `1_experiments/essential_bugs_round/results/phase1/bugs/B020_repro_output.txt`）。独立复算用正确的前向做精确中心差分（分段线性，步长远小于最大值与次大值
  之差），得 1.0；autograd 得 0.5。2.10 CPU/CUDA、nightly CPU 20261005、nightly CUDA 20260907，scatter_reduce 与 index_reduce、
  amax 与 amin，全部不一致（8 + 4 + 8 处）。
- 本轮网格（`1_experiments/essential_bugs_round/protocol_phase1.md`，`1_experiments/essential_bugs_round/results/phase1/classification_index.json.gz`）：index/scatter
  602 个条件中，**50 个条件的反向**在所有 eager 候选（CPU float64 / float32、CUDA float32、nightly CUDA）与 nightly Inductor 上
  偏离规格且彼此一致（「共有错误」）；2.10 的 Inductor 共有其中 48 个（另 2 个与 B016 叠加）。全部是 amax/amin、include_self=False、
  整数输入（被排除的 self 才会恰好等于结果）。

## 各组能否发现（W8）

| 组 | 能否发现 | 说明 |
|---|---|---|
| E（与 eager 比） | 否 | eager 本身就是出错的一方；Inductor 的反向分解用同一公式 |
| FP64 参照（eager CPU float64） | 否 | 同一公式 |
| F（独立规格） | 是 | 50 / 50 个条件 |
| P（不依赖参照的性质） | 部分 | 并列集合的次梯度检查（系数和为 1）在参与值本身有并列时发现（28 / 50 个条件）；参与值无并列、只是被排除的 self 等于结果时（22 / 50）预注册的性质没有覆盖——「梯度总量等于上游」这一条性质能覆盖，但不在本轮的 P 清单里 |
| FR（K_R） | 是 | Inductor 反向 kernel 的 e_sem 区间不含零（语义差异，不是舍入） |

## 测试判据（W5 的预览）

OpInfo 的 `sample_inputs_scatter_reduce` / `sample_inputs_index_reduce` 用 `make_tensor` 生成连续随机数（include_self 取 True
与 False），被排除的 self 几乎不可能恰好等于结果；梯度检查（含有限差分 gradcheck）因此从不经过这种情形。prod 有专门构造的含零
样例，amax/amin 没有对应的并列样例。`test/test_scatter_gather_ops.py` 没有 amax/amin 的梯度测试。

## 上游检索（W9，2026-10-07）

- issue / PR 关键词（`scatter_reduce include_self amax gradient`、`index_reduce amax backward`、`N_to_distribute` 等）：未见报告。
  相邻：#168358（Inductor 下 NaN，已关）与 PR #169263（N_to_distribute 为 0 时的 NaN，未合并），都不是本问题。
- main 的 `FunctionsManual.cpp`：未改。
- nightly：CPU 2.15.0.dev20261005、CUDA 2.15.0.dev20260907 均复现。
- 状态：新问题候选；上游草稿 `4_bugs_cases/bugs/upstream_drafts/B020_amax_amin_include_self_false_backward.md`，是否提交由用户决定。
