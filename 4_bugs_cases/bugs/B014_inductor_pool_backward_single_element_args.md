# B014：torch.compile 下池化反向对 1 元素的 kernel_size / stride / padding / dilation 直接崩溃

日期：2026-10-05。对象：PyTorch 2.10.0（`ka_main`），PyTorch main（2026-10-05 下载的 `torch/_inductor/lowering.py`）
代码相同。类：编译期崩溃（不是数值问题）。

## 现象

eager 接受 1 元素序列作为池化参数（表示各空间维取同一个值），例如 `nn.AvgPool2d(3, padding=(1,))`、
`F.max_pool2d(x, 3, 2, 1, dilation=(1,))`。`torch.compile` 下前向正常，反向（训练）编译失败：

    torch._inductor.exc.InductorError: LoweringException: AssertionError:
      target: aten.avg_pool2d_backward.default ... args[4]: [2]

## 机制

前向的 `_avg_poolnd`、`max_pool_checks` 先用 `pad_listlike(x, dim)` 把 1 元素列表扩展到 dim 维，再断言长度；
反向的三个 lowering 没有这一步，直接断言：

| lowering | 断言 |
|---|---|
| `avg_pool2d_backward` | `len(kernel_size) == 2`、`len(stride) == 2`、`len(padding) == 2` |
| `avg_pool3d_backward` | 同上，== 3 |
| `max_pool2d_with_indices_backward` | 同上，外加 `len(dilation) == 2` |

（max_pool3d 的反向走另一条路径，不受影响。）

## 发现过程

OpInfo × Inductor 筛查（`pre-reorg-20261009:scripts/tool_spec_cases_opinfo.py`，`OPINFO_SELECT=onesample`）里，
`oib_nn_functional_avg_pool2d_5`（OpInfo 自带的样例：kernel (4,4)、stride (2,2)、padding (2,)、ceil_mode、
count_include_pad）在工具运行时报 LoweringException。PyTorch 的 Inductor OpInfo 测试对 avg_pool1d/2d/3d、
max_pool1d 等只跑第一个样例（`inductor_one_sample["cuda"]`），所以这个样例从未在 CI 里被编译过。随后对
`lowering.py` 做静态扫描（断言参数长度但不调用 `pad_listlike` 的函数），找到另外两处，逐一实测确认。

## 证据

`4_bugs_cases/bugs/repro/B014_repro_inductor_pool_backward_single_element_args.py`（输出
`pre-reorg-20261009:results/tool_spec/final/inductor_pool_bwd/B014_repro_output.txt`；必须关闭 Inductor 缓存，否则命中先前成功的编译
产物会绕过 lowering）：5 种写法全部崩溃，max_pool3d 对照正常；在三个 lowering 的入口加上 `pad_listlike` 规整后
全部通过，梯度与 eager 逐位相同。

## 影响与状态

- 只影响用 1 元素序列传参的代码（模块写法 `padding=(1,)` 之类），常见的整数或完整元组写法不受影响；后果是
  训练编译直接失败，不会给出错误结果。
- 检索 pytorch/pytorch issue（avg_pool2d_backward / max_pool2d_with_indices_backward LoweringException、
  pad_listlike）：未见报告。
- nightly 实测（2026-10-06，2.15.0.dev20260907+cu126，CUDA）：`max_pool2d` 的两种单元素写法已通过（前向与梯度与 eager 相同，修复来源未查）；`avg_pool2d`（两种写法）与 `avg_pool3d` 仍然 `LoweringException: AssertionError`。GitHub 检索（2026-10-06，API）仍未见报告。
- 修法：三个反向 lowering 开头对 kernel_size / stride / padding（/ dilation）调用 `pad_listlike`，与前向一致。
