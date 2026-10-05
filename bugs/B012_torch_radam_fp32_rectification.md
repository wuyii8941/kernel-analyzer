# B012：torch.optim.RAdam 的 capturable 路径与 torch.compile 路径在 fp32 里算整流项，β₂ 接近 1 时更新出错

日期：2026-10-05。环境：torch 2.10.0+cu128（`ka_main`），RTX A6000；PyTorch main（2026-10-05 下载的
`torch/optim/radam.py`）代码相同。类：④（规格的公式以不稳定的方式在低精度下求值，加上 β₂ 的 fp32 常数取整）。

## 机制

RAdam 的整流项

    ρ_t = ρ_∞ − 2t·β₂ᵗ / (1 − β₂ᵗ),   ρ_∞ = 2/(1 − β₂) − 1,   ρ_t > 5 时才做自适应（整流）步

是两个接近 ρ_∞ 的数相减（β₂ = 0.999 时 ρ_∞ ≈ 1999，而前几步的 ρ_t 只有 5 左右）。

- 非 capturable 的 eager 路径：`step = _get_value(step_t)` 取 Python float，全程 float64，没有问题。
- `capturable=True`：`step = step_t`（float32 张量），`beta2**step`、`1 - beta2**step`、`rho_t` 都是 float32
  张量运算；β₂ 先被舍入到 float32（0.9999 → 0.99989998，1 − β₂ 的相对误差 1.7e-4），再经相消放大。
- `torch.compile(opt.step)`：`_get_value` 在编译时直接返回张量，所以走的是同样的 float32 计算。

结果是 ρ_t 的误差在 β₂ = 0.999 时约 4e-3（相对），β₂ = 0.9995 时 3e-2，β₂ = 0.9999 时 0.4，β₂ = 0.99999
时 46 倍；整流判据 ρ_t > 5 会翻转，整步更新都错。

## 证据

1. 工具：`results/tool_spec/final/radam/`（对规格检查，规格 = eager float64 优化器从同一状态走一步）。
   - 默认 β₂ = 0.999、第 7 步：param 的 e_sem 7.3e-7（相对参数，约为更新量的 7e-5），检出，分在"小"档；
   - β₂ = 0.9995、第 5 步：param 的 e_sem **1.45e-2**（≈ 整个更新量），检出；e_num 2.4e-7。说明仅 β₂ 的 fp32
     常数取整、按实数精确求值（K_R），就已经让判据翻转；
   - β₂ = 0.9999、第 2 步：e_sem 8.2e-3、e_num 8.2e-3，float32 算术相消（K − K_R）又贡献一份。
2. 复现 `bugs/repro/B012_repro_radam_fp32_rho.py`（输出 `results/tool_spec/final/radam/B012_repro_output.txt`），
   参数更新对 eager float64 的相对误差：

   | β₂ | fp32 下判据翻转的步 | capturable 最大误差 | 编译最大误差 | eager（非 capturable）fp32 |
   |---|---|---|---|---|
   | 0.999（默认） | 无 | 6.0e-3（第 7 步） | 5.7e-3（第 6 步） | 2.3e-4 |
   | 0.9995 | 5 | 0.99（第 5 步） | 0.99（第 5 步） | 3.3e-4 |
   | 0.9999 | 2、4、5 | 0.99；第 6 步 0.58、第 10 步 0.24、第 50 步 0.04 | 同 | 7.6e-4 |
   | 0.99999 | 1–5 | 13 倍（第 6 步） | 18 倍（第 6 步） | 2.3e-3 |

## 影响

- 默认 β₂ = 0.999：前几十步的更新偏 0.1%–0.6%，随后衰减，影响小但系统性。
- β₂ ≥ 0.9995：RAdam 设计用来处理的正是前几步（方差不可估计时不做自适应），fp32 求值恰恰在这几步把判据弄反；
  β₂ = 0.99999 时更新大一个数量级。
- 受影响的入口：`RAdam(..., capturable=True)`（CUDA graph 场景）、`torch.compile(optimizer.step)`
  （官方推荐的编译优化器用法）；单张量与 foreach 实现相同。
- 检索 PyTorch issue（RAdam capturable / rho_t / rectification / compile），没有相关报告。

## 修法方向

ρ_∞ 与 ρ_t 只依赖 β₂ 和步数，和参数无关：可以在 float64 里算（capturable 时用 float64 的 0 维张量，
或把 (1 − β₂) 作为精确常数并用 `expm1(t·log1p(−(1 − β₂)))` 求 1 − β₂ᵗ），再把整流系数转成参数精度。
