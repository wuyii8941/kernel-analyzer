# B024：Triton 浮点 `%`（`arith.remf`）在商超过 2²⁴ 时结果错误（sm_86，Triton 3.6.0）

日期：2026-10-13。对象：Triton 3.6.0（`ka_main`），RTX A6000（sm_86）。类：④（下降实现的不是声明的函数）。

## 现象

`x % y` 对浮点张量下降为 `arith.remf` → `llvm.frem`；NVPTX 没有 frem 指令，LLVM 把它展开为 `x − trunc(x / y) · y`
（PTX：`div.rn.f32`、`cvt.rzi.f32.f32`、`fma.rn.f32`）。只要 x / y 不能精确表示（|x / y| ≥ 2²⁴，或商被舍入），结果就不是
fmod，甚至符号和量级都不对：

| x | y | Triton `x % y` | `torch.fmod` |
|---|---|---|---|
| 1e20 | 3 | 2.2e12 | 2 |
| 1e30 | 1e-3 | −2.7e22 | 1.7e-4 |
| 3e10 | 0.7 | 510.9 | 0.597 |
| 67108864 | 3 | −2 | 1 |

f64、f16、bf16 同样受影响（原语表：`arith.remf` 四种类型都归为「近似」，f32 最大误差约 1e40 个 ulp）。libdevice 的
`__nv_fmodf` / `__nv_fmod` 是精确的（f32 只有 FTZ 造成的次正规冲零）。

## 证据

- 原语表 `1_experiments/composition_study/results/catalog.json`，`F1|arith.remf|default|*`：正确舍入比例 0.90–0.99，
  误差集中在宽分布与边界分布（商很大）；同一表中 `libdevice|__nv_fmod` 为正确舍入。
- 直接复现：`1_experiments/composition_study` 的 catalog 运行与上表（`tl.load(x) % tl.load(y)`，TTIR 为 `arith.remf`）。

## 上游状态

已知：triton-lang/triton issue #11986；修复 PR #11938（2026-09-24）、#11987（2026-09-27）均未合并（2026-10-13 检索）；
LLVM 侧 [NVPTX] Use ExpandIRInsts for frem（#224141）已于 2026-10-08 合并，但不在 Triton 3.6.0 锁定的 LLVM 中。
不另写草稿。

## 首次发现

组合研究的原语表（每个原语实例在设备上对精确值），之后才检索到上游记录。工具：仿真器对 `arith.remf` 原本就由设备求值
（近似指令的规则），所以组合单元逐位复现；偏差要对声明值才看得见：参照 K_R 是精确的 fmod，K − K_R 在商大的元素上为
整个量级。
