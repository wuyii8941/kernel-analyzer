# v2.5 草案（2026-10-16 规格包，隔离存放）

## 来源与状态

- **来源**：`kernel_spec_package_2026-10-16.tar.gz` 中的 `specification_v2_5/`，外加包里的 `SPEC_V2_5_PLAN.md`、
  `NEXT_STEPS.md`（存为 `NEXT_STEPS_20261016.md`）和包的 README（存为 `PACKAGE_README.md`）。
  - 压缩包 SHA-256：`21e578aafed2585c8c6e319557a97ade4be315d85577f061e9d468c3988dabb9`。
  - 包里的 `survey_9bb8abe/` 与 `1_experiments/library_survey/` 逐字节相同，没有复制。
  - 包里的其余部分已在仓库中：`specification_v2/`、`evidence/` 与仓库相同；`contracts/` 是 10-15 的旧草案，仓库里已有审阅后的版本。
- **状态**：草案，不进入任何结论（用户 2026-10-16 的决定）。
  - 运行时注册入口 `kernel_spec/registry.py` 只读 `contracts/*.yaml`，不会加载这里的任何文件。
  - 这里的 `check_v25.py` 会给出假通过，`CHECK_REPORT_v25.md` 与 `COVERAGE_v25.md` 中的计数都不是认证结果。

## 已复核属实的问题（10-16 外部审阅，`../../reviews/2026-10-16/REVIEW.md`）

| 问题 | 复核 |
|---|---|
| `check_v25.py` 报出矛盾仍退出 0 | 把 H1 的一个期望值改成 999：报告显示 contradiction，退出码 0 |
| 读法改错仍显示全部成立 | 把 B1 的 inclusive 读法改成 777：仍为 54/54 established |
| 隐藏容差 | 第 33–38 行用 10⁻⁴⁰ 容差判严格包围，宽度 ≤ 10⁻³⁰ 即判 established |
| 异常被当成数学结论 | 第 63–64 行把 `SpecNotEstablished` 转成 undefined；第 70–76 行把误读抛异常当作「不适用」 |
| MoE 权重区间丢掉真值 | 三个相等 logit、k = 3 时，返回的区间不含 1/3 |
| 加权归约丢掉参与者 | 两个参与者、一份权重：返回 `[1]`，没有拒绝 |
| 线性注意力默认缩放不一致 | d_k = 4 时，递推默认 1/2，分块默认 1 |
| A1 的完整性 | 合同主要定义未归一化的 Sᵀq，却引用含归一化的论文（审阅意见，未另行复核） |
| F2 的 profile | 把 OCP 允许的另一种取整当错误（审阅意见，未另行复核） |

## 怎样启用其中的一份

只有覆盖到测过算子的合同才会启用，例如线性注意力和 mamba 的状态更新、dropout 与采样、MoE 路由。启用时：
- 按 v3 的合同格式重写，放进 `contracts/`；
- 参照代码先修上面列出的问题，每个问题先写成失败的测试，再修；
- 过 `tools/check_contracts.py`。

计划与偏离见 `../../docs/PLAN_v3.md`。
