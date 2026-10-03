# blind_test_v1 交付材料（被测方 → 审阅方）

本分支只放交付用的数据包，与 `main` 的代码历史分开。生成这些包的代码与报告在 `main` 的提交 `0048ef5`：

- 阶段 1 报告（协议 v1.1）：`results/reference_eval/blind_test_v1/phase1_report_v1.1.md`
- 阶段 2 报告：`results/reference_eval/blind_test_v1/phase2/phase2_report.md`

检测器冻结于 `e43616c`，覆盖补充为 `cf7631c`；答案文件未打开。

## 内容

| 路径 | 用途 | 生成脚本（main） |
|---|---|---|
| `phase2_subset/blind_v1_subset_priority.tar.gz` | 阶段 2 发布文件第 6 节子集，优先部分（F1、F3、F5：prog_05、07、11、12、21、26、27、31） | `scripts/blind_test_v1_subset.py` |
| `phase2_subset/blind_v1_subset_others.tar.gz` | 第 6 节子集，其余（prog_01、14、15、16、19、32） | 同上 |
| `phase1_verification/blind_v1_verification.tar.gz.part00`–`part03` | 阶段 1 协议 v1.1 第 5 节的独立核实材料（分卷，合并后 146 MB） | `scripts/blind_test_v1_export.py` |

第 6 节子集：阶段 1 在 R2 或 R3 下检出的程序，seed 0 与 1 各一个 `.npz`（输入张量、运行时标量的 float64 值、输出 K），
键名说明、每个程序的家族与运行时标量的原始写法见 `index.json`（两个压缩包内各有一份，`phase2_subset/index.json` 是同一文件的副本，便于不解包查看）。

第 5 节材料：阶段 1 有检出的每个程序在 seed 0 与 32 上的输入张量、运行时标量、输出 K（`.npy` 与原始字节）、工具的 K_R
区间、残差区间与 R1–R5 下的端点保守投影（R5 方向另存为 `.npy`）。

## 合并与校验

GitHub 单文件上限 100 MB，第 5 节材料按 45 MiB 分卷：

```bash
cat phase1_verification/blind_v1_verification.tar.gz.part* > phase1_verification/blind_v1_verification.tar.gz
sha256sum -c SHA256SUMS
```

`SHA256SUMS` 的最后一行是合并后的整包，合并之前校验该行会报缺文件。
