# kernel-analyzer 规格包（2026-10-15）

## 内容

| 路径 | 是什么 |
|---|---|
| METHOD.md | 定稿的规格生成方法：两个参照（G_π 与 f）、三个对象（原文 → 合同 → 参照）、四种保证标签、三种检查结果 |
| NEXT_STEPS.md | 接下来要做的事，按顺序，每步有完成标志 |
| specification_v2/ | 规格 v2.4，原样未改（31 个角色的可执行参照、关系检查、测试） |
| contracts/ | 合同层：模板、三份试点合同（CE、ADAMW、RANKMEAN）、适配、检查器、检查报告 |
| evidence/ | 证据（不决定条款状态）：ONNX Runtime 对照、v1 错误重建、THEORY 撤回部分的核对、CE 与 Liger 声明公式的对照 |
| run_all.sh | 一条命令跑完全部检查 |

## 运行

依赖 python3 与 numpy、scipy、sympy、mpmath、pyyaml；onnxruntime 可选。

    ./run_all.sh

## 当前状态

- 规格 v2.4：31 个角色；测试 222 项全部通过。
- 合同层：3 份试点合同，共 11 条条款，全部 established；11 条误读全部被拒绝；三份合同都待审阅（draft-for-review）。
- 其余 28 个角色尚未迁移到合同层；在迁移并审过之前，工具对这些角色的语义判断一律报告「未建立」。
