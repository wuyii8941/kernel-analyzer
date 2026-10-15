# 三份规格合同：审阅后交付包

这个包补充并替代上传包的三份试点合同及实施口径，不改写原有数值规格源码。

**决定：可作为 Codex 实施依据；不是已经启用的 ground truth。**

## 阅读顺序

1. `CONTRACT_PROTOCOL.md`：固定口径，区分合同内容、实现验证和运行时接纳。
2. `contracts/ADAMW.yaml`、`CE.yaml`、`RANKMEAN.yaml`：完整观察、前态、域、公式、依据、profile 与独立见证。
3. `CODEX_TASK.md`：本轮要实现的适配器与检查器，含明确停止点。
4. `evidence/reviewed_witness_checks.json`：本次实际执行结果。

## 已完成

三份合同内容审查；固定 PyTorch 2.10 来源；明确 RANKMEAN 的任务选择；补完整后态、VJP、none/probability profiles、未定义与未支持分类；结构检查器及负例；直接调用原规格核验独立期望。
数目由 MANIFEST.json 记录。变异要求是实施任务，不是已经取得的检出成绩。

## 运行

```sh
python tools/validate_contracts.py
python -m unittest discover -s tests -v
python tools/check_witnesses.py --source-root /path/to/extracted/kernel_spec_package_2026-10-15 --output witness_results.json
```

要求 Python 3.10+、pyyaml、sympy、mpmath；调用旧规格时沿用其依赖。

`validate_contracts.py` 仅验证合同结构和引用。`--require-runtime-ready` 在当前包上必须返回 2，因为适配器和生产准入待完成，不可被理解成工具已可裁决。
`check_witnesses.py` 检查数学例子，不是生产适配器；明确跳过并登记待实现的非法输入/未支持检查。高精度点值复算不是严格区间证明。

## 边界

原始网页正文已通过 web 核对；容器未能下载原始 HTML，不声称已产生来源网页哈希。Codex 可在实施时留存版本化原文。
未执行 GPU、ONNX Runtime 旁证或完整仓库测试。本次没有改变模型任务、训练目标或继续扩大算子覆盖范围。
