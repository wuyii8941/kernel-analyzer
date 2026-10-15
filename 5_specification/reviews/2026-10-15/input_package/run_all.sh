#!/bin/sh
# 依赖：python3，numpy、scipy、sympy、mpmath、pyyaml；可选 onnxruntime
set -e
cd "$(dirname "$0")"
echo "== 1. 规格 v2.4 测试（应为 9 + 4 + 105 + 89 + 15 = 222 项通过）"
(cd specification_v2 && python3 test_relations.py 2>&1 | tail -1 && python3 test_oracle.py 2>&1 | tail -1)
(cd specification_v2/algorithm_specs/phase1 && python3 test_specs.py | tail -1)
(cd specification_v2/algorithm_specs/phase2 && python3 test_specs_phase2.py | tail -1)
(cd specification_v2/algorithm_specs/phase3 && python3 test_specs_phase3.py 2>&1 | tail -1)
echo "== 2. 合同检查"
python3 contracts/contract_check.py > contracts/CHECK_REPORT.md && tail -3 contracts/CHECK_REPORT.md
echo "== 3. 历史误读重建"
python3 evidence/historical_misreadings/meaning_relations.py
echo "== 4. 可选：ONNX Runtime 对照"
if python3 -c "import onnxruntime" 2>/dev/null; then python3 evidence/ort_witness/battery.py 2>/dev/null | tail -2; else echo "未安装 onnxruntime，跳过"; fi
