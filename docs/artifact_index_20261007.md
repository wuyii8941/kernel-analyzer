# 可复现包索引（阶段 D，2026-10-07）

环境：`/data1/tzh/envs/ka_main`（torch 2.10.0+cu128、Triton 3.6.0、gmpy2）；需要 torchao / bitsandbytes 的部分在
`/data1/tzh/envs/liger`；FPCore 求值在 `/data1/tzh/envs/fpcore`（titanfp 0.1.2）。缓存与临时文件放在仓库内 `.cache/`（见 `CLAUDE.md`
的环境变量）。版本标签：`eval-stage-a-20261006`（阶段 A 冻结版本）。

| 论文中的表 / 图 | 协议（运行前提交） | 复现命令 | 结果文件 |
|---|---|---|---|
| RQ1 盲测复算 | 盲测协议（`results/reference_eval/blind_test_records/`） | `python scripts/blind_records_recount.py` | `results/reference_eval/blind_test_records/recount.json` |
| RQ1/RQ2/RQ3 外部受控语料 | `docs/external_eval_protocol_20261006.md` | `python scripts/external_corpus.py list`；`tier1` / `tier2`（分片见脚本说明）；`blackbox --tier 1/2`；`ablation-a1 --tier 1/2`；`python scripts/external_corpus_summary.py` | `results/external/gpuemu/{summary.json, conditions_tier1.jsonl, tier2/, tier2_total/, baselines_*.jsonl, declared_check*.jsonl}` |
| RQ1 FPCore 交叉核验 | 同上第 7 节 | `python scripts/fpcore_crosscheck.py export`；`/data1/tzh/envs/fpcore/bin/python -I scripts/fpcore_crosscheck.py evaluate`；`... summary` | `results/fpcore/{summary.json, evaluated.jsonl, samples.jsonl}` |
| RQ1 严格包围验证 | 阶段 A 第 4 项 | `python scripts/strict_enclosure_check.py`（需要 `results/fpcore/fragments.jsonl`，由 export 生成） | `results/fpcore/strict_enclosure.json` |
| RQ3 核内定位（外部语料控制组） | 外部评价结果文档 | 见 `docs/external_eval_results_20261006.md`（`emulate.localize`，种子 32–47） | `results/external/gpuemu/localization_controls.json` |
| 同等预算比较 | `scripts/equal_budget.py` 的说明 | `python scripts/equal_budget.py` | `results/external/gpuemu/equal_budget.json` |
| Unsloth 子集 | `docs/unfamiliar_subset_protocol_20261006.md` | `python scripts/build_unsloth_shim.py`；`PYTHONPATH=.cache/pylibs/unsloth_shim:src:. python -m kernel_analyzer.cli check scripts/unsloth_subset.py --out results/external/unsloth --seeds 96`；`python scripts/unsloth_rounding_control.py` | `results/external/unsloth/`、`unsloth_holm.json`、`unsloth_rounding_control.json` |
| 统计校准、敏感性曲线 | `docs/statistics_calibration_20261006.md` | `python scripts/calibrate_equivalence.py`；`python scripts/sensitivity_curves.py [--reference mixed]` | `results/reference_eval/{calibration_equivalence.json, sensitivity_curves*.json, equivalence_zero_variance_recheck.json}` |
| RQ4 逐层表 | `docs/layer_table_protocol_20261006.md` | `python scripts/layer_table_m4.py`；`python scripts/layer_table_cuda_replay.py` | `results/reference_eval/layer_table/{table.json, engine.json, cuda_replay.json, localization.json}` |
| 从作用到重要性（标定、预测、δ） | `docs/importance_calibration_protocol_20261006.md` | `scripts/importance/run.py work --phase b1/b3`；`run.py single-step`；`scripts/importance/analyze.py curves / predict / summary / figure` | `results/importance/{runs/, runs_sealed/, single_step/, curves.json, predictions.json, summary.json, calibration.png}` |
| 第二规模的 ρ | 同上第 10 节 | `scripts/importance/scale_check.py train --ref R/Rp`；`measure`；`compare` | `results/importance/scale_check/` |
| 定向搜索与深案例 | `docs/directed_search_protocol_20261007.md` | `scripts/importance/shadow.py --group ka/liger/muon/c4deep --seed 0/1`（liger 组用 liger 环境，再在 ka_main 中 `--effects`）；`--group c10`；`scripts/importance/ema_deep.py --seed 0..3`；`scripts/importance/directed_summary.py` | `results/directed_search/` |
| 非精度定向搜索 | `docs/nonprecision_search_protocol_20261007.md` | `scripts/importance/nonprecision.py shadow --group D/F/R --seed 0/1`；`train --name D/N1/F/N3/N4/N5/R --seed …`；`shadow --group F_attr --seed 0`；`weights`；`n4-attribution`；`summary` | `results/nonprecision/` |
| B019 | `bugs/B019_averagedmodel_bf16_ema_stagnation.md` | `python bugs/repro/B019_repro_averagedmodel_bf16_ema.py` | `results/directed_search/B019_repro_output.txt` |
| 诊断对照、普查、成本 | — | 文档汇总已有结果 | `docs/diagnosis_comparison_20261006.md`、`docs/census_unified_20261006.md`、`docs/cost_equal_budget_20261006.md` |

逐位可重复：重要性标定与定向搜索的训练在 `torch.use_deterministic_algorithms(True)` 下逐位可重复（已用双次运行与影子测量的两遍
核对）；外部语料与 Unsloth 的 kernel 输出是确定的（没有 float atomic）。原始 tier-1 报告（27 MB）与 FPCore 片段（75 MB）只在本地，
仓库中保留按条件的汇总与样例。
