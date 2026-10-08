#!/usr/bin/env python3
"""Row 2 (specs/phase2/readjudication_procedure.md): every hit of the nine patterns in the conclusion texts of docs/ and
results/ gets one review row: original, which judgment it stood in for, rewrite, basis.  History files are not edited; the
rewrites go into the new report versions.  Completion check: every hit has a row and no rewrite contains a pattern.

    python scripts/closure/readjudication.py      # -> docs/readjudication_20261008.md, results/closure/readjudication.json
"""
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = re.compile(r"误差界内|在 .* 误差界|条件数|eager 也一样|与 eager 一致.*因此|正确舍入本身|不是实现问题|可以忽略|很小")
PI_S3 = ("精度不变性（S3 eager CPU float32 / float64 对，`results/closure/f_eval/s3_index.json`）：526,500 个元素中数值 294,268、一致 211,323、"
         "超过 float64 噪声的语义元素 0、未判定 2,736、异常 17,355（17,354 个在 float64 噪声内，1 个复核：s3idx062 seed 1 第 687 元素，"
         "d32 = 0、d64/|f| = 1.6e-12）")
PI_NORM = ("精度不变性（eager CPU 与 CUDA 的 float32 / float64 对，`results/closure/f_eval/normalization.json`）：10,800 个元素中数值 9,459、"
           "一致 1,329、超过 float64 噪声的语义元素 0、异常 7 个（全部在 float64 噪声内）；huge_offset 与 constant_rows 的条件全部为「数值」")
F_NORM = ("F（spec_normalization v0.1，`results/closure/f_eval/normalization.json`）：float32 超出 τ₃₂ 的元素 CPU 247、CUDA 250、Inductor 266，"
          "全部在三个 huge_offset 条件")
MEAN_NA = "平均作用：每个条件 3 个 seed，n < n_min = 16，记「无法判断（样本）」"

ROWS = {
    ("docs/census_20261005.md", 5): ("误差多大 / 是否偏离（用一 ulp 扰动的敏感度作判读底）",
        "判读的参照量为「每个参数挪一个 ulp 时输出的变化」（大小的描述）；是否存在语义偏离不再用它判定，按精度不变性或 K_R 给出", "历史普查，未重判；新报告不引用该判读"),
    ("docs/census_20261005.md", 29): ("是否偏离（敏感度底约为 1 时判不了）",
        "deepseek_v4 一 ulp 扰动下的输出变化约为 1（离散选择），该普查对它不给结论（未建立）", "历史普查，未重判"),
    ("docs/essential_bugs_phase2a_record_20261007.md", 48): ("语义归类与平均作用（用「两个顺序各自在误差界内」代替）",
        "顺序不变性性质的超差（8–11 / 342 个运行）：base 与 reverse 两个运行的语义归类见依据栏（无超过 float64 噪声的语义元素）；" + MEAN_NA, PI_S3),
    ("docs/essential_bugs_phase2b_record_20261007.md", 153): ("语义归类（用「数值（条件数）」代替）",
        "归一化 float32 的超出 τ₃₂：语义归类为数值（精度不变性，见依据）；超出的元素只在 huge_offset 条件（大小的描述）；" + MEAN_NA, PI_NORM + "；" + F_NORM),
    ("docs/essential_bugs_phase2b_record_20261007.md", 166): ("语义归类（「由条件数解释」）",
        "结论：没有语义偏离的元素（精度不变性）；超出 τ₃₂ 的大小见 F；平均作用无法判断（样本）", PI_NORM + "；" + F_NORM),
    ("docs/essential_bugs_phase2b_record_20261007.md", 263): ("是否违反（「来自正确舍入，不是实现的缺陷」代替了契约判断）",
        "gelu_tanh_bf16：非零平均作用在两组独立的 96 个 seed 上都检出（R1/R2/R3）；是否违反：契约 v2 与数值契约 v3 没有声明 bf16 输出允许的数值行为 → 只记录，不裁决；保留为研究对象，进入数值作用流",
        "G7 结果与复现（`results/essential/phase2b/g7_numerical_stream/`）；契约 v2「激活」行无数值行为条款"),
    ("docs/essential_bugs_phase2b_record_20261007.md", 303): ("语义归类与是否违反（「条件数造成」「来自正确舍入」）",
        "数值一栏改为：归一化 huge_offset / constant_rows 与 S3 index 前向的超出 τ₃₂ 经精度不变性归为数值；bf16 偏离只记录；swiglu / geglu 的 inf 输出（真实值超出 float32 表示范围）记第四栏；gelu_tanh_bf16 的平均作用为真实数值作用，契约未声明 → 只记录",
        PI_NORM + "；" + PI_S3),
    ("docs/essential_bugs_phase2b_record_20261007.md", 309): ("归因（「FR 把条件数与 bf16 的偏离归到 e_num」）",
        "归因：这些偏离在 FR 中落在 e_num；K_R 的区间包含同一输入上 float64 eager 的结果（交叉核验，不是包含精确值的证明）",
        "`results/essential/phase2b/normalization/fr_modeA.json`"),
    ("docs/external_eval_results_20261006.md", 49): ("误差多大（描述）",
        "相对 RMS 1.1·10⁻⁴：填充使每个原本量级在 10⁻⁴ 以下的概率略微变小", "大小的描述，保留数值"),
    ("docs/external_eval_results_20261006.md", 75): ("误差多大 / 是否漏报（「平均作用很小，不称漏报」）",
        "这些平均作用的大小为相对 10⁻⁹–10⁻⁷（比容差小 3–5 个数量级）；按评价计划只算「两类判定提供的不同信息」，不作为基线漏报计数", "大小的描述；计分规则不变"),
    ("docs/liger_single_boundary_collapse_experiment.md", 38): ("重要性（「很小的 loss 差异」）",
        "两边正常完成，loss 差异的数值见该文表格；不宣称稳定训练质量损害（重要性未标定）", "历史实验，未重判"),
    ("docs/method.md", 268): ("重要性（描述）", "数值量级在种子波动以下的分叉应同时……（按种子波动标定，不用「小」作判断）", "方法文档措辞"),
    ("docs/nonprecision_search_results_20261007.md", 56): ("平均作用（「不是可以忽略的量」）",
        "γ̂ 的区间跨过 0.05，相对 δ 无法判断等价，所以不能把它排除", "历史结果，措辞改写"),
    ("docs/nonprecision_search_results_20261007.md", 87): ("误差多大（「γ̂ 很小」）",
        "成立（γ̂ = −2.6e-5）；主要作用是固定方向（b̂ = 5.2e-3），γ̂ 比 b̂ 小两个数量级", "大小的描述"),
    ("docs/plan_20261006.md", 33): ("误差多大（引述旧 README）", "旧 README 写「工具找到的真实现象量级都在容差以下」（引述，待更新）", "计划文档引述"),
    ("docs/plan_20261006.md", 173): ("覆盖（「OpInfo 样例都很小」）", "OpInfo 样例的尺寸都在这些阈值以下（最长轴 ≤ 9）", "覆盖的描述"),
    ("docs/plan_20261006.md", 195): ("平均作用（「未检出即 95% 保证作用很小」）",
        "「未检出即有 95% 的作用上界」只对给出有效上界的投影成立", "统计结论的写法"),
    ("docs/protocol_closure_v3_20261008.md", 25): ("（协议本身列出禁用表述）", "第 1 节所列三种表述只能用于描述误差大小", "元陈述"),
    ("docs/protocol_essential_bugs_20261007.md", 64): ("非结论：表头指条件的个数", "条件个数", "非结论"),
    ("docs/protocol_essential_bugs_phase2_20261007.md", 147): ("非结论：指条件的个数", "完整条件个数", "非结论"),
    ("docs/protocol_essential_bugs_phase2_20261007.md", 148): ("非结论：指条件的个数", "试跑条件个数", "非结论"),
    ("docs/protocol_essential_bugs_phase2_20261007.md", 208): ("语义归类（事后规则「数值（条件数）」）",
        "事后规则作废，由数值契约 v3 的精度不变性取代；平移输入的辅助运行只作大小的描述", "数值契约 v3 第 2 节；" + PI_NORM),
    ("docs/protocol_essential_bugs_phase2_20261007.md", 213): ("误差多大（描述）",
        "输入使舍入误差放大或输出为 bf16 时它会超出（大小的描述）", "大小的描述"),
    ("docs/status_ledger_20261006.md", 54): ("语义归类（「全部在 float32 求和误差界内」）",
        "S3：index 前向超出 τ₃₂ 的元素经精度不变性归为数值（无超过 float64 噪声的语义元素）；" + MEAN_NA, PI_S3),
    ("docs/status_ledger_20261006.md", 57): ("语义归类（「全部由条件数解释」）",
        "归一化 23 个条件：float32 的超出 τ₃₂ 经精度不变性归为数值；FR 中落在 e_num", PI_NORM + "；" + F_NORM),
    ("docs/status_ledger_20261006.md", 58): ("是否违反（「来自 bf16 正确舍入，eager 相同」）",
        "G7：一个检出在新 seed 上不复现；另一个（gelu_tanh_bf16）复现，为真实数值作用，契约未声明 bf16 输出允许的数值行为 → 只记录，保留为研究对象",
        "G7 复现记录；契约 v2"),
    ("docs/talk_beyond_tolerance.md", 481): ("误差多大（演讲稿措辞）", "发现传统方法漏掉的：误差在容差以下，但正负分布不对称", "措辞"),
    ("docs/tool_changes_20261003.md", 43): ("统计（描述 p 值）", "p 值接近 0，Holm 因而把这类检验算作拒绝……", "历史工具变更记录"),
    ("docs/tool_changes_20261005.md", 109): ("是否偏离（敏感度底约为 1 时判不了）", "这类含离散选择的模型，一 ulp 扰动下输出变化约为 1，普查对它不给结论", "历史"),
    ("docs/tool_validation.md", 63): ("统计（描述）", "μ 接近 0 时）", "措辞"),
    ("docs/tool_validation.md", 440): ("误差多大（「正确舍入的 ±0.5 ulp 与 rsqrt.approx 同量级」）",
        "rsqrt 是每行一个标量，128 行里就近舍入引入的 ±0.5 ulp 与 rsqrt.approx 的误差同量级……（大小的比较）", "大小的描述"),
    ("results/reference_eval/blind_test_v1/phase2/phase2_report.md", 125): ("误差多大（描述）",
        "所以 K_R 是对的：这是程序声明语义与 f 之间真实的差异，相对量级见上表", "大小的描述（盲测计分不追溯）"),
    ("results/reference_eval/blind_test_records/blind_records/v2/blind_test_v2_final_scoring.md", 53): ("误差多大（描述）",
        "实际上有一个 R3 正分量 lr²(1−mu)·mu·‖g‖²/‖K_R‖ ≈ 5.3e-5（每轮标准误约 1.5e-5，z ≈ 3.5）", "大小的描述（盲测计分不追溯）"),
}
EXTRA = [  # guarantee table section 6 items that are not pattern hits
    ("「15 个家族全部跑完 / 覆盖完成」（2b 小结）", "覆盖声明", "冻结清单内的计划项状态表（已执行且可裁决 / 仅部分方法 / 不适用 / 未建立 / 环境不可用 / 超预算 / 未运行），附未覆盖原因",
     "`docs/essential_bugs_phase2b_record_v2_20261008.md` 的计划项状态表"),
    ("2b 各家族的「无定义」判定", "合法性", "对照契约 v2 的五类重分；契约未覆盖的 16 个条件记「契约外，待审阅」", "`results/closure/contract_classification.json`"),
    ("「归一化平移输入裁决」「模式 A 下 K_R 与 float64 的比较」等事后规则", "事后规则", "标为事后；前者由精度不变性取代；后者保留为交叉核验，在新输入上确认一次（三类组合验证）",
     "数值契约 v3；`docs/composition_checklists_20261008.md`"),
]


def hits():
    files = sorted(glob.glob(str(ROOT / "docs/**/*.md"), recursive=True))
    files += [str(ROOT / "results/reference_eval/blind_test_v1/phase2/phase2_report.md"),
              str(ROOT / "results/reference_eval/blind_test_records/blind_records/v2/blind_test_v2_final_scoring.md")]
    for f in glob.glob(str(ROOT / "results/**/*.json"), recursive=True):
        try:
            if P.search(open(f, encoding="utf-8", errors="ignore").read()):
                files.append(f)
        except OSError:
            pass
    out = []
    skip = {"docs/readjudication_20261008.md"}
    for f in files:
        rel = str(Path(f).relative_to(ROOT))
        if rel in skip:
            continue
        for i, line in enumerate(open(f, encoding="utf-8", errors="ignore"), 1):
            if P.search(line):
                m = P.search(line)
                a, b = max(0, m.start() - 60), min(len(line), m.end() + 60)
                out.append((rel, i, [x.group(0) for x in P.finditer(line)], line[a:b].strip().replace("|", "／")))
    return out


def main():
    rows, missing, bad = [], [], []
    for rel, ln, pats, ctx in hits():
        r = ROWS.get((rel, ln))
        if r is None:
            missing.append((rel, ln, ctx))
            continue
        if P.search(r[1]):
            bad.append((rel, ln))
        rows.append({"file": rel, "line": ln, "patterns": pats, "original": ctx, "replaced_judgment": r[0], "rewrite": r[1], "basis": r[2]})
    L = ["# 第 2 行：2b 及此前结论的复审表（2026-10-08）", "",
         "按 `specs/phase2/readjudication_procedure.md`：在 docs/ 与 results/ 的结论性文字中检索九个模式（`误差界内`、`在 .* 误差界`、`条件数`、"
         "`eager 也一样`、`与 eager 一致.*因此`、`正确舍入本身`、`不是实现问题`、`可以忽略`、`很小`），每一处一行。历史文件不改；改写进入新版本报告。"
         f"检索命中 {len(rows) + len(missing)} 处，复审 {len(rows)} 行，缺行 {len(missing)}，改写后仍含模式 {len(bad)}。"
         "（「条件数」在第一阶段与 2b 协议中指「条件的个数」的命中，标为非结论。）", "",
         "| # | 位置 | 原文 | 替代了哪类判断 | 改写后 | 依据 |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        L.append(f"| {i} | `{r['file']}:{r['line']}` | {r['original']} | {r['replaced_judgment']} | {r['rewrite']} | {r['basis']} |")
    L += ["", "## 保证与裁决表第 6 节列出的其余条目", "", "| 条目 | 类型 | 改写为 | 依据 |", "|---|---|---|---|"]
    for e in EXTRA:
        L.append("| " + " | ".join(e) + " |")
    (ROOT / "docs/readjudication_20261008.md").write_text("\n".join(L) + "\n")
    (ROOT / "results/closure/readjudication.json").write_text(json.dumps({"rows": rows, "missing": missing, "rewrite_contains_pattern": bad,
                                                                         "extra": EXTRA}, indent=1, ensure_ascii=False) + "\n")
    print("rows", len(rows), "missing", len(missing), "bad", len(bad))
    for m in missing:
        print("MISSING", m)


if __name__ == "__main__":
    main()
