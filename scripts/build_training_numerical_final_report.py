#!/usr/bin/env python3
"""Build the current report from verified records, not manually copied scores."""
import json
import argparse
from pathlib import Path
from scripts.run_training_numerical_v2 import BASE
from scripts.summarize_training_numerical_v2 import build_progress


NAMES = {
    'phi4_seq64_lmhead_dx': ('Phi', 'lm-head backward dX'),
    'liger_fused_ce_t128': ('Qwen / Liger', 'fused CE dW 累加'),
    'deepseek8b_seq256_backward_1714_in_out_ptr0': ('DeepSeek', 'normalization backward'),
    'deepseek8b_seq128_backward_1256_out_ptr0': ('DeepSeek', 'attention projection backward'),
    'llama32_text128_scan_0000': ('Llama', 'softmax backward'),
    'gemma4_text128_scan_0037': ('Gemma', '原内部行平方和输出'),
    'granite_fp32_expert_order_layer0': ('Granite', 'FP32 MoE expert 输出累加顺序'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=BASE / 'final_report.md')
    args = parser.parse_args()
    audit = build_progress()
    if not audit['selected_experiment_execution_complete']:
        raise SystemExit('Incomplete: ' + ', '.join(audit['remaining_acceptance']))
    verification = json.loads((BASE / 'report_recomputation_verification.json').read_text())
    temporal = json.loads((BASE / 'language_temporal_extension/summary.json').read_text())
    training = json.loads((BASE / 'language_training_confirmation_iid/summary.json').read_text())
    identity = json.loads((BASE / 'language_accumulation_identity_retry/result.json').read_text())
    lines = ['# 选定实验验收：原始记录自动汇总', '',
             '此页不代表整份 2026-09-05 实施计划完成；系统覆盖、统一对照和实际收益另行验收。', '',
             '本页由 `scripts/build_training_numerical_final_report.py` 生成。分析、成因和 loss 分别验收；有效测量不自动代表 bias 正例。', '',
             '## 实际参数写入复采', '',
             '| 模型 | 训练位置 | 确认集合更新差异 RMS / 正常更新 RMS | 声明的 1% 范围 |',
             '|---|---|---:|---|']
    for record in verification['reports']:
        result = json.loads((BASE / record['report']).read_text())
        model, position = NAMES.get(result['case_id'], (result['case_id'], '见原记录'))
        label = {'EQUIVALENT': '范围内', 'NON_EQUIVALENT': '超范围', 'INCONCLUSIVE': '边界未决'}[result['equivalence_decision']]
        lines.append(f"| {model} | [{position}]({record['report']}) | {100 * result['bias_analysis']['fixed_suite_total_rms']:.6f}% | {label} |")
    lines += ['', '各行属于各自固定状态集合和参数范围，不是一个随机总体，也不是 bias 严重程度排行榜。Granite 是新家族的小差异对照；其他六项是原案例复采。', '',
              '## 同一语言训练的成因、直接方向和 loss', '',
              f"- 真实 {identity['chunks']} 个分块乘积相同，实际 dW 差异可由累加舍入及最后转换精确重构。该恒等式解释来源，不是假设任意状态下舍入均值必定非零。",
              f"- 1024 步：{len(training['rows'])} 组新初始化中 {training['positive_pairs']} 组原实现验证 loss 更高，预声明单侧符号检验 p={training['p_value']:.8f}。精确数值和生成范围见[确认记录](language_training_confirmation_iid/summary.json)。",
              f"- 全部 {temporal['pairs']} 组、{temporal['trajectories']} 条原轨迹继续到 4096 步，保留真实参数和 AdamW 历史。",
              f"- {temporal['trajectories_with_later_window_means_in_first_window_halfspace']}/{temporal['trajectories']} 条轨迹的后期窗口均值与第一个窗口同向；{temporal['trajectories_with_each_window_split_means_aligned']}/{temporal['trajectories']} 条轨迹在每个窗口内前后半均值同向。",
              f"- 最终 {temporal['final_nonzero_loss_gaps']}/{temporal['pairs']} 组 loss 不同；其中 {temporal['final_candidate_loss_higher_pairs']} 组原实现较高，原实现减参考的平均差为 {temporal['final_loss_gap_mean_descriptive']:+.8f}。",
              f"- 两条轨迹同时满足上述全部窗口方向检查且最终 loss 不同的初始化组：{audit['pairs_with_observed_cross_window_direction_and_final_loss_difference']}。所有组都保留，重复组不是新增实现案例。", '',
              '直接检查的窗口为 1025–1056、2017–2048、3041–3072、4065–4096。该延续是在早期确认后开展的描述性验证，不把相关训练步骤当成独立样本，也不证明未采样步骤都同向。', '',
              '[完整延续结果](language_temporal_extension/summary.json)；[原坐标重算核验](report_recomputation_verification.json)。', '',
              '## 没有升级的主张', '',
              '不证明所有状态都有 bias、任意未来训练等价、持续质量恶化、训练崩溃，或方向分量独自解释全部 loss 差异。总体等价性检验未通过校准，因此仍关闭；固定集合分析不依赖这项未启用保证。', '']
    output = args.output
    with output.open('x') as handle:
        handle.write('\n'.join(lines))
    print(output)


if __name__ == '__main__':
    main()
