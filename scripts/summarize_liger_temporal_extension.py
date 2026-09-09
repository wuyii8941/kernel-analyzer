#!/usr/bin/env python3
"""Verify the full frozen extension and report all windows and loss outcomes."""
import json
import math
from scripts.continue_liger_language_temporal import OUT, WINDOWS, N
from scripts.run_liger_single_boundary_collapse import file_sha256
from scripts.run_training_numerical_v2 import save_new


def main():
    plan=json.loads((OUT/'plan.json').read_text());rows=[]
    for pair in range(N):
        results={};inputs=[];checkpoint_rows=[]
        for condition in ('candidate','reference'):
            root=OUT/f'pair{pair}'/condition
            if not (root/'summary.json').exists():raise SystemExit(f'Incomplete pair {pair}/{condition}; no partial-set conclusion')
            d=json.loads((root/'summary.json').read_text())
            if d['status']!='COMPLETE_TEMPORAL_EXTENSION' or d['plan_sha256']!=file_sha256(OUT/'plan.json'):raise RuntimeError('Protocol mismatch')
            steps=[json.loads(x) for x in (root/'steps.jsonl').read_text().splitlines()]
            direct=[json.loads(x) for x in (root/'direct.jsonl').read_text().splitlines()]
            if [r['step'] for r in steps]!=list(range(1025,4097)) or not all(math.isfinite(r['loss']) for r in steps):raise RuntimeError('Training incomplete or nonfinite')
            if [r['step'] for r in direct]!=[i for a,b in WINDOWS for i in range(a,b+1)]:raise RuntimeError('Direct windows incomplete')
            if [r['step'] for r in d['evaluations']]!=plan['evaluation_steps'] or not all(math.isfinite(r['shared_evaluation_loss']) for r in d['evaluations']):raise RuntimeError('Evaluation incomplete or nonfinite')
            if file_sha256(root/'checkpoint_4096.pt')!=d['checkpoint_sha256']:raise RuntimeError('Checkpoint digest mismatch')
            inputs.append([r['offsets'] for r in steps]);g=d['window_mean_gram']
            if len(g)!=4 or any(len(r)!=4 for r in g):raise RuntimeError('Window Gram dimensions differ')
            for index,(a,b) in enumerate(WINDOWS):
                w=d['windows'][index];part=[r for r in direct if a<=r['step']<=b]
                for field,key in [('effect_energy_sum','effect_energy'),('repair_energy_sum','repair_energy')]:
                    if not math.isclose(math.fsum(r[key] for r in part),w[field],rel_tol=1e-12):raise RuntimeError('Window does not reconstruct')
                if not math.isclose(g[index][index],w['effect_mean_energy'],rel_tol=1e-10,abs_tol=1e-30):raise RuntimeError('Mean Gram does not reconstruct')
            cosines=[g[0][i]/math.sqrt(g[0][0]*g[i][i]) if g[0][0]>0 and g[i][i]>0 else None for i in range(4)]
            results[condition]={'windows':d['windows'],'mean_cosine_to_first_window':cosines,
                'all_later_windows_same_halfspace':all(g[0][i]>0 for i in range(1,4)),
                'each_window_split_half_same_halfspace':all(w['split_half_mean_inner_product']>0 for w in d['windows']),
                'evaluations':d['evaluations'],'source_sha256':file_sha256(root/'summary.json')}
        if inputs[0]!=inputs[1]:raise RuntimeError('Paired inputs differ')
        gaps=[{'step':c['step'],'candidate_minus_reference_loss':c['shared_evaluation_loss']-r['shared_evaluation_loss']}
              for c,r in zip(results['candidate']['evaluations'],results['reference']['evaluations'])]
        rows.append({'pair':pair,'trajectory_diagnostics':results,'loss_gaps':gaps})
    directions=[r['trajectory_diagnostics'][c] for r in rows for c in ('candidate','reference')]
    final=[r['loss_gaps'][-1]['candidate_minus_reference_loss'] for r in rows]
    summary={'status':'COMPLETE_VERIFIED_TEMPORAL_EXTENSION','pairs':N,'trajectories':2*N,'rows':rows,
        'trajectories_with_later_window_means_in_first_window_halfspace':sum(d['all_later_windows_same_halfspace'] for d in directions),
        'trajectories_with_each_window_split_means_aligned':sum(d['each_window_split_half_same_halfspace'] for d in directions),
        'final_nonzero_loss_gaps':sum(x!=0 for x in final),'final_candidate_loss_higher_pairs':sum(x>0 for x in final),
        'final_loss_gap_mean_descriptive':math.fsum(final)/len(final),'plan_sha256':file_sha256(OUT/'plan.json'),
        'claim_boundary':'All predeclared windows and pairs retained. Empirical directional continuation across sampled evolving-state windows, not an iid-step confidence guarantee, convergence, or evidence every unmeasured step has the same sign. Extension after earlier outcomes is not a new independent prospective test.'}
    save_new(OUT/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
