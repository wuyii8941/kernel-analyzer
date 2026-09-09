"""Same-state magnitude, mean and aligned summaries; no new verdict thresholds."""
import math
import numpy as np


def stage_baselines(payload):
    ids = payload['state_ids']
    confirmation = payload['confirmation_state_ids']
    if len(set(ids)) != len(ids) or not confirmation or not set(confirmation) <= set(ids):
        raise ValueError('Invalid state identities')
    index = [ids.index(s) for s in confirmation]
    rows = []
    for name, statistics in payload.get('original_coordinate_statistics', {}).items():
        if name == 'ADAMW_UPDATE':
            continue  # proposed update is not a replacement for parameter write
        if len(statistics) != len(ids):
            rows.append({'stage':name, 'status':'ORIGINAL_STATISTICS_INCOMPLETE',
                         'recorded_states':len(statistics), 'expected_states':len(ids),
                         'claim_scope':'NOT_ASSESSED'})
            continue
        x = np.array([s['effect_energy'] for s in statistics], dtype=float)
        b = np.array([s['repair_energy'] for s in statistics], dtype=float)
        a = np.array([s['effect_repair_inner_product'] for s in statistics], dtype=float)
        if not all(np.isfinite(v).all() for v in (x,b,a)) or np.any(x < 0) or np.any(b < 0):
            raise ValueError('Invalid energy records')
        row = {'stage': name, 'state_count': len(index), 'coordinate_scope': payload.get('carrier'),
               'local_allclose': 'NOT_RECORDED_CANNOT_RECONSTRUCT_FROM_ENERGY',
               'mean_status': 'FULL_GRAM_UNAVAILABLE', 'claim_scope': 'FIXED_SUITE_DESCRIPTIVE'}
        energy = math.fsum(x[index]); reference = math.fsum(b[index])
        row.update(total_effect_energy=energy, total_reference_energy=reference)
        row.update(nonzero_effect_states=int(np.count_nonzero(x[index])),
                   largest_state_effect_energy_fraction=float(np.max(x[index]) / energy) if energy > 0 else 0.)
        if reference <= 0:
            row['status'] = 'ZERO_REFERENCE_ENERGY'
            rows.append(row)
            continue
        row.update(status='VALID', relative_rms=math.sqrt(energy/reference),
                   aligned_ratio_of_sums=math.fsum(a[index])/reference)
        exact = payload.get('stages', {}).get(name, {}).get('EXACT', {}).get('profile', {})
        gram = exact.get('suite', exact).get('joint_gram')
        if gram:
            gu, gr, gur = [np.asarray(gram[k], dtype=float) for k in ('effect_effect','repair_repair','effect_repair')]
            if any(g.shape != (len(ids),len(ids)) or not np.isfinite(g).all() for g in (gu,gr,gur)):
                raise ValueError('Malformed full Gram')
            if not (np.allclose(gu.diagonal(), x, rtol=1e-6, atol=1e-30)
                    and np.allclose(gr.diagonal(), b, rtol=1e-6, atol=1e-30)
                    and np.allclose(gur.diagonal(), a, rtol=1e-6, atol=1e-30)):
                row['mean_status'] = 'GRAM_ORIGINAL_STATISTICS_DISAGREE'
            else:
                n = len(index)
                row['mean_relative_magnitude'] = math.sqrt(max(0.,float(gu[np.ix_(index,index)].sum()))/(n*reference))
                row['mean_status'] = 'FULL_GRAM_FIXED_SUITE'
                if np.all(b[index] > 0):
                    g = np.ix_(index,index)
                    coefficients = a[index]/b[index]
                    qq = (gu[g] - gur[g]*coefficients[None,:]
                          - gur[g].T*coefficients[:,None]
                          + gr[g]*np.outer(coefficients,coefficients))
                    row['residual_mean_relative_magnitude'] = math.sqrt(max(0.,float(qq.sum()))/(n*reference))
                else:
                    row['residual_mean_status'] = 'ZERO_STATE_REFERENCE_ENERGY'
        rows.append(row)
    return rows
