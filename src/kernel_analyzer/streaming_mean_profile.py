"""Full-coordinate fixed-suite means without retaining all state vectors.

These are descriptive finite-suite quantities. They are neither a population
confidence bound nor proof that the error persists in a training trajectory.
"""
import math
import torch


class StreamingMeanProfile:
    def __init__(self, calibration_count, confirmation_count, repair_energy_floor=1e-30):
        if calibration_count<1 or confirmation_count<1:
            raise ValueError('Both partitions must be nonempty')
        if not math.isfinite(repair_energy_floor) or repair_energy_floor<0:
            raise ValueError('Invalid repair energy floor')
        self.calibration_count=calibration_count
        self.confirmation_count=confirmation_count
        self.floor=repair_energy_floor
        self.sums={}
        self.rows=[]
        self.perpendicular_valid=True

    def append(self, effect, repair):
        if len(self.rows)>=self.calibration_count+self.confirmation_count:
            raise ValueError('Too many states')
        u=torch.as_tensor(effect).detach().reshape(-1).to(device='cpu',dtype=torch.float64)
        r=torch.as_tensor(repair).detach().reshape(-1).to(device='cpu',dtype=torch.float64)
        if not u.numel() or u.shape!=r.shape or not bool(torch.isfinite(u).all() & torch.isfinite(r).all()):
            raise ValueError('Invalid full-coordinate vectors')
        if self.sums and u.shape!=self.sums['calibration_effect'].shape:
            raise ValueError('Coordinate count changed')
        if not self.sums:
            self.sums={k:torch.zeros_like(u) for k in ('calibration_effect','calibration_residual',
                                                      'confirmation_effect','confirmation_residual')}
        x=float(torch.dot(u,u)); b=float(torch.dot(r,r)); a=float(torch.dot(u,r))
        if not all(math.isfinite(v) for v in (x,b,a)):
            raise ValueError('Nonfinite vector statistics')
        q=u-r*(a/b) if b>self.floor else None
        self.perpendicular_valid &= q is not None
        confirmation=len(self.rows)>=self.calibration_count
        part='confirmation' if confirmation else 'calibration'
        row=dict(effect_energy=x,repair_energy=b,effect_repair_inner_product=a)
        if confirmation:
            for name,value in (('effect',u),('residual',q)):
                direction=self.sums['calibration_'+name]
                norm=float(torch.linalg.vector_norm(direction))
                row[name+'_calibration_projection']=(float(torch.dot(direction,value))/norm
                    if norm>0 and value is not None and (name=='effect' or self.perpendicular_valid) else None)
        self.sums[part+'_effect'].add_(u)
        if q is not None: self.sums[part+'_residual'].add_(q)
        self.rows.append(row)

    def finish(self):
        if len(self.rows)!=self.calibration_count+self.confirmation_count:
            raise ValueError('State count is incomplete')
        rows=self.rows[self.calibration_count:]
        b=math.fsum(r['repair_energy'] for r in rows)/self.confirmation_count
        result=dict(schema='full-coordinate-streaming-mean-v1',
            coordinate_count=self.sums['calibration_effect'].numel(),
            accumulation_dtype='float64', exact_real_arithmetic=False,
            calibration_count=self.calibration_count,confirmation_count=self.confirmation_count,
            repair_energy_floor=self.floor, rows=self.rows,
            population_guarantee=False,decision_role='DESCRIPTIVE_NOT_USED_FOR_EQUIVALENCE')
        if b<=0:
            result['status']='ZERO_REPAIR_ENERGY'; return result
        scale=math.sqrt(b)
        result.update(status='VALID',
            confirmation_mean_relative_magnitude=float(torch.linalg.vector_norm(self.sums['confirmation_effect']))/self.confirmation_count/scale,
            confirmation_residual_mean_relative_magnitude=(float(torch.linalg.vector_norm(self.sums['confirmation_residual']))/self.confirmation_count/scale
                if self.perpendicular_valid else None),
            residual_status='VALID' if self.perpendicular_valid else 'REPAIR_ENERGY_FLOOR_NOT_MET_NO_STATES_DROPPED',
            normalized_heldout_effect_projections=[None if r['effect_calibration_projection'] is None else r['effect_calibration_projection']/scale for r in rows],
            normalized_heldout_residual_projections=[None if not self.perpendicular_valid or r['residual_calibration_projection'] is None else r['residual_calibration_projection']/scale for r in rows])
        return result
