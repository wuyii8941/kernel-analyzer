"""Optional bounded vector retention; never changes the optimizer calculation."""
import torch


class SmallUpdateRecorder:
    def __init__(self, writer, max_elements=4096):
        self.writer = writer
        self.max_elements = max_elements
        self.calls = []

    def __call__(self, base, gradient, **kwargs):
        if base.numel() > self.max_elements:
            raise ValueError('Parameter too large for declared vector diagnostics')
        def vector(value):
            return None if value is None else value.detach().float().cpu().reshape(-1).tolist()
        # Save inputs before calling, and return the original writer result.
        record = {'base': vector(base), 'base_dtype': str(base.dtype), 'shape': list(base.shape),
                  'gradient': vector(gradient), 'first': vector(kwargs.get('first')),
                  'second': vector(kwargs.get('second')),
                  'optimizer_arguments': {k: v for k, v in kwargs.items() if k not in ('first', 'second')}}
        result = self.writer(base, gradient, **kwargs)
        record['actual_write'] = vector(result)
        self.calls.append(record)
        return result

    def finish(self, raw):
        ids = raw['state_ids']
        if len(self.calls) != 2 * len(ids):
            raise ValueError('Writer calls do not match the frozen single-case sequence')
        rows = []
        statistics = raw['original_coordinate_statistics']['PARAMETER_WRITE']
        for i, state in enumerate(ids):
            candidate, reference = self.calls[2*i:2*i+2]
            for field in ('base', 'base_dtype', 'shape', 'first', 'second', 'optimizer_arguments'):
                if candidate[field] != reference[field]:
                    raise ValueError('Unmatched optimizer inputs in diagnostic pair: ' + field)
            c = torch.tensor(candidate['actual_write'], dtype=torch.float32)
            r = torch.tensor(reference['actual_write'], dtype=torch.float32)
            effect = c-r
            observed = statistics[i]
            energy = float(effect.double().square().sum())
            if abs(energy-observed['effect_energy']) > 1e-10*max(energy, observed['effect_energy'], 1e-300):
                raise ValueError('Recorded vectors disagree with primary energy')
            coordinates = torch.nonzero(effect).reshape(-1).tolist()
            if len(coordinates) != observed['nonzero_effect_coordinates']:
                raise ValueError('Recorded vectors disagree with nonzero-coordinate count')
            rows.append({'state_id': state, 'candidate': candidate, 'reference': reference,
                         'nonzero_update_coordinates': coordinates})
        return {'schema': 'small-update-vector-diagnostics-v1', 'case_id': raw['case_id'],
                'parameter_scope': raw['carrier'], 'rows': rows,
                'role': 'Post-discovery mechanism replay, not new confirmation or loss evidence',
                'primary_analysis_changed': False}
