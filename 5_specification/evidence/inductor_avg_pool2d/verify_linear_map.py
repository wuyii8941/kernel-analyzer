"""Independent coefficient check of the captured pooling program, separate from task evaluation."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np
import torch

SPEC = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SPEC), str(SPEC / 'corpus')]
from calls.inductor_avg_pool2d import call
from kernel_analyzer import check


class BasisCase(check.Case):
    name = 'inductor_avg_pool2d_linear_coefficients'
    implementation = 'PyTorch 2.10 Inductor public forward and VJP'
    def inputs(self, seed):
        x, g = torch.zeros(1, 1, 4, 4, device='cuda'), torch.zeros(1, 1, 2, 2, device='cuda')
        (x if seed < 16 else g).view(-1)[seed if seed < 16 else seed - 16] = 1.
        return {'x': x, 'upstream': g}
    def launch(self, values):
        return call(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    # Independent task coefficients. No adapter, e2_pool or observed GPU output
    # enters this construction: enumerate each of the four real input windows.
    windows = [[r * 4 + c for r in range(top, min(top + 3, 4)) for c in range(left, min(left + 3, 4))]
               for top in (0, 2) for left in (0, 2)]
    task = [[Fraction(1, len(w)) if i in w else Fraction(0) for i in range(16)] for w in windows]
    backward = [[Fraction(1, 9) if i in w else Fraction(0) for i in range(16)] for w in windows]
    keep = {}
    diagnostic = check.run(BasisCase(), dev=list(range(16)), conf=list(range(16, 20)), repeats=1, keep=keep)
    observations = []
    for name, rows in keep.items():
        for row in rows:
            seed = row['seed']
            expected = ([task[o][seed] for o in range(4)] if seed < 16 else [Fraction(0)] * 4) if name == 'y' else (
                [Fraction(0)] * 16 if seed < 16 else backward[seed - 16])
            lo, hi = np.ravel(row['r_lo']), np.ravel(row['r_hi'])
            assert len(lo) == len(expected) and bool(np.asarray(row['ok']).all())
            assert not bool(np.asarray(row['premise']).any())
            assert all(Fraction(float(a)) <= value <= Fraction(float(b)) for a, b, value in zip(lo, hi, expected))
            assert all(float(b - a) < 1e-14 for a, b in zip(lo, hi))  # diagnostic precision, not equality certification
            observations.append({'output': name, 'seed': seed, 'expected': [str(v) for v in expected],
                'program_lower': lo.tolist(), 'program_upper': hi.tolist(), 'device': np.ravel(row['k']).tolist()})
    assert len(observations) == 40
    report = {'source': Path(__file__).read_text(), 'windows': windows,
        'task_forward_matrix': [[str(v) for v in row] for row in task],
        'program_backward_transpose': [[str(v) for v in row] for row in backward],
        'backward_minus_task_transpose': [[str(a - b) for a, b in zip(p, t)] for p, t in zip(backward, task)],
        'observations': observations, 'diagnostic': diagnostic,
        'scope': 'Basis consistency with an independently enumerated linear map; linearity and coordinate-only masks are separately reviewed in the captured TTIR. Not a population test or a general compiler theorem.'}
    with args.out.open('x') as file:
        json.dump(report, file, indent=2, default=str)


if __name__ == '__main__':
    main()
