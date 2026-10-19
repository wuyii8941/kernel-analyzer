"""Exact bounds on the reviewed forward formula; no reference VJP or device arithmetic used."""
import argparse
from fractions import Fraction
import gzip
import json
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    with gzip.open(root / 'development_observation_v1.json.gz', 'rt') as f:
        observation = json.load(f)
    with gzip.open(root / 'measurement_v1.json.gz', 'rt') as f:
        measured = json.load(f)
    launch = observation['launches'][0]
    parameters = {a['name']: a for a in launch['arguments']}
    assert parameters['N']['value'] == 7 and '%eps: f32' in launch['ttir']
    # Manual TTIR review establishes sum(x*x)/N, add eps, sqrt, reciprocal,
    # then x*r*weight. This script verifies the consequence of that reviewed
    # formula, not equivalence of arbitrary future IR to the formula.
    c = Fraction(struct.unpack('<f', struct.pack('<f', parameters['eps']['value']))[0])
    mappings = measured['binding_source']['tasks'][0]['input_map']
    e = Fraction(next(m['literal'] for m in mappings if m['target'] == '/eps'))
    declaration = measured['expanded_declaration']
    assert c > e > 0
    # The declaration fixes each input coordinate to this inclusive interval;
    # its endpoints are exactly representable in float32.
    bounds = declaration['inputs']['x']['sampler']['uniform']
    assert bounds == [-1/512, 1/512]
    max_moment = max(Fraction(x)**2 for x in bounds)
    # (m+e)/(m+c) increases with m, since its derivative is (c-e)/(m+c)^2.
    ratio_squared = (e/c, (max_moment+e)/(max_moment+c))
    ratio_bounds = (Fraction(109,1000), Fraction(534,1000))
    assert ratio_bounds[0]**2 <= ratio_squared[0] <= ratio_squared[1] <= ratio_bounds[1]**2
    report = {'program_epsilon_f32': str(c), 'task_epsilon': str(e),
              'mean_square_bounds': ['0', str(max_moment)],
              'forward_relation': 'G_y = f_y * sqrt((m+task_epsilon)/(m+program_epsilon)), m=sum(x*x)/7',
              'squared_ratio_bounds': [str(v) for v in ratio_squared],
              'certified_ratio_bounds': [str(v) for v in ratio_bounds],
              'relative_shrinkage_bounds': [str(1-ratio_bounds[1]), str(1-ratio_bounds[0])],
              'scope': 'Reviewed ideal forward program on the declared input range. Zero target coordinates remain zero. Does not describe hardware rounding, full gradients, other versions, or training impact.'}
    with args.out.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
