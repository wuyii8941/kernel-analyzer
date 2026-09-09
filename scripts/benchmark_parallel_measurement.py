#!/usr/bin/env python3
"""Benchmark bitwise-preserving CPU measurement, not model training speed."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scripts import run_training_bias_profile_v2_empirical as empirical
from kernel_analyzer.parallel_measurement import parallel_views


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Use a new output under /data1/tzh')
    rng=np.random.default_rng(20260906); records=[]
    for size in (1_000_003,16_000_003):
        value=rng.normal(size=size).astype(np.float32)
        expected,_=empirical._compact_views(value)
        for repeat in range(4):
            for mode in (('serial','parallel') if repeat%2==0 else ('parallel','serial')):
                start=time.perf_counter()
                if mode=='serial':
                    actual,_=empirical._compact_views(value)
                else:
                    actual,_=parallel_views(value,seeds=empirical.SKETCH_SEEDS,dimension=empirical.SKETCH_DIMENSION,
                                            cache=empirical._PACKED_SKETCH_CACHE,workers=3)
                elapsed=time.perf_counter()-start
                exact=all(np.array_equal(expected[k],actual[k]) for k in expected)
                records.append({'coordinates':size,'repeat':repeat,'mode':mode,'seconds':elapsed,'bitwise_match':exact})
    result={'schema':'parallel-measurement-benchmark-v1','status':'PASS' if all(r['bitwise_match'] for r in records) else 'FAIL',
            'scope':'CPU sketch measurement only, shared warm mapping cache; not training throughput',
            'records':records,'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                (Path(__file__),Path('src/kernel_analyzer/parallel_measurement.py'),Path('scripts/run_training_bias_profile_v2_empirical.py'))}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
