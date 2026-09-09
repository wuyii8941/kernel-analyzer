#!/usr/bin/env python3
"""Opt-in performance adapter for the unchanged bound-capture measurement.

Freeze this adapter separately. Do not use it to silently resume old protocols.
"""
import argparse
import sys
import numpy as np
import torch
from kernel_analyzer.parallel_measurement import parallel_views
from scripts import run_training_bias_profile_v2_empirical as empirical
from scripts import capture_bound_endpoint_bias_formation_v21 as capture


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--measurement-workers',type=int,default=3)
    args,remaining=p.parse_known_args()
    if not 1<=args.measurement_workers<=len(empirical.SKETCH_SEEDS):
        raise SystemExit('invalid measurement worker count')
    def compact(value):
        tensor=value if isinstance(value,torch.Tensor) else torch.from_numpy(np.asarray(value))
        host=tensor.detach().float().reshape(-1).cpu().numpy()
        return parallel_views(host,seeds=empirical.SKETCH_SEEDS,dimension=empirical.SKETCH_DIMENSION,
                              cache=empirical._PACKED_SKETCH_CACHE,workers=args.measurement_workers)
    empirical._compact_views=compact
    sys.argv=[sys.argv[0],*remaining]
    capture.main()


if __name__=='__main__': main()
