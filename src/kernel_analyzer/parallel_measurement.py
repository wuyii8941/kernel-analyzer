"""Parallelize independent sketch seeds without changing per-seed arithmetic.

Coordinate mapping, chunk order and float64 summation match the existing
empirical capture.  Version 3 retains float64 sketch storage so finite BF16
mask sentinels cannot overflow when several coordinates share a bucket.
Original-coordinate energy is untouched.
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .short_persistence import count_sketch_mapping


def parallel_views(host, *, seeds, dimension, cache, workers=3):
    host=np.asarray(host,dtype=np.float32).reshape(-1)
    if dimension!=4096 or not 1<=workers<=len(seeds):
        raise ValueError('This adapter preserves the existing 4096-bin protocol')
    coordinates=host.size
    if coordinates<=dimension:
        return {'EXACT':host.copy()}, coordinates
    def one(seed):
        key=(coordinates,seed)
        packed=cache.get(key)
        if packed is None:
            packed=np.empty(coordinates,dtype=np.uint16)
            for start in range(0,coordinates,1_000_000):
                stop=min(coordinates,start+1_000_000)
                buckets,signs=count_sketch_mapping(np.arange(start,stop,dtype=np.uint64),projection_dim=dimension,seed=seed)
                packed[start:stop]=buckets.astype(np.uint16)|((signs<0).astype(np.uint16)<<np.uint16(12))
            cache[key]=packed
        sketch=np.zeros(dimension,dtype=np.float64)
        for start in range(0,coordinates,1_000_000):
            stop=min(coordinates,start+1_000_000); codes=packed[start:stop]
            buckets=(codes & np.uint16(dimension-1)).astype(np.int64)
            signs=np.where((codes & np.uint16(1<<12))==0,1.,-1.)
            values=np.asarray(host[start:stop],dtype=np.float64)
            sketch+=np.bincount(buckets,weights=signs*values,minlength=dimension)
        if not np.isfinite(sketch).all():
            raise ValueError('CountSketch accumulation is nonfinite')
        return f'COUNT_SKETCH_V3_FLOAT64_SEED_{seed}',sketch
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one,seeds)),coordinates
