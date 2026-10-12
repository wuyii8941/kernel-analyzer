"""RQ3: the composition rule of an atomic fold (C14) on the locked build.

TTIR leaves the order of atomic updates to one address unspecified.  The candidate rule: the result is the sequential
fold of the updates, each rounded in the element type, in some order.  Checked here by repetition on the device:
every observed result must be one of the fold results over all orders (enumerated: 6 programs -> 720 orders), and
the integer folds must be order free.  The frequency of the observed results and their mean against the exact sum
show whether the order choice carries a systematic bias.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exact as X  # noqa: E402


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def fold(x, o, LANES: tl.constexpr):
        pid = tl.program_id(0)
        i = tl.arange(0, LANES)
        v = tl.load(x + pid * LANES + i)
        tl.atomic_add(o + i * 0, v, mask=i < LANES, sem="relaxed")
    return fold


def fold_results(vals, fmt):
    out = set()
    for perm in itertools.permutations(range(len(vals))):
        acc = F(0)
        for i in perm:
            r = X.round_frac(acc + F(vals[i]), fmt, "rne")
            acc = r if not isinstance(r, float) else acc
        out.add(float(acc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results/atomic_rule.json"))
    ap.add_argument("--repeats", type=int, default=400)
    a = ap.parse_args()
    import torch
    K = kernel()
    rng = np.random.default_rng(77)
    res = []
    for fmt, td in (("f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)):
        for case in range(4):
            P = 6
            mags = 2.0 ** rng.integers(-12, 12, P)
            vals = (rng.standard_normal(P) * mags) if case % 2 == 0 else np.abs(rng.standard_normal(P) * mags)
            x = torch.from_numpy(vals).to(td).cuda()
            vals = x.double().cpu().numpy()
            possible = fold_results(list(vals), fmt)
            seen = Counter()
            o = torch.zeros(1, device="cuda", dtype=td)
            try:
                for _ in range(a.repeats):
                    o.zero_()
                    K[(P,)](x, o, LANES=1)
                    torch.cuda.synchronize()
                    seen[float(o.double().item())] += 1
            except Exception as e:
                res.append({"fmt": fmt, "case": case, "status": f"not compiled / launched: {type(e).__name__}: "
                            f"{str(e).splitlines()[0][:160]}"})
                print(res[-1], flush=True)
                break
            exact = float(sum(F(v) for v in vals))
            mean = sum(k * c for k, c in seen.items()) / a.repeats
            rec = {"fmt": fmt, "case": case, "programs": P, "same_sign": case % 2 == 1,
                   "orders": 720, "distinct_fold_results": len(possible), "distinct_observed": len(seen),
                   "observed_in_fold_set": sum(c for k, c in seen.items() if k in possible), "repeats": a.repeats,
                   "mean_minus_exact_relative": (mean - exact) / abs(exact) if exact else None,
                   "fold_set_mean_minus_exact_relative": (np.mean(sorted(possible)) - exact) / abs(exact) if exact else None,
                   "most_frequent_share": max(seen.values()) / a.repeats}
            res.append(rec)
            print(rec, flush=True)
    # integer folds are order free
    xi = torch.randint(-1000, 1000, (6,), device="cuda", dtype=torch.int32)
    oi = torch.zeros(1, device="cuda", dtype=torch.int32)
    ok = 0
    for _ in range(100):
        oi.zero_()
        K[(6,)](xi, oi, LANES=1)
        ok += int(oi.item() == int(xi.sum().item()))
    res.append({"fmt": "i32", "repeats": 100, "equal_exact_sum": ok})
    print(res[-1])
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
