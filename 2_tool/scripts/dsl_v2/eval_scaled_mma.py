"""DSL v2 increment 13 evaluation: TTIR vs sm_100 TTGIR references of the 5 official scaled-MMA kernels on synthetic
random captures (positive inputs), and the sm_90 / sm_100 name coverage of the W0 observed inventory."""
import json
import sys
import numpy as np
sys.path.insert(0, "2_tool/tests")
import test_signatures_nvidia_scaled as T
from kernel_analyzer.reference_eval.ttir_mapping import rule_for

rows = []
rng = np.random.default_rng(2026)
cases = [("simple_dot_mxfp", None, None)] + [(v, n, s) for v, n, s in (("mxfp4_matmul_n128_s1", 128, 1),
         ("mxfp4_matmul_n128_s3", 128, 3), ("mxfp4_matmul_n256_s1", 256, 1), ("mxfp4_matmul_n256_s2", 256, 2))]
for name, n, stages in cases:
    if n is None:
        bufs = {"a_base": rng.integers(0, 0x7F, (128, 64)).astype(np.uint8),
                "b_base": rng.integers(0, 0x7F, (64, 128)).astype(np.uint8),
                "a_scale": rng.integers(122, 133, (128, 2)).astype(np.uint8),
                "b_scale": rng.integers(122, 133, (128, 2)).astype(np.uint8), "out": np.zeros(128 * 128, np.float32)}
        scalars, out_name = (), "out"
    else:
        K = 256 * max(2, stages)
        bufs = {"a_ptr": ("float8_e5m2", rng.integers(0, 0x7C, (128, K)).astype(np.uint8)),
                "b_ptr": rng.integers(0, 256, (n, K // 2)).astype(np.uint8),
                "output_ptr": np.zeros(128 * n, np.float32),
                "a_scale": rng.integers(122, 133, (128, K // 32)).astype(np.uint8),
                "b_scale": rng.integers(122, 133, (n, K // 32)).astype(np.uint8)}
        scalars = [("M", 128), ("N", n), ("K", K), ("stride_scale", K // 32), ("stride_am", K), ("stride_cm", n)]
        out_name = "output_ptr"
    g, gref = T._eval({"ttgir": T._ir(name, "sm100.ttgir")}, bufs, scalars)
    t, tref = T._eval({"ttir": T._ir(name, "ttir"), "ttgir": T._ir(name, "sm100.ttgir")}, bufs, scalars)
    lo1, hi1, s1 = g[out_name]
    lo2, hi2, s2 = t[out_name]
    both = (s1 == 0) & (s2 == 0)
    rows.append({"kernel": name, "elements": int(s1.size), "ttgir_complete": int((s1 == 0).sum()),
                 "ttir_complete": int((s2 == 0).sum()), "both": int(both.sum()),
                 "disjoint": int(((hi1[both] < lo2[both]) | (hi2[both] < lo1[both])).sum()),
                 "scaled_mma_executed": gref.rules.get("nvidia.tc_gen5_mma_scaled", 0),
                 "ttgir_aborted": sorted(set(gref.aborted.values()))})
obs = json.load(open("1_experiments/dsl_v2/w0/observed_e50b186e8bd2.json"))
nv = [r["id"] for r in obs["records"] if any(k.startswith("cuda:") for k in r["observations"])]
missing = [o for o in nv if (rule_for(o) is None or rule_for(o).status != "SUPPORTED")]
doc = {"kernels": rows, "nvidia_observed_ops": len(nv), "with_rule": len(nv) - len(missing), "missing": missing}
print(json.dumps(doc, indent=1))
json.dump(doc, open(sys.argv[1], "w"), indent=1)
