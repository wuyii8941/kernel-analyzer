#!/usr/bin/env python3
"""In-kernel localization by bitwise emulation (shadow execution).

1. Emulation coverage: every capture package under --captures is emulated
   (emulate.verify) and compared with the device output bit for bit.
2. Localization on kernels whose bias source is known: the mutation kernels
   (8 input draws per variant) and Liger cross-entropy launches.  For every
   rounding / approximating node j, c_j = K_emulated - K_(node j exact) on the
   bitwise-reproduced outputs; the per-unit mean of c_j is tested across units
   (t test, Holm over the nodes of that kernel variant).

    python scripts/run_inkernel_localization.py --captures .cache/version_ptx/captures \\
        --liger .cache/liger_kernels --out results/reference_eval/inkernel_localization.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kernel_analyzer.reference_eval.analysis import _holm  # noqa: E402
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, load_launch  # noqa: E402
from kernel_analyzer.reference_eval.emulate import HardwareOracle, localize, verify  # noqa: E402

DRAWS = 8
EXPECTED = {  # mutations whose locus and sign are known in advance
    ("scale", "rounding_down"): ("__nv_fmul_rd", -1), ("scale", "rounding_up"): ("__nv_fmul_ru", 1),
    ("sum4", "rounding_down_local"): ("__nv_fadd_rd", -1),
    ("accumulate", "rounding_down"): ("__nv_fadd_rd", -1), ("accumulate", "rounding_up"): ("__nv_fadd_ru", 1),
}


def coverage(captures: Path, oracle) -> list:
    rows = []
    for pkg in sorted(p for p in captures.iterdir() if (p / "launch.json").exists()):
        t0 = time.time()
        r = verify(load_launch(pkg), oracle=oracle)
        r["package"] = pkg.name
        r["seconds"] = round(time.time() - t0, 2)
        rows.append(r)
        print(pkg.name[:70], r["status"], {k: (v["bit_identical"], v["emulated"], v["written"])
                                                        for k, v in r["buffers"].items()}, flush=True)
    return rows


STATISTICS = {"mean": "mean contribution over output elements (additive bias)",
              "relative": "mean of c / K (multiplicative bias)",
              "aligned": "sum(c K) / |K| (component along the output itself, the mechanism direction of a scaling)"}


def unit_stats(units: list) -> list:
    """units: localize() results of one kernel variant -> per (node, buffer, statistic) t test across units,
    Holm over all of them."""

    from scipy.stats import t as tdist

    by_key = {}
    for res in units:
        for row in res["nodes"]:
            for buf, b in row["buffers"].items():
                e = by_key.setdefault((row["node"], buf), {"text": row["text"], "lowering": row["lowering"],
                                                           "mean": [], "relative": [], "aligned": [], "abs": [],
                                                           "endpoint_differ": 0, "endpoint_elements": 0,
                                                           "endpoint_max": 0.0})
                ec = b.get("endpoint_check")
                if ec:
                    e["endpoint_differ"] += ec["outputs_differ_by_half_ulp32_or_more"]
                    e["endpoint_elements"] += ec["elements"]
                    e["endpoint_max"] = max(e["endpoint_max"], ec["max_difference_in_ulp32"])
                e["mean"].append(b["mean"])
                e["relative"].append(b["mean_relative"])
                e["aligned"].append(b["aligned"])
                e["abs"].append(b["mean_abs"])
    out, tests = [], []
    for (node, buf), e in by_key.items():
        row = {"node": node, "buffer": buf, "text": e["text"], "lowering": e["lowering"], "units": len(e["mean"]),
               "mean_abs": float(np.mean(e["abs"])), "tests": {},
               "endpoint_check": {"elements": e["endpoint_elements"],
                                  "outputs_differ_by_half_ulp32_or_more": e["endpoint_differ"],
                                  "max_difference_in_ulp32": e["endpoint_max"]}}
        for stat in STATISTICS:
            m = np.array(e[stat])
            n = m.size
            sd = m.std(ddof=1) if n > 1 else 0.0
            if sd > 0:
                tstat = float(m.mean() / (sd / math.sqrt(n)))
                p = float(2 * tdist.sf(abs(tstat), n - 1))
            else:
                tstat, p = (math.inf if m.mean() != 0 else 0.0), (0.0 if m.mean() != 0 else 1.0)
            row["tests"][stat] = {"value": float(m.mean()), "t": tstat, "p": p}
            tests.append(row["tests"][stat])
        out.append(row)
    for test, rej in zip(tests, _holm([t["p"] for t in tests], 0.05) if tests else []):
        test["holm_significant"] = bool(rej)
    for row in out:
        best = max(row["tests"].items(), key=lambda kv: abs(kv[1]["t"]) if math.isfinite(kv[1]["t"]) else 1e300)
        row["strongest"] = {"statistic": best[0], **best[1]}
        row["significant"] = sorted(k for k, v in row["tests"].items() if v["holm_significant"])
    out.sort(key=lambda r: -(abs(r["strongest"]["t"]) if math.isfinite(r["strongest"]["t"]) else 1e300))
    return out


def mutation_part(oracle) -> list:
    import torch

    from scripts.run_method_comparison import specs

    rows = []
    for name, (make, table) in specs().items():
        for mut, kind in [(0, "original")] + list(table.items()):
            units, checks = [], []
            for s in range(DRAWS):
                tensors, launch_fn, shape = make(2000 + s)
                tensors = dict(tensors)
                tensors["Y"] = torch.empty(shape, device="cuda")
                rec = TritonLaunchRecorder(select=lambda n, i: n.startswith("m_"))
                with rec:
                    launch_fn(tensors, mut)
                    torch.cuda.synchronize()
                launch = rec.launches[-1]
                v = verify(launch, oracle=oracle)
                checks.append(v)
                if not v["bitwise_reproduced"]:
                    continue
                choices = v["lowering_choices"]
                units.append(localize(launch, oracle=oracle, ungrouped=choices["uncontracted"],
                                      swapped=choices["swapped"], endpoint_nodes="all"))
            stats = unit_stats(units)
            buffers = checks[0]["buffers"]
            row = {"kernel": name, "mutation": kind, "draws": DRAWS,
                   "bitwise_reproduced_draws": sum(c["bitwise_reproduced"] for c in checks),
                   "elements": {k: (sum(c["buffers"][k]["bit_identical"] for c in checks),
                                    sum(c["buffers"][k]["written"] for c in checks)) for k in buffers},
                   "not_emulable_nodes": checks[0]["not_emulable_nodes"],
                   "lowering_choices_from_output": [c["lowering_choices"] for c in checks
                                                    if c["fitted_to_output"]],
                   "nodes": stats}
            exp = EXPECTED.get((name, kind))
            sig = [r for r in stats if r["significant"]]
            if exp is not None:
                top = stats[0] if stats else None
                row["expected_locus"] = {"symbol": exp[0], "sign": exp[1], "statistic": "mean"}
                row["located"] = bool(top and exp[0] in top["text"] and "mean" in top["significant"]
                                      and np.sign(top["tests"]["mean"]["value"]) == exp[1])
            row["significant_nodes"] = [(r["text"][:80], r["significant"], r["strongest"]["statistic"],
                                         r["strongest"]["value"], r["strongest"]["t"]) for r in sig]
            rows.append(row)
            print(name, kind, "reproduced", row["bitwise_reproduced_draws"], "significant",
                  [(t[:50], st, f"{m:.2e}", f"{tt:.1f}") for t, _, st, m, tt in row["significant_nodes"]][:3],
                  "located" if row.get("located") else "", flush=True)
    return rows


def liger_part(root: Path, oracle, max_launches: int) -> dict:
    pkgs = sorted(p for p in root.iterdir() if (p / "launch.json").exists()
                  and json.loads((p / "launch.json").read_text())["kernel_name"] == "liger_cross_entropy_kernel")
    units, checks = [], []
    for pkg in pkgs[:max_launches]:
        launch = load_launch(pkg)
        v = verify(launch, oracle=oracle)
        checks.append(v)
        if v["bitwise_reproduced"]:
            units.append(localize(launch, oracle=oracle, endpoint_nodes="all"))
        print(pkg.name, v["bitwise_reproduced"], flush=True)
    n_non_ignore = sorted({next(a.value for a in load_launch(p).args if a.name == "n_non_ignore")
                           for p in pkgs[:max_launches]})
    return {"launches": len(units), "n_non_ignore": n_non_ignore,
            "bitwise_reproduced": sum(c["bitwise_reproduced"] for c in checks), "nodes": unit_stats(units)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--liger", type=Path, required=True)
    parser.add_argument("--liger-launches", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse-coverage", action="store_true")
    parser.add_argument("--only-coverage", action="store_true")
    args = parser.parse_args()
    oracle = HardwareOracle(options={"num_warps": 4, "enable_fp_fusion": True})
    report = {"schema": "kernel-analyzer-inkernel-localization-v2",
              "contribution": "c_j = K_emulated - K_(node j substituted): a model counterfactual. Every execution of "
                              "node j passes a float64 value of its exact result (midpoint of the enclosure) from its "
                              "emulated operands; all other nodes behave as on the device; approximate nodes downstream "
                              "keep their own error, hw(RN(x)) + f(x) - f(RN(x)). endpoint_check repeats the "
                              "substitution with both enclosure endpoints."}
    if args.reuse_coverage and args.out.exists():
        report["coverage"] = json.loads(args.out.read_text())["coverage"]
    else:
        report["coverage"] = coverage(args.captures, oracle)
    report["statistics"] = STATISTICS
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")  # coverage first
    if args.only_coverage and args.out.exists():
        old = json.loads(args.out.read_text())
        report["mutations"], report["liger_cross_entropy"] = old["mutations"], old["liger_cross_entropy"]
    else:
        report["mutations"] = mutation_part(oracle)
        report["liger_cross_entropy"] = liger_part(args.liger, oracle, args.liger_launches)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in report["liger_cross_entropy"].items() if k != "nodes"}))
    for r in report["liger_cross_entropy"]["nodes"][:8]:
        print(r["text"][:70], r["buffer"], {k: (f"{v['value']:.3e}", f"{v['t']:.1f}", v["holm_significant"])
                                             for k, v in r["tests"].items()})


if __name__ == "__main__":
    main()
