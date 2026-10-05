#!/usr/bin/env python3
"""Build the benchmark manifest for real bugs before and after their fixes (plan section 4, part 3).

Every unit: configuration (case group / case / environment / runtime patch), reference definition, measurement point,
proposition, and the independent answer basis it comes from (the bug report, its repro, a prediction recorded before
the run).  Positives and controls are taken from the bug reports, not from the tool's output.

    python scripts/benchmark_build.py && python scripts/benchmark_score.py results/benchmark/real_bugs_pre_post.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DET, CLEAN = ["candidate"], ["none", "constant-level"]
STRICT = {"mode": "B", "f": "strict enclosure (scripts/strict_specs.py)"}
SCREEN = {"mode": "B", "f": "float64 eager, declared 2^-40 budget (screening)"}
U = []


def unit(name, group, case, result, output, prop, text, ref, basis, env=None, preload=None):
    U.append({"unit": name, "part": "real_bug_pre_post",
              "config": {"group": group, "case": case, "env": env or {}, "preload": preload},
              "reference": ref, "measurement": {"output": output}, "proposition": prop, "proposition_text": text,
              "answer_basis": basis, "result": result})


# B016: mechanism 1 (unmasked atomic) is fixed by the mask patch, mechanism 2 (reader fused ahead) is not
b016 = "bugs/B016_inductor_scatter_into_single_element.md (mechanisms 1 and 2); bugs/repro/B016_*"
for case in ("sc1_mean_pool_one_graph", "sc1_scaled_count"):
    unit(f"B016/{case}/pre", "scatter1", case + "_strict", f"results/tool_spec/final/strict/{case}_strict.json", "out",
         {"sem_bins": DET}, "mechanism 1: e_sem detected", STRICT, b016)
    unit(f"B016/{case}/post", "scatter1", case, f"results/tool_spec/final/scatter1_patched/{case}.json", "out",
         {"sem_bins": CLEAN}, "mask patch: e_sem at most constant-level", SCREEN, b016,
         preload="bugs/repro/B016_mask_patch.py")
unit("B016/sc1_mean_pool_two_graphs/control", "scatter1", "sc1_mean_pool_two_graphs",
     "results/tool_spec/final/scatter1/sc1_mean_pool_two_graphs.json", "out", {"sem_bins": CLEAN},
     "control (two segments): not detected", SCREEN, b016)
for case, prop, text in [
        ("oib_scatter_reduce_mean_19", {"sem_bins": DET}, "mechanism 2: e_sem detected"),
        ("oib_scatter_reduce_amax_19", {"special_k_vs_f_min": 1}, "mechanism 2: K has inf/NaN where f is finite"),
        ("oib_index_reduce_amax_0", {"special_k_vs_f_min": 1}, "mechanism 2: K has inf/NaN where f is finite")]:
    for tag, d in (("pre", "results/tool_spec/opinfo_loose_rerun"), ("post-mask-patch", "results/tool_spec/final/scatter1_patched")):
        unit(f"B016/{case}/{tag}", "opinfo", case, f"{d}/{case}.json", "d0", prop,
             text + (" (unchanged by the mask patch)" if tag != "pre" else ""), SCREEN, b016,
             preload="bugs/repro/B016_mask_patch.py" if tag != "pre" else None)

# B012: predicted from the formula before the runs; fixed by evaluating rho_t and the bias corrections in float64
b012 = "docs/radam_prediction_20261006.md (prediction from the formula); bugs/B012_*; bugs/repro/B012_repro_radam_fp32_rho.py"
PRED = {("99985", 2): (CLEAN, 1e-3, "flip from fp32 arithmetic only: e_sem constant-level, e_num ~ whole update"),
        ("99985", 3): (DET, None, "flip from the fp32 constants: e_sem ~ whole update"),
        ("99985", 6): (DET, None, "same decision: e_sem ~ 0.55 x update"),
        ("9997", 5): (CLEAN, None, "unrectified in all three: e_sem ~ 2e-7 x update"),
        ("9997", 6): (["small"], None, "same decision: e_sem ~ 0.065 x update"),
        ("9993", 6): (["small"], None, "same decision: e_sem ~ 3.7e-3 x update"),
        ("99995", 2): (DET, None, "flip from the fp32 constants: e_sem ~ whole update")}
for (b, t), (bins, nmin, text) in PRED.items():
    case = f"opt_radam_b{b}_step{t}"
    prop = {"sem_bins": bins, **({"num_rel_min": nmin} if nmin else {})}
    unit(f"B012/{case}/pre", "inductor2", case + "_strict", f"results/tool_spec/final/strict/{case}_strict.json", "param",
         prop, text, STRICT, b012)
    unit(f"B012/{case}/post", "inductor2", case, f"results/tool_spec/final/radam_deep_fixed/{case}.json", "param",
         {"sem_bins": CLEAN}, "float64 fix: e_sem at most constant-level", SCREEN, b012,
         preload="bugs/repro/B012_radam_fp64_patch.py")

# B013: positives and controls as in the report's evidence table; the fixed kernel in .cache/pylibs/vllm_shim_b013fix
b013 = ("bugs/B013_vllm_unified_attention_bidir_swa.md (evidence table: positives and controls); "
        "bugs/repro/B013_repro_vllm_unified_attention_bidir_swa.py")
FIX = {"PYTHONPATH": ".cache/pylibs/vllm_shim_b013fix"}
for case in ("ua_bidir_sw8_mha", "ua_bidir_sw24_gqa4", "ua_perseq_causal_sw8"):
    unit(f"B013/{case}/pre", "vllm", case, f"results/tool_spec/final/vllm/{case}.json", "out", {"sem_bins": DET},
         "keys right of the window dropped: e_sem detected", SCREEN, b013)
    unit(f"B013/{case}/post", "vllm", case, f"results/tool_spec/final/vllm_b013fix/{case}.json", "out",
         {"sem_bins": CLEAN}, "fixed kernel: e_sem at most constant-level", SCREEN, b013, env=FIX)
for case, why in (("ua_bidir_sw8_qpkv16", "BLOCK_Q = 1"), ("ua_causal_sw24", "causal"), ("ua_bidir", "no window")):
    for tag, d, env in (("pre", "results/tool_spec/final/vllm", None), ("post", "results/tool_spec/final/vllm_b013fix", FIX)):
        unit(f"B013/{case}/control-{tag}", "vllm", case, f"{d}/{case}.json", "out", {"sem_bins": CLEAN},
             f"control ({why}): not detected", SCREEN, b013, env=env)

# B015 (no fix available)
unit("B015/oib_nn_functional_avg_pool3d_6/pre", "opinfo", "oib_nn_functional_avg_pool3d_6",
     "results/tool_spec/opinfo_onesample_rerun/oib_nn_functional_avg_pool3d_6.json", "d0", {"sem_bins": DET},
     "out-of-bounds windows averaged over the whole kernel volume: e_sem detected", SCREEN,
     "bugs/B015_inductor_avg_pool3d_ceil_backward.md; bugs/repro/B015_*")

out = ROOT / "results/benchmark/real_bugs_pre_post.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"title": "Real bugs before and after their fixes (benchmark part 3)", "units": U}, indent=1))
print(len(U), "units ->", out.relative_to(ROOT))
