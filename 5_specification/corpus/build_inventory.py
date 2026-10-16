#!/usr/bin/env python3
"""The denominator of spec v3 (PLAN_v3 section 1): every operator instance our experiments ran the tool on.

Reads the experiment records in 1_experiments (and the Inductor output code under .cache/inductor for the
three_way kernels) and writes

    corpus/instances.jsonl.gz    one line per tested instance: experiment, category (T / I / P), operator kind,
                                 operator, parameters, the f used at the time and the spec status recorded then
    corpus/tested_operators.json counts per experiment and kind, and the sources read

Category T: torch operators and compositions; I: programs defined by an issuer's formula; P: the program is the
task (no separate semantic reference applies). Run once and freeze; after the freeze rows may be added, not removed.

    python 5_specification/corpus/build_inventory.py
"""
from __future__ import annotations

import collections
import glob
import gzip
import hashlib
import json
import re
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
E = REPO / "1_experiments"
OUT = Path(__file__).resolve().parent
rows: list[dict] = []
sources: dict[str, str] = {}


def add(experiment, category, kind, operator, iid, params=None, f_then=None, status_then=None, regenerate=None):
    rows.append({"id": f"{experiment}/{iid}", "experiment": experiment, "category": category, "kind": kind,
                 "operator": operator, "params": params or {}, "f_then": f_then, "status_then": status_then,
                 "regenerate": regenerate})


def src(name, path):
    sources[name] = str(Path(path).relative_to(REPO)) if str(path).startswith(str(REPO)) else str(path)


def worst(statuses):
    order = ["no f", "refused", "outside", "X", "pending", "declined", "undefined", "ok"]
    def rank(s):
        s = str(s)
        for i, k in enumerate(order):
            if k in s:
                return i
        return len(order)
    return min(statuses, key=rank) if statuses else None


# ------------------------------------------------------------------ essential round phase 1 (+ closure reclassification)
P1 = E / "essential_bugs_round/results/phase1"
P1_KIND = {"ce": ("cross_entropy", "torch.nn.functional.cross_entropy"),
           "pool": ("pooling", "torch.nn.functional.avg_pool / max_pool"),
           "index": ("index_scatter", "index_add / index_reduce / scatter_reduce")}
for fam, (kind, op) in P1_KIND.items():
    path = P1 / f"classification_{fam}.json.gz"
    src(f"phase1 {fam}", path)
    d = json.load(gzip.open(path))
    for rec in d["records"]:
        c = rec["condition"]
        sts = [s.get("status") for cand in rec["candidates"].values() for part in (cand.get("class") or {}).values()
               for s in (part.get("seeds") or []) if isinstance(part, dict)]
        status = "spec undefined (domain)" if "spec_main_undefined" in sts else ("ok" if sts else None)
        add("essential_phase1", "T", kind, c.get("op", op) if fam != "ce" else op, c["id"],
            {k: v for k, v in c.items() if k != "id"}, "phase1 v0.4 spec (ff5e449:1_experiments/specs/phase1)", status,
            "pre-reorg-20261009:scripts/essential/conditions.py make_*_inputs(condition, seed), seeds 0-2")
# fused linear cross-entropy conditions (phase 1, Liger / Unsloth candidates); records are in the phase 1 summary only
flce_ids = [f"flce{i:03d}" for i in range(72)]
for i in flce_ids:
    add("essential_phase1", "T", "fused_linear_cross_entropy", "LigerFusedLinearCrossEntropy / unsloth fused CE", i,
        {}, "phase1 v0.4 spec (cross entropy of x W^T)", None,
        "pre-reorg-20261009:scripts/essential/conditions.py flce_conditions() / make_flce_inputs")
src("phase1 flce", "pre-reorg-20261009:scripts/essential/conditions.py (flce_conditions: 2 sizes x 3 ignore x 3 reductions x 2 eps x 2 weights)")

# ------------------------------------------------------------------ essential round phase 2b, as re-evaluated by the closure round
FE = E / "closure_round_v2/results/f_eval"
P2_KIND = {"activations": "activation", "attention": "attention", "clip_amp": "clip_and_scaler", "embedding": "embedding",
           "gather_layout": "gather_and_layout", "matmul_linear": "matmul", "moe": "moe", "normalization": "normalization",
           "optimizers": "optimizer", "packing": "packing", "reductions": "reduction", "rope": "rope",
           "schedulers": "scheduler", "s3_index": "index_scatter"}
for f in sorted(FE.glob("*.json")):
    fam = f.stem
    src(f"closure f_eval {fam}", f)
    d = json.load(open(f))
    by_cond = collections.defaultdict(list)
    for k, v in d.get("spec_status", {}).items():
        by_cond[k.rsplit("/", 1)[0]].append(v)
    for cond, sts in sorted(by_cond.items()):
        add("essential_phase2b" if fam != "s3_index" else "essential_phase1_s3", "T", P2_KIND[fam], d.get("spec", ""),
            cond, {"family": fam}, d.get("spec"), worst(sts),
            f"pre-reorg-20261009:scripts/essential/p2b_{'basic' if fam in ('matmul_linear', 'reductions', 'activations', 'gather_layout') else fam}.py")
P2B = E / "essential_bugs_round/results/phase2b"
for fam, kind in (("checkpoint", "checkpoint"), ("training_program", "training_program")):
    path = P2B / fam / "analysis.json"
    src(f"phase2b {fam}", path)
    d = json.load(open(path))
    conds = set()
    for cand in d["candidates"].values():
        conds |= set((cand.get("statuses") or {}).keys())
        for r in cand.get("rows") or []:
            conds.add(r.get("condition") or r.get("id") or json.dumps(r)[:40])
    if fam == "checkpoint" or not conds:   # checkpoint statuses are keyed by outcome, not by condition
        n = d["conditions"] if isinstance(d["conditions"], int) else len(d["conditions"])
        conds = {f"{fam}_{i:02d}" for i in range(n)}
    for cond in sorted(conds):
        add("essential_phase2b", "T", kind, fam, cond, {"family": fam}, None, "no f", f"pre-reorg-20261009:scripts/essential/p2b_{fam}.py")
for fam, kind in (("compile_sequences", "compiled_training_sequence"), ("g6_compositions", "composition"),
                  ("g7_numerical_stream", "numerical_stream")):
    path = P2B / fam / "results.json"
    src(f"phase2b {fam}", path)
    d = json.load(open(path))
    items = d.get("programs") or d.get("sequences") or []
    names = list(items) if isinstance(items, dict) else [(x.get("name") or x.get("id") or str(i)) if isinstance(x, dict)
                                                         else str(x) for i, x in enumerate(items)]
    for n in names:
        add("essential_phase2b", "T", kind, fam, n, {"family": fam}, None, "no f recorded", f"results {path.relative_to(REPO)}")

# ------------------------------------------------------------------ general round v3.1 regression cases and demos
GR = E / "general_round_v3_1/results/regression_v3_1"
GR_KIND = {"fa_ce_plain_bwd": "cross_entropy", "fla_gdr_chunk_basic": "linear_attention_recurrence",
           "flex_bwd_causal": "attention", "flex_fwd_softcap": "attention", "ind_avg_pool1d_ceil_bwd": "pooling",
           "ind_var_unbiased": "reduction", "mamba_ssu_plain": "ssm_state_update", "oib_nn_functional_avg_pool3d_6": "pooling",
           "opt_rprop": "optimizer", "sc1_mean_pool_one_graph": "index_scatter", "tut_attention_plain": "attention",
           "tut_layernorm_n1000_bwd": "normalization", "ua_bidir_sw8_mha": "paged_attention",
           "v2_lora_shrink_3slice": "lora", "v2_topkp_p_split": "sampling"}
for f in sorted(GR.glob("*.json")):
    d = json.load(open(f))
    src(f"general round {f.stem}", f)
    add("general_round_v3_1", "T", GR_KIND.get(f.stem, "unknown"), d.get("implementation", f.stem)[:120] if isinstance(
        d.get("implementation"), str) else f.stem, f.stem, {}, (d.get("specification") or "")[:160], "hand-written float64 reference",
        "pre-reorg-20261009:scripts/tool_spec_cases_*.py")
for demo, kind in (("bsr_dense_mm", "block_sparse_matmul"), ("bsr_softmax", "block_sparse_softmax"),
                   ("sampled_addmm", "sampled_addmm")):
    add("general_round_v3_1", "T", kind, f"torch.sparse {demo} (Triton)", f"demo_{demo}", {}, None, "no f (demo)",
        "1_experiments/general_round_v3_1/results/demo")
src("general round demos", E / "general_round_v3_1/README.md")

# ------------------------------------------------------------------ tolerance study
TL = E / "tolerance_study/results/ledger.json"
src("tolerance ledger", TL)
seen = set()
for r in json.load(open(TL))["rows"]:
    mech, impl = r.get("mechanism"), str(r.get("implementation"))
    if impl == "None" or (mech, impl) in seen:
        continue
    seen.add((mech, impl))
    kind = {"A": "optimizer", "B": "optimizer", "C": "matmul"}.get(mech, "unknown")
    op = {"A": "AdamW (bf16 parameters and states)", "B": "RAdam", "C": "linear / matmul (TF32)"}.get(mech, impl)
    add("tolerance_study", "T", kind, op, f"{mech}:{hashlib.sha1(impl.encode()).hexdigest()[:8]}", {"implementation": impl[:120]},
        {"A": "adamw_spec (calls_*.py)", "B": "radam_spec (calls_*.py)", "C": "linear_spec gamma_k enclosure"}.get(mech),
        "hand-written spec", "1_experiments/tolerance_study/calls_*.py")
add("tolerance_study", "T", "ema", "bf16 EMA (AveragedModel, timm)", "A:ema", {}, "fp64 model", "hand-written", "1_experiments/tolerance_study")

# ------------------------------------------------------------------ flash attention case
src("flash attention case", E / "flash_attention_case/README.md")
for v in ("tutorial_fwd", "planted_tf32", "planted_semantic_constant", "planted_bf16_accumulator", "planted_fp16_accumulator",
          "planted_stagnation", "planted_mask"):
    add("flash_attention_case", "T", "attention", "Triton tutorial 06 fused attention forward", v, {},
        "torch SDPA float64 (+ base-2 logsumexp)", "ok", "1_experiments/flash_attention_case/fa_inputs.py, make_planted.py")

# ------------------------------------------------------------------ three_way: Inductor kernels, bound by exact kernel source to their ATen graph fragment
UNITS = E / "three_way/results/units.jsonl"
src("three_way units", UNITS)
src("Inductor output code", REPO / ".cache/inductor (provenance comments: Original ATen, Graph fragment)")
blocks = {}
kc = re.compile(r"^(\w+) = async_compile\.triton\('(\w+)', '''", re.M)
for f in glob.glob(str(REPO / ".cache/inductor/*/*.py")):
    t = open(f, errors="ignore").read()
    if "Graph fragment" not in t:
        continue
    for m in kc.finditer(t):
        head = t.rfind("# Topologically Sorted Source Nodes:", 0, m.start())
        end = t.find("''', device_str=", m.end())
        if head < 0 or end < 0:
            continue
        comment = t[head:m.start()]
        orig = re.search(r"Original ATen: \[(.*?)\]", comment)
        frag = sorted(set(re.findall(r"target=torch\.ops\.(\w+\.\w+)", comment)))
        body = t[m.end():end]
        blocks.setdefault(hashlib.sha256(body.strip().encode()).hexdigest(), (m.group(2), orig.group(1) if orig else "", frag, f))
units = [json.loads(l) for l in open(UNITS)]
for u in units:
    unit = u["unit"].rstrip("/")
    model, config, name = unit.split("/")[-3:]
    prov, ttir = None, None
    cap = REPO / ".cache/three_way" / "/".join(unit.split("/")[-4:])
    srcs = list((cap / "compiled").glob("*.source")) if cap.is_dir() else []
    if srcs:
        loc = re.search(r'#loc = loc\("([^"]+\.py)"', srcs[0].read_text(errors="ignore"))
        if loc and Path(loc.group(1)).is_file():
            prov = blocks.get(hashlib.sha256(Path(loc.group(1)).read_text().strip().encode()).hexdigest())
    add("three_way", "T", "inductor_kernel", u.get("kernel", name), f"{model}/{config}/{name}",
        {"aten": prov[1] if prov else None, "graph_ops": prov[2] if prov else None, "verify": u.get("verify")},
        "plants only: f = G of the unmodified kernel",
        None if prov else ("binding pending: template kernel launched during autotuning, not in the final output code"
                           if "_tem_" in u.get("kernel", name) else "binding pending: kernel file missing from the cache"),
        "capture under .cache/three_way/capture; Inductor graph fragment in .cache/inductor")

# ------------------------------------------------------------------ composition study: library corpus (Liger, FlagGems) and the TF32 modification
CORP = E / "composition_study/results/corpus.json"
src("composition corpus", CORP)
for k in json.load(open(CORP))["kernels"]:
    for call in k.get("calls") or [k["call"]]:
        lib, op, dtype = call.split(".", 2)
        add("composition_corpus", "T", f"{lib}:{op}", call, f"{call}/{k['kernel']}", {"kernel": k["kernel"], "dtype": dtype},
            None, "no f (K vs G only)", "1_experiments/composition_study/corpus.py")
add("composition_study", "T", "matmul", "TF32 matmul in 6 Qwen layers (activations rounded to nearest)", "tf32_modification",
    {}, None, "no f (K vs G only)", "1_experiments/composition_study/tf32_attribution.py")

# ------------------------------------------------------------------ issuer-defined programs: blind tests v1 / v2, structure acceptance v1.1
for v in ("v1", "v2"):
    path = E / f"blind_tests/blind_records/{v}/blind_test_{v}_answer_key_UNSEALED.json"
    src(f"blind {v}", path)
    for pid, p in json.load(open(path))["programs"].items():
        add(f"blind_{v}", "I", f"blind_family_{p.get('family')}", f"blind {v} {pid}", pid,
            {"family": p.get("family"), "class": p.get("class"), "variant": p.get("variant")},
            f"issuer formula: blind_test_{v}_phase2_release.md section 1", "exact rational f (issuer)",
            f"1_experiments/blind_tests/blind_records/{v}")
ST = E / "structure_acceptance_v1_1/structure_v1_1_rc1_r20261008T2030.tar.gz"
src("structure acceptance", ST)
with tarfile.open(ST) as t:
    fams = {}
    for n in t.getnames():
        if "/jobs/" in n and n.endswith(".json"):
            d = json.load(t.extractfile(n))
            fams[d["program"]] = d.get("family")
for pid in sorted(fams):
    add("structure_v1_1", "I", f"structure_family_{fams[pid]}", f"structure {pid}", pid, {"family": fams[pid]},
        "package exact-rational f", "exact rational f (issuer)", "structure package (issuer); run record tarball")

# ------------------------------------------------------------------ P: the program is the task
src("dsl_v2", E / "dsl_v2/README.md")
add("dsl_v2", "P", "triton_language_primitives", "official Triton tests (test_core, tensor_descriptor, standard, random, "
    "conversions, libdevice): 12,766 launches, 233 test functions", "broad_capture", {"launches": 12766}, None, "not applicable",
    "1_experiments/dsl_v2")
src("composition primitives and programs", E / "composition_study/README.md")
add("composition_study", "P", "triton_primitive_forms", "986 primitive instances x 23 forms (17,687 cells)", "primitives",
    {"cells": 17687}, None, "not applicable", "1_experiments/composition_study/primitives.py, forms.py")
add("composition_study", "P", "random_triton_programs", "random programs: predictor dev and two held-out sets of 60",
    "random_programs", {}, None, "not applicable", "1_experiments/composition_study/programs.py")
B1 = E / "batch1_acceptance/items.json"
src("batch1", B1)
b1 = json.load(open(B1))
b1 = b1.get("items", b1) if isinstance(b1, dict) else b1
for i, it in enumerate(b1 if isinstance(b1, list) else []):
    add("batch1_acceptance", "P", "acceptance_probe", str(it.get("call") or it.get("kind") or it.get("id"))[:80],
        str(it.get("id", i)), {}, None, "not applicable", "1_experiments/batch1_acceptance/holdout_calls.py")

# ------------------------------------------------------------------ bugs found in tested operators (4_bugs_cases)
src("bugs", REPO / "4_bugs_cases/README.md")
BUGS = {"B001": ("rope", "Liger RoPE"), "B002": ("rope", "Liger RoPE"), "B003": ("moe", "Liger fused MoE experts (tl.dot TF32)"),
        "B004": ("normalization", "Qwen3-Next instance-patched RMSNorm"), "B005": ("optimizer", "pure-bf16 AdamW"),
        "B006": ("matmul", "tl.dot fp32 default TF32"), "B007": ("matmul", "bf16 tensor-core products, FP32 accumulation"),
        "B008": ("optimizer_quantized_state", "torchao AdamW4bit"), "B009": ("optimizer", "torchao _AdamW(bf16_stochastic_round)"),
        "B010": ("pooling", "compiled avg_pool2d ceil_mode backward"), "B011": ("attention", "Triton tutorial 06 attention backward"),
        "B012": ("optimizer", "RAdam"), "B013": ("attention", "vLLM sliding-window attention"),
        "B014": ("pooling", "Dynamo guard on max_pool"), "B015": ("pooling", "avg_pool"), "B016": ("index_scatter", "index_reduce"),
        "B017": ("pooling", "Dynamo guard on max_pool / unique"), "B018": ("attention", "SDPA rewrite"),
        "B019": ("ema", "EMA"), "B020": ("index_scatter", "index_reduce / scatter_reduce"), "B021": ("pooling", "pooling"),
        "B022": ("pooling", "pooling"), "B023": ("activation", "gelu(approximate='tanh') backward"),
        "B024": ("triton_float_mod", "Triton floating-point %")}
for b, (kind, op) in BUGS.items():
    add("bugs", "P" if kind == "triton_float_mod" else "T", kind, op, b, {}, None, None, "4_bugs_cases")

# ------------------------------------------------------------------ write
with gzip.GzipFile(OUT / "instances.jsonl.gz", "wb", mtime=0) as fh:
    for r in rows:
        fh.write((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode())
by_exp = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    by_exp[(r["experiment"], r["category"])][r["kind"]] += 1
summary = {
    "schema": "spec-v3-tested-operators-1",
    "built": "2026-10-16",
    "rule": "denominator of PLAN_v3 V3; frozen before any new contract; rows may be added after the freeze, never removed",
    "instances": len(rows),
    "by_category": dict(collections.Counter(r["category"] for r in rows)),
    "by_experiment": {f"{e} [{c}]": dict(sorted(k.items())) for (e, c), k in sorted(by_exp.items())},
    "three_way_provenance": {"units": sum(r["experiment"] == "three_way" for r in rows),
                             "bound_to_graph_fragment": sum(r["experiment"] == "three_way" and r["params"].get("graph_ops") is not None
                                                            for r in rows)},
    "three_way_graph_ops": sorted({o for r in rows if r["experiment"] == "three_way" for o in (r["params"].get("graph_ops") or [])}),
    "three_way_original_aten": sorted({o.strip() for r in rows if r["experiment"] == "three_way"
                                       for o in (r["params"].get("aten") or "").split(",") if o.strip()}),
    "sources": sources,
}
(OUT / "tested_operators.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
print(json.dumps({k: summary[k] for k in ("instances", "by_category", "three_way_provenance")}, indent=1))
for k, v in summary["by_experiment"].items():
    print(k, sum(v.values()), dict(list(v.items())[:12]))
