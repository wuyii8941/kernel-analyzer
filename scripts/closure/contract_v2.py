#!/usr/bin/env python3
"""Row 3 of the guarantee table: every condition that a phase-1 / phase-2 conclusion rests on, put into one of the five input
classes of input contract v2 (specs/phase2/contract_v2_all_families.md, ambiguities in specs/phase1/ambiguities.md and
specs/phase2/ambiguities_phase2.md).  The executor does not decide: a case the contract does not name is "契约外，待审阅"
(outside the contract, for review), and its results are recorded, never judged (``classify_condition`` is also the arrow-4
"contract not declared -> record only" path of the guarantee chain).

Classes (guarantee table section 3):
  A 合法且有定义                 all comparisons
  B 合法但文档未规定              all justified readings (clause ids)
  C 合法但实数目标无定义          conventions recorded, not judged (clause ids)
  D 不合法                       refused by the spec
  E 规格求值越出正规范围          spec not established (decided when the spec is evaluated; listed here when certain)
  X 契约外，待审阅
A condition can carry several classes (e.g. A for most elements and C for the elements of an empty window); the list is
ordered by how the condition is reported.  Data-dependent C / B flags of the phase-1 families come from the phase-1
classification records (spec readings / undefined statuses evaluated on the received inputs).

    python scripts/closure/contract_v2.py      # -> results/closure/contract_classification.json (+ summary table on stdout)
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "essential"))
OUT = ROOT / "results/closure/contract_classification.json"
CONTRACT = "input contract v2 (2026-10-08), specs_phase1 v0.4 + specs_phase2 v0.2"

A, B, C_, D, E, X = ("A 合法且有定义", "B 合法但文档未规定", "C 合法但实数目标无定义", "D 不合法（规格拒绝）",
                     "E 规格求值越出正规范围", "X 契约外，待审阅")


def entry(classes, clauses, note=""):
    return {"classes": classes, "clauses": clauses, "note": note}


# ------------------------------------------------------------------------------------------------ phase-1 families

def _phase1_flags():
    """condition id -> set of data-dependent categories seen in the phase-1 (and S3) classification records."""
    flags = defaultdict(set)
    for p in [ROOT / f"results/essential/phase1/classification_{f}.json.gz" for f in ("ce", "pool", "index")] + \
             [ROOT / f"results/essential/phase1_supplement/s3/classification_{f}.json.gz" for f in ("pool", "index")]:
        if not p.exists():
            continue
        d = json.load(gzip.open(p, "rt"))
        for r in d["records"]:
            cid = r["condition"]["id"]
            if r.get("held_doc_version_split"):
                flags[cid].add("held_doc_version_split")
            for cr in r["candidates"].values():
                for ce in cr.get("class", {}).values():
                    cat = ce.get("category", "")
                    for key in ("spec_main_undefined", "set_checks_only", "held_doc_version_split", "spec_error"):
                        if key in cat:
                            flags[cid].add(key)
                    for s in ce.get("seeds", []):
                        st = s.get("status", "")
                        if st and st != "ok":
                            flags[cid].add(f"seed_status:{st}")
    return flags


def ce_entry(c, fl):
    cls, cl = [A], ["CE legal domain", "CE-A6"]
    if c["eps"] and c["weight"] != "none":
        cls.append(B)
        cl.append("CE-A1 (weights with smoothing: R_A main, R_B)")
    if c["weight"] != "none" and c["reduction"] == "mean":
        if B not in cls:
            cls.append(B)
        cl.append("CE-A2 (denominator: D2 main, R_C diagnostic)")
    if "spec_main_undefined" in fl or (c["reduction"] == "mean" and (c["ignore"] == "all" or c["weight"] == "zeros")):
        cls.append(C_)
        cl.append("CE-A3")
    return entry(cls, cl)


def pool_entry(c, fl):
    cls, cl = [A], ["POOL legal domain"]
    if c["op"] == "avg_pool":
        cl.append("POOL-A1")
        if c.get("values") == "special":
            return entry([D], ["POOL legal domain: avg input must be finite"], "avg_pool with non-finite input")
    else:
        cl += ["POOL-A2", "POOL-A3"]
        if c.get("values") in ("ties", "ints") or "set_checks_only" in fl:
            cls.append(B)
    if "held_doc_version_split" in fl:
        cls.append(B)
        cl.append("MaxPool1d ceil: two documented formulas (held)")
    if c["op"] == "max_pool" and any(p * 2 > d * (k - 1) + 1 for p, d, k in zip(c["padding"], c.get("dilation", [1] * c["nd"]), c["kernel"])):
        return entry([D], ["POOL legal domain: 2p <= d(k-1)+1"])
    if c["op"] == "max_pool" and any(d > 1 for d in c.get("dilation", [1])) and any(p > 0 for p in c["padding"]):
        cls.append(C_)
        cl.append("POOL-A5 (geometrically empty windows possible)")
    return entry(cls, cl)


def index_entry(c, fl):
    cls, cl = [A], ["IDX legal domain"]
    if c.get("values") == "special_ints":
        cl.append("IDX-A3")
        cls.append(B)
    if c.get("reduce") in ("amax", "amin"):
        cl.append("IDX-A4")
        if "set_checks_only" in fl or c.get("values") in ("ints", "special_ints"):
            cls.append(B)
    # contract v2 (unlike v1) does not restrict sum / prod to finite inputs: non-finite values follow IDX-A3
    return entry(cls, cl)


def s3_pool_entry(c, fl):
    e = pool_entry(c, fl)
    if c["values"] in ("large", "small", "mixed"):
        e["clauses"].append("extreme magnitudes: E possible where the spec leaves the normal range")
    return e


def s3_index_entry(c, fl):
    e = index_entry(c, fl)
    if c["values"] in ("large", "small", "mixed"):
        e["clauses"].append("extreme magnitudes: E possible where the spec leaves the normal range")
    return e


# ------------------------------------------------------------------------------------------------ 2b families

def tp_entry(c):
    cls = [A]
    if c["tokens"] == "empty_microbatch":
        # a micro-batch without tokens is legal (counts >= 0); only a window without any token is ACC-A1
        return entry([A], ["ACC legal domain (token counts >= 0)", "PROG-A1"], "one empty micro-batch, window non-empty")
    return entry(cls, ["ACC legal domain", "PROG-A1"])


def matmul_entry(c):
    if c["shape"] == "batch-broadcast":
        return entry([D], ["BASE-A4: the spec supports 2-D and same-batch 3-D; batch broadcasting refused"])
    return entry([A], ["matmul/linear legal domain"], "inner dimension 0 gives an empty sum" if c["shape"].startswith("(m,0)") else "")


def reduction_entry(c):
    op, dims, v = c["op"], c["dims"], c["values"]
    if dims == "empty-extent":
        if op in ("sum", "prod"):
            return entry([A], ["empty reduction: sum = 0, prod = 1 (identity, documented)"])
        if op in ("amax", "amin", "mean"):
            return entry([C_], ["BASE-A1: empty amax/amin/mean undefined"])
        if op in ("var", "std"):
            return entry([C_], ["BASE-A1: N - correction <= 0"])
        return entry([X], [], f"{op} over an empty axis is not named by contract v2")
    if v == "neg_inf_row":
        if op == "logsumexp":
            return entry([A], ["logsumexp of an all -inf row = -inf (contract)"])
        if op in ("softmax", "log_softmax"):
            if dims == "all":                               # flattened: -inf entries in a row that is not all -inf
                return entry([A], ["softmax with some -inf entries: probability 0 (spec_base_ops.softmax)"])
            return entry([C_, A], ["BASE-A2: softmax of the all -inf row (row 0)", "the other rows: legal domain"])
        return entry([X], [], f"{op} of a row containing -inf: the contract does not name non-finite inputs to {op}")
    if op in ("amax", "amin") and v in ("small_ints", "all_equal"):
        return entry([A], ["ties: amax/amin evenly distribute the gradient (documented)"])
    if op in ("var", "std") and dims == "single" and c["size"] == "1":
        return entry([C_], ["BASE-A1: N - correction <= 0 (one element, correction 1)"])
    return entry([A], ["reductions legal domain"])


def activation_entry(c):
    cl = ["activations legal domain (finite input)"]
    cls = [A]
    if c["op"] in ("gelu_erf", "gelu_tanh", "geglu"):
        cl.append("BASE-A3 (declared formula per candidate; erf is a declared approximation in the spec)")
    if c["op"] == "relu" and c["values"] == "zeros":
        cls.append(B)
        cl.append("ReLU subgradient at 0 unspecified: spec returns the set {0, upstream}")
    if c["values"] == "huge" and c["op"] in ("swiglu", "geglu"):
        cl.append("exact output ~1e60 exceeds float32: candidate overflow goes to column 4")
    return entry(cls, cl)


def gather_entry(c):
    if c["op"] in ("gather", "index_select", "take_along_dim"):
        return entry([A], ["gather / index_select legal domain (indices in range, 2-D, dim in {0,1})"],
                     "take_along_dim read as gather along dim 1" if c["op"] == "take_along_dim" else "")
    return entry([X], [], f"{c['op']} (cat/split round trip, view aliasing) has no clause or spec function in contract v2")


def checkpoint_entry(c):
    if c["impl"] == "hf_gradient_checkpointing":
        return entry([A], ["保存与重算: dropout zero, same input and parameters (prop_checkpoint_equivalence)"],
                     "HF model in train mode with attention dropout 0 (default config)")
    return entry([A], ["保存与重算: dropout zero, same input and parameters (prop_checkpoint_equivalence)"])


def embedding_entry(c):
    cls, cl = [A], ["embedding legal domain"]
    if c.get("max_norm"):
        cl.append("EMB-A1")
        cls.append(B)
    if c["op"] == "embedding_bag":
        if c["bags"] == "empty":
            cl.append("empty bag -> zero (documented)")
        if c["padding_idx"] is not None and c["mode"] == "mean":
            cls.append(C_)
            cl.append("EMB-A2 (all-padding mean bag)")
        if c["per_sample_weights"] and c["mode"] != "sum":
            return entry([D], ["per_sample_weights only with mode=sum"])
    return entry(cls, cl)


def attention_entry(c):
    cls, cl = [A], ["attention legal domain"]
    if c["q_len_k_len"] != "equal" and c["mask"] in ("causal", "sliding_window"):
        cls.append(B)
        cl.append("ATT-C1 (alignment per declared convention)")
    if c["mask"] == "sliding_window":
        cls.append(B)
        cl.append("ATT-A2 (window boundary per declaration)")
    if c["mask"] == "fully_masked_row" or (c["mask"] == "sliding_window" and c["q_len_k_len"] == "q>k"):
        cls.append(C_)
        cl.append("ATT-A1 (rows without an allowed key)")
    if c["gqa"]:
        cl.append("ATT-C2")
    return entry(cls, cl)


def packing_entry(c):
    return entry([A, B], ["packing legal domain (document length >= 1)", "position reset start (0 declared)"])


def rope_entry(c):
    cl = ["RoPE legal domain (d even)", "ROPE-C1"]
    if c["scaling"]:
        cl.append("scaling variant per declared formula (linear: positions / 2)")
    return entry([A, B], cl)


def norm_entry(c):
    cls, cl = [A], ["normalization legal domain"]
    if c["op"] == "rms_norm":
        cl.append("NORM-A1 not applicable (eps given explicitly)")
    if c["op"].startswith("batch_norm") and c["batch"] * 10 == 1:
        cls.append(C_)
        cl.append("NORM-A2")
    if c["values"] == "tiny":
        cl.append("variance below eps: defined (eps inside the sqrt)")
    return entry(cls, cl)


def optimizer_entry(c):
    if c["opt"] == "adafactor":
        return entry([X], [], "Adafactor is not named by contract v2 / spec_optimizers (AdamW, Adam, SGD, RMSprop)")
    cls, cl = [A], ["optimizer legal domain"]
    if c["opt"] == "adam_amsgrad":
        cls.append(B)
        cl.append("OPT-A1 (amsgrad max object per locked docs; R_old alternative)")
    if c["maximize"] or c["weight_decay"]:
        cl.append("OPT-A2 (order per algorithm box)")
    if c["state"] == "grad_none":
        cl.append("grad=None -> skip (documented)")
    if c["state"] == "nonfinite_skip":
        cl.append("non-finite gradient handled by the AMP contract (AMP-A1, GradScaler skip)")
    return entry(cls, cl)


def scheduler_entry(c):
    n = c["steps"]
    if c["sched"] == "one_cycle":
        return entry([X], [], "OneCycleLR is not named by contract v2 / spec_training_program")
    if c["sched"] == "warmup_cosine_hf":
        warm = min(max(n // 10, 1), n)
        if not n > warm:
            return entry([D], [f"schedule legal domain: total > warmup (total {n}, warmup {warm})"])
        return entry([A, B], ["schedule legal domain", "SCH-A1", "SCH-A2"])
    return entry([A, B], ["schedule legal domain (T_max > 0)", "SCH-A1", "SCH-A2"])


def clip_entry(c):
    if c["op"] == "clip_value":
        return entry([X], [], "clip_grad_value_ is not named by contract v2 / spec_training_program")
    cl = [f"norm_type {c['norm_type']} in {{1, 2, inf}}", "CLIP-A1"]
    if c["op"] == "scaler_step":
        cl.append("AMP-A1")
    if c["grads"] in ("with_nan", "with_inf"):
        if c["op"] == "scaler_step":
            return entry([A], cl + ["GradScaler skips the step and backs off (documented)"])
        return entry([A], cl + ["error_if_nonfinite=False with a non-finite norm: scaled by the non-finite coefficient "
                                 "(documented) -> column 4"])
    return entry([A], cl)


def moe_entry(c):
    cls, cl = [A, B], ["MoE legal domain (1 <= k <= E, capacity >= 1)", "MOE-C1", "MOE-C3", "MOE-C4"]
    if c["routing"] == "ties":
        cl.append("MOE-C2 (top-k ties: set)")
    return entry(cls, cl)


# ------------------------------------------------------------------------------------------------ driver

def families():
    import conditions as P1
    import conditions_supplement as S3
    import p2b_attention as AT
    import p2b_basic as BA
    import p2b_clip_amp as CL
    import p2b_embedding as EM
    import p2b_normalization as NO
    import p2b_optimizers as OP
    import p2b_packing as PK
    import p2b_schedulers as SC
    import p2b_small_families as SM
    import p2b_training_program as TP
    fl = _phase1_flags()
    yield "cross_entropy (phase 1)", [(c["id"], c, ce_entry(c, fl[c["id"]])) for c in P1.ce_conditions() if c["op"] != "flce"]
    yield "pooling (phase 1)", [(c["id"], c, pool_entry(c, fl[c["id"]])) for c in P1.pool_conditions()]
    yield "index/scatter (phase 1)", [(c["id"], c, index_entry(c, fl[c["id"]])) for c in P1.index_conditions()]
    yield "pooling (S3)", [(c["id"], c, s3_pool_entry(c, fl[c["id"]])) for c in S3.pool_conditions()]
    yield "index/scatter (S3)", [(c["id"], c, s3_index_entry(c, fl[c["id"]])) for c in S3.index_conditions()]
    yield "training_program", [(c["id"], c, tp_entry(c)) for c in TP.conditions()]
    yield "matmul_linear", [(c["id"], c, matmul_entry(c)) for c in BA.FAMILIES["matmul_linear"].conditions()]
    yield "reductions", [(c["id"], c, reduction_entry(c)) for c in BA.FAMILIES["reductions"].conditions()]
    yield "activations", [(c["id"], c, activation_entry(c)) for c in BA.FAMILIES["activations"].conditions()]
    yield "gather_layout", [(c["id"], c, gather_entry(c)) for c in BA.FAMILIES["gather_layout"].conditions()]
    yield "checkpoint", [(c["id"], c, checkpoint_entry(c)) for c in SM.plan("checkpoint", SM.CK_KEYS)]
    yield "embedding", [(c["id"], c, embedding_entry(c)) for c in EM.conditions()]
    yield "attention", [(c["id"], c, attention_entry(c)) for c in AT.conditions()]
    yield "packing", [(c["id"], c, packing_entry(c)) for c in PK.conditions()]
    yield "rope", [(c["id"], c, rope_entry(c)) for c in SM.plan("rope", SM.RO_KEYS)]
    yield "normalization", [(c["id"], c, norm_entry(c)) for c in NO.conditions()]
    yield "optimizers", [(c["id"], c, optimizer_entry(c)) for c in OP.conditions()]
    yield "schedulers", [(c["id"], c, scheduler_entry(c)) for c in SC.conditions()]
    yield "clip_amp", [(c["id"], c, clip_entry(c)) for c in CL.conditions()]
    yield "moe", [(c["id"], c, moe_entry(c)) for c in SM.plan("moe", SM.MO_KEYS)]


def classify_condition(family, cond):
    """arrow 4 entry point: the contract class of one condition; outside the contract -> record only."""
    for fam, rows in families():
        if fam == family:
            for cid, c, e in rows:
                if cid == cond["id"]:
                    return e
    return entry([X], [], f"family {family} not named by contract v2")


PENDING = {   # clauses stopped by the seven-item doc check (docs/doc_check_seven_items_20261008.md), for review
    "schedulers": lambda c: ["SCH-A1 (T_cur >= T_max)"] if c["sched"] == "cosine_annealing" else [],
    "reductions": lambda c: ["BASE-A1 (var/std, N - correction <= 0)"] if c["op"] in ("var", "std") and
                  (c["dims"] == "empty-extent" or c["size"] == "1" and c["dims"] == "single") else [],
    "embedding": lambda c: ["E-D2 (max_norm scope)"] if c.get("max_norm") else [],
    "matmul_linear": lambda c: ["BASE-A4 label (broadcast: spec scope, not illegal input)"] if c["shape"] == "batch-broadcast" else [],
    # found in the F run (docs/spec_issues_phase2.md SPEC-ISSUE-1): O-D2 places maximize at the update, the locked 2.10 SGD box
    # negates g_t first; parameters agree when weight_decay = 0, the momentum buffer has the opposite sign
    "optimizers": lambda c: ["O-D2 (SGD maximize placement)"] if c["opt"].startswith("sgd") and c["maximize"] else [],
}


def main():
    out = {"contract": CONTRACT, "families": {}}
    print(f"{'family':26s} {'conds':>5s}  " + "  ".join(f"{k[:1]}" for k in (A, B, C_, D, E, X)))
    for fam, rows in families():
        tally = Counter()
        for cid, c, e in rows:
            for k in set(e["classes"]):
                tally[k[:1]] += 1
        pend = PENDING.get(fam, lambda c: [])
        out["families"][fam] = {"conditions": len(rows), "class_counts": dict(tally),
                                "rows": [{"id": cid, **e, "pending_clauses": pend(c)} for cid, c, e in rows]}
        print(f"{fam:26s} {len(rows):5d}  " + "  ".join(f"{tally.get(k[:1], 0)}" for k in (A, B, C_, D, E, X)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str) + "\n")
    outside = [(f, r["id"], r["note"]) for f, v in out["families"].items() for r in v["rows"] if X in r["classes"]]
    rec = {"path": "arrow 4: contract not declared -> record only (no judgment)", "contract": CONTRACT,
           "outside_contract_conditions": len(outside), "examples": outside[:20],
           "handling": "results of these conditions are recorded in the four-group tables but carry no defect / convention verdict"}
    (ROOT / "results/closure/degrade/arrow4_contract_not_declared.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    print("outside the contract:", len(outside))


if __name__ == "__main__":
    main()
