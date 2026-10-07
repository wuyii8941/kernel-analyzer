#!/usr/bin/env python3
"""2b family "training_program" (protocol v2 section 3; registry tier1.training_program): E, F, P on the accumulation window.

F is open for this family: ``specs/phase1/spec_accumulation.py`` (phase-1 package, reviewed) defines the window requirement
-- the per-token loss averaged over all non-ignored tokens of the GLOBAL batch (all micro-batches of the window, all ranks).

Conditions: the coverage plan's (tokens, ranks) combinations with the factor ``impl`` projected out -- (equal, 1),
(unequal, 1), (empty_microbatch, 1), (equal, 2), (unequal, 2) -- x 3 seeds.  The window is always the same 4 sequences
(lengths: equal 6/6/6/6, unequal 3/4/9/12, empty_microbatch 1/1/8/10 -- a length-1 sequence has no label after the shift),
k = 2 accumulation steps; 1 rank: micro-batches of 2 sequences; 2 ranks (gloo, CPU, torchrun): rank r takes sequences r and
r + 2, one per micro-batch.  Model: 1-layer Llama, random weights (torch.manual_seed(0) in each environment), float64 on CPU;
the Llama loss upcasts logits with ``.float()``, so the float32 contract applies (as in phase-1 W7).  One optimizer step with
lr = 0 and no clipping; the accumulated gradient is recorded inside the optimizer's step() (rank 0; ranks are checked equal).

Candidates (each run in its environment): HF Trainer 4.45.2 / 4.46.0 / 4.57.3 / 5.19.0; an Accelerate 1.7.0 loop as in the
Accelerate gradient-accumulation guide (``accelerator.accumulate`` + ``accelerator.backward(loss)``); a plain PyTorch loop as in
the PyTorch AMP / DDP examples (``loss / k``, DDP with ``no_sync`` on all but the last micro-batch).

Variants: ``base``; ``no_ga`` (E analogue: the same program without accumulation, micro-batch size x 2); ``regroup`` (P split
invariance: the sequences reordered 0, 2, 1, 3); ``padfree`` (P padding-free == padded: ``DataCollatorWithFlattening``,
1 rank, transformers >= 4.57 only -- older versions isolate packed sequences only under flash_attention_2, which needs CUDA).
P rank invariance compares (tokens, 2 ranks) with (tokens, 1 rank): the same window.

    /data1/tzh/envs/<env>/bin/python scripts/essential/p2b_training_program.py run --impl hf_trainer
    /data1/tzh/envs/liger/bin/python scripts/essential/p2b_training_program.py run --impl accelerate_loop
    /data1/tzh/envs/liger/bin/python scripts/essential/p2b_training_program.py run --impl own_loop
    python scripts/essential/p2b_training_program.py analyse
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "specs/phase1"))
CACHE = ROOT / ".cache/essential/p2b/training_program"
OUT = ROOT / "results/essential/phase2b/training_program"
SEEDS = (0, 1, 2)
K_ACC = 2
PATTERNS = {"equal": [6, 6, 6, 6], "unequal": [3, 4, 9, 12], "empty_microbatch": [1, 1, 8, 10]}
TAU32 = 2.0 ** -12


def conditions():
    plan = json.loads((ROOT / "results/essential/phase2b/coverage_plan.json").read_text())["tier1"]["training_program"]["conditions"]
    out, seen = [], {}
    for c in plan:
        key = (c["tokens"], c["ranks"])
        if key in seen:
            seen[key]["high_risk"] |= bool(c.get("high_risk"))
            continue
        d = {"id": f"tp_{c['tokens']}_r{c['ranks']}", "tokens": c["tokens"], "ranks": c["ranks"], "high_risk": bool(c.get("high_risk"))}
        seen[key] = d
        out.append(d)
    return out


def variants_for(cond, impl):
    v = ["base", "no_ga", "regroup"]
    if impl == "hf_trainer" and cond["ranks"] == 1:
        import transformers
        if tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 57):
            v.append("padfree")
    return v


# ------------------------------------------------------------------------------------------------ model and data

def cfg():
    from transformers import LlamaConfig
    return LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                       num_key_value_heads=2, max_position_embeddings=64)


_STATE = {}


def model():
    import torch
    from transformers import LlamaForCausalLM
    if "s" not in _STATE:
        torch.manual_seed(0)
        _STATE["s"] = {k: v.clone() for k, v in LlamaForCausalLM(cfg()).double().state_dict().items()}
    m = LlamaForCausalLM(cfg()).double()
    m.load_state_dict(_STATE["s"])
    return m


def sequences(cond, seed, variant):
    import torch
    g = torch.Generator().manual_seed(1000 * seed + 7)
    seqs = [torch.randint(1, 64, (n,), generator=g) for n in PATTERNS[cond["tokens"]]]
    if variant == "regroup":
        seqs = [seqs[0], seqs[2], seqs[1], seqs[3]]
    return seqs


def collate(batch):
    import torch
    L = max(len(s) for s in batch)
    ids = torch.zeros((len(batch), L), dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, s in enumerate(batch):
        ids[i, :len(s)], lab[i, :len(s)], att[i, :len(s)] = s, s, 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def flatten_collate(batch):
    from transformers import DataCollatorWithFlattening
    return DataCollatorWithFlattening()([{"input_ids": s.tolist()} for s in batch])


def rank_microbatches(seqs, ranks, rank, variant):
    """the micro-batches this rank sees in the window (k = 2), or one batch of double size without accumulation."""
    mine = seqs[0:4] if ranks == 1 else [seqs[rank], seqs[rank + 2]]
    if variant == "no_ga":
        return [mine]
    return [mine[0:2], mine[2:4]] if ranks == 1 else [[mine[0]], [mine[1]]]


def per_token_grads(seqs):
    """f's inputs: the float64 autograd gradient of each non-ignored token's loss (logits upcast as in the model's loss)."""
    import torch
    m = model()
    out = []
    for s in seqs:
        b = collate([s])
        logits = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        toks = []
        for c in range(len(s) - 1):
            m.zero_grad()
            torch.nn.functional.cross_entropy(logits[0, c][None].float(), s[c + 1][None]).backward(retain_graph=True)
            toks.append([Fraction(float(v)) for v in torch.cat([p.grad.reshape(-1) for p in m.parameters()])])
        out.append(toks)
    return out


# ------------------------------------------------------------------------------------------------ candidates

def _rec_opt():
    import torch

    class Rec(torch.optim.SGD):
        def step(self, closure=None):
            self.recorded = torch.cat([p.grad.detach().reshape(-1).clone() if p.grad is not None else torch.zeros(p.numel(), dtype=p.dtype)
                                       for p in self.param_groups[0]["params"]])
            return super().step(closure)
    return Rec


def grad_hf_trainer(seqs, cond, variant, rank, world):
    import torch
    from transformers import Trainer, TrainingArguments
    m = model()
    opt = _rec_opt()(m.parameters(), lr=0.0)
    mbs = rank_microbatches(seqs, cond["ranks"], rank, variant)
    bs, k = len(mbs[0]), len(mbs)
    mine = [s for mb in mbs for s in mb]
    coll = flatten_collate if variant == "padfree" else collate

    class T(Trainer):
        def get_train_dataloader(self):
            return torch.utils.data.DataLoader(mine, batch_size=bs, shuffle=False, collate_fn=coll)

    kw = dict(output_dir=str(CACHE / f"trainer_out_r{rank}"), per_device_train_batch_size=bs, gradient_accumulation_steps=k,
              max_steps=1, learning_rate=0.0, max_grad_norm=0.0, report_to=[], use_cpu=True, save_strategy="no",
              lr_scheduler_type="constant", remove_unused_columns=False, logging_steps=1, seed=0, dataloader_num_workers=0)
    if world > 1:
        kw["ddp_backend"] = "gloo"
    t = T(model=m, args=TrainingArguments(**kw), train_dataset=mine, optimizers=(opt, None))
    t.train()
    return opt.recorded.numpy()


def grad_accelerate(seqs, cond, variant, rank, world):
    from accelerate import Accelerator
    mbs = rank_microbatches(seqs, cond["ranks"], rank, variant)
    acc = Accelerator(gradient_accumulation_steps=len(mbs), cpu=True)
    m = model()
    opt = _rec_opt()(m.parameters(), lr=0.0)
    m, opt_w = acc.prepare(m, opt)
    for mb in mbs:
        with acc.accumulate(m):
            out = m(**collate(mb))
            acc.backward(out.loss)
            opt_w.step()
            opt_w.zero_grad()
    return opt.recorded.numpy()


def grad_own_loop(seqs, cond, variant, rank, world):
    import torch
    import torch.distributed as dist
    m = model()
    opt = _rec_opt()(m.parameters(), lr=0.0)
    net = m
    if world > 1:
        if not dist.is_initialized():
            dist.init_process_group("gloo")
        net = torch.nn.parallel.DistributedDataParallel(m)
    mbs = rank_microbatches(seqs, cond["ranks"], rank, variant)
    for i, mb in enumerate(mbs):
        ctx = net.no_sync() if world > 1 and i < len(mbs) - 1 else contextlib.nullcontext()
        with ctx:
            (net(**collate(mb)).loss / len(mbs)).backward()
    opt.step()
    return opt.recorded.numpy()


IMPLS = {"hf_trainer": grad_hf_trainer, "accelerate_loop": grad_accelerate, "own_loop": grad_own_loop}


def cand_id(impl):
    if impl == "hf_trainer":
        import transformers
        return f"hf_trainer_{transformers.__version__}"
    if impl == "accelerate_loop":
        import accelerate
        return f"accelerate_loop_{accelerate.__version__}"
    import torch
    return f"own_loop_torch_{torch.__version__.split('+')[0]}"


# ------------------------------------------------------------------------------------------------ driver

def worker(a):
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    cond = {c["id"]: c for c in conditions()}[a.cond]
    g = IMPLS[a.impl](sequences(cond, a.seed, a.variant), cond, a.variant, rank, world)
    np.save(f"{a.out}_rank{rank}.npy", g)


def distributed(impl, cond, seed, variant):
    tmp = CACHE / "dist"
    tmp.mkdir(parents=True, exist_ok=True)
    stem = tmp / f"{impl}_{cond['id']}_{seed}_{variant}_{os.getpid()}"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2", __file__, "worker",
           "--impl", impl, "--cond", cond["id"], "--seed", str(seed), "--variant", variant, "--out", str(stem)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=dict(os.environ, OMP_NUM_THREADS="1"))
    if r.returncode != 0:
        raise RuntimeError(f"torchrun rc={r.returncode}: {r.stderr[-600:]}")
    g0, g1 = np.load(f"{stem}_rank0.npy"), np.load(f"{stem}_rank1.npy")
    for x in (0, 1):
        os.remove(f"{stem}_rank{x}.npy")
    return g0, bool(np.array_equal(g0, g1))


def run(impl):
    import spec_accumulation as acc
    CACHE.mkdir(parents=True, exist_ok=True)
    cid = cand_id(impl)
    path = CACHE / f"{cid}.pkl"
    res = pickle.loads(path.read_bytes()) if path.exists() else {}
    t0 = time.time()
    for cond in conditions():
        for seed in SEEDS:
            key = (cond["id"], seed)
            if key in res and all(res[key].get(v, {}).get("status") == "ok" for v in variants_for(cond, impl)):
                continue
            rec = {"f": None}
            seqs = sequences(cond, seed, "base")
            ptg = per_token_grads(seqs)
            rec["f"] = np.array([float(v) for v in acc.window_grad(ptg)])
            rec["tokens"] = [len(t) for t in ptg]
            for var in variants_for(cond, impl):
                t1 = time.time()
                try:
                    if cond["ranks"] == 1:
                        g, same = IMPLS[impl](sequences(cond, seed, var), cond, var, 0, 1), True
                    else:
                        g, same = distributed(impl, cond, seed, var)
                    rec[var] = {"status": "ok", "grad": g, "ranks_equal": same, "seconds": round(time.time() - t1, 1)}
                except Exception as exc:  # noqa: BLE001
                    rec[var] = {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[-500:]}"}
            res[key] = rec
            path.write_bytes(pickle.dumps(res))
            print(cid, key, {v: rec[v]["status"] for v in variants_for(cond, impl)}, f"{time.time() - t0:.0f} s", flush=True)
    meta_p = CACHE / f"{cid}.meta.json"
    meta_p.write_text(json.dumps({"candidate": cid, "impl": impl, "python": sys.executable, "seconds": round(time.time() - t0, 1)}))


# ------------------------------------------------------------------------------------------------ analysis

def judge(k, r):
    k, r = np.asarray(k, float), np.asarray(r, float)
    bad = np.abs(k - r) > TAU32 * (1 + np.abs(r))
    return int(bad.sum()), float(np.abs(k - r).max() / max(np.abs(r).max(), 1e-300))


def analyse():
    out = {"conditions": [c["id"] for c in conditions()], "candidates": {}}
    for p in sorted(CACHE.glob("*.pkl")):
        res = pickle.loads(p.read_bytes())
        name = p.stem
        rows, tally = [], {"F": [0, 0], "E": [0, 0], "P_split": [0, 0], "P_padfree": [0, 0], "P_rank": [0, 0]}
        for (cid, seed), rec in sorted(res.items()):
            b = rec.get("base", {})
            if b.get("status") != "ok":
                rows.append({"condition": cid, "seed": seed, "status": b.get("status"), "reason": b.get("reason")})
                continue
            row = {"condition": cid, "seed": seed, "tokens_per_sequence": rec["tokens"], "ranks_equal": b["ranks_equal"]}
            nf, rf = judge(b["grad"], rec["f"])
            row["F_violations"], row["F_rel"] = nf, rf
            row["ratio_K_over_f"] = float(b["grad"] @ rec["f"] / (rec["f"] @ rec["f"]))
            tally["F"][0] += 1
            tally["F"][1] += int(nf > 0)
            for var, key in (("no_ga", "E"), ("regroup", "P_split"), ("padfree", "P_padfree")):
                v = rec.get(var)
                if v and v["status"] == "ok":
                    n, rel = judge(b["grad"], v["grad"])
                    row[f"{key}_violations"], row[f"{key}_rel"] = n, rel
                    tally[key][0] += 1
                    tally[key][1] += int(n > 0)
                elif v:
                    row[f"{key}_error"] = v.get("reason")
            if cid.endswith("_r2"):
                one = res.get((cid.replace("_r2", "_r1"), seed), {}).get("base")
                if one and one.get("status") == "ok":
                    n, rel = judge(b["grad"], one["grad"])
                    row["P_rank_violations"], row["P_rank_rel"] = n, rel
                    tally["P_rank"][0] += 1
                    tally["P_rank"][1] += int(n > 0)
            rows.append(row)
        out["candidates"][name] = {"tally": {k: f"{v[1]}/{v[0]}" for k, v in tally.items()}, "rows": rows}
        print(f"{name:28s}", {k: f"{v[1]}/{v[0]}" for k, v in tally.items()})
        for r in rows:
            if r.get("status"):
                print("   ", r["condition"], r["seed"], r["status"], (r.get("reason") or "")[:200])
                continue
            if r["seed"] == 0:
                print(f"    {r['condition']:24s} F {r['F_violations']:4d} rel {r['F_rel']:.3g} K/f {r['ratio_K_over_f']:.4f}"
                      f" | E {r.get('E_violations')} | split {r.get('P_split_violations')} | padfree {r.get('P_padfree_violations', '-')}"
                      f" | rank {r.get('P_rank_violations', '-')} | ranks equal {r['ranks_equal']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyse", "worker"])
    ap.add_argument("--impl", choices=list(IMPLS))
    ap.add_argument("--cond")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--variant")
    ap.add_argument("--out")
    a = ap.parse_args()
    {"run": lambda: run(a.impl), "analyse": analyse, "worker": lambda: worker(a)}[a.stage]()
