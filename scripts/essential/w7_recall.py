#!/usr/bin/env python3
"""W7 known-positive recall (docs/protocol_essential_bugs_20261007.md section 8): can each group E / F / P / FR see it?

    python scripts/essential/w7_recall.py b016        # ka_main, CUDA
    python scripts/essential/w7_recall.py b014        # ka_main, CUDA
    python scripts/essential/w7_recall.py hf-construct
    /data1/tzh/envs/hf_ga_4452/bin/python scripts/essential/w7_recall.py hf-measure   # and hf_ga_4460, liger (4.57.3)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (sets sys.path for the specs and src)

ROOT = common.ROOT
OUT = ROOT / "results/essential/phase1/w7"


def _save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str) + "\n")


# ------------------------------------------------------------------------------------------------ B016

def b016():
    import mpmath as mp
    import torch

    import spec_index_scatter as ix

    mp.mp.dps = 50
    dev = "cuda"
    res = {"torch": torch.__version__, "cases": {}}
    Ns = {0: 5, 1: 30, 2: 1000}

    # case 1: counting, single-element target, computed source (x + 1 with x = 0 -> ones)
    def count(x, idx):
        return {"out": torch.zeros(1, device=dev).scatter_add(0, idx, x + 1)}

    # case 2: multi-element target, in-graph constant index, computed source cos(x)
    def const_index(x):
        idx = torch.arange(x.numel(), device=dev) // 1000
        return {"out": torch.zeros(4, device=dev).index_add(0, idx, x.cos())}

    # case 3: single-element scatter_reduce(mean, include_self=True), backward
    def mean_bwd(t, s, idx):
        t = t.detach().requires_grad_(True)
        s = s.detach().requires_grad_(True)
        out = t.scatter_reduce(0, idx, s, "mean", include_self=True)
        out.sum().backward()
        return {"grad_self": t.grad, "grad_src": s.grad}

    def inputs(case, seed):
        g = torch.Generator(device="cpu").manual_seed(1000 + seed)
        N = Ns[seed]
        if case == "count":
            return dict(x=torch.zeros(N, device=dev), idx=torch.zeros(N, dtype=torch.long, device=dev))
        if case == "const_index":
            return dict(x=torch.randn(30 if seed == 0 else N, generator=g).to(dev))
        k = 3                                   # one shape for all seeds (the tool stacks seeds)
        return dict(t=torch.randn(1, generator=g).to(dev), s=torch.randn(k, generator=g).to(dev),
                    idx=torch.zeros(k, dtype=torch.long, device=dev))

    def spec(case, inp):
        """f from the independent specification (exact), as float64 enclosures."""
        if case == "count":
            N = inp["x"].numel()
            f = ix.scatter_reduce([0], 0, [0] * N, [1] * N, "sum")          # x + 1 = 1 exactly
            return {"out": common.enclose_array(f, (1,))}
        if case == "const_index":
            xs = inp["x"].double().cpu().tolist()
            cos = [Fraction(mp.nstr(mp.cos(mp.mpf(v)), 50)) for v in xs]   # 50 digits: far inside tau_32
            f = ix.index_add([0, 0, 0, 0], 0, [i // 1000 for i in range(len(xs))], cos)
            return {"out": common.enclose_array(f, (4,))}
        k = inp["s"].numel()
        g = Fraction(1, 1 + k)                                               # d mean / d each participant
        return {"grad_self": common.enclose_array([g], (1,)), "grad_src": common.enclose_array([g] * k, (k,))}

    fns = {"count": count, "const_index": const_index, "mean_bwd": mean_bwd}
    for case, fn in fns.items():
        torch._dynamo.reset()
        comp = torch.compile(fn, dynamic=False)
        rows = []
        for seed in (0, 1, 2):
            inp = inputs(case, seed)
            ke = {k: v.double().cpu().numpy() for k, v in fn(**inp).items()}
            try:
                kc = {k: v.double().cpu().numpy() for k, v in comp(**inp).items()}
            except Exception as exc:  # noqa: BLE001
                kc = {"error": repr(exc)[:300]}
            f = spec(case, inp)
            row = {"seed": seed, "size": int(next(iter(inp.values())).numel())}
            for name, (lo, hi) in f.items():
                for cand, k in (("eager_cuda_fp32", ke), ("inductor_cuda_fp32", kc)):
                    if name not in k:
                        row[f"{cand}/{name}"] = {"error": k.get("error")}
                        continue
                    dev_, _ = common.deviation(k[name], lo, hi, "float32")
                    row[f"{cand}/{name}"] = {"K": k[name].tolist()[:6], "deviates_from_f": bool(dev_.any())}
                row[f"f/{name}"] = lo.tolist()[:6]
                if name in ke and name in kc:     # E: compiled against eager, same device and dtype
                    e_dev, _ = common.deviation(kc[name], ke[name], ke[name], "float32")
                    row[f"E/{name}"] = bool(e_dev.any())
            # P: reference-free properties
            if case in ("count", "const_index"):
                n = inp["x"].numel()
                ones = dict(inp, x=torch.zeros_like(inp["x"]))           # x = 0 -> every contribution is 1
                for cand, g in (("eager_cuda_fp32", fn), ("inductor_cuda_fp32", comp)):
                    try:
                        got = g(**ones)["out"].double().cpu().tolist()
                    except Exception as exc:  # noqa: BLE001
                        row[f"P_counting/{cand}"] = {"error": repr(exc)[:200]}
                        continue
                    expect = [ix.prop_counting(n, 0)] + ([0] * 3 if case == "const_index" else [])
                    row[f"P_counting/{cand}"] = {"got": got, "expected": [float(v) for v in expect],
                                                 "violated": got != [float(v) for v in expect]}
            else:
                # backward consistency <B(v), u> = <v, J u>; J u from the spec (mean is linear in (self, src))
                k = inp["s"].numel()
                g = torch.Generator(device="cpu").manual_seed(77 + seed)
                ut, us = torch.randint(-3, 4, (1,), generator=g).double(), torch.randint(-3, 4, (k,), generator=g).double()
                Ju = ix.scatter_reduce([Fraction(int(ut[0]))], 0, [0] * k, [Fraction(int(v)) for v in us], "mean")[0]
                for cand, gfn in (("eager_cuda_fp32", fn), ("inductor_cuda_fp32", comp)):
                    try:
                        gr = gfn(**inp)
                    except Exception as exc:  # noqa: BLE001
                        row[f"P_adjoint/{cand}"] = {"error": repr(exc)[:200]}
                        continue
                    lhs = float((gr["grad_self"].double().cpu() * ut).sum() + (gr["grad_src"].double().cpu() * us).sum())
                    row[f"P_adjoint/{cand}"] = {"<B(v),u>": lhs, "<v,Ju>": float(Ju),
                                                "violated": abs(lhs - float(Ju)) > common.TAU["float32"] * (1 + abs(float(Ju)))}
            rows.append(row)
        # FR: the compiled kernels through the tool's mode B with f from the specification
        try:
            torch._dynamo.reset()
            comp_fr = torch.compile(fn, dynamic=False)
            rep, keep = common.fr_run(f"B016/{case}", lambda: None, lambda s: inputs(case, s),
                                      lambda inp: comp_fr(**inp), lambda inp: spec(case, inp))
            fr = {}
            for name, per in keep.items():
                cert = []
                for p, seed in zip(per, (0, 1, 2)):
                    lo, hi = spec(case, inputs(case, seed))[name]
                    ok = p["ok"]
                    s_lo, s_hi = p["r_lo"] - hi.reshape(-1), p["r_hi"] - lo.reshape(-1)
                    n_lo, n_hi = p["k"] - p["r_hi"], p["k"] - p["r_lo"]
                    cert.append({"seed": seed, "K": p["k"].tolist()[:4], "K_R": [p["r_lo"].tolist()[:4], p["r_hi"].tolist()[:4]],
                                 "e_sem_excludes_zero": bool(((s_lo > 0) | (s_hi < 0))[ok].any()),
                                 "e_num_excludes_zero": bool(((n_lo > 0) | (n_hi < 0))[ok].any()),
                                 "ok_elements": int(ok.sum()), "elements": int(ok.size)})
                fr[name] = cert
            fr_summary = {"outputs": fr, "outputs_not_written_by_triton": rep["outputs_not_written_by_triton"],
                          "ttir_coverage_complete": rep["ttir_coverage_complete"],
                          "launches": [l["kernel"] for l in rep["launches"]],
                          "float_atomics": [l["float_atomics"] for l in rep["launches"]]}
        except Exception as exc:  # noqa: BLE001
            fr_summary = {"error": repr(exc)[:400]}
        res["cases"][case] = {"rows": rows, "FR": fr_summary}
        print(case, json.dumps(rows[0], default=str)[:400], "\n FR", json.dumps(fr_summary, default=str)[:400], flush=True)
    _save("b016", res)


# ------------------------------------------------------------------------------------------------ B014 (avg_pool part)

def b014():
    """1-element sequence arguments of avg_pool2d/3d: Inductor's backward lowering asserts on their length.  Records
    the crash per group and which POOL-A1 reading the forward follows (the OpInfo sample has overhang windows)."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    import spec_pooling as sp

    dev = "cuda"
    cases = {
        "avgpool2d_k3_pad(1,)": dict(nd=2, size=(7, 7), kernel=3, stride=None, padding=(1,), ceil_mode=False, cip=True),
        "opinfo_avg_pool2d_5": dict(nd=2, size=(7, 7), kernel=(4, 4), stride=(2, 2), padding=(2,), ceil_mode=True, cip=True),
        "avgpool3d_k3_pad(1,)": dict(nd=3, size=(5, 5, 5), kernel=3, stride=None, padding=(1,), ceil_mode=False, cip=True),
    }
    res = {"torch": torch.__version__, "cases": {}}
    for name, c in cases.items():
        pool = F.avg_pool2d if c["nd"] == 2 else F.avg_pool3d
        nd = c["nd"]

        def full(v):
            return None if v is None else (list(v) * nd if isinstance(v, (tuple, list)) and len(v) == 1 else v)

        def _e_dev(a, b):     # E group under the float32 contract (protocol section 6), not bitwise
            b = b.double().cpu().numpy()
            return bool(common.deviation(a.double().cpu().numpy(), b, b, "float32")[0].any())

        def fn(x, v):
            x = x.detach().requires_grad_(True)
            y = pool(x, c["kernel"], c["stride"], c["padding"], ceil_mode=c["ceil_mode"], count_include_pad=c["cip"])
            (y * v).sum().backward()
            return y.detach(), x.grad
        rows = []
        for seed in (0, 1, 2):
            g = torch.Generator(device="cpu").manual_seed(500 + seed)
            x = torch.randint(-5, 6, (2,) + c["size"], generator=g).float()
            y0 = pool(x, c["kernel"], c["stride"], c["padding"], ceil_mode=c["ceil_mode"], count_include_pad=c["cip"])
            v = torch.randint(-3, 4, y0.shape, generator=g).float()
            row = {"seed": seed}
            ye, ge = fn(x.to(dev), v.to(dev))
            torch._dynamo.reset()
            try:
                yc, gc = torch.compile(fn, dynamic=False)(x.to(dev), v.to(dev))
                row["inductor"] = {"forward_E_deviates": _e_dev(yc, ye), "grad_E_deviates": _e_dev(gc, ge),
                                   "max_abs_forward_vs_eager": float((yc - ye).abs().max())}
            except Exception as exc:  # noqa: BLE001
                row["inductor"] = {"error": type(exc).__name__ + ": " + str(exc).splitlines()[0][:200]}
                try:   # forward alone compiles (the defect is in the backward lowering)
                    torch._dynamo.reset()
                    yf = torch.compile(lambda x: pool(x, c["kernel"], c["stride"], c["padding"], ceil_mode=c["ceil_mode"],
                                                      count_include_pad=c["cip"]), dynamic=False)(x.to(dev))
                    row["inductor"]["forward_only_E_deviates"] = _e_dev(yf, ye)
                    row["inductor"]["forward_only_max_abs_vs_eager"] = float((yf - ye).abs().max())
                except Exception as exc2:  # noqa: BLE001
                    row["inductor"]["forward_only_error"] = type(exc2).__name__
            xs = x.tolist()
            kw = dict(kernel=full(c["kernel"]), stride=full(c["stride"]), padding=full(c["padding"]), ceil_mode=c["ceil_mode"],
                      count_include_pad=c["cip"])
            for rd in ("R2", "R1"):
                f = sp.avg_pool(xs, reading=rd, **kw)
                lo, hi = common.enclose_array(f, tuple(ye.shape))
                dev_, _ = common.deviation(ye.double().cpu().numpy(), lo, hi, "float32")
                gf = sp.avg_pool_backward(v.tolist(), [2] + list(c["size"]), reading=rd, **kw)
                glo, ghi = common.enclose_array(gf, tuple(ge.shape))
                gdev, _ = common.deviation(ge.double().cpu().numpy(), glo, ghi, "float32")
                row[f"eager_vs_spec_{rd}"] = {"forward_deviates": bool(dev_.any()), "grad_deviates": bool(gdev.any()),
                                              "forward_elements_deviating": int(dev_.sum())}
            rows.append(row)
        res["cases"][name] = rows
        print(name, json.dumps(rows[0])[:600], flush=True)
    _save("b014", res)


# ------------------------------------------------------------------------------------------------ HF gradient accumulation

def hf_construct():
    """Main protocol section 7: two implementations of the mean of micro-batch means agree with each other (E sees 0)
    and both differ from the window requirement by -1/4 (token counts 1 and 3, gradient sums -1/2 and 3/2).  A
    construction checking the program logic of the three-way comparison, not a measurement of any HF version."""
    import torch

    import spec_accumulation as acc

    per_token = [[Fraction(-1, 2)], [Fraction(1, 2)] * 3]               # micro-batch gradient sums -1/2 and 3/2
    f = acc.window_grad(per_token)
    wrong_spec = acc.wrong_variant_mean_of_means(per_token)

    def loop_a(batches):                                                # mean per micro-batch, divided by k
        return sum(torch.tensor([float(v) for v in b], dtype=torch.float64).mean() for b in batches) / len(batches)

    def loop_b(batches):                                                # the same normalisation written differently
        return torch.stack([torch.tensor([float(v) for v in b], dtype=torch.float64).sum() / len(b) for b in batches]).mean()

    ka, kb = float(loop_a(per_token)), float(loop_b(per_token))
    res = {"f_window": str(f), "labelled_error_variant": str(wrong_spec), "impl_a": ka, "impl_b": kb,
           "E_a_minus_b": ka - kb, "F_a_minus_f": ka - float(f), "F_b_minus_f": kb - float(f),
           "P_split_invariance_a": [float(loop_a(per_token)), float(loop_a([[per_token[0][0], per_token[1][0]], per_token[1][1:]]))],
           "note": "construction of the comparison logic; not a measurement of any transformers version"}
    print(json.dumps(res))
    _save("hf_construct", res)


def hf_measure():
    """The real Trainer: tiny random Llama in float64 on CPU, k = 2 micro-batches of 2 sequences with unequal token
    counts (labels -100 on padding), one optimizer step with lr = 0 and no clipping; the accumulated gradient is
    recorded in the optimizer's step().  f = spec_accumulation.window_grad over per-token gradients (each the float64
    autograd gradient of one token's loss: the inputs of the requirement, not part of it)."""
    import numpy as np
    import torch
    import transformers
    from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

    import spec_accumulation as acc

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=32)
    base = LlamaForCausalLM(cfg).double()
    state = {k: v.clone() for k, v in base.state_dict().items()}
    g = torch.Generator().manual_seed(1)
    PAD = 0

    def seqs(lengths):
        return [torch.randint(1, cfg.vocab_size, (n,), generator=g) for n in lengths]

    patterns = {"unequal": [3, 4, 9, 12], "equal": [6, 6, 6, 6], "one_empty_microbatch": [1, 1, 8, 10]}
    data = {k: seqs(v) for k, v in patterns.items()}

    def collate(batch):
        L = max(len(s) for s in batch)
        ids = torch.full((len(batch), L), PAD, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, s in enumerate(batch):
            ids[i, :len(s)], lab[i, :len(s)], att[i, :len(s)] = s, s, 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    class Rec(torch.optim.SGD):
        def step(self, closure=None):
            self.recorded = torch.cat([p.grad.detach().reshape(-1).clone() for p in self.param_groups[0]["params"]])
            return super().step(closure)

    def trainer_grad(sequences, bs, k):
        model = LlamaForCausalLM(cfg).double()
        model.load_state_dict(state)
        opt = Rec(model.parameters(), lr=0.0)

        class T(Trainer):
            def get_train_dataloader(self):
                return torch.utils.data.DataLoader(sequences, batch_size=bs, shuffle=False, collate_fn=collate)

        args = TrainingArguments(output_dir=str(ROOT / ".cache/essential/hf_trainer"), per_device_train_batch_size=bs,
                                 gradient_accumulation_steps=k, max_steps=1, learning_rate=0.0, max_grad_norm=0.0,
                                 report_to=[], use_cpu=True, save_strategy="no", lr_scheduler_type="constant",
                                 remove_unused_columns=False, logging_steps=1, seed=0, dataloader_num_workers=0)
        t = T(model=model, args=args, train_dataset=sequences, optimizers=(opt, None))
        t.train()
        return opt.recorded.numpy()

    def per_token_grads(sequences, bs):
        """per micro-batch, the list of per-token gradient vectors (float64 autograd of each token's loss)."""
        model = LlamaForCausalLM(cfg).double()
        model.load_state_dict(state)
        out = []
        for i in range(0, len(sequences), bs):
            b = collate(sequences[i:i + bs])
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            lab = b["labels"][:, 1:]
            lg = logits[:, :-1]
            toks = []
            for r, c in zip(*torch.nonzero(lab != -100, as_tuple=True)):
                model.zero_grad()
                torch.nn.functional.cross_entropy(lg[r, c][None], lab[r, c][None]).backward(retain_graph=True)
                toks.append([Fraction(float(v)) for v in torch.cat([p.grad.reshape(-1) for p in model.parameters()])])
            out.append(toks)
        return out

    res = {"transformers": transformers.__version__, "torch": torch.__version__, "patterns": {}}
    for name, sequences in data.items():
        bs, k = 2, 2
        ptg = per_token_grads(sequences, bs)
        counts = [len(b) for b in ptg]
        f = np.array([float(v) for v in acc.window_grad(ptg)])
        means = [np.array([float(sum(col, Fraction(0)) / len(b)) for col in zip(*b)]) for b in ptg if b]
        wrong = sum(means) / len(means)                                   # the labelled error variant
        K = trainer_grad(sequences, bs, k)
        K_nogA = trainer_grad(sequences, bs * k, 1)                       # E analogue: no accumulation
        regroup = [sequences[0], sequences[2], sequences[1], sequences[3]]
        K_regroup = trainer_grad(regroup, bs, k)                          # P: split invariance

        def rel(a, b):
            return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))

        # the Llama head computes the loss on logits.float(): float32 even for a float64 model, so the Trainer is a
        # float32 candidate and the float32 contract applies (protocol section 6)
        tau = common.TAU["float32"]
        wrong_div_k = sum(means) / k                                      # sum of non-empty means / k (empty -> 0)

        res["patterns"][name] = {
            "valid_tokens_per_microbatch": counts,
            "F_rel_dev_from_window": rel(K, f), "F_deviates": rel(K, f) > tau, "contract": "float32 (tau_32)",
            "matches_labelled_error_variant_mean_of_nonempty_means": rel(K, wrong) <= tau,
            "matches_sum_of_means_over_k_empty_as_zero": rel(K, wrong_div_k) <= tau,
            "ratio_K_over_f_least_squares": float(K @ f / (f @ f)),
            "E_rel_dev_accumulated_vs_not": rel(K, K_nogA), "E_deviates": rel(K, K_nogA) > tau,
            "noGA_F_rel_dev": rel(K_nogA, f),
            "P_rel_dev_regrouped": rel(K, K_regroup), "P_violated": rel(K, K_regroup) > tau,
        }
        print(transformers.__version__, name, json.dumps(res["patterns"][name]), flush=True)
    _save(f"hf_measure_{transformers.__version__}", res)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    {"b016": b016, "b014": b014, "hf-construct": hf_construct, "hf-measure": hf_measure}[sys.argv[1]]()
