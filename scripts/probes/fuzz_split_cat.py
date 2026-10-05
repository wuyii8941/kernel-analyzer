#!/usr/bin/env python3
"""Fuzz Inductor's pre-grad split / cat passes (split_cat_fx_passes, on by default): random compositions of split,
chunk, unbind, getitem, cat, stack, squeeze, unsqueeze, reshape, permute, narrow and clamp, compiled vs eager.
These ops only move data (clamp selects), so the compiled result must equal eager bitwise.

    python scripts/probes/fuzz_split_cat.py --programs 300 --seed 0
"""
import argparse
import random
import traceback

import torch


def gen_program(rng: random.Random):
    """A program as a list of steps over a value list; values are tensors or lists of tensors."""
    steps = []
    nvals = 1  # value 0: the input tensor (4-D)
    shapes = {0: [4, 6, 8, 5]}
    lists = {}
    for _ in range(rng.randint(3, 8)):
        tensors = [i for i in range(nvals) if i in shapes and len(shapes[i]) > 0]
        lst = [i for i in range(nvals) if i in lists]
        op = rng.choice(["split", "chunk", "unbind", "cat", "stack", "squeeze_unsqueeze", "reshape", "permute",
                         "narrow", "clamp", "getitem"] if lst else
                        ["split", "chunk", "unbind", "reshape", "permute", "narrow", "clamp", "squeeze_unsqueeze"])
        if op in ("split", "chunk", "unbind"):
            src = rng.choice(tensors)
            shp = shapes[src]
            d = rng.randrange(len(shp))
            dd = d - len(shp) if rng.random() < 0.5 else d
            if op == "split":
                n = shp[d]
                if rng.random() < 0.5:
                    sz = rng.randint(1, n)
                    parts = [sz] * (n // sz) + ([n % sz] if n % sz else [])
                    arg = sz
                else:
                    cuts = sorted(rng.sample(range(1, n), min(n - 1, rng.randint(0, 3)))) if n > 1 else []
                    bounds = [0] + cuts + [n]
                    parts = [b - a for a, b in zip(bounds, bounds[1:])]
                    arg = parts
                steps.append(("split", src, arg, dd))
                lists[nvals] = [shp[:d] + [p] + shp[d + 1:] for p in parts]
            elif op == "chunk":
                k = rng.randint(1, max(1, shp[d]))
                n = shp[d]
                c = -(-n // k)
                parts = [c] * (n // c) + ([n % c] if n % c else [])
                steps.append(("chunk", src, k, dd))
                lists[nvals] = [shp[:d] + [p] + shp[d + 1:] for p in parts]
            else:
                steps.append(("unbind", src, None, dd))
                lists[nvals] = [shp[:d] + shp[d + 1:]] * shp[d]
            nvals += 1
        elif op == "getitem":
            src = rng.choice(lst)
            i = rng.randrange(len(lists[src]))
            steps.append(("getitem", src, i, None))
            shapes[nvals] = list(lists[src][i])
            nvals += 1
        elif op in ("cat", "stack"):
            src = rng.choice(lst)
            items = lists[src]
            idx = list(range(len(items)))
            rng.shuffle(idx)
            idx = idx[: rng.randint(1, len(idx))]
            sel = [items[i] for i in idx]
            nd = len(sel[0])
            if op == "cat":
                # cat needs equal shapes except along d
                ds = [d for d in range(nd) if all(s[:d] + s[d + 1:] == sel[0][:d] + sel[0][d + 1:] for s in sel)]
                if not ds:
                    continue
                d = rng.choice(ds)
                steps.append(("cat", src, idx, d - nd if rng.random() < 0.5 else d))
                shapes[nvals] = sel[0][:d] + [sum(s[d] for s in sel)] + sel[0][d + 1:]
            else:
                if any(s != sel[0] for s in sel):
                    continue
                d = rng.randint(0, nd)
                steps.append(("stack", src, idx, d - nd - 1 if rng.random() < 0.5 else d))
                shapes[nvals] = sel[0][:d] + [len(sel)] + sel[0][d:]
            nvals += 1
        elif op == "squeeze_unsqueeze":
            src = rng.choice(tensors)
            shp = shapes[src]
            d = rng.randint(0, len(shp))
            steps.append(("unsqueeze_squeeze", src, None, d))
            shapes[nvals] = list(shp)
            nvals += 1
        elif op == "reshape":
            src = rng.choice(tensors)
            shp = shapes[src]
            steps.append(("reshape", src, None, None))
            shapes[nvals] = [int(torch.tensor(shp).prod().item())]
            nvals += 1
        elif op == "permute":
            src = rng.choice(tensors)
            perm = list(range(len(shapes[src])))
            rng.shuffle(perm)
            steps.append(("permute", src, perm, None))
            shapes[nvals] = [shapes[src][p] for p in perm]
            nvals += 1
        elif op == "narrow":
            src = rng.choice(tensors)
            shp = shapes[src]
            d = rng.randrange(len(shp))
            start = rng.randrange(shp[d])
            length = rng.randint(1, shp[d] - start)
            steps.append(("narrow", src, (start, length), d))
            shapes[nvals] = shp[:d] + [length] + shp[d + 1:]
            nvals += 1
        else:  # clamp with scalar or keyword bounds
            src = rng.choice(tensors)
            lo, hi = sorted([rng.uniform(-1, 1), rng.uniform(-1, 1)])
            keep = rng.choice([(True, True), (True, False), (False, True)])  # at least one bound
            steps.append(("clamp", src, (lo if keep[0] else None, hi if keep[1] else None), rng.random() < 0.5))
            shapes[nvals] = list(shapes[src])
            nvals += 1
    outs = [i for i in range(nvals) if i in shapes][-3:]
    return steps, outs


def run(steps, outs, x):
    vals = [x]
    for op, src, arg, d in steps:
        v = vals[src]
        if op == "split":
            vals.append(list(torch.split(v, arg, dim=d)))
        elif op == "chunk":
            vals.append(list(torch.chunk(v, arg, dim=d)))
        elif op == "unbind":
            vals.append(list(torch.unbind(v, dim=d)))
        elif op == "getitem":
            vals.append(v[arg])
        elif op == "cat":
            vals.append(torch.cat([v[i] for i in arg], dim=d))
        elif op == "stack":
            vals.append(torch.stack([v[i] for i in arg], dim=d))
        elif op == "unsqueeze_squeeze":
            vals.append(torch.squeeze(torch.unsqueeze(v, d), d))
        elif op == "reshape":
            vals.append(v.reshape(-1))
        elif op == "permute":
            vals.append(v.permute(*arg))
        elif op == "narrow":
            vals.append(torch.narrow(v, d, arg[0], arg[1]))
        else:
            lo, hi = arg
            vals.append(torch.clamp(v, min=lo, max=hi) if d else torch.clamp(v, lo, hi))
    return [vals[i].reshape(-1) for i in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    x = torch.randn(4, 6, 8, 5, device="cuda")
    bad = 0
    for p in range(a.programs):
        steps, outs = gen_program(rng)
        fn = lambda t, s=steps, o=outs: run(s, o, t)  # noqa: E731
        torch._dynamo.reset()
        try:
            e = fn(x)
            c = torch.compile(fn)(x)
            ok = len(e) == len(c) and all(a_.shape == b_.shape and torch.equal(a_, b_) for a_, b_ in zip(e, c))
        except Exception as ex:  # noqa: BLE001
            ok = False
            print(f"program {p}: EXCEPTION {type(ex).__name__}: {str(ex).splitlines()[0][:100]}\n  steps={steps} outs={outs}")
            continue
        if not ok:
            bad += 1
            print(f"program {p}: MISMATCH\n  steps={steps} outs={outs}")
    print(f"{a.programs} programs, {bad} mismatches")


if __name__ == "__main__":
    main()
