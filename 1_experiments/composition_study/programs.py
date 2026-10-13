"""Random Triton programs over the composition forms, for the bias-prediction validation.

A program is drawn from a small grammar: an elementwise chain of catalogued operations (IEEE arithmetic, approximate
functions, conversions, continuous and discontinuous selects) placed in one form -- elementwise (C1 / C2), row
reduction (C3), scan (C4), loop accumulation (C8), dot (C15) -- with input dtypes, working dtypes and input
distributions drawn too.  ``generate(seed, n)`` is deterministic in the seed; the development set uses seed
"dev-<k>", the held-out set a seed fixed by the frozen predictor (predictor_protocol.md).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# chain steps: name -> (source with {v} the current value and {w} a second input, allowed working dtypes)
LOWP = ("f32", "bf16", "f16")
STEPS = {
    "add_w": ("{v} + {w}", LOWP), "sub_w": ("{v} - {w}", LOWP), "mul_w": ("{v} * {w}", LOWP),
    "fma_w": ("{v} * {w} + {v}", LOWP), "scale": ("{v} * 0.71875 + 0.125", LOWP), "square": ("{v} * {v}", LOWP),
    "abs": ("tl.abs({v})", LOWP), "relu": ("tl.maximum({v}, 0.0)", LOWP),
    "relu_where": ("tl.where({v} > 0, {v}, 0.0)", LOWP), "leaky": ("tl.where({v} > 0, {v}, {v} * 0.125)", LOWP),
    "step": ("tl.where({v} > 0.25, {v} + 1.0, {v})", ("f32",)),
    "floor": ("tl.floor({v} * 4.0) * 0.25", ("f32",)),
    "div_w": ("{v} / (tl.abs({w}) + 0.5)", ("f32",)),
    "exp": ("tl.exp(tl.minimum({v}, 8.0))", ("f32",)), "log": ("tl.log(tl.abs({v}) + 0.5)", ("f32",)),
    "sqrt": ("tl.sqrt(tl.abs({v}) + 0.25)", ("f32",)), "rsqrt": ("tl.rsqrt(tl.abs({v}) + 0.25)", ("f32",)),
    "sin": ("tl.sin({v})", ("f32",)), "cos": ("tl.cos({v})", ("f32",)),
    "sigmoid": ("1.0 / (1.0 + tl.exp(-{v}))", ("f32",)),
    "tanh": ("libdevice.tanh({v})", ("f32",)), "erf": ("libdevice.erf({v})", ("f32",)),
    "to_bf16": ("{v}.to(tl.bfloat16).to(tl.float32)", ("f32",)),
    "to_f16": ("{v}.to(tl.float16).to(tl.float32)", ("f32",)),
    "to_e5m2": ("{v}.to(tl.float8e5).to(tl.float32)", ("f32",)),
    "clamp": ("tl.clamp({v}, -2.0, 2.0)", ("f32",)),
}
DISTS = ("normal", "uniform01", "wide", "bf16valued")
TL = {"f32": "tl.float32", "bf16": "tl.bfloat16", "f16": "tl.float16"}


def _chain(rng, wd, n_steps, has_w):
    names = [s for s, (_, dts) in STEPS.items() if wd in dts and (has_w or "{w}" not in STEPS[s][0])]
    return [rng.choice(names) for _ in range(n_steps)]


def _body(steps, v="v", w="w", indent="    "):
    lines = []
    for i, s in enumerate(steps):
        lines.append(f"{indent}{v} = {STEPS[s][0].format(v=v, w=w)}")
    return lines


def generate(seed: str, n: int) -> list:
    rng = random.Random(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "little"))
    specs = []
    for i in range(n):
        form = rng.choice(["elementwise", "elementwise", "reduce", "scan", "loop", "dot", "dot"])
        name = f"p{i:03d}"
        if form == "dot":
            specs.append(_dot(rng, name))
            continue
        in_dt = rng.choice(["f32", "f32", "bf16", "f16"])
        wd = rng.choice(["f32", "f32", "f32", in_dt if in_dt != "f32" else "bf16"])
        has_w = rng.random() < 0.6
        steps = _chain(rng, wd, rng.randint(2, 6), has_w)
        dists = [rng.choice(DISTS), rng.choice(DISTS)]
        spec = {"name": name, "form": form, "in_dtype": in_dt, "work_dtype": wd, "steps": steps, "has_w": has_w,
                "dists": dists}
        if form == "elementwise":
            spec.update(B=1024, grid=8, out_dtype=rng.choice([wd, "f32"]))
            spec["source"] = _src_elementwise(spec)
        elif form == "reduce":
            spec.update(R=16, C=256, grid=8, acc_dtype=rng.choice(["f32", wd]))
            spec["source"] = _src_reduce(spec)
        elif form == "scan":
            spec.update(B=1024, grid=4, acc_dtype=rng.choice(["f32", wd]))
            spec["source"] = _src_scan(spec)
        else:
            spec.update(B=1024, T=rng.choice([16, 64]), grid=4, acc_dtype=rng.choice(["f32", "bf16", "f16"]))
            spec["source"] = _src_loop(spec)
        specs.append(spec)
    return specs


def _loads(spec, idx, indent="    "):
    wd = TL[spec["work_dtype"]]
    lines = [f"{indent}v = tl.load(x_ptr + {idx}).to({wd})"]
    if spec["has_w"]:
        lines.append(f"{indent}w = tl.load(w_ptr + {idx}).to({wd})")
    return lines


def _src_elementwise(s):
    body = [f"@triton.jit", f"def {s['name']}(x_ptr, w_ptr, y_ptr, B: tl.constexpr):",
            "    r = tl.program_id(0) * B + tl.arange(0, B)"]
    body += _loads(s, "r") + _body(s["steps"])
    body.append(f"    tl.store(y_ptr + r, v.to({TL[s['out_dtype']]}))")
    return "\n".join(body)


def _src_reduce(s):
    body = [f"@triton.jit", f"def {s['name']}(x_ptr, w_ptr, y_ptr, R: tl.constexpr, C: tl.constexpr):",
            "    rows = tl.program_id(0) * R + tl.arange(0, R)",
            "    idx = rows[:, None] * C + tl.arange(0, C)[None, :]"]
    body += _loads(s, "idx") + _body(s["steps"])
    body.append(f"    t = tl.sum(v.to({TL[s['acc_dtype']]}), axis=1)")
    body.append("    tl.store(y_ptr + rows, t.to(tl.float32))")
    return "\n".join(body)


def _src_scan(s):
    body = [f"@triton.jit", f"def {s['name']}(x_ptr, w_ptr, y_ptr, B: tl.constexpr):",
            "    r = tl.program_id(0) * B + tl.arange(0, B)"]
    body += _loads(s, "r") + _body(s["steps"])
    body.append(f"    t = tl.cumsum(v.to({TL[s['acc_dtype']]}), axis=0)")
    body.append("    tl.store(y_ptr + r, t.to(tl.float32))")
    return "\n".join(body)


def _src_loop(s):
    body = [f"@triton.jit", f"def {s['name']}(x_ptr, w_ptr, y_ptr, B: tl.constexpr, T: tl.constexpr):",
            "    r = tl.program_id(0) * B + tl.arange(0, B)",
            f"    acc = tl.zeros((B,), dtype={TL[s['acc_dtype']]})",
            "    for t in range(T):",
            "        idx = (tl.program_id(0) * T + t) * B + tl.arange(0, B)"]
    body += _loads(s, "idx", "        ") + _body(s["steps"], indent="        ")
    body.append(f"        acc += v.to({TL[s['acc_dtype']]})")
    body.append("    tl.store(y_ptr + r, acc.to(tl.float32))")
    return "\n".join(body)


def _dot(rng, name):
    dt = rng.choice(["f32", "f32", "f32", "bf16", "f16"])
    prec = rng.choice(["tf32", "tf32", "ieee", "tf32_rna"]) if dt == "f32" else "tf32"
    pre = rng.choice(["none", "none", "scale", "relu", "to_bf16"]) if dt == "f32" else "none"
    post = rng.choice(["none", "bias", "relu", "sigmoid"])
    spec = {"name": name, "form": "dot", "in_dtype": dt, "precision": prec, "pre": pre, "post": post,
            "dists": [rng.choice(["normal", "uniform01", "bf16valued"]) for _ in range(2)],
            "M": 64, "N": 64, "K": rng.choice([128, 256, 512]), "BK": 32, "grid": 4}
    pre_src = {"none": [], "scale": ["        a = a * 0.71875"], "relu": ["        a = tl.maximum(a, 0.0)"],
               "to_bf16": ["        a = a.to(tl.bfloat16).to(tl.float32)"]}[pre]
    if prec == "tf32_rna":
        pre_src = pre_src + ['        a = tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", "=r,r", [a], '
                             'dtype=tl.float32, is_pure=True, pack=1)']
    dot_prec = "ieee" if prec == "ieee" else "tf32"
    post_src = {"none": [], "bias": ["    acc = acc + tl.load(w_ptr + rn)[None, :]"],
                "relu": ["    acc = tl.maximum(acc, 0.0)"], "sigmoid": ["    acc = 1.0 / (1.0 + tl.exp(-acc))"]}[post]
    body = [f"@triton.jit",
            f"def {name}(x_ptr, b_ptr, w_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, "
            f"BK: tl.constexpr):",
            "    pm = tl.program_id(0) // 2", "    pn = tl.program_id(0) % 2",
            "    rm = pm * M + tl.arange(0, M)", "    rn = pn * N + tl.arange(0, N)",
            "    acc = tl.zeros((M, N), dtype=tl.float32)",
            "    for k0 in range(0, K, BK):", "        rk = k0 + tl.arange(0, BK)",
            "        a = tl.load(x_ptr + rm[:, None] * K + rk[None, :])",
            "        b = tl.load(b_ptr + rk[:, None] * (2 * N) + rn[None, :])"]
    body += pre_src
    body.append(f'        acc = tl.dot(a, b, acc, input_precision="{dot_prec}")')
    body += post_src
    body.append("    tl.store(y_ptr + rm[:, None] * (2 * N) + rn[None, :], acc)")
    spec["source"] = "\n".join(body)
    return spec


def materialize(specs: list, path: Path):
    """Write the kernels to one module file and import it (triton.jit needs the source on disk)."""
    head = ["import triton", "import triton.language as tl", "from triton.language.extra import libdevice", ""]
    path = Path(path)
    path.write_text("\n".join(head + [s["source"] + "\n" for s in specs]))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- inputs and launches

def draw(dist, n, dtype, gen):
    import torch
    if dist == "normal":
        v = torch.randn(n, device="cuda", generator=gen)
    elif dist == "uniform01":
        v = torch.rand(n, device="cuda", generator=gen)
    elif dist == "wide":
        v = torch.exp((torch.rand(n, device="cuda", generator=gen) - 0.5) * 8) * \
            torch.sign(torch.randn(n, device="cuda", generator=gen))
    else:                                       # bf16-valued
        v = torch.randn(n, device="cuda", generator=gen).bfloat16().float()
    return v.to({"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[dtype])


def launch_args(spec, sample_seed: int):
    """(grid, positional args, meta) for one input sample."""
    import torch
    gen = torch.Generator(device="cuda").manual_seed(sample_seed)
    if spec["form"] == "dot":
        M, N, K = spec["M"], spec["N"], spec["K"]
        a = draw(spec["dists"][0], 2 * M * K, spec["in_dtype"], gen)
        b = draw(spec["dists"][1], K * 2 * N, spec["in_dtype"], gen)
        bias = torch.randn(2 * N, device="cuda", generator=gen)
        y = torch.zeros(2 * M * 2 * N, device="cuda")
        return (spec["grid"],), [a, b, bias, y], {"M": M, "N": N, "K": K, "BK": spec["BK"]}
    if spec["form"] == "reduce":
        n, n_out, meta = spec["R"] * spec["C"] * spec["grid"], spec["R"] * spec["grid"], {"R": spec["R"],
                                                                                            "C": spec["C"]}
    elif spec["form"] == "loop":
        n, n_out, meta = spec["B"] * spec["T"] * spec["grid"], spec["B"] * spec["grid"], {"B": spec["B"],
                                                                                            "T": spec["T"]}
    else:
        n, n_out, meta = spec["B"] * spec["grid"], spec["B"] * spec["grid"], {"B": spec["B"]}
    x = draw(spec["dists"][0], n, spec["in_dtype"], gen)
    w = draw(spec["dists"][1], n, spec["in_dtype"], gen)
    out_dt = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[spec.get("out_dtype", "f32")]
    y = torch.zeros(n_out, device="cuda", dtype=out_dt)
    return (spec["grid"],), [x, w, y], meta


if __name__ == "__main__":
    for s in generate("dev-0", 12):
        print(json.dumps({k: v for k, v in s.items() if k != "source"}))
        print(s["source"])
