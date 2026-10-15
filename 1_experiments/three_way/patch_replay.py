"""Edit one line of a captured Inductor kernel's Python source, recompile it with the captured configuration and run it
on the captured inputs under the recorder: the patched launch is a new unit for analyze_units.analyze.

    run_patched(unit_dir, [(line, old, new)], out_dir) -> unit directory of the patched launch
    source_line(unit_dir, node) -> (path, line, text) of a TTIR node (for attribution sites)
    fix_edit(text) -> (old, new, mechanism) or None: the protocol's fix for the instruction on a source line
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def kernel_file(L) -> Path:
    m = re.search(r'#loc = loc\("([^"]+\.py)":(\d+):\d+\)', L.asm["ttir"])
    return Path(m.group(1))


def _module_text(path: Path, name: str, edits) -> str:
    lines = path.read_text().splitlines()
    for line, old, new in edits:
        text = lines[line - 1]
        if old not in text:
            raise ValueError(f"line {line} does not contain {old!r}: {text.strip()}")
        lines[line - 1] = text.replace(old, new, 1)
    d = next(i for i, l in enumerate(lines) if l.startswith(f"def {name}("))
    start = d
    while start > 0 and not lines[start - 1].startswith("@triton_heuristics"):
        start -= 1
    header = lines[:start - 1] if start > 0 else lines[:d]
    body = lines[d:]
    return "\n".join(header + ["@triton.jit"] + body) + "\n", d - len(header) - 1   # line shift of the body


def run_patched(unit_dir, edits, out_dir) -> Path:
    import torch

    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, load_launch, save_launch

    L = load_launch(Path(unit_dir))
    text, shift = _module_text(kernel_file(L), L.kernel_name, edits)
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    mod_path = HERE / "patched" / f"{L.kernel_name[:80]}_{digest}.py"
    mod_path.parent.mkdir(parents=True, exist_ok=True)
    mod_path.write_text(text)
    spec = importlib.util.spec_from_file_location(f"patched_{digest}", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, L.kernel_name)
    info = json.loads((Path(unit_dir) / "launch.json").read_text())["src_info"]
    constexprs = {info["arg_names"][k[0] if isinstance(k, list) else k]: v for k, v in info["constants"]}
    storages, call = {}, {}       # every argument by name: a specialised argument (e.g. xnumel == 1) is not positional
    for arg in L.args:
        if arg.name in constexprs or arg.constexpr:
            continue
        if arg.kind == "tensor":
            if arg.storage_ptr not in storages:
                storages[arg.storage_ptr] = arg.before.to("cuda").clone()
            raw = storages[arg.storage_ptr]
            dtype = getattr(torch, arg.dtype)
            offset = (arg.data_ptr - arg.storage_ptr) // arg.element_size
            numel = raw.numel() // arg.element_size
            flat = torch.empty(0, dtype=dtype, device="cuda").set_(raw.untyped_storage(), 0, (numel,))
            call[arg.name] = flat.as_strided(arg.shape, arg.stride, offset)
        elif arg.kind in ("int", "float", "bool"):
            call[arg.name] = arg.value
        elif arg.kind == "none":
            call[arg.name] = None
    opts = {k: L.metadata[k] for k in ("num_warps", "num_stages") if k in L.metadata}
    with TritonLaunchRecorder() as R:
        fn[tuple(L.grid)](**call, **constexprs, **opts)
        torch.cuda.synchronize()
    out = Path(out_dir)
    save_launch(R.launches[0], out)
    (out / "patch.json").write_text(json.dumps({"from": str(unit_dir), "edits": edits, "module": str(mod_path),
                                                "line_shift": shift}))
    return out


def source_line(L, node: str):
    from kernel_analyzer.reference_eval import emulate as E
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir

    module = parse_ttir(L.asm["ttir"])
    ops = {o.node_id: o for fn in module.funcs.values() for o in fn.walk()}
    locations = E.ttir_locations(L.asm["ttir"])
    defining = {}
    for line in L.asm["ttir"].splitlines():
        m = re.match(r"^\s*(%[\w#]+)(?::\d+)?\s*=", line)
        if m:
            defining.setdefault(m.group(1), line)
    loc = E.op_location(ops[node], locations, defining) if node in ops else None
    if not loc:
        return None
    path = Path(loc[0]) if Path(loc[0]).exists() else kernel_file(L)
    text = path.read_text().splitlines()[loc[1] - 1]
    return str(path), loc[1], text


FIXES = [   # (mechanism, pattern on the source line, replacement)
    ("rsqrt.approx", r"libdevice\.rsqrt\((\w+)\)", r"libdevice.div_rn(1.0, libdevice.sqrt_rn(\1))"),
    ("sqrt.approx", r"libdevice\.sqrt\((\w+)\)", r"libdevice.sqrt_rn(\1)"),
    ("sqrt.approx", r"tl\.sqrt\((\w+)\)", r"libdevice.sqrt_rn(\1)"),
    ("ex2.approx", r"tl_math\.exp\((\w+)\)", r"libdevice.exp(\1)"),
    ("ex2.approx", r"tl_math\.exp2\((\w+)\)", r"libdevice.exp2(\1)"),
    ("div.full", r"\((\w+) / (\w+|[0-9.e+-]+)\)", r"libdevice.div_rn(\1, \2)"),
    ("div.full", r"= (\w+) / (\w+|[0-9.e+-]+)$", r"= libdevice.div_rn(\1, \2)"),
    ("tf32 dot", r"allow_tf32=ALLOW_TF32", r'input_precision="ieee"'),
]


def fix_edit(text: str):
    for mech, pat, rep in FIXES:
        m = re.search(pat, text)
        if m:
            old = m.group(0)
            return old, re.sub(pat, rep, old), mech
    return None
