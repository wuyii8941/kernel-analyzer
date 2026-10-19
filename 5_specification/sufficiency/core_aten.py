#!/usr/bin/env python3
"""Sufficiency evaluation E2 (docs/SUFFICIENCY_PROTOCOL.md): the Core ATen operator set of the installed PyTorch.

Three layers, reported separately:
  1. inventory: every overload tagged torch.Tag.core in the installed torch (locked version, schemas recorded);
  2. correspondence: which contract names the operation (direct `aten.<name>` in covers, or an alias listed in
     ALIASES with its reason), or which non-arithmetic class carries it (structure/storage, random); name
     correspondence is not a claim of complete ground truth;
  3. probe: PyTorch's own OpInfo sample inputs are run on CPU under a dispatch recorder; every recorded call of a Core
     ATen overload that has a correspondence is queried through kernel_spec.registry.query (validated contracts only,
     the first profile that needs no task choice). Outcomes are counted per overload; values are additionally compared
     with the float64 PyTorch result as a numerical diagnostic (not a consistency verdict).

    python 5_specification/sufficiency/core_aten.py --out-dir NEW_DIR [--max-calls 50] [--budget 30]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import signal
import sys
import time
from fractions import Fraction
from pathlib import Path

import torch
import yaml
from torch.utils._python_dispatch import TorchDispatchMode

HERE = Path(__file__).resolve().parent
SPEC = HERE.parent
sys.path.insert(0, str(SPEC))

from kernel_spec import registry  # noqa: E402
from kernel_spec.num import ends  # noqa: E402

# An alias is used only when the contract's covered public function dispatches exactly this overload (checked
# empirically by the probe: the recorded call must come from that function's OpInfo samples or the overload itself).
ALIASES = {
    "avg_pool2d": ("POOL", "torch.nn.functional.avg_pool2d dispatches aten.avg_pool2d"),
    "avg_pool2d_backward": ("POOL", "the VJP of torch.nn.functional.avg_pool2d (profile finite_avg2d_autograd)"),
    "max_pool2d_with_indices": ("POOL", "torch.nn.functional.max_pool2d(return_indices=True) dispatches it"),
    "max_pool2d_with_indices_backward": ("POOL", "the VJP of max_pool2d (profile torch210_unique_max2d_autograd)"),
}
# Classes that do not carry an arithmetic formula; they still matter for the answer and are not dropped from the
# account: structure/storage semantics must be handled reliably elsewhere (to be verified, not assumed).
STRUCTURE = {"empty", "empty_strided", "as_strided", "copy", "fill", "full", "full_like", "scalar_tensor", "resize_",
             "sym_size", "sym_stride", "sym_numel", "sym_storage_offset", "sym_is_contiguous", "_local_scalar_dense",
             "arange", "diagonal", "repeat", "flip", "masked_scatter", "nonzero"}
RANDOM_OPS = {"rand", "randn", "randperm", "native_dropout"}

TAG = {torch.float64: "f64", torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16",
       torch.int64: "i64", torch.int32: "i32", torch.int16: "i16", torch.int8: "i8", torch.uint8: "u8",
       torch.bool: "b8"}
PROBE_DTYPES = (torch.float32, torch.bfloat16, torch.float64, torch.int64)


def core_overloads():
    out = {}
    for name in torch._C._dispatch_get_all_op_names():
        if not name.startswith("aten::"):
            continue
        base, _, ov = name[6:].partition(".")
        try:
            op = getattr(getattr(torch.ops.aten, base), ov or "default")
        except (AttributeError, RuntimeError):
            continue
        if torch.Tag.core in op.tags:
            out[f"{base}.{ov or 'default'}"] = str(op._schema)
    return dict(sorted(out.items()))


def aten_interface(cid):
    """True when the contract's adapter takes ATen calls {op, args, kwargs} (kernel_spec.aten_adapter)."""
    src = (SPEC / "adapters" / f"{cid}.py").read_text()
    return "make_reference" in src


def contract_table():
    """aten base name -> [(contract id, review status, implementation status, first free profile)]."""
    table = collections.defaultdict(list)
    for f in sorted((SPEC / "contracts").glob("*.yaml")):
        d = yaml.safe_load(f.read_text())
        free = [p for p, v in d.get("profiles", {}).items() if not v.get("task_choice_required")]
        info = (d["id"], d["review"]["status"], (d.get("activation") or {}).get("implementation_status"),
                free[0] if free else None)
        for c in d.get("covers") or []:
            if isinstance(c, str) and c.startswith("aten."):
                table[c[5:].split(".")[0]].append(info)
    return table


def correspondence(base, table):
    if base in table:
        return "direct", table[base], None
    if base in ALIASES:
        cid, why = ALIASES[base]
        rows = [r for rows in table.values() for r in rows if r[0] == cid][:1]
        if not rows:
            d = yaml.safe_load((SPEC / "contracts" / f"{cid}.yaml").read_text())
            free = [p for p, v in d["profiles"].items() if not v.get("task_choice_required")]
            rows = [(cid, d["review"]["status"], (d.get("activation") or {}).get("implementation_status"),
                     free[0] if free else None)]
        return "alias", rows, why
    if base in STRUCTURE:
        return "structure_storage", [], "no arithmetic formula; carried by structure/storage semantics (not verified here)"
    if base in RANDOM_OPS:
        return "random", [], "random stream; RANDOM contract conditions on the drawn values (forward not admitted)"
    return "none", [], None


# ------------------------------------------------------------------ probe
class Recorder(TorchDispatchMode):
    def __init__(self, wanted):
        super().__init__()
        self.wanted, self.calls = wanted, []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        key = f"{func._schema.name.split('::')[1]}.{func._overloadname}"
        if key in self.wanted:
            self.calls.append((key, func, _clone(args), _clone(kwargs)))
        return func(*args, **kwargs)


def _clone(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_clone(v) for v in x)
    if isinstance(x, dict):
        return {k: _clone(v) for k, v in x.items()}
    return x


class Skip(Exception):
    pass


def enc(x):
    """A recorded argument in the aten_adapter input format; Skip for what the format cannot carry."""
    if isinstance(x, torch.Tensor):
        if x.dtype not in TAG:
            raise Skip(f"dtype {x.dtype}")
        if x.dtype.is_floating_point and torch.isnan(x).any():
            raise Skip("NaN tensor value (no real-number encoding)")
        v = _inf_str(x.tolist())
        return {"t": TAG[x.dtype], "v": v, "shape": list(x.shape)}
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    if isinstance(x, dict):
        return {k: enc(v) for k, v in x.items()}
    if isinstance(x, torch.dtype):
        if x not in TAG:
            raise Skip(f"dtype {x}")
        return {"dtype": TAG[x]}
    if isinstance(x, float):
        if math.isnan(x):
            raise Skip("NaN scalar (no real-number encoding)")
        return x if math.isfinite(x) else ("inf" if x > 0 else "-inf")
    if isinstance(x, (bool, int, str)) or x is None:
        return x
    if isinstance(x, complex):
        raise Skip("complex scalar")
    raise Skip(f"argument type {type(x).__name__}")


def _inf_str(v):
    if isinstance(v, list):
        return [_inf_str(x) for x in v]
    if isinstance(v, float) and math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def non_default_kwargs(func, kwargs):
    """Drop keyword arguments equal to their schema default (they do not change the call)."""
    defaults = {a.name: a.default_value for a in func._schema.arguments if a.has_default_value()}
    return {k: v for k, v in kwargs.items() if not (k in defaults and _same(v, defaults[k]))}


def _same(a, b):
    try:
        return a == b and type(a) is type(b) or (a is None and b is None)
    except Exception:  # noqa: BLE001
        return False


def float64_result(func, args, kwargs):
    def up(x):
        if isinstance(x, torch.Tensor) and x.dtype.is_floating_point:
            return x.double()
        if isinstance(x, (list, tuple)):
            return type(x)(up(v) for v in x)
        return x
    try:
        return func(*up(args), **{k: up(v) for k, v in kwargs.items()})
    except Exception:  # noqa: BLE001
        return None


def _flat(x):
    if isinstance(x, (list, tuple)):
        for v in x:
            yield from _flat(v)
    else:
        yield x


def diagnostic(ref_out, t64):
    """(compared, max relative difference) between reference midpoints and the float64 result; None if not comparable."""
    if t64 is None:
        return None
    ts = [t for t in _flat(t64 if isinstance(t64, (list, tuple)) else [t64])]
    refs = ref_out if isinstance(t64, (list, tuple)) else [ref_out]
    worst, n = 0.0, 0
    for r, t in zip(refs, ts):
        if not isinstance(t, torch.Tensor):
            continue
        tv = t.double().flatten().tolist() if t.dtype.is_floating_point else t.flatten().tolist()
        rv = list(_flat(r)) if isinstance(r, list) else [r]
        if len(rv) != len(tv):
            return None
        for a, b in zip(rv, tv):
            m = _mid(a)
            if m is None or (isinstance(b, float) and not math.isfinite(b)):
                continue
            d = abs(m - float(b)) / max(abs(float(b)), 1e-12)
            worst, n = max(worst, d), n + 1
    return (n, worst) if n else None


def _mid(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, Fraction)):
        return float(v)
    if isinstance(v, str):
        return {"+inf": math.inf, "-inf": -math.inf}.get(v)
    if hasattr(v, "a") and hasattr(v, "b"):         # mpmath interval: exact endpoints, midpoint as a float
        lo, hi = ends(v)
        return float((lo + hi) / 2)
    return None


def _alarm(signum, frame):
    raise TimeoutError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-calls", type=int, default=50, help="queried calls per overload and dtype, in sample order")
    ap.add_argument("--budget", type=int, default=30, help="seconds per reference query")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=False)
    inv = core_overloads()
    table = contract_table()
    rows = {}
    for key, schema in inv.items():
        base = key.split(".")[0]
        kind, contracts, why = correspondence(base, table)
        rows[key] = {"overload": key, "schema": schema, "correspondence": kind, "reason": why,
                     "contracts": [dict(zip(("id", "review", "implementation", "profile"), c)) for c in contracts],
                     "probe": collections.Counter(), "dtypes": collections.defaultdict(collections.Counter),
                     "diagnostic": {"compared": 0, "max_rel": 0.0, "over_1e-9": 0}, "notes": []}
    (a.out_dir / "inventory.json").write_text(json.dumps(
        {"torch": torch.__version__, "overloads": inv, "base_names": sorted({k.split('.')[0] for k in inv})}, indent=1))

    probed = {k for k, r in rows.items() if r["correspondence"] in ("direct", "alias")}
    from torch.testing._internal.common_methods_invocations import op_db
    seen = collections.Counter()
    signal.signal(signal.SIGALRM, _alarm)
    for opinfo in op_db:
        for dtype in PROBE_DTYPES:
            if dtype not in opinfo.supported_dtypes("cpu"):
                continue
            try:
                samples = list(opinfo.sample_inputs("cpu", dtype, requires_grad=False))
            except Exception:  # noqa: BLE001
                continue
            for s in samples:
                rec = Recorder(probed)
                try:
                    with rec:
                        opinfo.op(s.input, *s.args, **s.kwargs)
                except Exception:  # noqa: BLE001
                    continue
                for key, func, args, kwargs in rec.calls:
                    tag = TAG.get(dtype)
                    if seen[(key, tag)] >= a.max_calls:
                        continue
                    seen[(key, tag)] += 1
                    row = rows[key]
                    c = next((c for c in row["contracts"] if c["implementation"] == "validated"), None)
                    if c is None:
                        outcome = "no validated contract"
                    elif row["correspondence"] == "alias" or not aten_interface(c["id"]):
                        outcome = "contract interface is not an ATen call (needs a call mapping)"
                    else:
                        kw = non_default_kwargs(func, kwargs)
                        try:
                            inp = {"op": f"aten.{key.split('.')[0]}", "args": enc(list(args)),
                                   "kwargs": enc(dict(kw)) if kw else {}}
                        except Skip as e:
                            outcome = f"input not representable: {e}"
                            inp = None
                        if inp is not None:
                            signal.alarm(a.budget)
                            t0 = time.time()
                            try:
                                r = registry.query(c["id"], c["profile"], inp)
                                outcome = r.outcome
                                if outcome != "value" and len(row["notes"]) < 6:
                                    row["notes"].append(f"{tag}: {outcome}: {(r.note or '')[:140]}")
                                if outcome == "value":
                                    dg = diagnostic(r.fields.get("out"), float64_result(func, args, kwargs))
                                    if dg:
                                        row["diagnostic"]["compared"] += dg[0]
                                        row["diagnostic"]["max_rel"] = max(row["diagnostic"]["max_rel"], dg[1])
                                        row["diagnostic"]["over_1e-9"] += int(dg[1] > 1e-9)
                            except TimeoutError:
                                outcome = "budget exceeded"
                            except Exception as e:  # noqa: BLE001 -- recorded, never hidden
                                outcome = "probe error"
                                row["notes"].append(f"{tag}: probe error {type(e).__name__}: {str(e)[:120]}")
                            finally:
                                signal.alarm(0)
                    row["probe"][outcome] += 1
                    row["dtypes"][tag][outcome] += 1
        print(opinfo.name, flush=True)
    for r in rows.values():
        r["probe"] = dict(r["probe"])
        r["dtypes"] = {k: dict(v) for k, v in r["dtypes"].items()}
    (a.out_dir / "core_aten.json").write_text(json.dumps(list(rows.values()), indent=1, default=str))
    summary = collections.Counter(r["correspondence"] for r in rows.values())
    print(json.dumps({"overloads": len(rows), "correspondence": summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
