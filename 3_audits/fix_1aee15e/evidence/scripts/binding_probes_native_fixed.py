"""The audit's run_binding_probes.py provenance cases (rounded compute, honest copy, ordinary compute) on the production
check.torch_intermediates and the production measure scope rule of the fixed tree (simulated capture metadata, CPU).
Expected after the F04 fix: none of the three is promoted to call level.  The probe's Gluon line was a dictionary
lookup on a stub; the real Gluon path is the GPU regression test_gluon_kernel_through_the_unified_entry."""
import json, sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np, torch
REPO = Path(sys.argv[1]); OUT = Path(sys.argv[2])
sys.path.insert(0, str(REPO / "src"))
from kernel_analyzer import check
x = torch.tensor([1., 2., 3.], dtype=torch.float32)
rows = []
for name, z in [("rounded_compute", (x[:1] + 2. ** -25).repeat(4)), ("honest_copy", x[:1].repeat(4)),
                ("ordinary_compute", (x[:1] + 0.125).repeat(4))]:
    raw = z.numpy().copy().view(np.uint8)
    inp = NS(kind="tensor", storage_ptr=101, name="upstream", dtype="float32", before=raw, after=raw.copy())
    out = NS(kind="tensor", storage_ptr=202, name="out", dtype="float32", before=np.zeros_like(raw), after=raw.copy())
    launch = NS(args=[inp, out], kernel_name="identity_kernel")
    ref = NS(loaded={101}, loaded_any={101}, stored={202})
    mixed = check.torch_intermediates([launch], NS(launches=[ref]), {"x": x}).get(202, [])
    # measure.run_level since the fix: any non-Triton upstream buffer keeps the reference kernel-level
    rows.append({"case": name, "upstream_tags": mixed, "classification": "kernel-level" if mixed else "call-level"})
OUT.write_text(json.dumps({"scope": "production torch_intermediates, simulated metadata, CPU", "rows": rows}, indent=1) + "\n")
print(json.dumps(rows))
assert all(r["classification"] == "kernel-level" for r in rows)
