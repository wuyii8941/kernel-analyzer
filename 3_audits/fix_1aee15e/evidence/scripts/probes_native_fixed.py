"""The audit's run_probes.py probes (unsigned compare, integer CAS, scaled dot, two-order CAS) on the production
functions of the fixed tree.  run_probes.py --repo checks AST equality with its baseline excerpts first, which by design
fails on changed code, so its native branch is reproduced here without that gate.  Expected-correct outcomes are
asserted (the probe script itself asserts that the defects reproduce)."""
import ast, importlib.util, json, sys
from pathlib import Path
PKG = Path(sys.argv[1]); REPO = Path(sys.argv[2]); OUT = Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("run_probes", PKG / "repro/run_probes.py")
rp = importlib.util.module_from_spec(spec); sys.modules["run_probes"] = rp; spec.loader.exec_module(rp)
env, _ = rp.load_functions(None, False)
names = {n.name for n in ast.parse((PKG / "repro/production_excerpts.py").read_text()).body if isinstance(n, ast.FunctionDef)}
sys.path.insert(0, str(REPO / "src"))
from kernel_analyzer.reference_eval import ttir_eval as native
from kernel_analyzer.reference_eval.ttir_parser import PtrType
rp.TV, rp.PtrType, rp._ftv = native.TV, PtrType, native._ftv
for n in names:
    env[n] = getattr(native, n, None) or getattr(native.KernelReferenceEvaluator, n)
# harness adaptation: the probe's stub evaluator lacks the read record added by the F04 follow-up (_note_read); it
# has no effect on the CAS value or its status
_cas = env["_op_atomic_cas"]
def _cas_with_stub(ctx, *a):
    if not hasattr(ctx, "_note_read"):
        ctx._note_read = lambda *x: None
    return _cas(ctx, *a)
env["_op_atomic_cas"] = _cas_with_stub
out = {"scope": "production functions of the fixed tree, isolated (rule level); no GPU", "functions": sorted(names),
       "unsigned_compare": rp.cmp_probe(env), "integer_cas": rp.cas_probe(env),
       "scaled_dot": rp.scaled_probe(env, True), "two_order_assumption": rp.order_probe(env)}
OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")
ok = {"unsigned_compare": all(r["matches"] for r in out["unsigned_compare"]),
      "integer_cas": all(r["matches"] for r in out["integer_cas"]),
      "scaled_interval": out["scaled_dot"]["interval_propagation"]["covers_target_upper"],
      "scaled_scale_conditional": out["scaled_dot"]["scale_dependency"]["output_conditional"],
      "order_not_a_point": out["two_order_assumption"]["observed_status"] != 0}
print(json.dumps(ok))
assert all(ok.values()), ok
