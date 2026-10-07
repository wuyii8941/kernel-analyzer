import sys, json
sys.path.insert(0, "scripts/essential")
import torch, common
import p2b_g6_g7 as G
res = {}
for name in ("sum_long_bf16", "gelu_tanh_bf16", "layer_norm_bf16"):
    _, dt, make, fn = [p for p in G.g7_programs() if p[0] == name][0]
    for label, seeds in (("original_0_95", tuple(range(0, 96))), ("fresh_1000_1095", tuple(range(1000, 1096)))):
        st = {}
        def setup():
            torch._dynamo.reset(); st["fn"] = torch.compile(fn, dynamic=False); launch(inputs_(seeds[0])); torch.cuda.synchronize()
        def inputs_(s): return {"args": [a.to("cuda", dt) for a in make(s)], "_seed": s}
        def launch(t): return {"out": st["fn"](*t["args"])}
        rep = common.fr_run(f"g7rep/{name}", setup, inputs_, launch, lambda _t: None, seeds=seeds)[0]
        num = rep["outputs"]["out"]["numerical"]
        rules = [{k: r.get(k) for k in ("rule", "verdict", "p_value", "p_dev", "p_conf", "statistic", "estimate", "direction") if k in r} for r in num.get("rules", [])]
        res[f"{name}/{label}"] = rules
        print(name, label, [(r["rule"], r["verdict"], {k: v for k, v in r.items() if k.startswith("p")}) for r in rules], flush=True)
json.dump(res, open("results/essential/phase2b/g7_numerical_stream/replication.json", "w"), indent=1, default=str)
