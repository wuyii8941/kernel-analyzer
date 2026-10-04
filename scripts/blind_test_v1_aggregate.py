"""Aggregation for blind_test_v1 phase 1: Holm over program x rule, family-wise bitwise comparison,
localization of detected effects, and the report in the protocol's CSV template."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval.analysis import _holm  # noqa: E402

RULES = ("R1", "R2", "R3", "R4", "R5")


def _localize(pid, family, rule, out_name, work: Path) -> dict:
    """Rank nodes by their share of the detected projection, on the two saved seeds (0 and 32).

    c_j = K - K_(node j substituted by its exact value) is projected on the detected rule's direction
    (R1: -1, R2: -sign(K), R3: -K/|K|, R4: the F4 pattern; for R5 the seed's own residual direction)."""

    from kernel_analyzer.reference_eval.capture import load_launch
    from kernel_analyzer.reference_eval.emulate import _outputs, emulate, rounding_nodes, verify
    from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir

    shares, statuses, texts, per_seed = {}, [], {}, {}
    for seed_dir in sorted((work / pid).glob("seed*")):
        launch = load_launch(seed_dir)
        v = verify(launch, ungrouped=[], swapped=[])
        statuses.append(f"{seed_dir.name}:{v['status']}")
        if v["status"] != "bit_identical":
            per_seed[seed_dir.name] = {"emulation": v["status"]}
            continue
        seed_values = {}
        base, emu = emulate(launch)
        outs = _outputs(base, launch)
        name = max(outs, key=lambda n: outs[n][0].sum())
        w_, k, _, st, act, elem, bits = outs[name]
        sel = w_ & (st == 0)
        kk = k[sel]
        ref = KernelReferenceEvaluator(parse_ttir(launch.asm["ttir"])).evaluate(launch)
        rb = next(b for b in ref.buffers.values() if b.name == name)
        e = kk - 0.5 * (rb.lo + rb.hi)[sel]
        if family == "F3" and out_name in ("mean", "rstd"):
            col = 0 if out_name == "mean" else 1
            idx = rb.global_indices()[sel]
            keep = (idx % 2) == col
        else:
            keep = np.ones(kk.size, dtype=bool)
        if rule == "R1":
            wdir = -np.ones(kk.size)
        elif rule == "R2":
            wdir = -np.sign(kk)
        elif rule == "R3":
            wdir = -kk
        elif rule == "R4":
            cols = 256
            pattern = np.zeros(cols)
            pattern[:64], pattern[64:128] = -1.0, 1.0
            wdir = np.tile(pattern, kk.size // cols)
        else:
            wdir = e.copy()
        wdir = np.where(keep, wdir, 0.0)
        wdir = wdir / (np.linalg.norm(wdir) or 1.0)
        total = float(e @ wdir)
        for node in rounding_nodes(emu):
            res, _ = emulate(launch, exact_nodes=[node], oracle=emu.oracle)
            o = _outputs(res, launch)[name]
            ok = o[3][sel] == 0
            c = np.where(ok, kk - o[1][sel], 0.0)
            shares.setdefault(node, []).append(float(c @ wdir))
            seed_values[node] = float(c @ wdir)
        texts.update({o.node_id: o.text.strip()[:120] for fn in emu.module.funcs.values() for o in fn.walk()})
        top = max(seed_values, key=lambda n: abs(seed_values[n])) if seed_values else None
        tot = sum(abs(x) for x in seed_values.values()) or 1.0
        per_seed[seed_dir.name] = {"emulation": "bit_identical", "top_node": top,
                                   "top_share": abs(seed_values[top]) / tot if top else None,
                                   "residual_projection": total}
    if not shares:
        return {"status": statuses, "localized": None}
    means = {n: float(np.mean(v)) for n, v in shares.items()}
    ranked = sorted(means.items(), key=lambda kv: -abs(kv[1]))
    total_abs = sum(abs(v) for v in means.values()) or 1.0
    cum, region = 0.0, 0
    for _, v in ranked:
        cum += abs(v)
        region += 1
        if cum >= 0.9 * total_abs:
            break
    node, val = ranked[0]
    tops = [d["top_node"] for d in per_seed.values() if d.get("top_node")]
    agree = sum(t == node for t in tops)
    grade = "stable" if agree >= 2 else ("clue" if agree == 1 else "inconsistent")
    failed = [k for k, d in per_seed.items() if d["emulation"] != "bit_identical"]
    return {"status": statuses, "localized": texts.get(node, node), "node": node,
            "share_of_projected_effect": abs(val) / total_abs, "region_size_ops": region,
            "top3": [(texts.get(n, n), v) for n, v in ranked[:3]],
            "grade": grade, "seeds_agreeing": agree, "seeds_emulated": len(tops), "emulation_failed_on": failed,
            "per_seed": {k: {**d, "top_node": texts.get(d.get("top_node"), d.get("top_node"))} for k, d in per_seed.items()}}


def aggregate(package: Path, out: Path, work: Path = None):
    manifest = json.loads((package / "manifest.json").read_text())
    reports = {}
    for entry in manifest["programs"]:
        p = out / "programs" / f"{entry['id']}.json"
        reports[entry["id"]] = json.loads(p.read_text()) if p.exists() else {"program": entry["id"],
                                                                             "family": entry["family"],
                                                                             "error": "not run"}
    # Holm over program x rule (every output of every program; R4 only on F4)
    tests = [(pid, o, rule) for pid, r in reports.items() if "outputs" in r for o in r["outputs"]
             if isinstance(o.get("rules"), dict) and "verdict" not in o["rules"]
             for rule in RULES if rule in o["rules"] and "p" in o["rules"][rule]]
    rejects = _holm([o["rules"][rule]["p"] for _, o, rule in tests], 0.05)
    for (pid, o, rule), rej in zip(tests, rejects):
        o["rules"][rule]["holm_reject"] = bool(rej)
        o["rules"][rule]["final_verdict"] = o["rules"][rule]["verdict"] if rej else "NOT_CONFIRMED"
    # the tool's own default detector: Holm over all its tests of all programs, per family of targets
    for fam in ("vector_mean", "alignment"):
        dt = [(o, t) for r in reports.values() if "outputs" in r for o in r["outputs"] if "default_detector" in o
              for t in o["default_detector"][fam]["tests"] if t.get("p") is not None]
        rej = _holm([t["p"] for _, t in dt], 0.05)
        for (o, t), rr in zip(dt, rej):
            supported = t.get("diagnostics", {}).get("tail_assumption_supported", True)
            t["final_verdict"] = ("DETECTED" if supported else "EXPLORATORY_ONLY") if rr else "NOT_CONFIRMED"
        for r in reports.values():
            for o in r.get("outputs", []):
                if "default_detector" in o:
                    vs = [t["final_verdict"] for t in o["default_detector"][fam]["tests"] if t.get("p") is not None]
                    o["default_detector"][fam]["final_verdict"] = (
                        "DETECTED" if "DETECTED" in vs else "EXPLORATORY_ONLY" if "EXPLORATORY_ONLY" in vs
                        else "NOT_CONFIRMED")
    # family-wise bitwise comparison over all 96 seeds
    for pid, r in reports.items():
        same = [q for q, s in reports.items() if q != pid and s.get("family") == r.get("family")
                and r.get("output_hashes") and s.get("output_hashes") == r.get("output_hashes")]
        r["bitwise_identical_to"] = same
    # localization of the strongest detected rule
    work = work or ROOT / ".cache" / "blind_v1_work"
    for pid, r in reports.items():
        detected = [(abs(o["rules"][rule]["mean"]) / max(abs(o["rules"][rule]["mu_interval"][1] - o["rules"][rule]["mu_interval"][0]), 1e-300),
                     rule, o["output"]) for o in r.get("outputs", []) if isinstance(o.get("rules"), dict)
                    for rule in RULES if o["rules"].get(rule, {}).get("final_verdict", "").startswith("DETECTED")]
        if detected:
            _, rule, name = max(detected)
            try:
                r["localization"] = {"rule": rule, **_localize(pid, r["family"], rule, name, work)}
            except Exception as exc:
                r["localization"] = {"rule": rule, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    (out / "phase1_report.json").write_text(json.dumps(reports, indent=2, default=float) + "\n")
    # CSV in the template's columns
    header = ["program", "family", "coverage_complete_frac", "coverage_conditional_frac",
              "coverage_not_established_frac", "unsupported_ops", "bitwise_identical_to", "residual_pos_frac",
              "residual_mean", "residual_max_width"] + \
             [f"mu_{r}_{k}" for r in RULES for k in ("interval", "verdict")] + \
             ["localized_node", "region_size_ops", "seconds", "edits_made"]
    template = package / "report_template_phase1.csv"
    if template.exists():  # the package's own columns (v2 adds the constant and scalar inventories)
        header = template.read_text().splitlines()[0].split(",")
    rows = []
    for pid in sorted(reports):
        r = reports[pid]
        if "outputs" not in r:
            rows.append({"program": pid, "family": r.get("family"), "edits_made": "none",
                         "unsupported_ops": r.get("error", "")})
            continue

        def cell(fn):
            vals = [fn(o) for o in r["outputs"]]
            return vals[0] if len(vals) == 1 else " | ".join(f"{o['output']}: {v}" for o, v in zip(r["outputs"], vals))

        row = {"program": pid, "family": r["family"],
               "coverage_complete_frac": cell(lambda o: f"{o['coverage_classes']['complete_composed']:.4f}"),
               "coverage_conditional_frac": cell(lambda o: f"{o['coverage_classes']['conditional_local']:.4f}"),
               "coverage_not_established_frac": cell(lambda o: f"{o['coverage_classes']['not_established']:.4f}"),
               "unsupported_ops": ";".join(r["unsupported"]) or "none",
               "bitwise_identical_to": ";".join(r["bitwise_identical_to"]) or "none",
               "residual_pos_frac": cell(lambda o: f"{o['residual']['positive_frac']:.3f}"),
               "residual_mean": cell(lambda o: f"{o['residual']['mean']:.3e}"),
               "residual_max_width": cell(lambda o: f"{o['residual']['max_width']:.2e}"),
               "seconds": r["seconds"], "edits_made": "none"}
        for rule in RULES:
            def interval(o, rule=rule):
                x = o["rules"].get(rule) if isinstance(o.get("rules"), dict) else None
                return "" if not x or "mu_interval" not in x else f"[{x['mu_interval'][0]:.3e}, {x['mu_interval'][1]:.3e}]"

            def verdict(o, rule=rule):
                x = o["rules"].get(rule) if isinstance(o.get("rules"), dict) else None
                return "" if not x else x.get("final_verdict", x.get("verdict", ""))
            row[f"mu_{rule}_interval"] = cell(interval)
            row[f"mu_{rule}_verdict"] = cell(verdict)
        inv = r.get("interface")
        if inv:
            consts = [c for c in inv["float_constants"] if c.get("value") is not None]
            rounded = [f"{c['value']}~{c['rounded_from']['candidate']} ({c['rounded_from']['relative_rounding']:.2e})"
                       for c in inv["rounded_compile_time_constants"]]
            row["compile_time_constants"] = (f"{len(consts)} float constants; rounded: " + ("; ".join(rounded) or "none"))
            row["interface_scalars"] = "; ".join(
                f"{x['name']}: {x['passed']} -> {x['received']} ({x['parameter_type']}"
                + (", exact)" if x.get("exact") else ", compile-time)" if x.get("compile_time") else
                   f", rounded {x.get('relative_rounding', 0):.2e})")
                for x in inv["runtime_scalars"]) or "none"
        loc = r.get("localization") or {}
        row["localized_node"] = loc.get("localized") or (loc.get("error") or "")
        row["region_size_ops"] = loc.get("region_size_ops", "")
        rows.append(row)
    with open(out / "phase1_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"{len(rows)} programs written")
