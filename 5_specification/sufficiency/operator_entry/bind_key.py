#!/usr/bin/env python3
"""Path A, one recorded call key (docs/OPERATOR_ENTRY_PROTOCOL.md sections 3-5, 7): candidate contracts and
profiles, generation (rule v1), T2 replay check, one measurement per candidate profile, profile selection, class.

    bind_key.py --workload REC_DIR --key K --spec SCRATCH_SPEC --out RESULT.json [--timeout 700]

Writes one JSON result: {"key", "op", "class", "reason", "profiles": [...], "verdict", ...}.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import record as R  # noqa: E402

R.install_launch_hook()
import torch  # noqa: E402
import yaml  # noqa: E402

import instantiate as I  # noqa: E402

TOOL = HERE.parents[2] / "2_tool/scripts/measure.py"
EVID, NEWC, MATCH, EXTEND, SCOPE, USABLE = ("run or evidence insufficient", "new contract needed",
                                             "specification matching needed (step 2)", "generation rule extension needed",
                                             "outside contract scope", "usable as is")


def candidates(spec, base):
    """(validated contracts covering base, drafted-only covering ids)."""
    val, other = [], []
    for f in sorted((spec / "contracts").glob("*.yaml")):
        c = yaml.safe_load(f.read_text())
        if base not in (c.get("covers") or []):
            continue
        if (c.get("activation") or {}).get("implementation_status") == "validated":
            val.append(c)
        else:
            other.append(c["id"])
    return val, other


def aten_interface(spec, cid):
    p = spec / "adapters" / f"{cid}.py"
    return p.is_file() and "make_reference" in p.read_text()


def load_module(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def replay_names(spec, name, tensors):
    """T2: kernel names launched by one execution of the generated call."""
    call = load_module(spec / "corpus/calls" / f"{name}.py", f"pa_call_{name}")
    gen = load_module(spec / "corpus/calls" / f"{name}_inputs.py", f"pa_gen_{name}")
    inputs = {t: getattr(gen, t)(0, None, None) for t in tensors}
    call.call(inputs)                      # warm-up (JIT); launches of the second execution are compared
    torch.cuda.synchronize()
    R.reset_launches()
    call.call(inputs)
    torch.cuda.synchronize()
    return [n for n, _, _ in R.all_launches()]


def verdict(report):
    sys.path.insert(0, str(HERE.parent))
    import run_e4
    return run_e4.verdict(report)


def level_failure(report):
    """(failure_class, reason) of the first level that did not finish, or (None, None)."""
    for lv in (report or {}).get("levels", []):
        if lv.get("status") not in ("ok",):
            return lv.get("failure_class"), lv.get("reason") or lv.get("status")
    return None, None


def reference_outcome(spec, decl_path, values, budget=300):
    """Outcome of the registry query on the mapped recorded inputs (used when the measurement only says the binding
    was not retained): 'value' / 'domain_undefined' mean the profile applies; 'unsupported' / 'invalid_input' mean
    the call is outside it."""
    import signal
    from kernel_analyzer.contract_binding import _tensor_record
    from kernel_spec import registry
    sys.path.insert(0, str(HERE.parents[1] / "corpus"))
    import call_binding
    decl = json.loads(decl_path.read_text())
    plan = json.loads((decl_path.parent / decl["compare"]["binding_plan"]).read_text())
    rec = torch.load(decl_path.parent / decl["call"].split(":")[0].replace(".py", "_values.pt"), map_location="cuda",
                     weights_only=False)
    obs = {"tensor_arguments": {k: {"before": _tensor_record(t)} for k, t in rec["tensors"].items()}}
    task = plan["tasks"][0]
    inputs, _, _ = call_binding._mapped(task["input_map"], obs)
    def alarm(*_):
        raise TimeoutError
    old = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(budget)
    try:
        r = registry.query(task["contract"], task["profile"], inputs)
        return r.outcome, (r.note or "")[:200]
    except TimeoutError:
        return "budget_exceeded", ""
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def evaluated(report):
    return any(o.get("status") == "evaluated" for lv in (report or {}).get("levels", [])
               for o in (lv.get("outputs") or {}).values())


def over_budget(report):
    return any(lv.get("status") == "over budget" for lv in (report or {}).get("levels", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--key", type=int, required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=700)
    a = ap.parse_args()
    run = json.loads((a.workload / "run.json").read_text())
    k = next(json.loads(l) for l in open(a.workload / "keys.jsonl") if json.loads(l)["key"] == a.key)
    res = {"workload": a.workload.name, "key": a.key, "op": k["op"], "calls": k["calls"], "launches": k["launches"]}

    def done(cls, reason, **extra):
        res.update({"class": cls, "reason": reason}, **extra)
        a.out.write_text(json.dumps(res))
        return 0

    if not run["t1_launch_sequences_agree"]:
        return done(EVID, "T1: recording changed the launch sequence of this workload")
    if not k["launch_sets_agree"]:
        res["note"] = "calls sharing this key launched different kernel sequences; the first call's is used"
    if a.key in run.get("keys_over_capture_cap", []):
        return done(EVID, "capture size limit (200 MB)")
    base = "aten." + k["op"].split(".")[1]
    val, other = candidates(a.spec, base)
    if not val:
        return done(NEWC, "no validated contract covers " + base + (f" (not validated: {other})" if other else ""))
    usable = [c for c in val if aten_interface(a.spec, c["id"])]
    if not usable:
        return done(MATCH, "contract interface is not an ATen call: " + ",".join(c["id"] for c in val))
    cands = [(c, p) for c in usable for p, pd in c["profiles"].items() if pd.get("task_choice_required") is False]
    choice = [(c["id"], p) for c in usable for p, pd in c["profiles"].items() if pd.get("task_choice_required")]
    if not cands:
        return done(MATCH, f"only task-choice profiles: {choice}")
    import importlib.metadata as md
    rt = {"distribution": "flag_gems", "version": md.version("flag_gems")}
    values = a.workload / "values" / f"{a.key}.pt"
    profiles, refusals, t2 = [], [], None
    for c, p in cands:
        name = f"pa_{a.workload.name}_{a.key}_{c['id']}_{p}".replace("-", "_").replace(".", "_")
        try:
            d = I.generate(a.spec, name, k["op"], values, c, p, rt)
        except I.Refuse as e:
            refusals.append({"contract": c["id"], "profile": p, "stage": e.stage, "why": e.why})
            continue
        if t2 is None:
            tensors = sorted(json.loads(d.read_text())["inputs"])
            try:
                got = replay_names(a.spec, name, tensors)
            except Exception as err:  # noqa: BLE001 -- recorded
                return done(EVID, f"T2 replay execution error: {type(err).__name__}: {str(err)[:200]}")
            t2 = got == k["launches"]
            res["t2_replay_launches"] = got
            if not t2:
                return done(EVID, "T2: the replayed call launches a different kernel sequence")
        rp = a.out.with_name(a.out.stem + f"__{c['id']}_{p}.report.json")
        try:
            pr = subprocess.run([sys.executable, str(TOOL), "--declaration", str(d), "--out", str(rp)],
                                capture_output=True, text=True, timeout=a.timeout)
            rc = pr.returncode
            err = pr.stderr[-300:]
        except subprocess.TimeoutExpired:
            rc, err = "budget_exceeded", ""
        report = json.loads(rp.read_text()) if rp.exists() else None
        fclass, freason = level_failure(report)
        entry = {"contract": c["id"], "profile": p, "rc": rc, "report": rp.name if report else None,
                 "evaluated": evaluated(report), "over_budget": rc == "budget_exceeded" or over_budget(report),
                 "failure_class": fclass, "failure_reason": (freason or "")[:300],
                 "verdict": verdict(report) if report else ("not_established", err)}
        entry["applies"] = True if entry["evaluated"] else None      # None = undecided
        if not entry["evaluated"] and freason and "did not retain an admitted specification binding" in freason:
            outcome, note = reference_outcome(a.spec, d, values)
            entry["reference_outcome"], entry["reference_note"] = outcome, note
            if outcome in ("unsupported", "invalid_input"):
                entry["applies"] = False
        profiles.append(entry)
    res["profiles"], res["generation_refusals"] = profiles, refusals
    if not profiles:
        return done(EXTEND, "; ".join(f"{r['stage']}: {r['why']}" for r in refusals)[:300])
    app = [x for x in profiles if x["applies"]]
    undecided = [x for x in profiles if x["applies"] is None]
    if len(app) > 1:
        return done(MATCH, "more than one profile applies: " + ",".join(f"{x['contract']}/{x['profile']}" for x in app))
    if not app:
        if undecided:
            x = undecided[0]
            why = "over budget" if x["over_budget"] else f"{x['failure_class']}: {x['failure_reason']}"
            return done(EVID, f"applicability not decided ({x['contract']}/{x['profile']}): {why}"[:300])
        if choice:
            return done(MATCH, f"no profile without task choice applies; task-choice profiles exist: {choice}")
        notes = [f"{x['contract']}/{x['profile']}: {x.get('reference_outcome')}: {x.get('reference_note', '')[:100]}"
                 for x in profiles]
        return done(SCOPE, "no candidate profile applies: " + " | ".join(notes)[:300])
    if undecided:
        return done(EVID, "profile uniqueness not decided: " + ",".join(f"{x['contract']}/{x['profile']}" for x in undecided)
                    + f" undecided next to {app[0]['contract']}/{app[0]['profile']}")
    x = app[0]
    v, why = x["verdict"]
    if v == "not_established":
        return done(EVID if x["over_budget"] else SCOPE, f"reference not complete under {x['contract']}/{x['profile']}: {why}"[:300],
                    verdict=v, selected=f"{x['contract']}/{x['profile']}")
    return done(USABLE, f"{x['contract']}/{x['profile']}: {why}", verdict=v, selected=f"{x['contract']}/{x['profile']}")


if __name__ == "__main__":
    sys.exit(main())
