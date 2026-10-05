#!/usr/bin/env python3
"""Score the tool against the benchmark manifest (plan section 4): every unit is a configuration + reference definition +
measurement point + proposition + independent answer basis; the result JSON is checked against the proposition.

    python scripts/benchmark_score.py results/benchmark/real_bugs_pre_post.json

Propositions (all on one output of one report):
    sem_bins: allowed e_sem bins (tool_spec_summary.sem_bin)
    special_k_vs_f_min: at least this many K-vs-f special-value class mismatches
    num_rel_min / num_rel_max: bounds on the e_num relative RMS
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tool_spec_summary import sem_bin  # noqa: E402


def check(unit):
    path = ROOT / unit["result"]
    if not path.exists():
        return None, "result missing"
    rep = json.loads(path.read_text())
    if "error" in rep:
        return False, "error: " + rep["error"].splitlines()[0][:80]
    e = rep["outputs"].get(unit["measurement"]["output"])
    if e is None:
        return False, "output not evaluated"
    prop = unit["proposition"]
    sv = e.get("special_values") or {}
    b = sem_bin(e.get("semantic"), bool(e.get("depends_on_non_triton_intermediates")),
                (e.get("total") or {}).get("relative_rms"), e.get("semantic_elementwise"))
    num = ((e.get("numerical") or {}).get("scale") or {}).get("relative_rms")
    why = [f"bin={b}", f"e_num={num:.2g}" if num is not None else "e_num=n/a",
           f"special(K vs f)={sv.get('k_vs_f_class_mismatch')}"]
    ok = True
    if "sem_bins" in prop:
        ok &= b in prop["sem_bins"]
    if "special_k_vs_f_min" in prop:
        ok &= (sv.get("k_vs_f_class_mismatch") or 0) >= prop["special_k_vs_f_min"]
    if "num_rel_min" in prop:
        ok &= num is not None and num >= prop["num_rel_min"]
    if "num_rel_max" in prop:
        ok &= num is not None and num <= prop["num_rel_max"]
    return bool(ok), ", ".join(why)


def main():
    manifest = json.loads(Path(sys.argv[1]).read_text())
    rows, n_ok, n_all = [], 0, 0
    for u in manifest["units"]:
        ok, why = check(u)
        n_all += ok is not None
        n_ok += bool(ok)
        rows.append(f"| {u['unit']} | {u['proposition_text']} | {'—' if ok is None else ('pass' if ok else 'FAIL')} | {why} |")
    print(f"# {manifest['title']}\n\n| unit | proposition | result | observed |\n|---|---|---|---|")
    print("\n".join(rows))
    print(f"\npassed {n_ok} of {n_all} units with results ({len(manifest['units'])} in the manifest)")


if __name__ == "__main__":
    main()
