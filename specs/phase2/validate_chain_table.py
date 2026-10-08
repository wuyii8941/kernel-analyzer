"""Check that a filled chain table meets the completion standard of row 1 of the guarantee table.
Usage: python3 validate_chain_table.py chain_table.json [repo_root]
Fails (exit 1) if any code/test/trigger field is empty, if a referenced file does not exist under repo_root,
or if a referenced test name does not occur in the referenced file."""
import json, os, sys

path = sys.argv[1]; root = sys.argv[2] if len(sys.argv) > 2 else "."
t = json.load(open(path))
problems = []
for a in t["arrows"]:
    for c in a["code"]:
        if not c["function"] or not c["file"]:
            problems.append(f"arrow {a['id']}: empty code location")
        elif not os.path.exists(os.path.join(root, c["file"])):
            problems.append(f"arrow {a['id']}: file {c['file']} not found")
        elif c["function"] not in open(os.path.join(root, c["file"]), errors="ignore").read():
            problems.append(f"arrow {a['id']}: function {c['function']} not in {c['file']}")
    for tst in a["tests"]:
        if not tst["name"] or not tst["file"]:
            problems.append(f"arrow {a['id']}: empty test")
        elif not os.path.exists(os.path.join(root, tst["file"])):
            problems.append(f"arrow {a['id']}: test file {tst['file']} not found")
        elif tst["name"] not in open(os.path.join(root, tst["file"]), errors="ignore").read():
            problems.append(f"arrow {a['id']}: test {tst['name']} not in {tst['file']}")
    for d in a["degrade_paths"]:
        if not d["triggered_record"]:
            problems.append(f"arrow {a['id']}: degrade '{d['condition']}' has no triggered record")
        elif not os.path.exists(os.path.join(root, d["triggered_record"])):
            problems.append(f"arrow {a['id']}: record {d['triggered_record']} not found")
print(f"{len(problems)} problems")
for p in problems:
    print(" -", p)
sys.exit(1 if problems else 0)
