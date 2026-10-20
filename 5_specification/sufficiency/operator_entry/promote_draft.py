#!/usr/bin/env python3
"""Promote an independently reviewed path A2 draft pair to the runtime contract (CONTRACT_AUTHORING section 8).

    promote_draft.py CID --yaml-sha SHA --adapter-sha SHA --review reviews/.../A2_package_X_review.md [--date YYYY-MM-DD]

Refuses unless the draft files have exactly the sha256 the reviewer confirmed. Writes contracts/<CID>.yaml and
adapters/<CID>.py from the drafts with: review.status reviewed_for_implementation, the review named in review.reviewer,
review.date, activation.implementation_status validated, every 'status: drafted' (decisions) set to reviewed; then
retains byte-identical copies in validated/ and checks that the registry admits every profile.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

SPEC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SPEC))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cid")
    ap.add_argument("--yaml-sha", required=True)
    ap.add_argument("--adapter-sha", required=True)
    ap.add_argument("--review", required=True)
    ap.add_argument("--date", default="2026-10-21")
    ap.add_argument("--draft-tag", default="path_a_A2_20261020")
    a = ap.parse_args()
    dy = SPEC / "contracts/draft" / f"{a.cid}_{a.draft_tag}.yaml"
    da = SPEC / "contracts/draft" / f"{a.cid}_adapter_{a.draft_tag}.py"
    if sha(dy) != a.yaml_sha or sha(da) != a.adapter_sha:
        raise SystemExit(f"draft sha256 differs from the reviewed files: {sha(dy)} {sha(da)}")
    if not (SPEC / a.review).is_file():
        raise SystemExit(f"no review note {a.review}")
    text = dy.read_text()
    head, sep, rest = text.partition("\nactivation:")
    if not sep:
        raise SystemExit("no activation block")
    head = re.sub(r"(?m)^  status: drafted$", "  status: reviewed_for_implementation", head, count=1)
    note = f"independent review {a.date} ({a.review})"
    if re.search(r"(?m)^  reviewer: ", head):
        head = re.sub(r"(?m)^  reviewer: (.*)$",
                      lambda m: "  reviewer: '" + (m.group(1).strip("'\"") + "; " + note).replace("'", "''") + "'",
                      head, count=1)
    else:
        head = re.sub(r"(?m)^(  status: reviewed_for_implementation)$",
                      lambda m: m.group(1) + "\n  reviewer: '" + note + "'", head, count=1)
    head = re.sub(r"(?m)^  date: .*$", f"  date: '{a.date}'", head, count=1)
    rest = re.sub(r"(?m)^  implementation_status: pending$", "  implementation_status: validated", rest, count=1)
    out = head + sep + rest
    out = out.replace("status: drafted", "status: reviewed")
    target_y, target_a = SPEC / "contracts" / f"{a.cid}.yaml", SPEC / "adapters" / f"{a.cid}.py"
    target_y.write_text(out)
    shutil.copyfile(da, target_a)
    shutil.copyfile(target_y, SPEC / "validated/contracts" / f"{a.cid}.yaml")
    shutil.copyfile(target_a, SPEC / "validated/adapters" / f"{a.cid}.py")
    from kernel_spec import registry
    c = registry.load(a.cid)
    if c.data["review"]["status"] != "reviewed_for_implementation" or \
            c.data["activation"]["implementation_status"] != "validated":
        raise SystemExit("promoted contract does not carry the reviewed / validated status")
    for profile, pd in c.data["profiles"].items():
        binding = {profile: "promotion check"} if pd.get("task_choice_required") else None
        ok, reasons, _ = registry.activation(c, profile, task_bindings=binding)
        print(f"{a.cid}/{profile}: {'admitted' if ok else 'REFUSED ' + '; '.join(reasons)}")
        if not ok:
            raise SystemExit(1)
    if "not yet reviewed" in out:
        print("note: the text still says 'not yet reviewed' somewhere; edit the scope sentence by hand")


if __name__ == "__main__":
    main()
