#!/usr/bin/env python3
"""Rebuild only the wrapper/inventory release for an existing model/shape.

This is used when a harmless observer-code change makes an old wrapper release
stale.  It performs one warm F+B compile, writes a new release, and reuses the
old semantic task plan only after checking that all task IDs still exist in the
new inventory/campaign.  It does not run a scientific screen.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "archive/round1_code/src")]

from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import (  # noqa: E402
    freeze_or_validate_release,
    load_model,
    wrapper_modules,
)


def load_json(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--old-release", type=Path, required=True)
    parser.add_argument("--new-release", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-graph-breaks", action="store_true",
                        help="rebind a release using the repository's graph-break-compatible runner")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="validate a release already created by a prior warm compile")
    args = parser.parse_args()
    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if not states:
        raise RuntimeError("input bank is empty")
    if args.new_release.exists() and any(args.new_release.iterdir()) and not args.reuse_existing:
        raise RuntimeError("new release directory is non-empty")

    if args.reuse_existing:
        inventory = args.new_release / "inventory.json.gz"
        campaign = args.new_release / "campaign.json.gz"
        capture = args.new_release / "capture.json"
        if not all(path.exists() for path in (inventory, campaign, capture)):
            raise RuntimeError("existing release is incomplete")
    else:
        configure_candidate_runtime(24000)
        device = torch.device(args.device)
        model = load_model(args.architecture, args.model, device)
        start = len(PyCodeCache.modules)
        candidate = torch.compile(
            LossStep(model), backend="inductor",
            fullgraph=not args.allow_graph_breaks, dynamic=False,
        )
        tokens = states[0].get("input_ids", states[0].get("token_ids"))
        warm = torch.tensor([tokens], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        candidate(warm).backward()
        torch.cuda.synchronize(device)
        wrappers = wrapper_modules(list(PyCodeCache.modules[start:]))
        inventory, campaign = freeze_or_validate_release(
            modules=wrappers, release=args.new_release,
            architecture=args.architecture, input_bank=args.input_bank,
            state=states[0], allow_graph_breaks=args.allow_graph_breaks,
        )

    old_plan = args.old_release / "same_dtype_tasks.json.gz"
    new_plan = args.new_release / "same_dtype_tasks.json.gz"
    shutil.copy2(old_plan, new_plan)
    plan = load_json(new_plan)
    inventory_payload = load_json(inventory)
    campaign_payload = load_json(campaign)
    available = {
        (str(row.get("region_id")), str(row.get("symbol")))
        for row in campaign_payload.get("rows", [])
    }
    generated = inventory_payload.get("generated_regions", {}).get("inventory", {})
    available |= {
        (str(row.get("region_id")), str(row.get("symbol")))
        for row in generated.get("regions", [])
    }
    direct = inventory_payload.get("direct_runtime_calls", {})
    available |= {
        (str(row.get("region_id")), str(row.get("symbol")))
        for row in direct.get("rows", [])
    }
    requested = {
        (str(row.get("candidate_region_id")), str(row.get("symbol")))
        for row in plan.get("rows", [])
    }
    if not requested <= available:
        missing = sorted(requested - available)
        raise RuntimeError(f"new release lost {len(missing)} frozen task IDs")
    print(json.dumps({"new_release": str(args.new_release), "tasks": len(requested),
                      "inventory": str(inventory), "campaign": str(campaign)}))


if __name__ == "__main__":
    main()
