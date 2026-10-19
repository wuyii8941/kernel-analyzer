#!/usr/bin/env python3
"""Held-out model selection for sufficiency evaluation E3 (docs/SUFFICIENCY_PROTOCOL.md section E3), fixed before any
held-out measurement. Rule, applied in order to the Hugging Face Hub listing sorted by downloads:

  - pipeline tag text-generation or fill-mask, library transformers, not gated, has safetensors weights;
  - model_type is supported natively by the installed transformers (CONFIG_MAPPING, no remote code);
  - model_type differs from every development model (DEV_TYPES, the 23 three_way models) and from every model
    already selected;
  - at most MAX_PARAMS parameters (safetensors metadata);
  - take the first N such models.

The raw listing pages and every candidate's decision are written next to the selection, so the selection can be
re-derived. Selected models stay in the denominator whatever happens later (download, compile or capture failure).

    python 5_specification/sufficiency/select_heldout.py --out NEW.json [--n 6]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DEV_TYPES = {"bert", "bloom", "deberta-v2", "electra", "falcon", "gemma3", "gemma3_text", "gemma4", "gpt2", "gpt_neo",
             "gpt_neox", "granitemoe", "llama", "mamba", "mistral3", "ministral3", "olmoe", "opt", "phi3", "qwen2",
             "qwen3", "qwen3_vl", "roberta", "rwkv", "t5"}
MAX_PARAMS = 2_000_000_000
API = "https://huggingface.co/api/models"


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--pages", type=int, default=10, help="listing pages of 100 per pipeline tag")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; selections are never overwritten")
    from transformers import CONFIG_MAPPING
    import transformers
    native = set(CONFIG_MAPPING.keys())
    listing = []
    for tag in ("text-generation", "fill-mask"):
        for page in range(a.pages):
            q = urllib.parse.urlencode({"pipeline_tag": tag, "library": "transformers", "sort": "downloads",
                                        "direction": -1, "limit": 100, "skip": 100 * page})
            try:
                rows = get(f"{API}?{q}")
            except Exception as e:  # noqa: BLE001
                listing.append({"tag": tag, "page": page, "error": str(e)})
                break
            listing += [{"tag": tag, "id": r["id"], "downloads": r.get("downloads", 0)} for r in rows]
            if len(rows) < 100:
                break
    cands = sorted({r["id"]: r for r in listing if "id" in r}.values(), key=lambda r: -r["downloads"])
    chosen, decisions, used = [], [], set(DEV_TYPES)
    for c in cands:
        if len(chosen) >= a.n:
            break
        d = {"id": c["id"], "downloads": c["downloads"]}
        try:
            info = get(f"{API}/{c['id']}?expand[]=config&expand[]=safetensors&expand[]=gated&expand[]=siblings")
        except Exception as e:  # noqa: BLE001
            d["decision"] = f"skip: metadata unavailable ({e})"
            decisions.append(d)
            continue
        mt = (info.get("config") or {}).get("model_type")
        params = (info.get("safetensors") or {}).get("total")
        d.update(model_type=mt, params=params, gated=info.get("gated"))
        if info.get("gated"):
            d["decision"] = "skip: gated"
        elif not params:
            d["decision"] = "skip: no safetensors parameter count"
        elif mt not in native:
            d["decision"] = "skip: model_type not native to the installed transformers"
        elif mt in used:
            d["decision"] = "skip: model_type already used (development or earlier selection)"
        elif params > MAX_PARAMS:
            d["decision"] = "skip: more than 2B parameters"
        else:
            d["decision"] = "selected"
            chosen.append(d)
            used.add(mt)
        decisions.append(d)
    a.out.write_text(json.dumps({"rule": __doc__, "transformers": transformers.__version__,
                                 "dev_types": sorted(DEV_TYPES), "selected": chosen, "decisions": decisions,
                                 "listing": listing}, indent=1))
    print(json.dumps(chosen, indent=1))
    return 0 if len(chosen) == a.n else 1


if __name__ == "__main__":
    sys.exit(main())
