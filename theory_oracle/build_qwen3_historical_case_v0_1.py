#!/usr/bin/env python
"""Materialize a patch-free Qwen3 varlen-attention case from a fixed reference."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-source-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--case-id", default="qwen3_attention_varlen_layout_case_001")
    parser.add_argument("--attn-type", choices=["varlen", "sdpa"], default="varlen")
    parser.add_argument("--flavor", choices=["toy", "1.7B"], default="toy")
    args = parser.parse_args()
    import torch

    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.fixed_source_root.resolve()))
    model_mod = importlib.import_module("torchtitan.models.qwen3.model.model")
    args_mod = importlib.import_module("torchtitan.models.qwen3.model.args")
    sys.path.remove(str(args.fixed_source_root.resolve()))
    config = {
        "dim": 2048 if args.flavor == "1.7B" else 64,
        "n_layers": 28 if args.flavor == "1.7B" else 1,
        "n_heads": 16 if args.flavor == "1.7B" else 4,
        "n_kv_heads": 8 if args.flavor == "1.7B" else 2,
        "vocab_size": 151936 if args.flavor == "1.7B" else 128,
        "head_dim": 128 if args.flavor == "1.7B" else 16,
        "hidden_dim": 6144 if args.flavor == "1.7B" else 128,
        "norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "qk_norm": True,
        "max_seq_len": 4096 if args.flavor == "1.7B" else 16,
        "depth_init": True,
        "attn_type": args.attn_type,
        "attn_mask_type": "causal",
        "eos_id": 127,
        "enable_weight_tying": False,
        "dtype": "torch.float16",
    }
    seed = 20260723
    torch.manual_seed(seed)
    subject = model_mod.Attention(args_mod.Qwen3ModelArgs(**{k: v for k, v in config.items() if k != "dtype"}))
    weights = out / "weights.pt"
    torch.save({key: value.detach().cpu() for key, value in subject.state_dict().items()}, weights)
    batch, seq, dim = 2, 8, config["dim"]
    x = torch.randn(batch, seq, dim, dtype=torch.float16)
    tokens = torch.tensor([[1, 2, 127, 4, 5, 6, 7, 8], [9, 10, 11, 12, 127, 14, 15, 16]], dtype=torch.long)
    rope = model_mod.precompute_rope_cache(config["head_dim"], seq, config["rope_theta"])
    inputs = out / "inputs.pt"
    torch.save({"x": x, "tokens": tokens, "rope_cache": rope}, inputs)
    manifest = {
        "schema_version": "forkcert.qwen3-opaque-case.v0.1",
        "case_id": args.case_id,
        "visibility": "patch_free_opaque_case",
        "reference_role": "declared_reference_execution",
        "contract": {"endpoint": "attention output tensor", "candidate_must_match_reference": True},
        "subject": {"family": "Qwen3", "component": "Attention", "config": config, "regions": ["wq", "wk", "wv", "q_norm", "k_norm", "inner_attention", "wo", "__root__"]},
        "input": {"path": "inputs.pt", "sha256": sha256_file(inputs), "seed": seed, "shape": [batch, seq, dim]},
        "artifacts": {"inputs": "inputs.pt", "weights": "weights.pt"},
        "locator_exclusions": ["issue identifier", "fixed revision", "patch", "pull-request discussion", "root-cause notes"],
    }
    (out / "case_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # Import after the manifest exists so the runner uses exactly the declared case.
    from theory_oracle.qwen3_historical_case_runner_v0_1 import execute_case
    reference_run = out / "reference_run"
    execute_case(args.fixed_source_root, out, reference_run)
    run = json.loads((reference_run / "run.json").read_text())
    reference_endpoint = out / "reference_endpoint.pt"
    shutil.copy2(reference_run / run["endpoint"]["path"], reference_endpoint)
    reference_regions = out / "reference_regions"
    reference_regions.mkdir()
    for name, row in run["regions"].items():
        source = reference_run / row["path"]
        shutil.copy2(source, reference_regions / Path(row["path"]).name)
        row["path"] = str(Path("reference_regions") / Path(row["path"]).name)
    shutil.copy2(reference_run / run["trace"]["path"], out / "reference_trace.json")
    manifest["reference"] = {
        "endpoint": {"path": str(reference_endpoint.relative_to(out)), "sha256": sha256_file(reference_endpoint)},
        "regions": run["regions"],
        "trace": {"path": "reference_trace.json", "sha256": sha256_file(out / "reference_trace.json")},
    }
    (out / "case_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"case_dir": str(out), "case_id": manifest["case_id"], "reference": manifest["reference"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
