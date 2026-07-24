#!/usr/bin/env python
"""Build a patch-hidden Qwen3-shaped higher-order-gradient case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--case-id", default="opaque_qwen3_grad_case_a")
    parser.add_argument("--actual-boundary", action="store_true")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.manual_seed(args.seed)
    weight_source = "deterministic_qwen3_1p7b_shape"
    checkpoint_revision = None
    tokens = None
    input_source = "deterministic_hidden_state"
    if args.snapshot_dir:
        from safetensors.torch import load_file

        snapshot = args.snapshot_dir.resolve()
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        shard = snapshot / index["weight_map"]["model.layers.0.self_attn.q_proj.weight"]
        q_weight = load_file(str(shard), device="cpu")["model.layers.0.self_attn.q_proj.weight"]
        args.hidden_size = int(q_weight.shape[0])
        # Save the transposed weight so x @ weight is algebraically the
        # Qwen3 q_proj F.linear(x, q_proj.weight) operation.
        weight = q_weight.t().contiguous().float()
        weight_source = "Qwen3-1.7B-layer0-self_attn-q_proj-weight"
        checkpoint_revision = snapshot.name
        if args.actual_boundary:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                snapshot,
                dtype=torch.float32,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            tokens = (torch.arange(args.tokens, dtype=torch.long)[None, :] * 17 + 11) % int(model.config.vocab_size)
            with torch.no_grad():
                embedded = model.model.embed_tokens(tokens)
                x = model.model.layers[0].input_layernorm(embedded)[0].contiguous().float()
            del model, embedded
            input_source = "Qwen3-1.7B-layer0-actual-embedding-input-layernorm-boundary"
    else:
        # The dimensions match Qwen3-1.7B's hidden size and a short token span.
        weight = torch.randn(args.hidden_size, args.hidden_size, dtype=torch.float32)
    if tokens is None:
        x = torch.randn(args.tokens, args.hidden_size, dtype=torch.float32)
    torch.save(x, out / "inputs.pt")
    torch.save(weight, out / "weights.pt")
    manifest = {
        "schema_version": "forkcert.qwen3-opaque-compiler-grad-case.v0.1",
        "case_id": args.case_id,
        "visibility": "patch_free_opaque_case",
        "locator_exclusions": [
            "issue identifier", "fixed revision", "patch", "pull-request discussion", "root-cause notes"
        ],
        "subject": {
            "family": "Qwen3",
            "scale": "1.7B hidden-size projection",
            "hidden_size": args.hidden_size,
            "tokens": args.tokens,
            "region": "attention_query_projection_equivalent",
            "operator_contract": "matrix projection followed by differentiable first derivative",
            "weight_source": weight_source,
            "checkpoint_revision": checkpoint_revision,
            "input_source": input_source,
        },
        "input": {
            "seed": args.seed,
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "sha256": sha256(out / "inputs.pt"),
        },
        "artifacts": {
            "inputs": "inputs.pt",
            "weights": "weights.pt",
            "weights_sha256": sha256(out / "weights.pt"),
        },
        "endpoint": {
            "name": "higher_order_gradient_contract",
            "fields": ["numeric_value", "requires_grad", "has_grad_fn", "backward_succeeds"],
        },
    }
    if tokens is not None:
        torch.save(tokens, out / "tokens.pt")
        manifest["artifacts"]["tokens"] = "tokens.pt"
        manifest["input"]["token_shape"] = list(tokens.shape)
    (out / "case_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"case_dir": str(out), "input_sha256": manifest["input"]["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
