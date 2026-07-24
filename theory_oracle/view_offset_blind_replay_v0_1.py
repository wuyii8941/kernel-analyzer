#!/usr/bin/env python
"""Blind replay for case_002; no issue or patch metadata is consumed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from view_offset_probe_v0_1 import compile_callable, delta, exact, hash_tensor, provenance, view_chain, view_first, view_second


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--input-artifact", required=True)
    parser.add_argument("--reference-artifact", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import torch

    manifest_path = Path(args.case_manifest).resolve()
    input_path = Path(args.input_artifact).resolve()
    reference_path = Path(args.reference_artifact).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if file_sha256(input_path) != manifest["input"]["sha256"]:
        raise RuntimeError("input hash does not match manifest")
    if file_sha256(reference_path) != manifest["reference"]["sha256"]:
        raise RuntimeError("reference hash does not match manifest")
    artifact = torch.load(input_path, map_location="cuda")
    expected = torch.load(reference_path, map_location="cuda")
    warm, target = artifact["warm"].cuda(), artifact["target"].cuda()
    root = Path(args.cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    chain = compile_callable(torch, view_chain, root / "chain")
    first = compile_callable(torch, view_first, root / "first")
    second = compile_callable(torch, view_second, root / "second")
    for row in warm:
        chain(row)
        first(row)
        second(view_first(row))
    torch.cuda.synchronize()
    rows = []
    for index, row in enumerate(target):
        eager_first = view_first(row)
        eager_chain = view_chain(row)
        eager_second = view_second(eager_first)
        compiled_first = first(row)
        compiled_chain = chain(row)
        compiled_second = second(eager_first)
        compiled_second_on_compiled_boundary = second(compiled_first)
        rows.append({
            "index": index,
            "input": hash_tensor(row),
            "expected_reference": hash_tensor(expected[index]),
            "eager_chain": hash_tensor(eager_chain),
            "compiled_chain": hash_tensor(compiled_chain),
            "compiled_matches_reference": exact(compiled_chain, expected[index]),
            "first_boundary_exact": exact(eager_first, compiled_first),
            "second_same_input_production": not exact(eager_second, compiled_second),
            "second_production_delta": delta(eager_second, compiled_second),
            "second_boundary_mediation": not exact(compiled_second, compiled_second_on_compiled_boundary),
            "chain_delta": delta(eager_chain, compiled_chain),
        })
    report = {
        "schema_version": "forkcert.view_offset_blind_replay.v0.1",
        "case_id": manifest["case_id"],
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))},
        "case_manifest_sha256": file_sha256(manifest_path),
        "rows": rows,
        "provenance": {name: provenance(root / name) for name in ("chain", "first", "second")},
        "gates": {
            "complete_witness": any(not row["compiled_matches_reference"] for row in rows),
            "first_boundary_control_exact": all(row["first_boundary_exact"] for row in rows),
            "same_input_local_production": any(row["second_same_input_production"] for row in rows),
            "boundary_mediation_observed": any(row["second_boundary_mediation"] for row in rows),
            "kernel_provenance_present": any(
                item["kernel_paths"]
                for name in ("chain", "first", "second")
                for item in provenance(root / name)["artifacts"]
            ),
        },
        "allowed_claim_level": "LOCAL_INJECTION_WITH_WRAPPER_STOP",
        "claim_scope": "blind local production evidence only; no root-cause or fixed-patch claim",
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"case_id": report["case_id"], "gates": report["gates"], "allowed_claim_level": report["allowed_claim_level"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
