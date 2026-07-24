#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from forkcert.config import load_config


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if any(ch in text for ch in [":", "#", "{", "}", "[", "]", ","]) or text.strip() != text:
        return json.dumps(text)
    return text


def write_simple_yaml(path: Path, data: dict[str, Any]) -> None:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {yaml_scalar(sub_value)}")
        else:
            lines.append(f"{key}: {yaml_scalar(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rewrite_model_paths(cfg: dict[str, Any], checkpoint: str) -> dict[str, Any]:
    out = json.loads(json.dumps(cfg))
    for key in ["policy", "path_ref", "path_alt"]:
        item = out.get(key)
        if isinstance(item, dict) and "model_name_or_path" in item:
            item["model_name_or_path"] = checkpoint
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create phase configs that point at a Phase 0 trained checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="results/configs")
    parser.add_argument("--config", action="append", required=True)
    args = parser.parse_args()

    checkpoint = str(Path(args.checkpoint))
    out_dir = Path(args.out_dir)
    written = []
    for config_path in args.config:
        src = Path(config_path)
        cfg = rewrite_model_paths(load_config(str(src)), checkpoint)
        out = out_dir / src.name.replace(".example", ".phase0_final")
        write_simple_yaml(out, cfg)
        written.append(str(out))
    manifest = out_dir / "phase0_final_configs.json"
    manifest.write_text(
        json.dumps({"checkpoint": checkpoint, "configs": written}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"checkpoint": checkpoint, "configs": written, "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
