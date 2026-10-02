#!/usr/bin/env python3
"""Build a small frontier of natural operator families not yet represented.

This is an experiment scheduler, not a bias verdict.  It uses only local model
configuration and the current problem-group ledger.  A family is promoted only
after a real state-bank replay and source-local intervention elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("/data1/tzh/models")
LEDGER = ROOT / "results/property/case_causal_audit_v1/root_cause_closure_current.json"
OUTPUT = ROOT / "results/property/case_causal_audit_v1/natural_semantic_gap_frontier.json"


def _feature_candidates(config: dict[str, Any]) -> list[str]:
    model_type = str(config.get("model_type") or "").lower()
    sections = [config]
    for section_name in ("text_config", "vision_config", "audio_config"):
        section = config.get(section_name)
        if isinstance(section, dict):
            sections.append(section)

    def has_key(*keys: str) -> bool:
        return any(any(key in section for key in keys) for section in sections)

    def value(key: str) -> Any:
        for section in sections:
            if key in section:
                return section[key]
        return None

    features: list[str] = []
    if bool(value("relative_attention")) or str(value("position_embedding_type") or "").startswith("relative"):
        features.append("disentangled_or_relative_attention")
    if (
        bool(value("alibi"))
        or bool(value("use_alibi"))
        or "alibi" in model_type
        or any(str(section.get("model_type") or "").lower() in {"bloom", "falcon"} for section in sections)
    ):
        features.append("alibi_position_bias")
    if value("attention_window") is not None or value("sliding_window") is not None:
        features.append("local_or_sliding_attention")
    if has_key("num_local_experts", "num_experts", "num_experts_per_tok"):
        features.append("mixture_of_experts_routing")
    if model_type in {"mamba", "mamba2", "rwkv"}:
        features.append("state_space_recurrence")
    if has_key("rope_theta", "rotary_pct", "rotary_dim", "rope_parameters"):
        features.append("rotary_position_encoding")
    if any(token in model_type for token in ("vision", "vl", "siglip", "gemma3", "mistral3")) or isinstance(config.get("vision_config"), dict):
        features.append("vision_or_multimodal_path")
    if isinstance(config.get("audio_config"), dict) or "audio_token_id" in config:
        features.append("audio_encoder_path")
    return sorted(set(features))


def _has_weights(model_dir: Path) -> bool:
    return any(
        path.is_file()
        for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth")
        for path in model_dir.glob(pattern)
    )


def _current_groups() -> list[str]:
    if not LEDGER.is_file():
        return []
    data = json.loads(LEDGER.read_text(encoding="utf-8"))
    return [str(row.get("problem_group")) for row in data.get("active_problem_groups", [])]


def build() -> dict[str, Any]:
    groups = _current_groups()
    model_records: list[dict[str, Any]] = []
    feature_models: dict[str, list[dict[str, Any]]] = {}
    for config_path in sorted(MODEL_ROOT.rglob("config.json")):
        model_dir = config_path.parent
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        features = _feature_candidates(config)
        attention_window = config.get("attention_window", config.get("sliding_window", config.get("window_size")))
        if attention_window is None:
            for section_name in ("text_config", "vision_config", "audio_config"):
                section = config.get(section_name)
                if isinstance(section, dict):
                    attention_window = section.get(
                        "attention_window", section.get("sliding_window", section.get("window_size"))
                    )
                    if attention_window is not None:
                        break
        record = {
            "model": str(model_dir),
            "model_type": config.get("model_type"),
            "features": features,
            "attention_window": attention_window,
            "has_local_weights": _has_weights(model_dir),
        }
        model_records.append(record)
        for feature in features:
            feature_models.setdefault(feature, []).append(record)

    gap_rows: list[dict[str, Any]] = []
    for feature, records in sorted(feature_models.items()):
        if feature == "disentangled_or_relative_attention" and any("deberta_disentangled" in group for group in groups):
            status = "EXISTING_FAMILY_NEEDS_NEW_ROOT_ONLY"
        elif feature == "alibi_position_bias" and any("bloom_alibi" in group for group in groups):
            status = "EXISTING_FAMILY_NEEDS_NEW_ROOT_ONLY"
        elif feature == "audio_encoder_path" and any("audio" in group for group in groups):
            status = "EXISTING_FAMILY_NEEDS_NEW_ROOT_ONLY"
        elif feature == "local_or_sliding_attention":
            # The local window is currently a semantic mask choice, not a
            # same-target arithmetic implementation variant.  Do not turn
            # the existence of a sliding mask into a new numerical problem
            # group without a reference that computes the same mask semantics.
            status = "NO_SAME_TARGET_NUMERICAL_VARIANT_IDENTIFIED"
        elif feature == "mixture_of_experts_routing" and any("router" in group or "moe" in group for group in groups):
            status = "EXISTING_FAMILY_NEEDS_NEW_ROOT_ONLY"
        elif feature == "state_space_recurrence" and any("mamba" in group for group in groups):
            status = "EXISTING_FAMILY_NEEDS_NEW_ROOT_ONLY"
        elif feature == "rotary_position_encoding" and any("rope" in group or "rotary" in group for group in groups):
            status = "EXISTING_FAMILY_NEEDS_NEW_ROOT_ONLY"
        elif feature == "vision_or_multimodal_path" and any("vision" in group or "convolution" in group or "position_interpolation" in group for group in groups):
            status = "EXISTING_FAMILY_NEEDS_NEW_ROOT_ONLY"
        else:
            status = "UNREPRESENTED_SEMANTIC_FAMILY"
        gap_rows.append({
            "semantic_feature": feature,
            "status": status,
            "model_count": len(records),
            "models_with_local_weights": sum(bool(row["has_local_weights"]) for row in records),
            "models_with_effective_window_and_weights": sum(
                bool(row["has_local_weights"])
                and isinstance(row["attention_window"], (int, float))
                and row["attention_window"] < 1024
                for row in records
            ),
            "models": [
                {
                    "model": row["model"],
                    "model_type": row["model_type"],
                    "attention_window": row["attention_window"],
                    "has_local_weights": row["has_local_weights"],
                }
                for row in records
            ],
            "promotion_gate": [
                "real model weights and declared input bank",
                "same-local-input candidate/reference replay",
                "single semantic intervention",
                "parameter-write confirmation across held-out states",
                "deduplication against the current problem-group ledger",
            ],
        })

    return {
        "schema": "natural-semantic-gap-frontier-v1",
        "status": "TRIAGE_ONLY_NOT_A_PROBLEM_GROUP_COUNT",
        "model_root": str(MODEL_ROOT),
        "model_count": len(model_records),
        "active_problem_group_count": len(groups),
        "semantic_gap_count": sum(row["status"] == "UNREPRESENTED_SEMANTIC_FAMILY" for row in gap_rows),
        "models": model_records,
        "gaps": gap_rows,
        "interpretation": (
            "A gap is a candidate semantic family. Missing local weights or a missing "
            "same-input intervention prevents any natural-training promotion."
        ),
    }


def main() -> None:
    result = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "model_count": result["model_count"],
        "active_problem_group_count": result["active_problem_group_count"],
        "semantic_gap_count": result["semantic_gap_count"],
        "gaps": [
            {
                "semantic_feature": row["semantic_feature"],
                "status": row["status"],
                "models_with_local_weights": row["models_with_local_weights"],
                "models_with_effective_window_and_weights": row[
                    "models_with_effective_window_and_weights"
                ],
            }
            for row in result["gaps"]
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
