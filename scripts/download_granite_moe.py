#!/usr/bin/env python3
"""Download the pinned small Granite MoE base checkpoint."""

from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="ibm-granite/granite-3.1-1b-a400m-base",
    revision="408b6e90baab8cf24f4aa9f8e19703ffa0a53b29",
    local_dir="/data1/tzh/models/ibm-granite/granite-3.1-1b-a400m-base",
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "model-*.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
)
