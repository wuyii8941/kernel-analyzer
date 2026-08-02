#!/usr/bin/env python3
"""Run one natural multimodal Qwen3-VL loss forward/backward smoke step."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_step(
    processor: Any,
    image_path: Path,
    *,
    width: int,
    height: int,
    question: str = "What technical mechanism is shown in this diagram?",
    answer: str = "The diagram explains token dropping in a mixture-of-experts model.",
    pad_length: int | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    image = Image.open(image_path).convert("RGB").resize((width, height))

    user_content = [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]
    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    ]
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    common = {"images": [image], "return_tensors": "pt", "do_resize": False}
    prompt = processor(text=[prompt_text], **common)
    full_options: dict[str, Any] = {}
    if pad_length is not None:
        full_options = {
            "padding": "max_length",
            "max_length": pad_length,
            "truncation": True,
        }
    inputs = processor(text=[full_text], **common, **full_options)
    prompt_length = int(prompt["input_ids"].shape[1])
    if not torch.equal(
        inputs["input_ids"][:, :prompt_length], prompt["input_ids"]
    ):
        raise RuntimeError("assistant-loss prefix is not identical to prompt input")

    labels = inputs["input_ids"].clone()
    labels[:, :prompt_length] = -100
    labels[inputs["attention_mask"] == 0] = -100
    if int((labels != -100).sum()) == 0:
        raise RuntimeError("assistant loss mask is empty")
    metadata = {
        "question": question,
        "answer": answer,
        "resized_image": [width, height],
        "sequence_length": int(inputs["input_ids"].shape[1]),
        "padded_sequence_length": pad_length,
        "prompt_length": prompt_length,
        "loss_tokens": int((labels != -100).sum()),
        "pixel_values_shape": list(inputs["pixel_values"].shape),
        "image_grid_thw": inputs["image_grid_thw"].tolist(),
    }
    return dict(inputs), labels, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=224)
    args = parser.parse_args()

    if args.width % 32 or args.height % 32:
        raise ValueError("width and height must be divisible by 32")
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    inputs, labels, input_metadata = prepare_step(
        processor,
        args.image,
        width=args.width,
        height=args.height,
    )

    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.config.use_cache = False
    model.train()
    model.zero_grad(set_to_none=True)

    device_inputs = {key: value.to(device) for key, value in inputs.items()}
    device_labels = labels.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    outputs = model(
        **device_inputs,
        labels=device_labels,
        use_cache=False,
        return_dict=True,
    )
    loss = outputs.loss
    if loss is None or not bool(torch.isfinite(loss)):
        raise RuntimeError("forward loss is absent or nonfinite")
    loss.backward()

    gradient_parameters = 0
    nonfinite_gradient_parameters = 0
    gradient_l2_squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient_parameters += 1
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            nonfinite_gradient_parameters += 1
        gradient_l2_squared += float(torch.sum(gradient * gradient).cpu())

    payload = {
        "schema": "kernel-analyzer.round2-vl-smoke.v1",
        "status": "COMPLETE" if nonfinite_gradient_parameters == 0 else "NONFINITE",
        "scope": "one natural Qwen3-VL multimodal loss forward/backward step",
        "model": {
            "path": str(args.model.resolve()),
            "config_sha256": sha256_file(args.model / "config.json"),
            "dtype": "bfloat16",
            "attention": "eager",
        },
        "input": {
            **input_metadata,
            "image_path": str(args.image.resolve()),
            "image_sha256": sha256_file(args.image),
        },
        "result": {
            "loss": float(loss.detach().cpu()),
            "gradient_parameters": gradient_parameters,
            "nonfinite_gradient_parameters": nonfinite_gradient_parameters,
            "global_gradient_l2": gradient_l2_squared**0.5,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "tf32": False,
            "seed": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["result"], sort_keys=True))


if __name__ == "__main__":
    main()
