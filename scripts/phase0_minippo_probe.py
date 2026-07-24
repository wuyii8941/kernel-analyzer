#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from forkcert.config import load_config
from forkcert.io import read_jsonl
from forkcert.logprob_runner import PathConfig, _tokenize_prompt_response, configure_determinism, load_hf_path


def path_config(data: dict) -> PathConfig:
    item = data.get("policy", data.get("path_ref", data))
    return PathConfig(
        name=item.get("name", "phase0-policy"),
        model_name_or_path=item["model_name_or_path"],
        dtype=item.get("dtype", "bf16"),
        device=item.get("device", "cuda"),
        compile_model=item.get("compile_model", False),
        attn_implementation=item.get("attn_implementation"),
        attention_backend=item.get("attention_backend"),
        logits_upcast_fp32=item.get("logits_upcast_fp32", True),
    )


def stable_advantage_sign(case_id: str, token_index: int) -> int:
    score = sum(ord(ch) for ch in case_id) + token_index
    return 1 if score % 2 == 0 else -1


def select_trainable_parameters(model, mode: str):
    if mode == "all":
        for param in model.parameters():
            param.requires_grad_(True)
        return [param for param in model.parameters() if param.requires_grad]
    if mode == "lm_head":
        for param in model.parameters():
            param.requires_grad_(False)
        head = model.get_output_embeddings()
        if head is None:
            raise ValueError("model has no output embeddings/lm_head to train")
        for param in head.parameters():
            param.requires_grad_(True)
        return [param for param in head.parameters() if param.requires_grad]
    raise ValueError(f"unsupported trainable mode: {mode}")


def token_logprobs_for_sample(tokenizer, model, config: PathConfig, sample: dict) -> tuple[list[int], list[str], torch.Tensor]:
    encoded = _tokenize_prompt_response(tokenizer, sample["prompt"], sample["response"], config.device)
    input_ids = encoded["input_ids"]
    prompt_len = encoded["prompt_len"]
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    if config.logits_upcast_fp32:
        logits = logits.float()
    log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    values = []
    token_ids = []
    token_texts = []
    for full_pos in range(prompt_len, input_ids.shape[1]):
        pred_pos = full_pos - 1
        token_id = int(input_ids[0, full_pos].item())
        token_ids.append(token_id)
        token_texts.append(tokenizer.decode([token_id], clean_up_tokenization_spaces=False))
        values.append(target_log_probs[0, pred_pos])
    return token_ids, token_texts, torch.stack(values)


@torch.no_grad()
def collect_old_logps(tokenizer, model, config: PathConfig, samples: list[dict]) -> dict[tuple[str, int], dict]:
    old = {}
    for sample in samples:
        token_ids, token_texts, logps = token_logprobs_for_sample(tokenizer, model, config, sample)
        for i, (token_id, token_text, logp) in enumerate(zip(token_ids, token_texts, logps)):
            sign = stable_advantage_sign(sample["case_id"], i)
            old[(sample["case_id"], i)] = {
                "case_id": sample["case_id"],
                "token_index": i,
                "token_id": token_id,
                "token_text": token_text,
                "old_logp": float(logp.item()),
                "advantage": float(sign),
                "advantage_sign": sign,
            }
    return old


def dump_minibatch_state(
    fh,
    tokenizer,
    model,
    config: PathConfig,
    samples: list[dict],
    old: dict,
    epoch: int,
    minibatch: int,
    *,
    step: int | None = None,
    state_label: str = "pre_minibatch",
) -> None:
    with torch.no_grad():
        for sample in samples:
            token_ids, token_texts, logps = token_logprobs_for_sample(tokenizer, model, config, sample)
            for i, (token_id, token_text, new_logp) in enumerate(zip(token_ids, token_texts, logps)):
                old_state = old[(sample["case_id"], i)]
                row = {
                    "case_id": sample["case_id"],
                    "token_index": i,
                    "token_id": token_id,
                    "token_text": token_text,
                    "old_logp": old_state["old_logp"],
                    "new_logp": float(new_logp.item()),
                    "advantage": old_state["advantage"],
                    "advantage_sign": old_state["advantage_sign"],
                    "epoch": epoch,
                    "minibatch": minibatch,
                    "optimizer_step": step,
                    "state": state_label,
                }
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()


def ppo_minibatch_loss(tokenizer, model, config: PathConfig, samples: list[dict], old: dict, eps: float) -> torch.Tensor:
    losses = []
    for sample in samples:
        _token_ids, _token_texts, logps = token_logprobs_for_sample(tokenizer, model, config, sample)
        for i, logp in enumerate(logps):
            state = old[(sample["case_id"], i)]
            old_logp = torch.tensor(state["old_logp"], dtype=logp.dtype, device=logp.device)
            advantage = torch.tensor(state["advantage"], dtype=logp.dtype, device=logp.device)
            ratio = torch.exp(logp - old_logp)
            clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
            surrogate = torch.minimum(ratio * advantage, clipped * advantage)
            losses.append(-surrogate)
    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 0 margin dumps with a minimal PPO-style probe.")
    parser.add_argument("--config", default="configs/phase0_minippo.example.yaml")
    parser.add_argument("--samples", default="data/prompt_pairs.jsonl")
    parser.add_argument("--out-jsonl", default="data/phase0_minippo_dump.jsonl")
    parser.add_argument("--final-rollout-jsonl", default=None, help="Optional final-state rollout dump for Phase 4/5 natural scans.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--target-steps", type=int, default=None, help="Override epochs and keep cycling minibatches until this many optimizer steps are collected.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--trainable", choices=["lm_head", "all"], default="lm_head")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-dir", default="data/phase0_checkpoints")
    parser.add_argument("--save-final-checkpoint", action="store_true", help="Save the final policy for Phase 1/4 logprob rescoring.")
    parser.add_argument("--final-checkpoint-dir", default="data/phase0_policy_final")
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    configure_determinism(seed=0)
    cfg = load_config(args.config)
    policy = path_config(cfg)
    samples = read_jsonl(args.samples)[: args.max_samples]
    if not samples:
        raise ValueError("Phase 0 requires at least one prompt-response sample.")
    tokenizer, model = load_hf_path(policy)
    trainable_params = select_trainable_parameters(model, args.trainable)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    old = collect_old_logps(tokenizer, model, policy, samples)

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir)
    if args.save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt_dir = Path(args.final_checkpoint_dir)
    metadata_path = out.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "config": cfg,
                "sample_count": len(samples),
                "epochs": args.epochs,
                "target_steps": args.target_steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "eps": args.eps,
                "trainable": args.trainable,
                "trainable_parameter_count": int(sum(param.numel() for param in trainable_params)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with out.open("w", encoding="utf-8") as fh:
        step = 0
        epoch = 0
        while True:
            for start in range(0, len(samples), args.batch_size):
                if args.target_steps is not None and step >= args.target_steps:
                    break
                minibatch = samples[start : start + args.batch_size]
                dump_minibatch_state(fh, tokenizer, model, policy, minibatch, old, epoch, start // args.batch_size, step=step)
                optimizer.zero_grad(set_to_none=True)
                loss = ppo_minibatch_loss(tokenizer, model, policy, minibatch, old, args.eps)
                loss.backward()
                optimizer.step()
                step += 1
                if args.save_checkpoints and step % int(cfg.get("checkpoint_every", 20)) == 0:
                    model.save_pretrained(ckpt_dir / f"step_{step:06d}")
                    tokenizer.save_pretrained(ckpt_dir / f"step_{step:06d}")
            epoch += 1
            if args.target_steps is not None:
                if step >= args.target_steps:
                    break
            elif epoch >= args.epochs:
                break
    if args.save_final_checkpoint:
        final_ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_ckpt_dir)
        tokenizer.save_pretrained(final_ckpt_dir)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["final_checkpoint_dir"] = str(final_ckpt_dir)
        metadata["total_optimizer_steps"] = step
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.final_rollout_jsonl:
        final_rollout = Path(args.final_rollout_jsonl)
        final_rollout.parent.mkdir(parents=True, exist_ok=True)
        with final_rollout.open("w", encoding="utf-8") as fh:
            dump_minibatch_state(fh, tokenizer, model, policy, samples, old, epoch, -1, step=step, state_label="final")
    print(f"wrote Phase 0 PPO-style dump to {out}")


if __name__ == "__main__":
    main()
