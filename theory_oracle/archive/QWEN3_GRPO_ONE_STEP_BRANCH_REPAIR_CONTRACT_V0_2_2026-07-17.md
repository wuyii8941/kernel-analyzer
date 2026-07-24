# Qwen3 GRPO One-Step Branch-Repair Contract v0.2 — 2026-07-17

## Status and correction

Frozen before the corrected A/B/C run. It incorporates all estimands and claim
limits from v0.1.

The first execution attempt is `INVALID`: its independent batch scorer computed
full logits without the Trainer's `logits_to_keep` forward argument. Eager happened
to reproduce the selected anchor, but compiled differed from the observed compiled
endpoint by about `0.0107` and selected the eager clipping branch before repair.
Consequently B and C were identical. That is endpoint-realization failure, not a
zero branch effect.

## Corrected realization gate

The corrected executor must reproduce the exact Trainer scoring path:

- identical left-padded prompt/response tensors and attention mask;
- `use_cache=False`;
- Qwen `logits_to_keep = completion_length + 1`;
- exclude the final next-token logits and retain exactly the completion positions;
- TRL `selective_log_softmax` at temperature 1;
- identical FP16 autocast and SDPA MATH context.

In addition to the v0.1 gates, A's selected log-probability must exactly equal the
confirmed eager anchor and B/C must exactly equal the confirmed compiled anchor.
Failure is `INVALID`. No tolerance is introduced.

The state, event, SGD probe, intervention and endpoint remain unchanged. New output
paths preserve the invalid first attempt.

