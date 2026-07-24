# Qwen3 case 001 — blind localization result

## Scope

This case uses the TorchTitan Qwen3 `Attention` module with
`attn_type="varlen"`.  The blind locator received:

- a small deterministic Qwen3 Attention input and state artifact;
- a declared reference endpoint and reference boundary artifacts;
- the buggy TorchTitan source worktree;
- generic natural module boundaries and an unfiltered TorchDispatch trace.

It did **not** receive the issue number, PR, commit IDs, patch text, or the
developer root-cause explanation.  Those are stored only in the external bug
registry and were used after the blind report was frozen.

The current Tesla T4 cannot execute the historical varlen FlashAttention
kernel directly.  The harness therefore uses the same per-document SDPA
compatibility fallback for both source worktrees.  This preserves the packed
layout interface but means this is a Qwen3 implementation-localization case,
not a claim about the T4 FlashAttention kernel.

## Blind Oracle result

The endpoint Oracle observed a stable, finite implementation/reference
discrepancy.  The two repeated buggy executions were exactly identical, so the
case does not look like within-state runtime noise.

Natural region comparisons were exact through `inner_attention`.  The first
declared named output that differed was the subsequent `wo` output; the full
Attention output also differed.  This distinguishes the attention backend
from the downstream projection as a producer/propagation boundary.

The generic operation traces contained two extra operations in the buggy run.
Their first sequence mismatch was:

```text
reference: view
candidate:  transpose
```

No operation name was hard-coded into the locator as a target.  The result was
therefore frozen as:

```text
OPERATION_INTERVAL_CANDIDATE
```

not as a root-cause claim.

## Post-reveal score

After freezing the blind report, the historical PR was revealed.  Its changed
source path is `torchtitan/models/qwen3/model/model.py`, and its patch moves a
transpose into only the flex/sdpa branches.  The blind operation interval
covered that mechanism and the changed source path was covered.

The strongest supported result is:

```text
HISTORICAL_PATCH_COVERED_OPERATION_INTERVAL
```

This is positive evidence that the Oracle plus generic local trace can narrow
this case to the correct Attention operation interval without a bug-specific
heuristic.  It is not proof of a unique causal root, a compiler pass, or a
generated GPU kernel.  The external issue and fix are the independent ground
truth used only for scoring.

## Reproduction artifacts

- Opaque case: `data/qwen_bug_sources/qwen3_attention_varlen_layout_case_001/`
- Blind report: `results/operator_oracle/qwen3_case001_blind_locator.json`
- Post-reveal score: `results/operator_oracle/qwen3_case001_post_reveal_score.json`
- Generic runner: `theory_oracle/qwen3_historical_case_runner_v0_1.py`
- Generic locator: `theory_oracle/qwen3_blind_locator_v0_1.py`
- Blind-package audit: `results/operator_oracle/qwen3_case001_blind_protocol_audit.json`

The protocol audit found no issue/fix/root-cause metadata keys, no source
worktree or reference-run files, and all declared locator exclusions present.
