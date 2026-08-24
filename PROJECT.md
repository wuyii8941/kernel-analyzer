# Project map and live status

## Purpose

The project asks how a local operator or fused-region difference becomes a
directional training error.  The unit is one concrete forward invocation and
the actual backward program that consumes its saved tensors and cotangent.
The strict Flash-style chain is T1 local difference, T2 endpoint repair and
parameter reach, T3 complete coherent carrier, and T4 paired weight
accumulation.  A T1 screen or a carrier alone is not a case.

## Live denominator

- Four models × three shapes: 12 cells.
- Full-coordinate directional endpoints: 1,562 / 1,562 audited.
- T1: 1,390 pass, 172 reject, 0 pending.
- Historical short-horizon headline: **3 operator-local source/transport
  direction records** (Liger fused CE, Phi `lm_head dX`, and Qwen
  `lm_head dX`). It is not the current long-horizon count.
- Current long-horizon audit: **23 unique matrix IDs, 29 matrix records, 11
  historical candidates plus two held-out family-replication rows, and 28
  merged audit rows**. Six final cases are confirmed under the current rule:
  five direct 4096-step cases (Liger, Phi `lm_head dX`, Qwen `lm_head dX`,
  Llama `lm_head dX`, and Ministral `lm_head dX`) plus Qwen3-VL SiLU, whose
  long-run feedback separation has a paired loss gap. Mamba, saved-P and both
  `v_proj` rows do not have robust long direct direction; layer-23, Gemma4
  and DeepSeek `dV` remain unresolved for different replay/formation
  reasons. Llama and Ministral are same-family model replications, not new
  implementation classes.
- Historical audit registries contain 6 strict F+B/repair/carrier/trajectory
  records and 8 paired-separation records. Those are provenance counts, not
  counts of confirmed persistent operator-local bias.
- The historical 48-endpoint pre-deduplication snapshot is superseded as a
  strict-case count; endpoint instances never substitute for independent
  mechanisms or complete current gates.
- Mamba seq64 T2: 43 complete (9 pass, 34 reject); T3: 0 carrier survivors.
- Mamba seq256 T2: the 58-row small shard is complete (57 pass, 1 reject).
  The 524-row large shard is still pending; therefore the seq256 cell has no
  cell-level T2 completion marker and must not enter T3 yet.

These numbers are intentionally conservative.  Pending rows remain in the
denominator and are not called normal controls or cases.

The corrected short-screen evaluation contains 15 declared rows. Its 16-step
parameter-update direction score is a retrospective prioritization result,
not a long-horizon label or universal accuracy. Qwen did not escalate in its
cold-start short window but is robust in the separate warm-state 4096-step
review, so non-escalation cannot be interpreted as long-term safety. The
current `NEW_IMPL` Gemma validation is source-negative with Adam-state feedback
and adds no long-horizon direct positive.

## Directory map

| Path | Role | Retention |
|---|---|---|
| `src/` | reusable analyzer library | keep |
| `scripts/` | capture, proof, audit, T1--T4 runners | keep |
| `tests/` | unit tests | keep |
| `case.md`, `cases_flash_style.md` | case registry and Flash-style logic | keep |
| `docs/coverage.md`, `docs/denominator.md`, `docs/bias_protocol.md` | compact protocol/status references | keep |
| `docs/current_mainline.md`, `docs/claims.md`, `docs/gate_history.md` | current result, claim ledger, and gate/count provenance | keep |
| `docs/bias_properties.md` | candidate bias-formation tournament | keep |
| `docs/l23_qproj_tile.md` | detailed mathematical case derivation | keep |
| `results/coverage/` | frozen inputs, releases, ledgers, T1--T4 evidence | keep; remove only validated intermediates |
| `results/final/` | compact historical derivations and summaries | keep |
| `archive/` | old round material, ignored by git | optional historical archive |

The large T1 artifacts and runtime releases are required to resume the
remaining Mamba seq256 T2 denominator.  They must not be removed during a
cleanup.

## Safe cleanup policy

Delete generated build products, Python bytecode caches, stale partial files,
and intermediate artifacts whose final union is present and whose filenames
are not referenced by the runners.  Never delete a frozen input bank, a
runtime release, a full-coordinate artifact, an F+B proof ledger, a causal /
carrier / trajectory result, or a mathematical derivation merely because it
is large.
