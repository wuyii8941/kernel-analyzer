# Project map and live status

## Purpose

The project asks how a concrete LLM training implementation difference becomes
a parameter-update direction and a paired training consequence. The unit is
one concrete forward invocation and the actual backward program that consumes
its saved tensors and cotangent. Historical T1--T4 labels remain in artifacts;
the current paper order is formation decomposition, local/gradient/update
measurement, short screening, and long-run consequence.

## Live denominator

- Four models × three shapes: 12 cells.
- Full-coordinate directional endpoints: 1,562 / 1,562 audited.
- T1: 1,390 pass, 172 reject, 0 pending.
- Historical short-horizon headline: **3 operator-local source/transport
  direction records** (Liger fused CE, Phi `lm_head dX`, and Qwen
  `lm_head dX`). It is not the current long-horizon count.
- Current long-horizon machine audit: **23 unique matrix IDs and 301 row-level
  records**. There are 43 rows with long-run bias evidence plus paired loss
  separation: 3 direct rows with exported late windows, 8 aggregate direct
  rows without exported late-window statistics, and 32 feedback-sustained
  rows. Only 4 rows currently have explicit late rolling-window confirmation.
  Five more rows have paired loss separation without robust long direct bias.
  The broader outcome-relevant count is 105, but it includes historical
  candidates whose 4096-step persistence has not been measured and must not be
  called final persistent cases. Forty-five rows are unresolved or abstain.
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
| `docs/current_mainline.md`, `docs/method.md`, `docs/claims.md`, `docs/gate_history.md` | current mainline, method, claim ledger, and gate/count provenance | keep |
| `docs/effective_antithetic_symmetry.md`, `docs/persistence_property_protocol.md` | detailed formation decomposition and reduction-only orbit predictor | keep |
| `docs/three_mechanism_profiles.md` | one protocol applied to normalization, softmax backward, and attention BMM | keep |
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
