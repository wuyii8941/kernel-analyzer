# Coverage status

The declared denominator is four models × three sequence lengths = 12 cells:
Qwen3-1.7B, Mamba-130M, Phi-4-mini, and DeepSeek-R1-Qwen3-8B at 64, 128,
and 256 tokens.

## Closed coverage layers

| Layer | Status |
|---|---:|
| execution and F+B origin accounting | 12 / 12 cells |
| concrete analytic F+B witnesses | 12 / 12 cells |
| candidate-to-F+B binding | 12 / 12 cells |
| precision and same-dtype Oracles | 12 / 12 cells |
| full-coordinate T1 endpoint audit | 1,562 / 1,562 endpoints |

The full-coordinate T1 split is 1,390 passes, 172 rejects, and zero pending.
A T1 pass is eligibility for causal follow-up, not a Flash-style case.

## Causal funnel

Mamba seq64 has completed T2 for all 43 T1 survivors (9 pass, 34 reject),
with no T3 carrier survivors.  Mamba seq256 has completed its 58-row small
shard (57 pass, 1 reject); its 524-row large shard remains pending.  No
cell-level seq256 T2 marker is valid until both shards are joined.  T3 and T4
must consume only a complete T2 cell.

The authoritative live files are:

- `results/coverage/four_model_full_operator_status.json`
- `results/coverage/cases/full_coordinate_audit.json.gz`
- `results/coverage/cases/causal/`
- `results/coverage/cases/carrier/`
- `results/coverage/cases/trajectory/`
