# Phase-4 Megatron asset audit v0.1

## Scope

Phase 4 is a frozen **Megatron matched training step**, not a Qwen substitute,
free-running trajectory, or a generic language-model smoke test.  Before any
installation or model construction, the required assets must exist so the
result can bind a stable semantic-contract violation to a specific training
state.

## Audit performed

On 2026-07-24, the workspace-wide filename census found no Megatron/MCore
source tree, training runner, checkpoint, frozen optimizer state, or matched
step artifact under `/data1/tzh`.  The intended PyTorch runtime
`/data1/tzh/pt211_fresh_env` has PyTorch `2.11.0+cu126` and no importable
`megatron` package.

This is an asset/configuration finding, not a CUDA finding.  CUDA-capable T4
hardware and the project CUDA environments have already run the current
calibration and historical GPU controls.

## Required Phase-4 handoff

Before Phase 4 can start, provide either an existing Megatron experiment path
or authorization to construct a fresh Megatron-Core subject, plus:

1. source revision and dependency/environment lock;
2. a state ID with model parameters, optimizer state, scheduler/loss-scale
   state, batch/token inputs, and all relevant RNG states;
3. a declared semantic contract exhibiting a stable one-step symptom under
   the same eager/compiled conditions;
4. two or more frozen matched states: one discovery state and held-out states;
5. provenance/debug-artifact capture permissions for the bound compiler
   configuration.

Without these, installing Megatron alone would produce a new subject but not
the required complex-training localization case.  It must not be represented
as a recovered historical Megatron failure.

## Ordering boundary

Phase 3 also requires independently withheld historical packages.  Its
evaluator-owned truth cannot be reconstructed by the analyst after reading
public issue material.  The project is therefore waiting on two explicit
external inputs, rather than advancing out of order or substituting a simpler
case:

- two evaluator-held historical packages for Phase 3;
- Megatron source/state assets (or explicit authorization to build a fresh
  subject) for Phase 4.
