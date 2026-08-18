# Kernel Analyzer

Kernel Analyzer audits every concrete operator invocation in one complete
training forward plus its actual backward.  Its scientific unit is one exact
forward/backward pair; an operator name or tensor shape is never used to infer
the backward.

The Flash-style evidence chain is:

```text
local same-dtype implementation difference
  -> exact endpoint repair
  -> real parameter-gradient carrier
  -> cross-state coherent direction
  -> paired weight accumulation
```

See [PROJECT.md](PROJECT.md) for the live status and retention map,
[cases_flash_style.md](cases_flash_style.md) for the case standard, and
[case.md](case.md) for the current case registry.

The main code is in `src/` and `scripts/`.  Scientific artifacts are kept
under `results/coverage/` and compact derivations under `results/final/`.
Raw model weights, compiler caches, and temporary run products live outside
this repository under `/data1/tzh`.

The declared coverage scope is four models (Qwen3-1.7B, Mamba-130M, Phi-4,
DeepSeek-R1-Qwen3-8B) at sequence lengths 64, 128, and 256.  Full-coordinate
T1 is closed for all 1,562 directional endpoints.  The causal T2--T4 funnel
is intentionally still fail-closed and is reported in `PROJECT.md`.
