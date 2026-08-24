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
  -> trajectory-local directional persistence
  -> paired weight accumulation
```

Independent-state coherence is reported on the separate generalizable-bias
track.  Failing that stronger gate does not revoke a complete trajectory-local
Flash-style case.

Start from the [documentation index](docs/README.md) and
[current main result](docs/current_mainline.md). They define the only current
claim and count. See [PROJECT.md](PROJECT.md) for the
coverage and retention map,
[cases_flash_style.md](cases_flash_style.md) for the historical case standard,
and [case.md](case.md) for the historical case registry. Paper-facing claims and
their exact evidence boundaries are tracked in
[docs/claims.md](docs/claims.md); count and gate changes are recorded in
[docs/gate_history.md](docs/gate_history.md).  The completed property-search
scope is summarized in
[docs/bias_property_search_completion.md](docs/bias_property_search_completion.md).
Older round notes are retained only as experiment history; they must not be
used for the current case count or headline conclusion. The current
deliverable is a two-level workflow: a bounded 16/32-step AdamW direction
screen followed by a 4096-step warm-state review. Short runs rank what to test;
only the long run assigns the current long-horizon direction label. This is not
a universal safety classifier, and a 4096-step direct-effect result is not a
claim that training loss has converged to a different endpoint.

The current long-horizon audit contains 23 unique matrix case IDs, 11
historical candidates, and 26 merged audit rows. It currently confirms four
final cases: three direct long-horizon cases (Liger fused CE, Phi `lm_head dX`,
and Qwen `lm_head dX`) plus one feedback-sustained Qwen3-VL SiLU case with a
paired loss gap. The full row-by-row status is in
`docs/all_bias_long_horizon_audit.md`.

The main code is in `src/` and `scripts/`.  Scientific artifacts are kept
under `results/coverage/` and compact derivations under `results/final/`.
Raw model weights, compiler caches, and temporary run products live outside
this repository under `/data1/tzh`.

The declared coverage scope is four models (Qwen3-1.7B, Mamba-130M, Phi-4,
DeepSeek-R1-Qwen3-8B) at sequence lengths 64, 128, and 256.  Full-coordinate
T1 is closed for all 1,562 directional endpoints.  The causal T2--T4 funnel
is intentionally still fail-closed and is reported in `PROJECT.md`.
