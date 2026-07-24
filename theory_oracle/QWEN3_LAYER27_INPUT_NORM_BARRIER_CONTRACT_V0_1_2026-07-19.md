# Qwen3 layer-27 input-RMSNorm barrier-control contract v0.1

## Estimand

For the frozen Qwen3-0.6B step-29 matched state, estimate the selected-token
log-probability effect of compiling versus eagerly executing the layer-27
`input_layernorm`, while holding an explicit graph-break boundary fixed.

This is a **barrier-conditioned operator effect**.  It is not automatically an
effect in the original whole-compiled candidate, because the original candidate
fuses the preceding layer residual with this normalization.

## Arms

- `E_b`: eager body/head, fixed boundary, eager target RMSNorm;
- `E_bI`: eager body/head, same boundary, compiled target RMSNorm (injection);
- `C_b`: compiled body/head, same boundary, compiled target RMSNorm;
- `C_bR`: compiled body/head, same boundary, eager target RMSNorm (repair).

The original whole-eager and frozen whole-compiled hashes are retained only as
transport anchors.  `C_b != C_0` does not invalidate the within-barrier
contrasts, but it forbids transport to the original candidate.

## Validity gates

1. The eager boundary arm reproduces the frozen eager anchor exactly.
2. All repeated executions are exact under the deterministic protocol.
3. Repair and candidate use the same compiled outer body/head objects and differ
   only in the runtime target mode.
4. Injection and reference use the same eager outer body/head objects and differ
   only in the runtime target mode.
5. The compiled target is actually invoked in both compiled-target arms.

## Contrasts and claims

- `E_bI - E_b`: barrier-conditioned injection effect;
- `C_b - C_bR`: barrier-conditioned compiled-context target effect;
- `C_b - C_0`: boundary disturbance, if the frozen candidate anchor is
  available.

No contrast identifies a primitive (`mean`, `rsqrt`, cast, multiply) inside
RMSNorm.  No correctness or multi-state claim is authorized.
