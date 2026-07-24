# Case 001 Localization Certificate v0.1

Machine-readable certificate:

`results/operator_oracle/case001_localization_certificate_v0_1.json`

## Allowed claim

`INTERVENTION_DEPENDENT_ATTRIBUTION`

The certificate is valid for the declared case package. It does **not** claim
that the target kernel is the unique root cause or that the tested expression
is the developer's historical patch.

## Evidence chain

1. The eager/reference and compiled endpoint differ reproducibly; the compiled
   output is exact across independent processes.
2. The pool boundary is exact under isolated eager/compiled replay.
3. The reduction suffix differs on identical pool inputs. This is local
   production evidence for the `flatten+sum` region.
4. The pool boundary mediation test is negative, so the pool is not promoted
   to a producer or mediator merely because it is upstream.
5. Inductor provenance maps the region to `aten.view`, `aten.sum`, and
   `triton_red_fused_sum_view_1`.
6. A separately loaded generated-kernel hypothesis changes only the declared
   batch-stride expression. The non-target generated wrapper context is
   invariant, and two independent intervention processes agree.
7. The large wrong-result error is reduced to the control-level residual
   (`3.05e-05` versus `8.72` before intervention).

## Remaining boundary

This is a credible operator/kernel intervention certificate, not a correctness
proof. The historical patch remains hidden for later blind scoring. A stronger
claim would require compiler-stage bisection, runtime/autotuning identity, and
comparison with the external developer-confirmed repair.
