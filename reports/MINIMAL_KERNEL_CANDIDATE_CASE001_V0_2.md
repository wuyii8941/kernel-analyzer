# Minimal op/kernel localization slice: case_001

## Result

The first small candidate is the frozen `case_001` witness:

```text
adaptive_avg_pool2d -> flatten/view -> sum
```

It is a suitable Inductor/Triton subject because the complete eager/compiled
output differs, the pool boundary is unchanged on identical input, and the
compiled reduction suffix produces a reproducible local discrepancy.

## Evidence

- Local replay audit: `results/operator_oracle/case001_minimal_local_replay_audit_v0_2.json`
- Kernel intervention audit: `results/operator_oracle/case001_minimal_kernel_intervention_audit_v0_2.json`
- Frozen case package: `theory_oracle/blind_cases/case_001/`
- Live PyTorch 2.11 stage matrix: `results/operator_oracle/case001_stage_screen_pt211_20260723.json`
- Live PyTorch 2.11 local replay: `results/operator_oracle/case001_live_local_replay_pt211_20260723.json`

The local audit independently checks two reports and confirms:

- complete compiled witness repeats exactly;
- eager matches the frozen reference artifact;
- the pool-only boundary is exact;
- the compiled `flatten/view + sum` suffix differs when given the same eager
  pool tensor;
- generated-kernel provenance contains the relevant ATen source relation.

The intervention audit independently checks two same-schema intervention runs
and confirms:

- target kernel: `triton_red_fused_sum_view_1`;
- only the declared generated expression changed in the captured wrapper;
- non-target wrapper signatures and provenance summaries are invariant;
- the endpoint error is reduced from `8.7196044921875` to a residual below the
  declared `4e-5` control tolerance.

## Live stage screen

On host GPU 0 (Tesla T4), with PyTorch `2.11.0+cu126`, the same frozen input
gave:

| Stage | Full endpoint | Pool boundary | Same-input suffix |
|---|---|---|---|
| eager | exact reference | exact | exact |
| `aot_eager` | exact reference | exact | exact |
| `inductor` | wrong (`max_abs=8.7196044921875`) | exact | differs (`max_abs=3.0517578125e-05`) |

This is the required stage gate for a kernel-oriented subject: the AOT path
passes while the Inductor path fails. The pool boundary is not numerically
different in this witness. The isolated reduction suffix is a reproducible
local producer of a small discrepancy; the much larger full-program error is
linked to the generated reduction kernel only by the separate controlled
kernel intervention, not by the suffix delta alone.

The same case was also run with PyTorch
`2.13.0.dev20260609+cu126`. There the large historical error did not reproduce
and only the small suffix discrepancy remained. The candidate is therefore
version-specific and must always be bound to its compiler version and cache
artifacts.

## Allowed claim

This is an `INTERVENTION_DEPENDENT_ATTRIBUTION` candidate for a generated
reduction kernel. It is not a unique root cause, compiler-stage proof, or
developer-confirmed historical patch. The local replay separately establishes
`LOCAL_PRODUCER_WITH_PROVENANCE`; it does not prove that the reduction suffix
is the only producer in the full program.

The intervention is a generated-code hypothesis. The next method step is to
reproduce this evidence through a generic subject screener and operation/kernel
candidate interface, without hard-coding this case's operator names.
