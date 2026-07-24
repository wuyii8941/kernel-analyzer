# PyTorch #122260 — GPU generated-kernel mechanism audit v0.1

## Purpose and scope

This is a **mechanism/pipeline case**, not a hidden historical-patch accuracy
case.  The public issue was read while selecting the candidate, and its linked
ghstack PR stack is closed/unmerged.  It therefore cannot contribute to the
two Phase-3 external-validation scores.

It is retained because it supplies a real, deterministic GPU correctness
witness on the available hardware and lets us test the full lower-level path
without inventing a seeded compiler fault.

## Bound witness

On PyTorch `2.2.0+cu121`, CUDA 12.1, Tesla T4, with the frozen scalar float32
inputs, the declared endpoint is:

```text
exp((x * scale) - (x * scale)) == finite 1
```

The stage screen is stored in
`results/historical_candidate_screen/fma_context_122260_v0_1/screen.json`.
Two repeats give:

| execution | result | contract |
|---|---:|---|
| eager | 1 | passes |
| Dynamo + eager | 1 | passes |
| AOT eager | 1 | passes |
| Inductor | inf | fails |

This is a backend observation, **not** a proof that Inductor is the unique
first-bad compiler pass.

## Provenance

The debug replay captures an FX graph with `mul`, `mul_1`, `sub`, and `exp`.
The actual generated wrapper contains source-node annotations for those nodes
and invokes the compiler-emitted Triton symbol:

```text
triton_poi_fused_exp_mul_sub_0
```

The machine-readable provenance artifact is
`results/historical_blind/fma_context_122260_v0_1/post_certificate_provenance.json`.
This is an auditable FX-to-generated-kernel relation.  It is not a one-to-one
mapping from any individual FX operation to the kernel.

## Same-input replay and controlled intervention

The generated wrapper was copied verbatim, replayed with the exact same two
scalar inputs, and compared with a byte-identical no-op copy.  Then only this
compiler-emitted line was changed:

```text
tmp5 = tmp4 - tmp4
```

to

```text
tmp5 = tl.zeros_like(tmp4)
```

The resulting report is
`results/historical_blind/fma_context_122260_v0_1/generated_kernel_intervention/intervention_report.json`.

Controls and outcome:

| arm | output |
|---|---:|
| generated wrapper baseline | inf |
| byte-identical no-op wrapper | inf |
| one-expression repair | 1 |

The input fingerprints match, baseline replay is repeatable, provenance is
captured, and the report verifier passes.  Thus the emitted wrapper locally
produces the discrepancy on same inputs and the declared expression repair
changes the endpoint.

## Allowed claim

`INTERVENTION_DEPENDENT_ATTRIBUTION` is the strongest allowed claim.

The target is the whole fused wrapper.  After excluding it, there are no
non-target graph/kernel artifacts left to compare; equality of that empty set
must not be upgraded to operator-level context invariance.  This prevents a
vacuous `OPERATOR_LEVEL_EFFECT` claim.

Not established:

- a unique root cause or source line;
- a strict first-bad compiler stage;
- an individual `mul`, `sub`, or `exp` as the unique faulty operation;
- a general FMA diagnosis beyond this concrete emitted expression;
- independent external-patch agreement.

## Method consequence

This is positive evidence that a real GPU compiler witness can be carried
through:

```text
semantic contract
→ backend observation
→ FX provenance
→ compiler-emitted Triton wrapper
→ same-input local replay
→ no-op / one-expression intervention
→ claim-gated certificate
```

It does **not** close Phase 3.  The next required evidence remains an
independently merged, withheld lower-level patch case, followed by the complex
Megatron matched-step evaluation.
