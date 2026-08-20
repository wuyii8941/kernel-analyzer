# Source-aligned repair audit

## Correction

The historical `FP32 MM -> BF16 cast` arm is only a kernel-arithmetic repair.
It retains deterministic output rounding.  Therefore its trajectory separation
cannot be cited as repair success for an output-rounding source.

## Results

| Case | Proven directional sources | Scope | Full-source arm | Repair verdict | Downstream carrier |
|---|---|---|---|---|---|
| qwen64_vproj_mm | kernel, output_rounding | POPULATION | JOINT | SOURCE_DEBIASED_IN_EXPECTATION_LOCAL_ONLY | FAIL_CAUSAL_NONCOHERENT |
| qwen128_vproj_mm | output_rounding | POPULATION | ROUNDING_ONLY | SOURCE_DEBIASED_IN_EXPECTATION_LOCAL_ONLY | FAIL_CAUSAL_NONCOHERENT |
| mamba_seq64_input_proj | kernel, output_rounding | CAUSAL_PILOT | JOINT | SOURCE_DEBIASED_IN_EXPECTATION_DOWNSTREAM_UNRESOLVED | UNRESOLVED_4_STATE_PILOT |

## Interpretation

`ROUNDING_ONLY` and `JOINT` preserve the declared low-precision ABI and make
the quantizer coordinate-wise centered in conditional expectation.  The saved
finite-repeat certificate reports Monte Carlo residual separately.  Thus a
successful result is a **source-debiasing causal repair**, not a claim that one
random BF16 realization equals FP32.

Candidate--repair trajectory drift means the repaired source affects training.
It does not mean the repair remains wrong, and it also does not establish that
the repair equals a full high-precision reference.  End-to-end FP32-reference
equivalence is explicitly `NOT_MEASURED` for all three cases.
