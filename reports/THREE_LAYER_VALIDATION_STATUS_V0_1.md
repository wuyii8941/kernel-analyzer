# Three-layer validation status

## Seeded-fault calibration

`results/operator_oracle/seeded_fault_conformance_v0_1.json` is valid and
covers four controls: local producer, benign continuous mutation,
propagation-only boundary, and no-op.  It is pipeline conformance evidence,
not compiler-bug evidence.

## Historical blind localization

Case 004 is the compliant historical candidate.  Its pre-reveal certificate is
frozen at `OBSERVATION`; post-reveal fixed validation and an IR repair produce
the final certificate at `INTERVENTION_DEPENDENT_ATTRIBUTION`.  The claim stops
before kernel localization because the repair rebuilds compiler context and no
fixed non-target suffix is available.

## Unknown/prospective case

The Qwen3 and adaptive-pool cases remain prospective mechanism cases.  They
can generate implementation-relative discrepancy evidence, but without an
independent specification/fix they must not be scored as historical root-cause
localization.  Their existing certificates remain separate from Case 004.

The three layers therefore have different roles: calibrate the evidence
logic, evaluate it against an external fixed case, and test discovery on an
unresolved case.  None of them justifies a global operator ranking.
