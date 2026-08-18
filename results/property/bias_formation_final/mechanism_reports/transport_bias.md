# Transport bias

## Verdict

`CASE_CANDIDATE_NOT_VALIDATED` for Phi MM. The open-loop formation map is `LOCAL_CENTERED -> PARAMETER_GRADIENT_BIASED -> EFFECTIVE_UPDATE_BIASED`. A row-pairing intervention changes the gradient population from `BIASED` to `CENTERED` while preserving local residual norms, but the current analytic RMSNorm-only transport reconstruction has relative error 0.32--0.60.

## Evidence

- `phi_transport_decomposition.json`
- `results/property/bias_formation/interventions/phi4_mm_transport_pairing.json`

## Boundary

The pairing result motivates transport analysis; it does not establish a complete transport mechanism or a cross-operator property. The missing semantic VJP terms must be closed before promotion.
