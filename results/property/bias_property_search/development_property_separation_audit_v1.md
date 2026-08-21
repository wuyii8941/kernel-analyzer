# Development property separation audit

Status: `COMPLETE_WITH_CONTROL_GAPS`.

A property is not admitted to a held-out predictor merely because it has a positive case; centered controls must be measured under the same property definition.

| property | known positives | measured controls | separation | oracle eligibility |
|---|---:|---:|---|---|
| `source_asymmetry` | 3 | 5 | True | `CANDIDATE_FORMATION_INPUT` |
| `source_transport_coupling` | 1 | 0 | False | `CASE_LEVEL_ONLY_UNVALIDATED` |
| `transport_concentration` | 3 | 5 | False | `SUPPORTING_FEATURE_ONLY` |
| `carrier_stability` | 6 | 0 | False | `SHORT_SCREEN_UNVALIDATED` |

Concentration is supporting-only by design.  Source--transport and carrier stability remain explicitly unvalidated against the five scope-extension controls until their required measurements are captured.
