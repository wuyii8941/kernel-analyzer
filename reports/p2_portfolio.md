# P2 Portfolio Oracle

## Confound Checklist
- same gated 15-mutation catalog as authoritative RQ5 analysis: PASS
- same legal eager/compile population: PASS
- matched alarm budget `425` for every strategy: PASS
- artificial mutation labels kept distinct from historical/certified bugs: PASS

## Delta Self Control
No new model execution is performed. The legal and mutation inputs retain their authoritative self/canary gates.

## External Validity
The catalog is Qwen3-0.6B, T4 FP16, and 15 artificial altered operations. Family coverage is catalog-conditioned.

## Main Result
| strategy | alarms | mutation_tp | precision | token_recall | family_coverage |
| --- | --- | --- | --- | --- | --- |
| delta_only | 425 | 425 | 1.0 | 0.055338541666666664 | 1 |
| fork_only | 425 | 420 | 0.9882352941176471 | 0.0546875 | 11 |
| family_oracle_upper_bound | 425 | 425 | 1.0 | 0.055338541666666664 | 15 |
| 50_50_portfolio | 425 | 421.0498 | 0.990705411764629 | 0.054824192708337736 | 10.2606 |

The 50-50 row reports the mean over randomized fork ties; its 95% intervals are stored in `results/p2_portfolio.json`.

## Interpretation
The 50-50 portfolio does not exceed the best single signal on token recall.
It does not exceed the best single signal on mutation-family coverage.
The family oracle is an unattainable label-aware upper bound, not a deployable ForkCert result.
