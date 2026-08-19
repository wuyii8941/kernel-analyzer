# Four-model F+B bias-finding coverage audit

The denominator is the compiler-bound semantic equivalence inventory; every
cell remains in the denominator even when its carrier or closure is unresolved.

- F+B semantic cells: **791**.
- Screened task coordinates (including newly bound screens): **747**.

| status | cells |
|---|---:|
| SCREENED | 727 |
| SCREENED_VIA_EXACT_DOWNSTREAM_CLOSURE | 40 |
| SEMANTIC_REGION_PENDING | 24 |

## Interpretation

The deepest module-stack binding recovered Mamba `dt_proj.bias` and
`conv1d.bias` paths that were previously mislabeled ambiguous. Short-screen
ratios never become cases without disjoint confirmation. Current formal case
count is recorded in the mechanism candidate map, not inferred from this audit.
