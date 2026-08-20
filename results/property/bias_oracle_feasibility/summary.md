# Low-cost Bias Oracle Feasibility

This is a screening-method study, not a new correctness certificate.
The exact conditional antithetic experiment remains the escalation path.

## Synthetic mechanism controls

| Control | Expected | sketch status | relative error | odd norm | even norm |
|---|---|---:|---:|---:|---:|
| centered_linear_safe | NONE | SCREEN | 0 | 0 | 0 |
| source_mean_linear_transport | TRANSPORTED_MEAN | SCREEN | 0 | 1 | 0 |
| centered_variance_quadratic_rectification | CURVATURE_RECTIFICATION | SCREEN | 0 | 0 | 3 |
| mixed_source_and_rectification | BOTH | SCREEN | 0 | 1 | 2.25 |
| nonsmooth_support_switch | ESCALATE | ESCALATE_NONLOCAL_OR_NONSMOOTH_RESPONSE | 0 | 0 | 0.4 |

Interpretation: transported mean covers source/pairing asymmetry; the
Rademacher antithetic curvature sketch covers smooth response rectification.
The amplitude gate deliberately escalates the support-switch control.

## Shared all-block HVP experiment

Status: **PASS**.

The synthetic coupled F+B response recovered both blocks' first-order
and curvature terms with one shared forward/reverse graph plus eight coded
HVPs. Cross-block coupling was present and canceled through the codes.
This is the only tested route whose probe count does not multiply by op count.

## Coded group-intervention experiment

This alternative uses only ordinary F+B runs and keeps the full response
vector. Random masks switch groups of units between candidate and repair;
vector-valued sparse recovery localizes contributors, while held-out masks
test the required additivity assumption.

| Scenario | mask runs | support recall | held-out residual |
|---|---:|---:|---:|
| SPARSE_ADDITIVE | 8 | 0.487 | 0.644 |
| SPARSE_ADDITIVE | 16 | 0.975 | 0.062 |
| SPARSE_ADDITIVE | 24 | 1.000 | 0.008 |
| SPARSE_ADDITIVE | 32 | 1.000 | 0.008 |
| SPARSE_WITH_INTERACTION | 8 | 0.350 | 1.312 |
| SPARSE_WITH_INTERACTION | 16 | 0.725 | 1.059 |
| SPARSE_WITH_INTERACTION | 24 | 0.900 | 0.947 |
| SPARSE_WITH_INTERACTION | 32 | 0.963 | 1.001 |
| DENSE_ADDITIVE | 8 | 0.115 | 0.933 |
| DENSE_ADDITIVE | 16 | 0.168 | 0.833 |
| DENSE_ADDITIVE | 24 | 0.200 | 0.738 |
| DENSE_ADDITIVE | 32 | 0.200 | 0.678 |

## Retrospective repeat-budget ablation

Each entry compares a deterministic prefix/even/random subset with the
already frozen full-repeat verdict, using the unchanged 2,000-bootstrap policy.

### qwen64_vproj

| repeats | agreement | unresolved | opposite resolved | comparisons |
|---:|---:|---:|---:|---:|
| 4 | 0.929 | 0.071 | 0.000 | 240 |
| 6 | 0.975 | 0.025 | 0.000 | 240 |
| 8 | 0.975 | 0.025 | 0.000 | 240 |
| 12 | 1.000 | 0.000 | 0.000 | 240 |
| 16 | 1.000 | 0.000 | 0.000 | 80 |

### qwen128_vproj

| repeats | agreement | unresolved | opposite resolved | comparisons |
|---:|---:|---:|---:|---:|
| 4 | 0.933 | 0.067 | 0.000 | 240 |
| 6 | 0.975 | 0.025 | 0.000 | 240 |
| 8 | 1.000 | 0.000 | 0.000 | 80 |

### mamba64_input_proj

| repeats | agreement | unresolved | opposite resolved | comparisons |
|---:|---:|---:|---:|---:|
| 4 | 0.871 | 0.121 | 0.000 | 240 |
| 6 | 0.900 | 0.100 | 0.000 | 240 |
| 8 | 0.917 | 0.075 | 0.000 | 240 |
| 12 | 0.992 | 0.000 | 0.000 | 240 |
| 16 | 1.000 | 0.000 | 0.000 | 80 |

## Random output-projection ablation

A shared HVP returns one scalar update projection at a time. The table
tests how many Gaussian output coordinates preserve the frozen full-vector
conditional verdict when projected directly from the retained Grams.

### qwen64_vproj

| sketch dimensions | agreement | unresolved | opposite resolved |
|---:|---:|---:|---:|
| 4 | 0.550 | 0.450 | 0.000 |
| 8 | 0.669 | 0.331 | 0.000 |
| 16 | 0.775 | 0.225 | 0.000 |
| 32 | 0.812 | 0.188 | 0.000 |
| 64 | 0.944 | 0.056 | 0.000 |

### qwen128_vproj

| sketch dimensions | agreement | unresolved | opposite resolved |
|---:|---:|---:|---:|
| 4 | 0.369 | 0.631 | 0.000 |
| 8 | 0.581 | 0.419 | 0.000 |
| 16 | 0.688 | 0.312 | 0.000 |
| 32 | 0.781 | 0.219 | 0.000 |
| 64 | 0.794 | 0.206 | 0.000 |

### mamba64_input_proj

| sketch dimensions | agreement | unresolved | opposite resolved |
|---:|---:|---:|---:|
| 4 | 0.512 | 0.487 | 0.000 |
| 8 | 0.637 | 0.350 | 0.000 |
| 16 | 0.731 | 0.263 | 0.000 |
| 32 | 0.769 | 0.231 | 0.000 |
| 64 | 0.919 | 0.081 | 0.000 |

## Decision boundary

- A first/second-moment screen is dimension-independent in probe count, but
  it is not exact for nonsmooth or nonlocal responses.
- Repeat reduction is acceptable only as a sequential screen. An unresolved
  prefix escalates; it is never imputed as centered.
- A new operator still needs a declared F+B perturbation boundary. Units with
  no faithful antithetic/source perturbation must abstain.
- No result here changes an existing case verdict.
