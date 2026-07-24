# Direct generated-kernel reduction intervention

This experiment is the first slice that modifies generated kernel code itself,
rather than replacing the kernel output after it returns.

## Intervention

The original output code for runtime call 16 was copied without changing the
original artifact. In the copied kernel, only

```text
tl.sum(tmp7, 1)
```

was replaced by two masked 512-element reductions whose results are added.
The intervention kept the original kernel ABI, launch arguments, input
buffers, output buffer, generated-kernel family and downstream compiled
suffix.

## Evidence

- [runner result](</data1/tzh/forkcert/results/operator_oracle/qwen3_step236_layer16_attention_mlp_slice_v0_6/result.json>);
- [independent audit](</data1/tzh/forkcert/results/operator_oracle/qwen3_step236_layer16_attention_mlp_slice_v0_6/audit.json>).

The audit reports:

- `valid: true`;
- `evidence_level: DIRECT_GENERATED_KERNEL_CODE_INTERVENTION_AND_FIXED_SUFFIX_MEDIATION`;
- `live_generated_kernel_code_intervention_exact: true`;
- no backend recompile of the original candidate graph;
- exact repeated live-variant records and provenance hashes.

The direct reduction-tree change produced a reproducible continuous output
difference (`nonzero = 190`, L2 approximately `0.0523`) but did not change the
clipping event in this particular state. This is an informative negative
result: a reduction-tree perturbation is visible numerically, yet the chosen
perturbation is not sufficient to cross the endpoint boundary here.

## Interpretation

This validates that the pipeline can intervene at generated-kernel code level
and distinguish:

1. direct kernel-code sensitivity;
2. continuous discrepancy;
3. semantic endpoint mediation.

It does not prove that the two-512 reduction is the compiler's actual reduction
tree or that it is the source of the eager–compiled discrepancy. The stronger
`reference_reduce` result remains a stage-level explanatory counterfactual,
while this direct intervention establishes the required kernel-level
observability and intervention mechanism.
