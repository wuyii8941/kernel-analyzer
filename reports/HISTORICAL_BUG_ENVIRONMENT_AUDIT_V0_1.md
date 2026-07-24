# Historical Bug Environment Audit v0.1

This audit records only environment instantiation and endpoint reproducibility.
It intentionally does not inspect or summarize any fixed patch, pull-request
discussion, or root-cause analysis.

## Candidate selected for the first blind slice

`pytorch_adaptive_avgpool_flatten_sum` (`pytorch/pytorch#180956`)

Reproducer source (input contract only):

`/data1/tzh/paper/confirmed_bugs/pytorch/bug_006_adaptive_avgpool_flatten_sum/minimal_repro.py`

Target environment observed through the GPU service:

| Field | Observed value |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu126 |
| Triton | 3.6.0 |
| CUDA runtime | 12.6 |
| GPU | Tesla T4, compute capability 7.5 |
| Declared input | `[4, 2049, 8, 8]`, float32; fp64 control |
| Compiled backend | TorchInductor |

## First target run

The minimal program produced a stable eager output and a different Inductor
output. The observed absolute per-batch differences were approximately:

`[1.53e-05, 5.72e-01, 2.17e+00, 8.72e+00]`

with maximum absolute difference approximately `8.72`. The program's shape
variation also showed the same qualitative pattern for batch sizes 2, 3, 4,
and 8. The fp64 control remained discrepant (maximum approximately `11.38`),
so this candidate is not presently explained as a float32 tolerance issue.

The reproducer's two fusion-breaking controls completed with only small
floating-point-level differences (approximately `3.05e-05`). These controls
are recorded as negative controls for the endpoint, not as proof of a compiler
mechanism.

## Qualification decision

`REPRODUCIBLE_BLIND_ELIGIBLE`: two independent GPU processes produced the same
reference and compiled vectors, the same per-batch error pattern, and the same
control outcomes. Before localization, the raw input/output logs still need to
be copied into an immutable case package with hashes; the service journal is
only the screening record.

This qualifies the case because:

- the endpoint is a compact per-batch tensor relation;
- eager and Inductor runs are both executable;
- the discrepancy is repeatable and large relative to the declared exact
  relation;
- the case exercises reduction/fusion/indexing-shaped compiler behavior;
- the control programs provide a nontrivial negative comparison.

This does **not** yet establish:

- the first compiler stage where the discrepancy appears;
- a local discrepancy producer;
- a faulty operator or kernel;
- correctness of any repair;
- a unique root cause.

Those claims require the frozen blind protocol and the later production,
mediation, provenance, and context-invariance checks.

## Rejected/held-out candidates at this stage

- `pytorch_expanded_index_add` currently fails closed with an explicit
  alias-write error in the available nightly; it is not a silent wrong-result
  witness on that environment.
- `pytorch_layernorm_reciprocal` is reproducible and useful as an
  infrastructure fallback, but is not the first reduction/indexing case.
