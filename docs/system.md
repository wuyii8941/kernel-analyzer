# Automated analysis system

## Runtime release identity

A generated candidate is identified by both exact forward/backward wrapper
bytes and its Python, PyTorch, CUDA, Transformers, and Triton environment. A
legacy release may acquire an `environment.json` only after recompilation has
reproduced every wrapper SHA-256 exactly. A different compiler environment is
a different candidate release, even when it loads the same model and input.

This rule caught an execution-plan error: compiling the Mamba seq128 release
under PyTorch 2.10 changed its generated backward from 3,265 to 6,337 MM calls.
The original PyTorch 2.13.0.dev20260521+cu126 environment reproduced both
frozen wrappers exactly and is now the only admitted environment for that
release.

The code is designed so a model step or wrapped single operator can be supplied
without an LLM. A spec provides a model factory, frozen states, a scalar-loss
`StepExecution`, an exact semantic registry and candidate backend plugins.

The reference provider records execution and exact forward/backward origins.
The semantic registry uses exact dispatcher overloads only. It marks a unit
`ANALYTICALLY_PROVED` only when all concrete program-proof checks are present;
an executable formula witness alone remains
`FORMULA_REGISTERED_EXECUTABLE_WITNESS_ONLY`.

The common candidate engine—not a plugin assertion—validates same-dtype program
identity or same-semantics precision provenance before local promotion.
Total-error-only observations cannot assign a cause. Causal repair and paired
trajectory gates produce the Flash-style verdict; independent-state raw,
relative and analytic-factor hypotheses produce the separate generalization
verdict with the hypothesis family included in multiplicity control.

Before reading T1--T4 outcomes, the engine now invokes an optional Signed
Transport Coherence provider on every F+B unit. A provider supplies only
reference operands, the declared arithmetic schedule, analytic event-to-gradient
transport and a reference margin. The engine derives compact event-factor
certificates, rejects candidate/verdict/identity leakage, and writes an explicit
abstention for every unit whose factors are unavailable. T4 is not accessible as
a property label.

Run identity hashes provider/backend source code and every retained evidence
file. Resume rejects stale provenance. Reports store proof units, unresolved
records, stage inputs and case certificates separately.

```bash
kernel-analyzer analyze examples/qwen_retained_spec.py
kernel-analyzer verify results/system_runs/RUN_ID
kernel-analyzer resume examples/qwen_retained_spec.py
```

The retained Qwen adapter is now deliberately fail-closed: its old local rows
lack typed same-dtype program provenance and complete local-direction evidence,
so all 3,459 denominator units remain T1-unresolved. Historical strict
root-arithmetic and closed semantic-region cases are replayed through a
separate audited case adapter.

Complete-coordinate vectors are reduced through temporary float64 chunks under
`/data1/tzh`; only their hashes, complete Gram matrix, U-statistic and bootstrap
interval are retained. The current property denominator has complete concrete
analytic F+B proofs for 1,562/1,562 exact endpoints. Event-level reference
arithmetic factors remain queued and fail-closed until captured.
