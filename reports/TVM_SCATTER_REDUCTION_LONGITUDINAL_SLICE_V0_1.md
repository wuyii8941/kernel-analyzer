# TVM ScatterElements reduction: longitudinal slice v0.1

This is a pipeline validation case, not a blind accuracy benchmark.  The
candidate's upstream fix was read while screening the TVM corpus, so the
status is `PATCH_EXCLUDED_REPLAY`.

## Witness and environment

The input is a deterministic ONNX `ScatterElements` opset-18 node with
`reduction="add"`, run with ONNX Runtime as the semantic reference.  The same
input is run through two separately built TVM checkouts:

| checkout | result against ONNX Runtime |
|---|---|
| parent of the fix (`dfc9fc03...`) | exact mismatch, max absolute error 1.0 |
| fix (`378c4f304...`) | exact match |

The buggy checkout was run twice.  The output and converted Relax IR hashes
were stable across runs, so this slice has a reproducible witness on CPU/LLVM.

## Evidence chain

1. **Observation:** the compiled result differs from the independent ONNX
   Runtime result.
2. **IR-boundary production evidence:** the ONNX frontend converts the node to
   `relax.scatter_elements(..., reduction="update")`, while the input
   specification requires `reduction="add"`.  This is a local semantic
   discrepancy at the ONNX-to-Relax boundary, not a same-input numeric replay
   of a Python operator.
3. **Provenance:** the ONNX node is associated with the Relax
   `scatter_elements` call and a generated TIR `scatter_elements` prim_func
   after `LegalizeOps`, followed by LLVM lowering.  The buggy TIR writes the
   update directly; the fixed TIR adds the existing output value.  Kernel
   identity/source location is not captured, so provenance is still
   incomplete below TIR.
4. **Controlled intervention:** rebuilding a minimal Relax module with only
   the reduction attribute set to `add` restores the exact reference output.
   This is intervention-dependent attribution.  It is not a root-cause proof:
   rebuilding can change generated code and this slice does not establish
   kernel/context invariance.
5. **Version validation:** the fixed checkout preserves `reduction="add"` in
   the converted Relax IR and matches the reference without the repair.

## What this proves, and what it does not

The slice proves that the current repository can produce an auditable,
reproducible chain from semantic witness to an IR-boundary discrepancy and a
minimal semantic intervention.  It does **not** prove a unique faulty kernel,
an operator-level causal effect under unchanged compiler context, or a
correctness claim for eager-versus-compiled training in general.

The allowed claim is therefore:

> `IR_BOUNDARY_PRODUCTION_PLUS_INTERVENTION_DEPENDENT_ATTRIBUTION`

The pre-reveal machine-readable certificate is assembled only from the two
buggy-run artifacts; it does not take the fixed checkout as input.  Its
`EvidenceGates` deliberately leave same-input numeric local replay,
provenance completeness, and non-target context invariance false.

The next strengthening step is to capture the post-LegalizeOps/generated
function and compare non-target artifacts for a same-context replay.  If that
cannot be done without rebuilding or changing fusion, the analysis must stop
at the TIR/Relax boundary rather than invent a kernel root cause.

## Machine-readable evidence

- `results/operator_oracle/tvm_scatter_reduction/buggy.json`
- `results/operator_oracle/tvm_scatter_reduction/buggy_case003.json`
- `results/operator_oracle/tvm_scatter_reduction/fixed.json`
- `results/operator_oracle/tvm_scatter_reduction/fixed_case003.json`
- `results/operator_oracle/tvm_scatter_reduction/case003_pre_reveal_certificate.json`
- `theory_oracle/blind_cases/case_003/case_manifest.json`
- `theory_oracle/tvm_scatter_reduction_case_v0_1.py`
- `theory_oracle/assemble_tvm_scatter_blind_certificate_v0_1.py`
