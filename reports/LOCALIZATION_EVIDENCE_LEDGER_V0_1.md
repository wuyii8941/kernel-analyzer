# Localization evidence ledger v0.1

This ledger is the current authority for localization claims.  It supersedes
status language in older phase reports, which remain retained as reproducible
historical records.  A case is classified by the narrowest claim supported by
its bound artifacts, not by how suggestive its mechanism appears.

## Claim levels

| level | permitted statement | explicitly not established |
|---|---|---|
| `OBSERVATION` | two implementations violate a declared endpoint contract | source, propagation, or compiler stage |
| `LOCAL_PRODUCTION` | identical boundary input yields distinct local outputs | that the local difference reaches the endpoint |
| `BOUNDARY_MEDIATION` | replacing a boundary value changes a fixed-suffix endpoint | that the upstream region produced the value |
| `INTERVENTION_DEPENDENT_ATTRIBUTION` | a declared intervention changes the endpoint while captured non-target context is invariant | unique cause, necessity, sufficiency, historical patch agreement |
| `STAGE_OPERATION_CANDIDATE` | the earliest observed stage and a candidate region/op set are supported | source line, unique operation, generated-kernel cause |
| `EXTERNAL_PATCH_VALIDATED` | a frozen blind certificate covers an independently verified external patch mechanism | general correctness beyond that benchmark |

## Current cases

| case | role now | bound environment / artifacts | strongest allowed claim | not allowed |
|---|---|---|---|---|
| Qwen higher-order-gradient (A/B/C) | development and regression case; correct stopping example | `results/operator_oracle/qwen3_compiler_grad_case_{a,b,c}_*_pt211.json`; `theory_oracle/locate_qwen3_compiler_grad_case_v0_1.py` | `STAGE_OPERATION_CANDIDATE`: Dynamo eager preserves the contract while `aot_eager` fails; `mm` is in the generic candidate inventory | external-patch coverage, Inductor/kernel claim, full-training claim |
| Case 001: adaptive pool → view/flatten → sum | live PyTorch-2.11 Inductor regression and generated-kernel plumbing regression | `results/operator_oracle/case001_stage_screen_pt211_20260723.json`; `case001_live_local_replay_pt211_20260723.json`; `case001_minimal_{local_replay,kernel_intervention}_audit_v0_2.json`; case package `theory_oracle/blind_cases/case_001/` | `INTERVENTION_DEPENDENT_ATTRIBUTION` for the declared captured generated-kernel expression; local same-input suffix discrepancy is separately recorded | independent historical-patch accuracy, unique sum/op/kernel root cause, first-bad-stage proof |
| Case 004 TVM Gather | historical attempted higher-level stopping case; **invalid for frozen-core scoring** | `reports/CASE004_TVM_GATHER_NEGATIVE_BLIND_LOCALIZATION_V0_1.md`; `reports/TVM_GATHER_RERUN_INVALID_V0_1.md`; `results/operator_oracle/tvm_gather_negative*/` | no current localization claim: the fresh buggy witness is not repeatable | Phase-3 validation, stage candidate, TIR/kernel root cause |
| Qwen GRPO matched states | Oracle development and numerical semantic-impact evidence | frozen-state artifacts and `theory_oracle/qwen3_grpo_*` runners | implementation-relative continuous/event/update measurements under their stated protocol | external correctness, generic operator/localization accuracy |
| PyTorch #105929 TorchDispatch rewrite | **negative external calibration** | `results/historical_candidate_screen/torchdispatch_105929_v0_1/screen.json`; `results/historical_blind/torchdispatch_105929_v0_1/pre_reveal_certificate.json`; `reports/PYTORCH_105929_PRE_POST_REVEAL_AUDIT_V0_1.md` | a backend observation only: the bound old runtime exposes the contract violation with Inductor while the public merged fix is in Dynamo capture | Phase-3 success, first-bad-stage claim, FX reduction claim, op/kernel attribution |
| PyTorch #122260 FMA context | GPU lower-level mechanism and intervention case | `results/historical_candidate_screen/fma_context_122260_v0_1/screen.json`; `results/historical_blind/fma_context_122260_v0_1/{post_certificate_provenance.json,generated_kernel_intervention/intervention_report.json}`; `reports/PYTORCH_122260_GPU_KERNEL_MECHANISM_AUDIT_V0_1.md` | `INTERVENTION_DEPENDENT_ATTRIBUTION` for a declared expression in the captured whole fused Triton wrapper | Phase-3 external score, individual-op cause, strict first-bad stage, root cause |
| PyTorch #141538 FractionalMaxPool lowering | retrospective lower-level development case; public issue and merged patch were read before local analysis | `results/historical_candidate_screen/fractional_maxpool*_141538_v0_1/screen.json`; `results/historical_blind/fractional_maxpool_141538_v0_1/local_evidence_report.json`; `results/historical_post_reveal/fractional_maxpool_141538_v0_1/fixed_runtime_control.json`; `reports/PYTORCH_141538_RETROSPECTIVE_AUDIT_V0_1.md` | `LOCAL_INJECTION`: fixed explicit boundary inputs reproduce a discrepancy and exported `aten.fractional_max_pool2d` has compiler-emitted wrapper provenance | Phase-3 blind score, mediation, context-preserving intervention, unique lowering/kernel root cause |

## Environment caveat

The host currently has CUDA-capable Tesla T4 devices.  Historical documents
that report unavailable CUDA/Inductor reflect their original execution
environment and must not be used as current capability evidence.  Every new
case must bind its exact Torch version, GPU, compiler configuration, input and
raw artifacts in its own certificate.

## Consequence for the active method

No current case is a held-out external-patch benchmark that has passed the
method score.  The calibration pair and generic locator are frozen; Phase 3
now requires a deterministic witness whose pre-reveal stage semantics and
post-reveal merged patch mechanism agree at the licensed level.  #105929 is
retained precisely because it shows why that gate is necessary.
