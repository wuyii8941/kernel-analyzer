# Validation provenance and limits

The numerical kernels ran on real RTX A6000 GPUs. Baseline jobs import the exact original numerical module, while retaining the same regression tests and other repository source. They do not replace GPU math with CPU substitutes. The FLA and Liger numerical baseline modules are byte-identical at the newer pinned heads; the latest modified-source checks were repeated there. Earlier records remain unchanged locally.

- B007_before_v1 and B038_before_v1 failed to import dependencies; neither is numerical evidence.
- B007_before_v2 used an overly strong activation-output-dtype assertion for explicit FP32 initial Z. Those two control failures reflected old output promotion, not the numerical bug. The corrected test preserves these controls; B007_before_v3 gives two numerical/problem cases and two passing controls. Output-policy behavior is disclosed in the RFC/PR text, rather than hidden by the test change.
- In the final four-case regression the numeric output/state assertions precede the new state-dtype assertion. This makes the original implicit-state failures direct constant-preservation failures, rather than failures of the proposed dtype policy. The four cases were rerun on both original and modified modules after this assertion-order change. The 41-case run used the same assertions in the previous order; the numerical implementation and successful pass conditions are unchanged.
- B007_after_v1 is an interrupted full-file attempt with 22 passes and four allocator-cap OOM failures. It is not a passing suite. No numerical assertion failure is reported by those four OOM records. Later affected-path checks do not replace or erase that result.
- B007_latest_affected terminated with SIGSEGV (-11), without JUnit, while the padded-FP16 layer-gradient test was running. That diagnostic run had an extra periodic faulthandler dump timer. The timer was removed before independent original/modified FP16 probes and the final affected-path rerun. The cause of that process failure is not established; successful later checks must not erase it or be represented as a demonstrated fix for the crash.
- B038_existing_v1 is an interrupted broader run. The later existing-function selection still has a BF16 sliding-window failure; the exact unmodified case reproduces the same values. The submission text discloses it and does not claim a full-suite pass.
- B054 baseline is restricted to safe contiguous-score cases, not a full denominator for the 43 patched tests. The legacy runtime adapter changes API names/signatures, not numerical formulas. Historical-stack native validation remains unperformed.
- B047's 348 passing pytest parameterizations include upstream early-return cases; some do not execute the GPU kernel.
- The CPU-only v1 review is historical. Its “GPU unrun” status is superseded by this package, not retroactively edited.

JUnit copies replace local absolute workspace/environment paths with placeholders. The full original logs and results remain in local validation_v2/results. The public records retain numerical values, failures, selectors, and counts. Some original declarations predate the extra numerical-source hash field; that missing field is left null rather than invented.

No issue, upstream PR, fork, upstream commit, or maintainer message is created by this review export. The separately authorized publication is a documentation/patch review branch in the user's own kernel-analyzer repository. Draft PRs would notify maintainers and still require user review.

## Follow-up

The same complete linear-attention file now passes 58 tests using an 80% allocator cap on an idle GPU. Product code, comparisons, references and tolerances are unchanged. This closes the earlier complete-file validation gap; it does not delete the old OOM/SIGSEGV evidence or establish the cause of the diagnostic SIGSEGV. Direct Based/ReBased layer checks and the exact make checkstyle result accompany this version.

The FLA PR text now follows the current PR template. Its default-normalization scope is corrected using the actual public signatures and layer callsites. FlagGems/Trident checklists preserve upstream wording. Liger's full-repository test and mini-model-convergence boxes remain unchecked; only the actual checkstyle command is newly checked.

## Independent-review corrections

[Fixed follow-up evidence](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/README.md) records the PR8969 overlap, nonzero-Q/K/V matrix, DKV-only rollback, declared dV format budget and initial BF16 comparison failures. Current FlagGems patch base is the unmerged PR head, not master. The corrected Attorch selector includes six new cases and passes 354 parameterizations. An optional broader Liger run was interrupted after 1,311 passes/100 skips; no convergence job started. That is not a whole-repository pass.
