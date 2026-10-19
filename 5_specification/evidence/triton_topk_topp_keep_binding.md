# Direct Triton float32 top-k then top-p keep-mask binding review — 2026-10-19

This review covers `corpus/calls/triton_topk_topp_keep.py` and
`corpus/call_bindings/triton_topk_topp_keep.json`. The call receives 64 rows of 64 finite, pairwise-distinct float32
logits, one integer `k` and one float32 `p` per row. One Triton program handles each row and returns all 4,096 Boolean
keep decisions. The binding maps every input and output and checks `logits`, `k` and `p` unchanged. It binds to the
reviewed vLLM 0.16.0 deterministic filtering task through the full call boundary; a sampling-related kernel name
alone is insufficient.

Five structured boundary batches and twelve random batches, together containing 69,632 decisions, matched the public
vLLM PyTorch formula exactly. This check exposed one real implementation boundary: deriving a token's higher-logit
mass by subtraction can round that ratio to one when `p=1`. The call now preserves the source-level `p=1` rule
explicitly, so the whole top-k remainder survives. All inputs remained unchanged.

An initial formal run used unconstrained normal draws and was correctly rejected after a physical float32 tie
appeared. Its report is retained at
`.cache/spec_trial_20261019_sampling/triton_topk_topp_keep/baseline_report.json`. The admitted measurement now uses a
versioned input generator whose 24 batches were separately checked to contain no ties and to stay inside the declared
range.

The successful formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were reviewed and
usable; all 98,304 Boolean task coordinates were complete and resolved, and every coordinate matched the admitted
task. Repeated launches were bitwise identical. Because every residual was exactly zero, the generic statistical
rules correctly report a degenerate sample rather than claiming proof of no bias; the exact Boolean comparison is
98,304 matches and zero mismatches. The report is
`.cache/spec_trial_20261019_sampling/triton_topk_topp_keep/unique_input_baseline_report.json`.

A temporary controlled candidate flipped token 0 in every row after computing the correct mask. The Boolean
comparison found exactly 1,536 mismatches out of 98,304 coordinates, 64 in every input: 1,222 false positives and 314
false negatives. All 1,536 program-to-task coordinates excluded zero. Both fixed-direction and aligned comparisons
reported a nonzero average effect for the program-to-task and complete candidate-to-task differences. The command
summary displayed the separate zero-residual execution layer, so the detailed semantic and total fields are the
relevant verdicts for this injection. Temporary source and binding files were removed; the report remains at
`.cache/spec_trial_20261019_sampling/triton_topk_topp_keep/controlled_flip_token0_report.json`.

This binding does not cover another shape, dtype, tie policy, parameter source, masked-logit output, probability
output, random sampling, alternate backend, special value, gradient, layout effect, or a kernel selected only by its
name.
