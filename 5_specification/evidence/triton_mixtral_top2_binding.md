# Direct Triton float32 Mixtral top-2 binding review — 2026-10-19

This review covers `corpus/calls/triton_mixtral_top2.py` and
`corpus/call_bindings/triton_mixtral_top2.json`. The call receives float32 router logits `[128,8]` and supplied expert
outputs `[128,8,32]`. One Triton launch writes the complete Boolean selection mask, complete dense routing weights and
complete combined output. The plan binds every task input and output and checks both input tensors unchanged. It binds
to the reviewed Mixtral top-2 profile through the full call boundary; a kernel name containing `moe` would not be
enough.

A direct GPU 3 comparison against the public PyTorch formula selected exactly the same experts. Maximum absolute
differences were `5.96e-8` for routing weights and `2.38e-7` for the combined output; mean absolute differences were
`5.14e-9` and `2.96e-8`. Both inputs remained unchanged.

The formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were reviewed and usable. The
selection mask and weights each supplied 24,576 complete, resolved coordinates, and the combined result supplied
98,304. All 24,576 Boolean coordinates matched. No program-to-task interval excluded zero; its mean absolute residual
was about `4.19e-17` for the combined result. Device-to-program mean absolute residuals were about `4.89e-9` for
weights and `3.06e-8` for the combined result. The float32 routing weights showed a small repeatable effect relative
to the exact real task, and aligned tests also detected a small effect in the combined output. This is reported as
measured numerical bias rather than erased by comparison with another float32 implementation.

The evaluator model checked the Triton 3.6 reduction tree bit for bit, but still records the chosen lowering order as
an assumption. Consequently the overall program-semantics verdict remains `not_established`; the complete admitted
task reference and candidate-to-task measurements remain usable. The retained baseline is
`.cache/spec_trial_20261019_mixtral_top2/triton_mixtral_top2/baseline_report.json`.

A temporary controlled candidate added 0.25 only to the final combined output. The selection and weight results stayed
unchanged, while all 98,304 combined-output program-to-task intervals excluded zero. Their mean absolute difference
was exactly 0.25, and both fixed semantic rules detected a nonzero effect. Device-to-program mean absolute residual
remained about `3.21e-8`, so the injected term was assigned to the task-formula difference rather than ordinary device
rounding. Temporary source and binding files were removed; the report remains at
`.cache/spec_trial_20261019_mixtral_top2/triton_mixtral_top2/controlled_out_plus_quarter_report.json`.

This binding does not cover another token count, expert count, expert width, dtype, tie rule, capacity policy,
auxiliary loss, router noise, expert implementation, dispatch layout, distributed combination, or a kernel selected
only by its name.
