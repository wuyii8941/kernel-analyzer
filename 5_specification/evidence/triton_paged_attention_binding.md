# Direct Triton float32 paged-attention binding review — 2026-10-19

This review covers `corpus/calls/triton_paged_attention.py` and
`corpus/call_bindings/triton_paged_attention.json`. The call receives query `[5,4,16]`, K/V caches `[6,8,2,16]`
and a supplied int32 block table `[2,2]`. It fixes query lengths `[3,2]`, KV lengths `[13,10]`, scale 0.25,
bottom-right causal masking, inclusive left window 4, contiguous GQA and no softcap. Twenty Triton programs, one per
query row and head, write the complete float32 output.

The binding maps every received tensor, all logical metadata, every output coordinate, the single launch and all four
unchanged input states. The input generator varies the physical block assignment by seed. The wrapper checks shape,
dtype, device, contiguity, finiteness, range and block indices. Four direct GPU comparisons with an independent dense
gathered PyTorch formula had maximum absolute difference at most `1.19e-7`, mean absolute difference between
`1.16e-8` and `1.61e-8`, finite outputs and unchanged inputs.

The formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were reviewed and usable; all
7,680 output coordinates had complete references at the requested resolution, and repeated launches were bitwise
identical. No program-to-task coordinate excluded zero. Device-to-program mean and maximum absolute differences were
about `1.28e-8` and `1.20e-7`. The aligned device-to-program rules detected a small negative numerical effect, while
all separate program-to-task rules remained unconfirmed. The report is
`.cache/spec_trial_20261019_pagedattn/triton_paged_attention/baseline_report.json`.

A temporary controlled candidate added 0.25 only to the final output. Every one of the 7,680 program-to-task
coordinates excluded zero, with mean absolute difference 0.25; both fixed and aligned semantic rule families detected
a nonzero effect. Device-to-program error remained at ordinary float32 scale. The reviewed source was restored and
matches the retained validated copy; the controlled report remains at
`.cache/spec_trial_20261019_pagedattn/triton_paged_attention/controlled_plus_quarter_report.json`.

This binding does not extend to other shapes, dtypes, block sizes, sequence layouts, windows, softcap, quantized
caches, segmented execution, optional biases, or a kernel selected only by its name.
