# Direct Triton f16 packed-varlen binding review — 2026-10-19

This review covers corpus/calls/triton_pack_varlen.py and
corpus/call_bindings/triton_pack_varlen.json. The call receives q [5,4,16] and k/v [7,2,16], fixes q lengths
[2,3], k/v lengths [4,3], scale 0.25, bottom-right causal masking, contiguous GQA and no optional features.
Twenty Triton programs, one per query row and head, write the complete float16 output.

The binding maps all three input tensors, both cumulative arrays, both maxima, every public option, the complete
output and all unchanged input states. Four direct GPU comparisons with an independently written dense PyTorch
formula produced finite outputs and unchanged inputs. Maximum absolute differences ranged from 2.20e-4 to
4.88e-4; mean absolute differences ranged from 3.24e-5 to 4.27e-5.

The formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were reviewed and usable; all
7,680 output coordinates had complete references at the requested resolution, and repeated launches were bitwise
identical. No program-to-task coordinate excluded zero. Program-to-task mean and maximum absolute differences were
about 1.78e-17 and 1.67e-16. The complete device-to-task mean and maximum absolute differences were about
4.02e-5 and 4.92e-4; none of the four declared projections was confirmed at alpha 0.05. The report is
.cache/spec_trial_20261019_pack/triton_pack_varlen/baseline_report.json.

A temporary controlled kernel added 0.25 at every final store. Every one of the 7,680 program-to-task coordinates
excluded zero, with semantic mean absolute difference 0.25; both fixed and aligned semantic rule families confirmed
a nonzero effect. The reviewed source was restored and matches the retained validated copy. The controlled report is
.cache/spec_trial_20261019_pack/triton_pack_varlen/controlled_plus_quarter_report.json.

This binding does not extend to other shapes, dtypes, sequence partitions, scales, bidirectional execution, empty
causal rows, local windows, softcap, ALiBi, paged K/V, optional outputs, backward or a kernel selected only by its
name.
