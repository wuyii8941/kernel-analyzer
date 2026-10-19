# Direct Triton unique-row sort binding review — 2026-10-19

This review covers `corpus/calls/triton_sort_unique.py` and `corpus/call_bindings/triton_sort_unique.json`. It applies only to one contiguous physical finite float32 input of shape `[32, 32]`, with pairwise distinct values in each row, and the complete returned values and int64 original indices of the same shape.

The wrapper checks the declared input, shape, dtype, device, layout, finite range, and row uniqueness. One launch uses 32 Triton programs. Each program loads one complete row, applies Triton ascending sort, finds the unique original column for every result, and writes every coordinate of both outputs. The plan binds the full input pre-state, dimension one, ascending direction, both complete outputs, and the unchanged input state to the reviewed SORT `finite_unique` profile. A GPU 3 sanity run was bit identical to `torch.sort` for both outputs.

The formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were reviewed and usable; all 24,576 value coordinates and 24,576 index coordinates matched the admitted task specification exactly. The program-level evaluator could not establish the bitonic reduction combiner used by Triton sort, so this case has no K−Gπ result. The report preserves that refusal while still giving complete K−f results from the independently admitted task specification. It is stored at `.cache/spec_trial_20261019_sort/triton_sort_unique/baseline_report.json`.

A temporary controlled candidate added 0.25 to every returned sorted value after computing the correct order and indices. The complete value K−f comparison had mean absolute residual about 0.25 and the fixed-mean rules detected a nonzero average effect; all index residuals remained zero. The temporary candidate and binding were removed after the run. Its report is `.cache/spec_trial_20261019_sort/triton_sort_unique/controlled_plus_quarter_report.json`.

This binding does not cover other shapes, dtypes, layouts, tied or special values, descending order, top-k, unsorted selection, output buffers, or another Triton kernel with a similar name.
