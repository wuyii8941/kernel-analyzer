# Real bugs before and after their fixes (benchmark part 3)

| unit | proposition | result | observed |
|---|---|---|---|
| B016/sc1_mean_pool_one_graph/pre | mechanism 1: e_sem detected | pass | bin=candidate, e_num=9.8e-08, special(K vs f)=0 |
| B016/sc1_mean_pool_one_graph/post | mask patch: e_sem at most constant-level | pass | bin=none, e_num=1.2e-07, special(K vs f)=0 |
| B016/sc1_scaled_count/pre | mechanism 1: e_sem detected | pass | bin=candidate, e_num=6e-08, special(K vs f)=0 |
| B016/sc1_scaled_count/post | mask patch: e_sem at most constant-level | pass | bin=none, e_num=7.9e-08, special(K vs f)=0 |
| B016/sc1_mean_pool_two_graphs/control | control (two segments): not detected | pass | bin=none, e_num=8.1e-08, special(K vs f)=0 |
| B016/oib_scatter_reduce_mean_19/pre | mechanism 2: e_sem detected | pass | bin=candidate, e_num=0, special(K vs f)=0 |
| B016/oib_scatter_reduce_mean_19/post-mask-patch | mechanism 2: e_sem detected (unchanged by the mask patch) | pass | bin=candidate, e_num=0, special(K vs f)=0 |
| B016/oib_scatter_reduce_amax_19/pre | mechanism 2: K has inf/NaN where f is finite | pass | bin=unresolved, e_num=n/a, special(K vs f)=9 |
| B016/oib_scatter_reduce_amax_19/post-mask-patch | mechanism 2: K has inf/NaN where f is finite (unchanged by the mask patch) | pass | bin=unresolved, e_num=n/a, special(K vs f)=9 |
| B016/oib_index_reduce_amax_0/pre | mechanism 2: K has inf/NaN where f is finite | pass | bin=unresolved, e_num=n/a, special(K vs f)=9 |
| B016/oib_index_reduce_amax_0/post-mask-patch | mechanism 2: K has inf/NaN where f is finite (unchanged by the mask patch) | pass | bin=unresolved, e_num=n/a, special(K vs f)=9 |
| B012/opt_radam_b99985_step2/pre | flip from fp32 arithmetic only: e_sem constant-level, e_num ~ whole update | pass | bin=constant-level, e_num=0.0082, special(K vs f)=0 |
| B012/opt_radam_b99985_step2/post | float64 fix: e_sem at most constant-level | pass | bin=constant-level, e_num=2.5e-08, special(K vs f)=0 |
| B012/opt_radam_b99985_step3/pre | flip from the fp32 constants: e_sem ~ whole update | pass | bin=candidate, e_num=4.9e-06, special(K vs f)=0 |
| B012/opt_radam_b99985_step3/post | float64 fix: e_sem at most constant-level | pass | bin=constant-level, e_num=2.5e-08, special(K vs f)=0 |
| B012/opt_radam_b99985_step6/pre | same decision: e_sem ~ 0.55 x update | pass | bin=candidate, e_num=3e-06, special(K vs f)=0 |
| B012/opt_radam_b99985_step6/post | float64 fix: e_sem at most constant-level | pass | bin=none, e_num=2.5e-08, special(K vs f)=0 |
| B012/opt_radam_b9997_step5/pre | unrectified in all three: e_sem ~ 2e-7 x update | pass | bin=constant-level, e_num=2.5e-08, special(K vs f)=0 |
| B012/opt_radam_b9997_step5/post | float64 fix: e_sem at most constant-level | pass | bin=constant-level, e_num=2.5e-08, special(K vs f)=0 |
| B012/opt_radam_b9997_step6/pre | same decision: e_sem ~ 0.065 x update | pass | bin=small, e_num=1.6e-06, special(K vs f)=0 |
| B012/opt_radam_b9997_step6/post | float64 fix: e_sem at most constant-level | pass | bin=constant-level, e_num=2.5e-08, special(K vs f)=0 |
| B012/opt_radam_b9993_step6/pre | same decision: e_sem ~ 3.7e-3 x update | pass | bin=small, e_num=2.4e-07, special(K vs f)=0 |
| B012/opt_radam_b9993_step6/post | float64 fix: e_sem at most constant-level | pass | bin=constant-level, e_num=2.5e-08, special(K vs f)=0 |
| B012/opt_radam_b99995_step2/pre | flip from the fp32 constants: e_sem ~ whole update | pass | bin=candidate, e_num=8.2e-06, special(K vs f)=0 |
| B012/opt_radam_b99995_step2/post | float64 fix: e_sem at most constant-level | pass | bin=constant-level, e_num=2.5e-08, special(K vs f)=0 |
| B013/ua_bidir_sw8_mha/pre | keys right of the window dropped: e_sem detected | pass | bin=candidate, e_num=0.00025, special(K vs f)=0 |
| B013/ua_bidir_sw8_mha/post | fixed kernel: e_sem at most constant-level | pass | bin=none, e_num=0.00025, special(K vs f)=0 |
| B013/ua_bidir_sw24_gqa4/pre | keys right of the window dropped: e_sem detected | pass | bin=candidate, e_num=0.00025, special(K vs f)=0 |
| B013/ua_bidir_sw24_gqa4/post | fixed kernel: e_sem at most constant-level | pass | bin=none, e_num=0.00025, special(K vs f)=0 |
| B013/ua_perseq_causal_sw8/pre | keys right of the window dropped: e_sem detected | pass | bin=candidate, e_num=0.00023, special(K vs f)=0 |
| B013/ua_perseq_causal_sw8/post | fixed kernel: e_sem at most constant-level | pass | bin=none, e_num=0.00023, special(K vs f)=0 |
| B013/ua_bidir_sw8_qpkv16/control-pre | control (BLOCK_Q = 1): not detected | pass | bin=none, e_num=0.00025, special(K vs f)=0 |
| B013/ua_bidir_sw8_qpkv16/control-post | control (BLOCK_Q = 1): not detected | pass | bin=none, e_num=0.00025, special(K vs f)=0 |
| B013/ua_causal_sw24/control-pre | control (causal): not detected | pass | bin=none, e_num=0.00023, special(K vs f)=0 |
| B013/ua_causal_sw24/control-post | control (causal): not detected | pass | bin=none, e_num=0.00023, special(K vs f)=0 |
| B013/ua_bidir/control-pre | control (no window): not detected | pass | bin=none, e_num=0.00026, special(K vs f)=0 |
| B013/ua_bidir/control-post | control (no window): not detected | pass | bin=none, e_num=0.00026, special(K vs f)=0 |
| B015/oib_nn_functional_avg_pool3d_6/pre | out-of-bounds windows averaged over the whole kernel volume: e_sem detected | pass | bin=candidate, e_num=8.2e-08, special(K vs f)=0 |

passed 38 of 38 units with results (38 in the manifest)
