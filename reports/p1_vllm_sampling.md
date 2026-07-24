# P1 HF-vLLM Sampling Fork Scan

## Confound Checklist
- same checkpoint and fixed response token IDs: PASS
- temperature=1.0 before top-k/top-p: PASS
- common random numbers shared by case/token/draw: PASS
- independent-process self decisions: PASS

## Delta Self Control
HF decision self failures=0; vLLM decision self failures=0.

## External Validity
T4 FP16, vLLM 0.9.2 V0 and XFormers only; BF16/V1/FlashAttention remain external replication work.

## Summary
| schema_version | tokens | samples | top_k_candidate_set_forks | top_p_candidate_set_forks | top_k_sampling_fork_states | top_p_sampling_fork_states | top_k_first_draw_sampling_forks | top_p_first_draw_sampling_forks | top_k_sampling_fork_draws | top_p_sampling_fork_draws | hf_self_failures | vllm_self_failures | all_regions_unknown | rate_ci_method | top_k_candidate_set_rate | top_k_candidate_set_cluster_ci95_low | top_k_candidate_set_cluster_ci95_high | top_p_candidate_set_rate | top_p_candidate_set_cluster_ci95_low | top_p_candidate_set_cluster_ci95_high | top_k_sampling_state_rate | top_k_sampling_state_cluster_ci95_low | top_k_sampling_state_cluster_ci95_high | top_p_sampling_state_rate | top_p_sampling_state_cluster_ci95_low | top_p_sampling_state_cluster_ci95_high | top_k_first_draw_rate | top_k_first_draw_cluster_ci95_low | top_k_first_draw_cluster_ci95_high | top_p_first_draw_rate | top_p_first_draw_cluster_ci95_low | top_p_first_draw_cluster_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forkcert.p1.hf-vllm-sampling.v1 | 512 | 4 | 512 | 305 | 413 | 362 | 178 | 161 | 10980 | 10360 | 0 | 0 | True | case_id cluster bootstrap, 10000 draws, seed=0 | 1.0 | 1.0 | 1.0 | 0.595703125 | 0.404296875 | 0.78125 | 0.806640625 | 0.658203125 | 0.9375 | 0.70703125 | 0.52734375 | 0.84375 | 0.34765625 | 0.25390625 | 0.43359375 | 0.314453125 | 0.21875 | 0.400390625 |
