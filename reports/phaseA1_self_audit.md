# Phase A1 Self-Run Independence Audit

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- independent OS processes/CUDA contexts: PASS
- same-GPU warm-cache runtime determinism: PASS
- second physical T4 measured: PASS
- compile warm-cache and cold-cache variants separated: PASS

## Delta Self Control
| path | variant | tokens | pid_left | pid_right | independent_processes | gpu_left | gpu_right | device_uuid_left | device_uuid_right | delta_mean | delta_p50 | delta_p99 | delta_max | bitwise_equal | cache_left | cache_right |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eager_fp16 | independent_process_same_gpu_warm_cache | 51200 | 2882036 | 2882961 | True | 3 | 3 | cedc3c05-23f7-24ad-9cbe-9d37906526c7 | cedc3c05-23f7-24ad-9cbe-9d37906526c7 | 0.0 | 0.0 | 0.0 | 0.0 | True | /data1/tzh/forkcert/cache/torchinductor | /data1/tzh/forkcert/cache/torchinductor |
| eager_fp16 | independent_process_cross_gpu | 51200 | 2882036 | 2883510 | True | 3 | 5 | cedc3c05-23f7-24ad-9cbe-9d37906526c7 | c11c5970-8a41-05e3-4c02-3134af39695b | 0.0 | 0.0 | 0.0 | 0.0 | True | /data1/tzh/forkcert/cache/torchinductor | /data1/tzh/forkcert/cache/torchinductor |
| compile_fp16 | independent_process_same_gpu_warm_cache | 51200 | 2884471 | 2885217 | True | 3 | 3 | cedc3c05-23f7-24ad-9cbe-9d37906526c7 | cedc3c05-23f7-24ad-9cbe-9d37906526c7 | 0.0 | 0.0 | 0.0 | 0.0 | True | /data1/tzh/forkcert/cache/torchinductor | /data1/tzh/forkcert/cache/torchinductor |
| compile_fp16 | independent_process_cross_gpu | 51200 | 2884471 | 2886244 | True | 3 | 5 | cedc3c05-23f7-24ad-9cbe-9d37906526c7 | c11c5970-8a41-05e3-4c02-3134af39695b | 0.0 | 0.0 | 0.0 | 0.0 | True | /data1/tzh/forkcert/cache/torchinductor | /data1/tzh/forkcert/cache/torchinductor |
| eager_attention_fp16 | independent_process_same_gpu_warm_cache | 51200 | 2887154 | 2887643 | True | 3 | 3 | cedc3c05-23f7-24ad-9cbe-9d37906526c7 | cedc3c05-23f7-24ad-9cbe-9d37906526c7 | 0.0 | 0.0 | 0.0 | 0.0 | True | /data1/tzh/forkcert/cache/torchinductor | /data1/tzh/forkcert/cache/torchinductor |
| eager_attention_fp16 | independent_process_cross_gpu | 51200 | 2887154 | 2888143 | True | 3 | 5 | cedc3c05-23f7-24ad-9cbe-9d37906526c7 | c11c5970-8a41-05e3-4c02-3134af39695b | 0.0 | 0.0 | 0.0 | 0.0 | True | /data1/tzh/forkcert/cache/torchinductor | /data1/tzh/forkcert/cache/torchinductor |
| sdpa_math_fp16 | independent_process_same_gpu_warm_cache | 51200 | 2888633 | 2889137 | True | 3 | 3 | cedc3c05-23f7-24ad-9cbe-9d37906526c7 | cedc3c05-23f7-24ad-9cbe-9d37906526c7 | 0.0 | 0.0 | 0.0 | 0.0 | True | /data1/tzh/forkcert/cache/torchinductor | /data1/tzh/forkcert/cache/torchinductor |
| sdpa_math_fp16 | independent_process_cross_gpu | 51200 | 2888633 | 2889630 | True | 3 | 5 | cedc3c05-23f7-24ad-9cbe-9d37906526c7 | c11c5970-8a41-05e3-4c02-3134af39695b | 0.0 | 0.0 | 0.0 | 0.0 | True | /data1/tzh/forkcert/cache/torchinductor | /data1/tzh/forkcert/cache/torchinductor |
| compile_fp16 | independent_process_independent_cold_compile_cache | 51200 | 2890124 | 2891329 | True | 3 | 3 | cedc3c05-23f7-24ad-9cbe-9d37906526c7 | cedc3c05-23f7-24ad-9cbe-9d37906526c7 | 0.0 | 0.0 | 0.0 | 0.0 | True | /data1/tzh/forkcert/cache/phaseA1_inductor_cold_90257b7811a649bd8eb0821be5166c07/a | /data1/tzh/forkcert/cache/phaseA1_inductor_cold_90257b7811a649bd8eb0821be5166c07/b |

## External Validity
This audit runs on Tesla T4 FP16. It tests process, device, and compile-cache reproducibility on this platform; it does not establish BF16-kernel reproducibility. A BF16-hardware replication remains required.

## Conclusion
none; both cold compilations selected bitwise-equivalent execution
