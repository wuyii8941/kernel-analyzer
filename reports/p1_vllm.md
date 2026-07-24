# P1 HF-vLLM T4 Completion Report

## Claim Scope
This P1 establishes observed cross-engine decision forks for one frozen Qwen3-0.6B FP16 state on Tesla T4. It does not classify any case as a certified bug or establish BF16/V1/FlashAttention behavior.

## Confound Checklist
- checkpoint SHA-256 `f28ce3f7f7da92f7230438acae3f50f0adb83e13207558d03c1a93c3b9e31f11`: PASS
- tokenizer SHA-256 `be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506`: PASS
- exact prompt/response token IDs and response offsets: PASS
- optimizer step 5, policy iteration 2, rollout 1, pre-minibatch state: PASS
- raw/processed processor-free identity: PASS
- old_logp and advantage joined by case/token with zero mismatches: PASS
- no legal analytic bound B: all regions remain `unknown`

## Delta Self Control
Independent-process HF and vLLM self p99 are both exactly 0. Request-order reversal is bitwise identical over 1024 tokens. Sampling decisions also have zero HF and vLLM self failures.

## P1 Completion Scope (P1 完成口径)
| category | item | evidence |
| --- | --- | --- |
| full | same-state HF-vLLM scoring | 512 tokens, 4 responses; hashes and state fields match |
| full | independent-process self | HF p99=0; vLLM p99=0 |
| full | signed bias | cluster bootstrap, including advantage groups |
| full | clipping + sampling scans | same checkpoint/data; CRN sampling |
| equivalent | raw/processed mode | processor-free identity: 1320/1320 bitwise equal |
| equivalent | batch invariance | independent process + reversed order: 1024/1024 bitwise equal |
| hardware gap | FLASH_ATTN / TORCH_SDPA / V1 / BF16 | deferred to Ampere+; reports/future_work.md |
| hardware gap | chunked-prefill attribution | Triton LLIR PassManager failure on T4 XFormers prefix-prefill |

## Raw/Processed Identity
With all 12 request-level processor conditions disabled, `1320` tokens were compared: 0 bitwise mismatches, max absolute delta 0. This is recorded as identity proof in place of a vLLM 0.9.2 selector.

## Signed Bias
| group | tokens | clusters | mean_alt_minus_ref | cluster_ci95 |
| --- | --- | --- | --- | --- |
| all | 512 | 4 | -0.14295375091677176 | [-0.18747488160463366, -0.09979610116909754] |
| positive | 128 | 1 | -0.08333350927100014 | [-0.08333350927100014, -0.08333350927100014] |
| negative | 384 | 3 | -0.16282716479869563 | [-0.20530095444291874, -0.13399666308977842] |

The all-token signed mean is nonzero in this four-cluster frozen sample. The positive-advantage group contains only one response, so no strong advantage-sign association claim is made.

## Decision Forks
| mechanism | count/denominator | rate | cluster_ci95 |
| --- | --- | --- | --- |
| clipping actual | 139/512 | 0.271484375 | [0.18359375, 0.365234375] |
| top-k actual sampling state | 413/512 | 0.806640625 | [0.658203125, 0.9375] |
| top-p actual sampling state | 362/512 | 0.70703125 | [0.52734375, 0.84375] |

Top-k candidate sets differ at 512/512 states; top-p candidate sets differ at 305/512. Actual sampling uses 64 deterministic common-random-number draws per state; the state-level rates above are primary because draws within a state are correlated.

## Attribution Switches
| switch | canary | activity | changed_tokens | forks_repaired | status |
| --- | --- | --- | --- | --- | --- |
| enforce_eager true -> false | PASS (1024 tokens) | CUDA graph capture confirmed; prompt logprobs unchanged | 0 | 0 | active but no effect on this prefill score path |
| explicit XFORMERS -> AUTO | PASS (1024 tokens) | AUTO selected XFORMERS | 0 | 0 | no-op / equivalent completion |
| chunked-prefill off -> on | NOT REACHED | scheduler enabled 128-token chunks; kernel compile failed | n/a | n/a | hardware/software-stack gap |

No executable T4 singleton switch repaired a fork. This is reported as incomplete attribution under the accepted hardware scope, not evidence that the cross-engine difference is irreducible.

## Interpretation
P1 supports cross-engine observed clipping and actual sampling forks under a frozen, reproducible FP16 state. It also shows a directional HF-vLLM signed difference in this small four-response sample. It does not identify a violating implementation, provide a legal error bound, or localize the difference beyond the available T4 switch set.

## Artifacts
- `results/p1_vllm/p1_summary.json`: machine-readable summary
- `results/p1_vllm/clipping_certificates.jsonl`: clipping certificates
- `results/p1_vllm/sampling_certificates.jsonl`: CRN sampling certificates
- `results/p1_vllm/raw_processed_identity.json`: identity audit
- `results/p1_vllm/compare_*.json`: no-op and resource controls
- `logs/p1_vllm_*.log`: engine/backend/compiler evidence

## Next Decision
GO to P2. P1 is complete under the user-approved T4 scope; Ampere+ gaps remain future work and do not block the portfolio experiment.
