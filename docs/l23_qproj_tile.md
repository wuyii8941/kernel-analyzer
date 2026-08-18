# Layer-23 q_proj tile

## Verdict

The fixed tile
`model.layers.23.self_attn.q_proj.weight[1152:1280,1664:1792]` is a natural,
directional, cumulative **closed semantic-region carrier**. Its complete numerical boundary
is closed, one local causal component is isolated, and paired live-weight
accumulation is demonstrated. It is therefore counted as the third natural
complete F+B directional-carrier case found by the project, but not as a fully
single-kernel-attributed case.

## Forward and actual backward

Rows 1152:1280 are query head 9; columns 1664:1792 are input block 13. The
exact forward and AOT backward are

\[
Y=HW^\top,\qquad dW=G^\top H,\qquad dH=GW.
\]

Forward `mm_161` and backward `mm_267/mm_268` are bound by saved-primal
identity and the same cotangent. Same-input EXTERN GEMMs are bitwise exact.
Across five held-out checkpoints and 32 states, exact BF16 hybrid GEMMs give:

| factor | mean projection | bootstrap 95% CI |
|---|---:|---:|
| total tile | 3.5079e-4 | [1.9075e-4, 5.0129e-4] |
| query cotangent G | 3.4534e-4 | [1.8712e-4, 4.9210e-4] |
| input H | 5.4537e-6 | [-7.7980e-6, 1.9446e-5] |

Thus the carrier enters through the query cotangent, not H or the final GEMM.

## Exact attention factorization

The real AOT chain is

\[
G_q=S_{bwd}K,\qquad
S_{bwd}=\alpha\,J_{softmax}(P)^\top U,\qquad
U=DV^\top,\qquad
D=G_oW_o.
\]

The bound calls are `bmm_76`, the fused softmax backward, `bmm_74`, and
`mm_262`. Every eager replay at these boundaries is bitwise exact. Candidate
clone sham differences and algebraic Shapley closure errors are zero.

| split | stable factor / total | other factor / total | joint residual / total |
|---|---:|---:|---:|
| `bmm_76`: S vs K | S 98.62% | K 0.95%, CI crosses 0 | 0.43%, CI crosses 0 |
| softmax: U vs P/program | U 84.92% | P/program 14.17%, CI crosses 0 | 0.91%, CI crosses 0 |
| `bmm_74`: D vs V | D 72.11% | V 9.87%, CI crosses 0 | 18.02%, CI crosses 0 |
| `mm_262`: Go replacement | removes 65.49% | -- | 34.51%, still significant |

Replacing reference S and K at actual `bmm_76` eliminates the total frozen
direction: residual mean 1.5049e-6 with 95% CI
[-1.2343e-5, 1.5717e-5]. This closes the numerical carrier boundary.

The deeper result is different: most stable signal arrives through the
downstream output cotangent `Go`. The current compile was rebound dynamically
and the old unresolved downstream source was recursively split.

## Downstream residual decomposition

At layer 23,

\[
G_o=R_{23}+M_{23},
\]

where `R23` is the direct residual cotangent and `M23` is the
post-attention-RMSNorm + MLP VJP. `R23` explains 62.76% of the original carrier
(95.82% of the `Go` removal) with interval
\([1.0545\times10^{-4},3.3386\times10^{-4}]\); `M23` explains 2.74% and its
interval crosses zero.

The same exact two-stage split was then applied to layers 24--27. For each
layer, the actual input cotangent is first split into the attention-output
residual path and the input-RMSNorm + attention VJP, then the residual path is
split into its direct and local-MLP terms. Every eager sum replay and RR
endpoint is bitwise exact; every sham is zero; all Shapley closure errors are
zero up to `1.14e-13`.

| layer | direct downstream / total | local MLP / total | attention VJP / total |
|---:|---:|---:|---:|
| 24 | +64.61% | +0.06%, CI crosses 0 | +5.75%, CI crosses 0 |
| 25 | +72.23% | -2.92%, CI crosses 0 | +5.72%, CI crosses 0 |
| 26 | +74.83% | +0.37%, CI crosses 0 | **-9.97%, CI below 0** |
| 27 | **+116.73%** | **-34.30%, CI below 0** | -3.31%, CI crosses 0 |

These fractions are nested causal cuts and are not additive. They show a
stable residual-stream carrier, plus two attenuation paths: layer-26 attention
and layer-27 MLP oppose the final direction rather than create it.

The layer-27 direct term is the terminal cotangent

\[
G_z=\nabla_{logits}\operatorname{NLL}(\log\operatorname{softmax}(logits)),
\quad D_n=G_zW_{lm},\quad
T=J_{RMSNorm}(H)^TD_n.
\]

Nested repairs at the actual fused logits VJP, `mm_198`, and final-RMSNorm VJP
give:

| terminal stage | removal / total | conclusion |
|---|---:|---|
| upstream forward logits | **+76.23%**, CI above 0 | dominant |
| fused NLL/log-softmax VJP on the same logits | +0.64%, CI crosses 0 | not a local cause |
| `lm_head` input-VJP MM | 0.00% exactly | not a cause in this seq1024 protocol |
| final-RMSNorm backward | -0.58%, CI crosses 0 | not a cause |

The analytic same-logits VJP is bitwise equal to eager in all 160 runs. A
complete final-RMSNorm forward+backward materialization repair explains only
5.01% and its interval crosses zero. Thus the dominant downstream component is
an already-changed forward logit signal transported backward through the
terminal loss and residual stream, not a terminal NLL, GEMM, or RMSNorm bug.

## Isolated local component

Layer-23 key RMSNorm+RoPE forward `forward:1352` removes 51.5% of the total
carrier across the same five checkpoints and 32 states. Its complete F+B group
is `forward:1352 + backward:{156,158,159,160,164}`; joint F+B and forward-only
repairs agree to 3.6e-11 at state 8, and candidate-restoration sham is 4.51e-10.

The key calculation is

\[
r=(\operatorname{mean}x^2+\epsilon)^{-1/2},\qquad
k=\operatorname{RoPE}(\operatorname{RN}_{bf16}(xr)w).
\]

Eager materializes BF16 at normalization, weight, and rotary boundaries.
Inductor retains these intermediates in FP32 registers until the final store.
Restoring only eager materialization is statistically indistinguishable from
the full key-forward repair; changing the RMS reduction schedule is not. This
proves one local mechanism:

\[
\text{fusion-delayed BF16 materialization}
\to\Delta K\to\Delta S_{bwd}\to\Delta G_q\to\Delta dW_q.
\]

Query RMSNorm+RoPE backward and local softmax backward have real arithmetic
differences but fail the 5x32 coherent-direction gate.

## Weight accumulation

Two paired 32-step AdamW trajectories start at checkpoint 1024 and use states
8:39, updating only layer-23 q_proj.

| repair | FP32 tile L2 | FP32 projection | BF16 changed coordinates |
|---|---:|---:|---:|
| isolated key F+B | 4.3329e-4 | +1.2246e-5 | 636 |
| complete S/K boundary | 6.0825e-4 | +3.7518e-5 | 1007 |

The complete-boundary FP32 projection increases at every recorded step after
initialization. This proves that the closed attention-state semantic-region
carrier accumulates into live weights. It does not turn the overlapping
upstream contributors into one local kernel bug.

## Claim boundary

Supported: a strict natural complete F+B semantic-region case, a directional
carrier rooted at (S_{bwd}), one local fusion/materialization contributor,
an exact S-only repair and conservative joint S/K repair with exact sham, and
live-weight accumulation. The paired trajectory uses the wider S/K repair;
K's independent directional interval crosses zero.

Not supported: a unique single-kernel cause for the whole carrier or property
generalization. The causal root is closed at the attention-state semantic
region, but its upstream contributors overlap. The isolated key-forward
mechanism and terminal-logit repair are nested because changing the key also
changes later logits; their percentages must not be added.

Machine-auditable attribution:
`results/coverage/cases/l23_qproj_attention_state_region.json`.

## Compact evidence

- `results/final/l23_go_step{64,256,1024,2048,4096}.json`
- `results/final/l23_go_summary.json`
- `results/final/l23_go_path_summary.json`
- `results/final/l23_{residual,go}_l{24,25,26,27}_summary.json`
- `results/final/l23_terminal_summary.json`
- `results/final/l23_final_norm_summary.json`
- `results/final/l23_attention_live_weight.json`
- `results/final/l23_key_actual_scale_materialization_32states.json`
- `results/final/l23_key_forward_backward_{joint,sham}_s8.json`
- `results/final/l23_key_live_weight_adamw.json`
