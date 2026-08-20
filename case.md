# Directional-bias cases

## Canonical evidence counting (2026-08-20)

The eight-case Bias Formation roster is an **audit denominator**, not eight
persistent-bias positives.  Four evidence layers are reported separately:

1. `FORMATION_POSITIVE`: a matched intervention identifies a conditional
   event/pairing asymmetry or response-even component;
2. `TRAJECTORY_SEPARATION`: a closed candidate/repair live-weight run separates
   above its initial distance;
3. `DIRECTIONAL_PERSISTENCE`: a predeclared or calibration-frozen
   trajectory-local signed component does not cancel;
4. `SAME_CONTRAST_FULL_CHAIN`: the formation and persistence results use the
   same repair contrast, or a declared closed semantic-region superset.

Parameter-distance growth alone is not called directional bias.  A fixed
absolute direction across unrelated natural inputs is not required, but a
trajectory-local persistence witness is required for a Flash-style claim.

| case | formation | paired separation | directional persistence | contrast alignment | same-contrast full chain |
|---|---|---|---|---|---|
| Liger fused CE | matched positive | yes | confirmed | aligned | yes |
| Phi seq64 `lm_head dX` | matched positive | yes | confirmed | aligned | yes |
| Qwen64 `v_proj` | matched JOINT positive | yes | confirmed for historical KERNEL_ONLY arm | mismatch | no |
| Qwen128 `v_proj` | matched ROUNDING_ONLY positive | yes | not confirmed | mismatch | no |
| Qwen saved-P | matched positive | yes | confirmed | aligned | yes |
| Qwen3-VL SiLU | matched optimizer-response positive | yes | not confirmed | aligned base contrast | no |
| Mamba `in_proj` | partial JOINT formation | yes | confirmed for KERNEL_ONLY arm | mismatch | no |
| layer-23 attention region | semantic-region mechanism; parity follow-up unresolved | yes | confirmed | aligned semantic superset | yes |

Canonical counts are therefore:

- paired separation: **8/8**;
- directional persistence: **6/8**;
- matched formation mechanism: **6/8**;
- current formation-to-persistence same-contrast full chain: **4/8**.

The six formation positives and six persistence positives are not the same
six.  Qwen128 and SiLU form bias without a confirmed persistent direction;
Mamba and layer-23 have persistent trajectory evidence while their current
formation-map follow-up is partial or bounded.  The project-wide strict
Flash-style registry is a separate denominator and includes Qwen `lm_head dX`,
which is not in this eight-case formation roster.

## Conditional moving-frame case (2026-08-20)

The frozen bias-risk oracle found one new formation case outside its six-case
development roster: DeepSeek layer-35 attention `dV`
(`backward:676:output_0`). The complete F+B relation is

```text
O = P V,
dV = P^T dO,
dW_v = dV^T H.
```

Only the compiled BF16 `dV` BMM output is replaced by its FP32-recomputed,
BF16-ABI reference; the exact repair reaches the complete
`model.layers.35.self_attn.v_proj.weight` gradient. On 32 new natural states,

```text
alpha = <g_candidate - g_repair, g_repair> / ||g_repair||^2
```

has mean `-0.003698` and bootstrap 95% CI
`[-0.006637, -0.000592]`. Thus the candidate systematically contracts the
same-state repair-gradient component even though unrelated states need not
share one absolute parameter direction. Two other promoted candidates failed
the frozen confirmation and two sign-changing controls were not flagged.

This is a strict **conditional bias-formation** case with a closed backward
endpoint repair. It is not yet counted as a complete Flash-style/SEUP
trajectory-accumulation case. Compact evidence is in
`results/property/bias_oracle_recovery/confirmation/summary.md`; the frozen
oracle is `results/property/bias_oracle_recovery/oracle.md`.

## Source-aligned repair correction (2026-08-20)

The presence of a candidate--repair trajectory is not itself evidence that the
repair removed the source named by a separate decomposition.  The historical
`FP32 MM -> BF16 cast` arm repairs MM kernel arithmetic but necessarily retains
deterministic BF16 output rounding.

The three affected MM cases are now handled by source identity:

- Qwen seq128 `v_proj`: output rounding only, so the valid arm is
  `ROUNDING_ONLY`;
- Qwen seq64 `v_proj`: the completed 32-state decomposition finds both kernel
  arithmetic and output rounding directional, so the full arm is `JOINT`;
- Mamba seq64 `in_proj`: both sources are directional; kernel-only,
  rounding-only, and joint arms form a factorial intervention, with `JOINT` as
  the full observed-source arm.

Rounding repair uses adjacent BF16 values with coordinate-wise probabilities
whose conditional expectation equals the FP32 value.  It therefore removes
deterministic rounding bias in expectation while preserving the BF16 ABI.  It
does not claim that one BF16 realization equals FP32 or that the repaired full
training step is end-to-end FP32-equivalent.  Nor does a noncoherent direction
across unrelated states prove downstream centering.  The next measurement
holds one state fixed, repeats only repair randomness, and separately tests
the local repair residual and the candidate effect removed after backward and
optimizer mapping.  The exact definitions and claim boundary are in
`docs/source_aligned_repair.md`.

The formal Qwen seq128 `v_proj` follow-up now holds 16 natural states fixed
one at a time and changes only the stochastic rounding draw (8 repeats per
state). The repair's declared local rounding residual is centered in 16/16
conditions. Natural candidate minus repair-ensemble effects are biased in
16/16 conditions at the local output, actual `v_proj.weight` gradient,
stateless-SGD update, and zero-moment AdamW-step1 update. This is a new
conditional source-formation case even though the effects do not share one
direction across unrelated states. It does not prove absolute downstream
repair bias is zero, and it is not joined to the historical `KERNEL_ONLY`
trajectory. Compact evidence is in
`results/property/conditional_debias/qwen128_vproj.md`.

Qwen seq64 `v_proj` now has the same fixed-state closure under its correct
`JOINT` repair.  The initial 8-draw campaign left one condition unresolved, so
the threshold was not relaxed; an independent 16-draw campaign with a new seed
bank was run over all 16 conditions.  It centers the repair local residual in
16/16 and finds the candidate-minus-repair local, real gradient, stateless-SGD,
and zero-moment AdamW effects biased in 16/16.  This upgrades Qwen64 from a
local-only repair to a conditional source-formation positive.  Evidence is in
`results/property/conditional_debias/qwen64_vproj.md`.

The layer-23 `S_bwd` follow-up also exposed a real representability boundary.
Direct reflection of a BF16 residual about the eager reference is not always
an exact BF16 residual.  A predeclared nearest-BF16 projection produced exact
`+epsilon/-epsilon` pairs and exact shams in 16/16 conditions, but the projected
source missed the frozen 90% natural-source-fidelity gate in some conditions.
The nearby exact pairs nevertheless show a 2.9--16.6% F+B response-even
component and a 67.2--72.5% zero-moment AdamW response-even component.  These
are bounded response-geometry results, not permission to relabel the natural
layer-23 source as a marginal-preserving matched mechanism.

Mamba seq64 `in_proj` was then run as a complete 16-condition, 16-draw
cross-architecture confirmation under `JOINT`.  Its repair local residual is
centered in 16/16 conditions, while the natural local effect and zero-moment
AdamW-step1 effect are biased in 16/16.  The actual backward and stateless-SGD
effect are biased in 13/16 and unresolved in three; none is certified centered.
This is deliberately retained as a partial mixed case.  It proves local source
formation and a bounded optimizer effect, but it does not pass an all-layer
conditional F+B gate.  Compact evidence is in
`results/property/conditional_debias/mamba_seq64_input_proj.md`.

## Bias Formation Map update (2026-08-20)

The denominator remains **eight unique closed F+B paired-separation
artifacts**; it is not an eight-case persistent-bias count.  A new optimizer
experiment on Qwen3-VL SiLU is evidence for that existing case, not a ninth
case. Six cases have matched formation-mechanism evidence, while a different
set of six has directional-persistence evidence:

- Liger and Phi support **event/pairing asymmetry**: the natural schedule or
  residual--transport pairing is directional, while a semantic-orbit or
  marginal-preserving control restores cancellation.
- Qwen seq64/128 `v_proj` independently support **conditional source
  asymmetry**: the correct deterministic source is removed by a locally
  centered stochastic repair, and the removed effect reaches real
  gradient/update in every fixed condition.
- Qwen saved-P and Qwen3-VL SiLU support **optimizer response rectification**:
  exact equal-norm `+delta_g/-delta_g` pairs at the same Adam state do not map
  to opposite effective updates.  Their accumulated non-oddness ratios are
  `0.6817` and `0.6956`.  Energy weighting shows that sign-crossing coordinates
  carry `99.48%` and `99.87%` of the response-even energy, and steps 1--2
  generate more than `99.5%` of it in both cases.  This locates the shared
  rectification at Adam's cold-start small-gradient/sign boundary.

For a predeclared semantic antithetic operation, the common equation is

```text
E[F(epsilon)|c]
  = integral p_s(epsilon) F_e(epsilon)
  + integral p_a(epsilon) F_o(epsilon).
```

The first term is response rectification; the second is event/pairing
asymmetry.  If the event population is antithetically closed and the complete
F+B/optimizer response is odd, both terms vanish regardless of error variance.
This is the current testable property; SEUP remains the downstream persistence
condition. Qwen64/128 are now fixed-state conditional source-formation
positives, but their new JOINT/ROUNDING_ONLY formation repairs are not joined
to the historical KERNEL_ONLY trajectories.  SiLU remains a response-
rectification positive and a directional-persistence negative under its frozen
trajectory gate. Missing global 32-state carriers are not safety results. Mamba has
a complete local conditional result but a mixed real-backward result, so it
remains partial. Layer-23 retains its semantic-region boundary and failed
natural-fidelity gate.

The canonical per-case audit is
`results/property/bias_formation_systematic/scientific_summary.md`; the full
derivation is in `docs/effective_antithetic_symmetry.md`.

## Endpoint re-screen update (2026-08-17)

The previous eight-row audit was a stale snapshot, not the completed endpoint
case registry.  Rejoining the exhaustive artifacts finds **41 additional unique
exact endpoints** that already pass complete concrete F+B proof, full-coordinate
T1, causal repair T2, complete-carrier T3, and paired 32-step accumulation T4.
They comprise 32 DeepSeek, eight Qwen, and one Phi-4 endpoint and cover `add`,
`bmm`, `rsqrt`, `mm`, and `sum` roots.  Together with seven previously retained
strict cases, the invocation-level count is **48 before mechanism
deduplication**.

All 41 have unique candidate IDs, exact AOT endpoints, and generated regions.
They reduce to nine provisional model/phase/root/carrier recurrence patterns;
those patterns are not yet nine proved causal mechanisms.  Because the new
repairs are made at exact AOT endpoints inside generated regions, they are
strict endpoint/closed-region cases, not automatically single-instruction
kernel attributions.

The authoritative exhaustive join is
`results/coverage/endpoint_case_rescreen.json.gz`, with a compact table in
`results/coverage/endpoint_case_rescreen.md`. That case-rescreen snapshot still
contains the older 557 pending-T1 entries, so it must not be used as the live T1
denominator. The authoritative full-coordinate reconciliation now audits all
1,562 endpoints: 1,390 pass T1, 172 reject T1, and 0 remain pending. The newly
reconciled 557 Mamba seq256 rows contain 524 T1 survivors and 33 rejects, but
have no T2/T3/T4 follow-up artifacts yet; therefore they add **zero strict
cases** at this point. Normal non-accumulating controls remain in the denominator
but are not called cases.

The property study is now correctly defined at the complete F+B/T3 layer, not
by T4 accumulation.  Its 1,562-endpoint population currently contains 57 T3
coherent F+B endpoints, 588 completed normal references, and 917 unresolved
endpoints.  The earlier 41 T4-pass versus 15 T4-fail split is not a property
comparison because all 56 already pass T3.  The candidate is Signed Transport
Coherence: schedule-derived signed arithmetic residuals transported through
complete analytic F+B derivatives have a common cross-state component above a
nonlinear remainder and reference margin.  It is formalized but not yet
claimed; see `results/property/hypothesis_matrix.json` and
`docs/property_hypotheses.md`.

## Dual-track audit status

The audit now separates trajectory-local `FLASH_STYLE_CASE` from cross-state
`GENERALIZABLE_BIAS`.  A failure across unrelated natural inputs does not
revoke a complete causal F+B mechanism that accumulates along a paired
training trajectory.

There are **six strict Flash-style cases**: four root-arithmetic cases—Qwen
seq128 `lm_head dX`, Liger fused linear CE, Phi-4 seq64 `lm_head dX`, and
Mamba seq64 layer-0 `in_proj`—plus two causally closed semantic-region cases:
the layer-23 q-projection tile and layer-27 softmax saved-state `dS` region.
Phi's trajectory is bounded to an evolving final-norm weight; both semantic
regions are closed boundaries, not unique single-kernel attributions. Only
Liger and Phi pass the separate cross-state concrete-
mechanism gate.  This is still not a cross-operator property claim.

The old Qwen `lm_head` 32-state all-parameter interval crosses zero.  That
measurement remains valid and now means `GENERALIZABLE_BIAS=FAIL`, while its
complete F+B, causal repair, frozen carrier controls, and 32-step paired
trajectory give `FLASH_STYLE_CASE=PASS`.  The FlashAttention literature case
is not counted as repository-generated evidence.

The four-model × three-shape discovery denominator is **12/12 complete**. The
exhaustive same-dtype ledger retains 1,562 directional endpoints in 804
generated-region/F+B-owner groups. The stricter complete-coordinate audit is
now complete for T1: 1,562/1,562 endpoints have a disposition (1,390 pass T1,
172 reject T1, 0 pending). Passing T1 does not promote an endpoint to a case.
Every survivor still needs exact sham/repair T2, complete 32-state real-carrier
T3, and—only after T3 passes—a paired 32-step T4 trajectory. The authoritative
live denominator is `results/coverage/cases/full_coordinate_audit.json.gz`.

The F+B mathematics/program layer is complete for all 1,562 endpoints. They
reduce to ten analytic roots (`add`, `mm`, `bmm`, `rsqrt`, `sum`, `mul`,
`clone`, `permute`, `cat`, and cast), and every endpoint is bound through exact
compiler-carried provenance to a concrete witness containing saved origins,
non-tensor arguments, the real cotangent, actual backward nodes, and output
edges. The registry is
`results/coverage/cases/directional_candidate_math_registry.json.gz`. This
proof does not substitute for the numerical T1--T4 gates.

The Qwen seq128 layer-3 q-norm `rsqrt_13` endpoint is the first newly completed
negative disposition in this stricter pipeline. Full-coordinate T1 and causal
repair pass, and the repair produces real gradient and BF16-materialized weight
divergence. However, the complete q-projection carrier has bootstrap lower
bound `-3.549260788e-7`, and its paired projection rises through step 16 then
falls at step 32. It therefore fails both coherent-carrier T3 and directional-
accumulation T4 and is not a strict Flash-style case. The signed rejection is
`results/coverage/cases/qwen128_rsqrt13_strict_case.json.gz`.

The previously established Phi-4 seq64 `lm_head` input-gradient MM remains a
strict bounded case with complete F+B proof, corrected full-coordinate T1,
causal repair, coherent downstream carrier, and paired 32-step trajectory.

The authoritative audit is
`results/coverage/existing_case_reaudit.json`; the dual-track rules are in
`results/coverage/case_classification_protocol.json`.  The old frozen
cross-state rules remain unchanged in
`results/coverage/directional_bias_protocol.json` and
`docs/bias_protocol.md`.

## Four-model candidate follow-up (2026-08-11)

The earlier valid non-Triton queue contained 83 sampled positives derived from
the frozen four-model screen. Its completed follow-up found:

- 38 backward MM endpoints have only 16 output coordinates, so the original
  32-state screen already covered every coordinate. Across seq64, seq128, and
  seq256, a four-state exact-call follow-up found that the same-operation FP32
  reference changes zero coordinates after casting back to the declared BF16
  output. These are precision-only residuals, not candidate-added causal bias.
- Three forward scan-output candidates were nominated by 64 sampled
  coordinates. Full 1536-coordinate checks reject all three: seq64 fails on
  32 states; seq128 and seq256 fail the predeclared four-state pilot gate.

The corrected distinct-cluster bootstrap rejects all four Mamba seq128 targets
and the Phi-4 seq256 target. It also confirms two precision-only T1 candidates:
Qwen3-1.7B seq128 layer-0 `v_proj` output and Mamba-130M seq64 first
input-projection output. Both now have exact concrete F+B proofs and 32-state
mechanism decompositions. Together with the already complete Phi backward MM,
they form a replicated mechanism class within the MM operator family. For
Phi-4 seq64 the corrected bootstrap rejects one target and confirms
one: `phi4_seq64_backward_497_output`, over all 196,608 coordinates and 32
states, has U-statistic \(9.6268\times10^{-8}\) and bootstrap 95% CI
\([8.6147\times10^{-8},1.0690\times10^{-7}]\). Its local F+B proof is complete,
and its downstream causal/accumulation gates now pass within the frozen-other-
parameters T4 boundary. Qwen seq64 supplies same-generated-region support,
while Qwen seq256 rejects the available targets. DeepSeek seq64 and seq128 now
both complete 32 natural states: all six candidates have confidence intervals
crossing zero and fail `FAIL_CAUSAL_NONCOHERENT`. Thus all 83 valid non-Triton
candidates have live dispositions, although typed Triton and full analytic-
proof coverage remain open, so no all-op numerical-coverage claim is made.
The earlier 41-item queue was not exhaustive and is superseded.
The historical 18,164 Triton sampled-boundary positives are not valid candidates.
An ABI audit found that all 18,164 came from passing FP32 allocations to
already compiled kernels whose pointer signatures remained `*bf16`/`*fp16`.
That reinterprets bytes; it does not create an FP32 implementation. A decisive
counterexample was a pure transpose/copy kernel: its legitimate FP32-to-BF16
residual must be zero, while the old replay changed about 196.5k of 196.6k
coordinates and then changed a convolution-weight gradient. That apparent
causal case is therefore `INVALID_REFERENCE_ABI`, not a case. The audit is
`results/coverage/triton_reference_abi_audit.json`.

Independently typed FP32 Triton programs have since been generated. Their
valid screen exposes a new DeepSeek seq128 attention-softmax F+B candidate:
semantic owner `vl-fb-0881` binds generated `forward:191` and
`backward:1529`, and both endpoints have positive 32-state directional
intervals. The complete derivation is in
`results/coverage/cases/deepseek128_attention_softmax_fb.md`. It is a new
local F+B directional-bias case, but it is not counted among the six strict
Flash-style cases until causal repair, a coherent parameter carrier, and
accumulation pass.

The machine-readable queue is
`results/coverage/bias_candidate_queue.json`, with batch evidence in
`results/coverage/mamba_seq{64,128,256}_*_cast_gate.json`.

## Counting rule

A complete case must bind one real forward to its actual backward, show a
directional local error, trace that error to a coherent parameter-gradient or
weight carrier, and pass a causal intervention. A nonzero forward residual is
not required: an exact forward followed by a biased, exactly bound backward
edge is still a complete forward/backward case. Here, "natural" means that the
case occurs on unmodified model states and inputs rather than an injected
sentinel or a controlled synthetic domain.

The table retains the literature anchor and all project cases so their
mathematical evidence is not deleted. The dual-track audit counts **6 strict
Flash-style passes: 4 root-arithmetic + 2 closed semantic regions**. Separately, only **2
concrete mechanisms** pass cross-state confirmation and **0 cross-operator
properties** are claimed.

| Case | Origin | Biased endpoint | Isolated cause | Accumulation evidence |
|---|---|---|---|---|
| FlashAttention | Qiu and Yao | attention backward / query-weight gradient | low-precision online-softmax output reused in backward | coherent weight error across tokens and steps |
| seq128 `lm_head` MM | this project; Flash-style pass, cross-state fail | actual input VJP `dX` | BF16 GEMM reduction and arithmetic path | causal carrier and 32-step paired weight divergence; no cross-state property |
| Phi-4 seq64 `lm_head` MM | this project; bounded complete case | actual input VJP `dX` | BF16 GEMM arithmetic path; same-dtype arm is exactly zero | coherent final-norm gradient and paired 32-step evolving-weight divergence |
| Qwen seq128 layer-0 `v_proj` MM | this project; fixed-state conditional source-formation positive | forward output `Y` | deterministic FP32-to-BF16 output rounding; unbiased-rounding arm centers the local source in 16/16 fixed conditions, while candidate-minus-repair gradient/update effects are biased in 16/16 | historical accumulation trajectory is a different contrast; persistence for the rounding repair remains unmeasured |
| Qwen seq64 layer-0 `v_proj` MM | this project; fixed-state conditional source-formation positive | forward output `Y` | kernel arithmetic and output rounding are both directional; joint source-debiased arm centers the local source and removes a biased real F+B/update effect in 16/16 fixed conditions | historical trajectory used a different contrast; persistence for the joint repair remains unmeasured |
| Mamba seq64 layer-0 `in_proj` MM | this project; partial cross-architecture conditional result | forward output `Y` | joint repair centers kernel arithmetic plus output rounding locally in 16/16; local/Adam effects are biased 16/16, while real gradient/SGD is biased 13/16 and unresolved 3/16 | historical 32-step trajectory closes only kernel arithmetic; the joint all-layer conditional gate remains unresolved |
| Qwen seq128 layer-27 softmax | this project; strict semantic-region pass | actual backward `dS`, then real q/k VJPs | backward reconstructs probability from BF16 logits/max/sum instead of the true FP32 forward probability | exact saved-P repair/sham; paired q/k 32-step projection grows at every checkpoint |
| Liger fused linear CE | this project | actual `dW` | 64 chunk contributions stored and added in BF16 | tied-weight carrier and 32-step repaired trajectory |
| layer-23 `q_proj` tile | this project; strict semantic-region pass | actual `dW_q` tile | restoring attention-backward state S_bwd alone closes the carrier; one upstream contributor is fusion-delayed BF16 materialization in key RMSNorm+RoPE | exact S-only and joint repair/sham; paired 32-step AdamW divergence uses the conservative S/K boundary |

## 1. FlashAttention reference case

For

\[
S=\alpha QK^T,\qquad P=\operatorname{softmax}(S),\qquad O=PV,
\]

and upstream cotangent \(G=\partial L/\partial O\), the backward is

\[
dV=P^TG,\qquad dP=GV^T,
\]

\[
\delta=\operatorname{rowsum}(G\odot O),\qquad
D=P\odot(dP-\delta),
\]

\[
dQ=\alpha DK,\qquad dK=\alpha D^TQ,
\qquad dW_Q=dQ^TX\quad\text{when }Q=XW_Q^T.
\]

The paper identifies a directional low-precision error in the tiled
online-softmax output \(O\). Because backward reuses \(O\) through \(\delta\),
the error becomes a structured query-gradient error. Its projection onto the
query-weight update is coherent across tokens and training steps, so it does
not cancel. Recomputing the relevant forward quantity at higher precision, or
changing the mathematically equivalent online-softmax scaling, removes the
cause and restores training stability.

This is a natural, complete forward/backward and long-trajectory case reported
by the paper, not a discovery of this repository.

## 2. seq128 `lm_head` input-gradient MM

The concrete Qwen3-1.7B invocation is

\[
Y=XW^T,
\]

with actual backward

\[
dX=GW,\qquad dW=G^TX.
\]

The proof binds the real forward to the real input-VJP edge using the same
saved \(W\) and upstream cotangent \(G\). The forward result is unchanged by
the backward-only repair; the directional error is in `dX`. This still meets
the complete F+B definition because the forward, saved values, cotangent, and
actual backward program are closed as one unit.

Disabling BF16 reduced-precision reduction removes about **91.05%** of the
local residual RMS. The remaining error is consistent with the GEMM
FMA/reduction tree and accumulation order. The independent 32-step experiment
was already a paired baseline-versus-analytic-VJP-repair live-weight
trajectory, not an observational baseline-only run:

- the parameter-gradient carrier is nonzero in 32/32 steps;
- FP32 master weights and materialized BF16 weights diverge in 32/32 steps;
- final pairwise L2 distances are 0.00487622 (FP32 master) and 0.00536579
  (materialized BF16).

Therefore `lm_head` **is a natural, paper-level complete F+B case**. It is not
the FlashAttention kernel or the same source bug; it reproduces the same
causal form with a different local arithmetic mechanism.

## 2b. Phi-4 seq64 `lm_head` input-gradient MM

For the exact Phi-4-mini invocation,

\[
Y=XW^T,
\quad X\in\mathbb R^{64\times3072},
\quad W\in\mathbb R^{200064\times3072},
\]

the actual AOT backward is

\[
dX=QW,\qquad dW=Q^TX.
\]

The forward root `mm(view,t(primals_2))` has sequence number 13520 and exports
the exact saved `X` and `W^T`. The backward obtains `Q` from the two real loss
cotangents, computes `mm_2(Q,W)` for `dX` and `mm_1(Q^T,X)` for `dW`, and both
edges reach the backward outputs. The generated call at source SHA-256
`441bcc...c7c72` is compiler-bound to `mm_2`; no name or shape matching is
used.

Across 32 natural states, the complete 196,608-coordinate BF16-minus-FP32
carrier has maximum absolute error \(7.1546\times10^{-5}\), U-statistic
\(9.6268\times10^{-8}\), and 95% CI
\([8.6147\times10^{-8},1.0690\times10^{-7}]\). Two repeats are exact. The
same-dtype candidate-minus-reference arm is identically zero.

The type-compatible repair computes only this MM in FP32 and then casts back
to the required BF16 output. Across 32/32 states it reduces local error against
the same FP32 reference (mean 5.63%, range 3.16%--7.79%), leaves loss exact,
and changes the same 194 downstream parameter gradients. The same-dtype sham
keeps loss and every parameter-gradient digest exact in 32/32 states. The
topologically immediate `model.norm.weight` carrier is coherent over all 3072
coordinates: U-statistic \(7.8696\times10^{-6}\), 95% CI
\([6.7460\times10^{-6},9.0099\times10^{-6}]\).

Finally, a paired 32-step FP32-master SGD trajectory evolves only this
final-norm weight while freezing all other parameters. At both arms' current
weights, standard and repair loss remain exact and the same-weight carrier
inner product is positive in 32/32 steps. The master-arm distance grows
monotonically from \(2.1279\times10^{-6}\) to \(9.1859\times10^{-5}\); the
final materialized BF16 distance is \(6.1035\times10^{-5}\). This closes a
bounded Flash-style causal chain, not a full-model optimizer trajectory. The
machine proof is `results/coverage/cases/phi4_seq64_lmhead_dx.json`.

## 2c. Qwen seq128 layer-0 `v_proj` output rounding

The exact invocation is

\[
Y=XW^T,\quad X\in\mathbb R^{128\times2048},\quad
W\in\mathbb R^{1024\times2048},
\]

with actual AOT backward

\[
dX=QW,\qquad dW=Q^TX.
\]

The proof binds generated `buf14`, forward root `mm_2`, saved `X` and `W^T`,
the real cotangent `view_1240`, and both actual VJP edges. Across 32 natural
states its complete-coordinate precision error has U-statistic
\(6.6813\times10^{-5}\) and 95% CI
\([5.5071\times10^{-5},7.9540\times10^{-5}]\); the same-dtype optimization
error is exactly zero.

For the identical BF16 operands, the local error closes exactly as

\[
Y_{\mathrm{low}}-Y_{32}=
\bigl(Y_{\mathrm{low}}-\operatorname{bf16}(Y_{32})\bigr)+
\bigl(\operatorname{bf16}(Y_{32})-Y_{32}\bigr).
\]

The first term, the generated-MM kernel difference at fixed operands and ABI,
is noncoherent: U-statistic \(2.4578\times10^{-9}\), 95% CI
\([-4.0306\times10^{-11},1.2968\times10^{-8}]\). The second term,
deterministic BF16 output rounding, reproduces essentially all of the T1
direction: U-statistic \(6.6812\times10^{-5}\), 95% CI
\([5.5338\times10^{-5},7.9173\times10^{-5}]\). Mean L2 norms are 0.00142
and 0.08794 respectively. A same-ABI FP32-accumulation intervention is therefore
correctly rejected as the cause. This is a rigorously sourced T1 F+B case, not
yet a complete Flash-style case: an output-rounding intervention, coherent
downstream carrier, and live T4 accumulation remain open. Machine evidence is
in `results/coverage/cases/qwen128_vproj.json` and
`results/coverage/cases/qwen128_vproj_precision_decomposition.json`.

## 2d. Replicated low-precision MM mechanism class

Three independent concrete F+B units now pass the complete-coordinate T1 and
exact source-decomposition gates:

- Qwen seq128 layer-0 `v_proj` forward MM: only deterministic output rounding
  is coherent;
- Mamba seq64 layer-0 `in_proj` forward MM: both the local kernel difference
  and output rounding are coherent;
- Phi seq64 `lm_head` input-VJP backward MM: only the local kernel difference
  is coherent.

Thus low-precision MM directional bias is a replicated class, but not a single
universal bug. It contains at least two additive, invocation-dependent
mechanisms. Kernel arithmetic is independently supported by Mamba and Phi;
output rounding is independently supported by Qwen and Mamba. This evidence
spans three models and forward/backward endpoints, but all units implement
matrix multiplication. It therefore does **not** meet the preregistered bar for
cross-operator-family property generalization. The Phi and Mamba members now
close causal repair, downstream carrier, and T4. For Mamba, the repaired local-
MM carrier is not cross-state coherent in the four-state pilot, but its paired
trajectory projection at steps 1/8/16/32 is
(0.004031,0.004571,0.004634,0.004653); final master L2 is (0.008289), with
52,150 BF16 coordinates diverged. The aggregate machine
certificate is `results/coverage/cases/mm_precision_mechanism_class.json`.

## 3. Liger fused-linear cross-entropy `dW`

The closed terminal region is

\[
Z=HW^T,\qquad
L=-N^{-1}\sum_t\log\operatorname{softmax}(Z)_{t,a_t},
\]

\[
G_{t,v}=N^{-1}
\left(\operatorname{softmax}(Z)_{t,v}-\mathbf1[v=a_t]\right),
\qquad dH=GW,\qquad dW=G^TH.
\]

The eager region reproduces the full-model loss, and its logits VJP equals the
cotangent captured from the real full backward. The candidate is Liger's actual
custom autograd program. At \(T=128,D=2048,V=151936\), it processes two tokens
per chunk and makes 64 sequential additions into a BF16 `dW` accumulator.

Changing only that accumulator to FP32 leaves loss and `dH` bitwise identical
in all 24 held-out states, while the default-minus-FP32 `dW` carrier is positive
in 24/24. Mean `dW` RMS errors against the same FP32 region reference are

\[
6.6562\times10^{-6}\;\text{(default)},\quad
5.5979\times10^{-6}\;\text{(FP32 accumulator)},\quad
5.5504\times10^{-6}\;\text{(eager)}.
\]

The intervention removes **95.7%** of the mean candidate-added `dW` error. In
a disjoint full-step confirmation, only the tied
`model.embed_tokens.weight` gradient changes; loss, terminal `dH`, and the
other 309 parameter gradients are bitwise exact. Its frozen carrier is positive
in 24/24 states with bootstrap 95% CI \([0.168,0.220]\).

A frozen 32-step paired trajectory now closes the live-weight consequence. At
each arm's evolving weights, both accumulator implementations were evaluated
before the selected arm update. All 64 same-weight contrasts retained exact
loss, hidden state, labels, terminal `dH`, and 309 untied gradients; only the
tied-weight gradient changed, and all 64 projections onto the previously
frozen carrier were positive. With stateless SGD and an FP32 master, the
default-minus-repair master distance grew from \(8.5868\times10^{-6}\) after
step 1 to \(2.2394\times10^{-3}\) after step 32. The corresponding materialized
BF16 distance grew from \(6.3461\times10^{-5}\) to \(3.2074\times10^{-3}\).
This proves live-weight feedback from the repaired accumulator cause; it does
not claim AdamW behavior or catastrophic loss instability.

Appending mathematically ignored zero rows changes the chunk schedule from
64 two-token chunks to 32 four-token chunks. The `dW` effect remains directional
with BF16 accumulation (24/24), but is incoherent with FP32 accumulation
(13/11). This establishes a **chunk geometry x accumulation precision**
interaction. It refines this case and is not counted as a fourth case.

## 4. seq1024 key-RMSNorm structural bias factor (not an independent complete case)

The long-horizon screen found a repeat-exact natural Inductor carrier at
`model.layers.19.self_attn.k_norm.weight`.  For one head vector
(x\in\mathbb R^{128}), key RMSNorm and rotary embedding are

\[
r=(128^{-1}\sum_j x_j^2+\epsilon)^{-1/2},\qquad
y_j=w_jx_jr,\qquad k=R_\theta y .
\]

For the real upstream key cotangent (g_k), the closed backward is

\[
g_y=R_\theta^Tg_k,\qquad
\frac{\partial L}{\partial w_j}
=\sum_{t,h}(g_y)_{t,h,j}x_{t,h,j}r_{t,h}.
\]

The forward, saved (x,r,w), rotary transpose VJP, weight-product partials,
and final reduction are exact-bound as one F+B chain.  All 949 changed closed
units are also mapped to all 310 reachable trainable parameters, so this
carrier was not selected from a tied-embedding-only subset.

The mechanism exclusions are sharp.  Eager and AOT-eager have bitwise-equal
loss and all 128 key/query RMSNorm gradient coordinates in both repeats; the
error first appears in Inductor code generation.  Replacing the layer-19 key
backward pointwise boundary or its closed final weight reduction removes
approximately 0% of the direction.  Replacing all 55 independently closed
query/key forward regions together removes only 3.70% of the key direction.
Thus no single tested q/k kernel or final reduction explains the carrier.

Two graph-level controls do identify a stable implementation-structure cause.
Across the frozen held-out checkpoints 64, 256, 1024, 2048, and 4096, eight
states, and two exact repeats:

- limiting fusion size and disabling epilogue fusion removes a mean absolute
  frozen-direction projection of (8.79\times10^{-5}), with clustered 95%
  lower bound (1.65\times10^{-5}>0);
- additionally forcing reused intermediates to materialize removes
  (7.54\times10^{-5}), with lower bound (3.22\times10^{-7}>0).

Both pass in four of five checkpoint means and pass the prespecified clustered
lower-bound gate.  Split-reduction disabling alone does not explain the effect.
The supported conclusion is therefore: **fusion and intermediate
materialization placement causally induce part of a coherent key-RMSNorm
gradient bias**.  This is distinct from choosing BF16 as the dtype, although
its numerical action is still mediated by finite-precision rounding points.

It is not counted as an independent project case because the intervention is
graph-wide, does not restore the full carrier to eager equivalence, and has not
yet been propagated through a paired live-weight trajectory.  The analogous
query-RMSNorm carrier fails the frozen-direction bootstrap gate and remains a
negative result.

## 5. layer-23 query-projection attention-state region (strict semantic-region case)

The old large-parameter screen sampled only 256 coordinates per parameter and
could miss a head-local direction. The replacement screen partitions every
one of the model's 1,720,574,976 parameter coordinates exactly once into
128-element vector blocks or 128x128 matrix tiles, with a separate global
parameter direction. Step 0 and eight states define the directions; steps 64,
256, 1024, 2048, and 4096 are held out.

The discovery screen produced 120 candidates. A second campaign froze those
120 directions and evaluated 32 disjoint validation states. One direction
survived the exact one-sided sign test and Holm family-wise correction:

- parameter: `model.layers.23.self_attn.q_proj.weight`;
- query output rows 1152:1280 (query head 9 of 16);
- input-feature columns 1664:1792;
- positive in 27/32 independent state-averaged measurements;
- exact sign-test p=5.65e-5, below the first Holm threshold 4.17e-4;
- all five checkpoint means positive, minimum mean cosine 0.151;
- clustered 95% lower projection bound 1.88e-4.

This is stronger than the old sparse carrier screen and is explicitly local
to one query head and one input-feature block. Exact 5-checkpoint x 32-state
factorization binds the actual chain

\[
dW=G^TH,\quad G_q=S_{bwd}K,\quad
S_{bwd}=\alpha J_{softmax}(P)^TU,\quad U=DV^T,\quad D=G_oW_o.
\]

The stable fractions of the total direction are 98.62% from `S_bwd`, 84.92%
from U, 72.11% from D, and 65.49% removable by replacing downstream cotangent
`Go`. Direct K, P/local-softmax-program, and V effects are not independently
significant. Reference S/K replacement at actual `bmm_76` closes the total
carrier: residual mean 1.5049e-6 with interval
[-1.2343e-5, 1.5717e-5]. All same-input eager replays are bitwise exact and all
candidate-restoration shams are zero.

One local component is isolated: layer-23 key RMSNorm+RoPE forward contributes
51.5%. Inductor delays BF16 materialization across normalization, weight, and
rotary arithmetic; restoring eager materialization reproduces the complete
key-forward repair. Its complete F+B intervention and 32-step AdamW trajectory
pass, with 636 BF16 tile coordinates diverging.

The complete S/K boundary repair also passes a paired 32-step AdamW trajectory:
final FP32-master tile L2 is 6.0825e-4, frozen-direction projection is positive
3.7518e-5, and 1007 BF16 coordinates diverge. This proves total-carrier
accumulation.

The old downstream `UNRESOLVED` boundary has now been recursively decomposed.
At layer 23, the direct residual cotangent explains 62.76% of the original
carrier, while the local MLP path explains 2.74% with a confidence interval
crossing zero. Exact residual/attention and residual/MLP splits through layers
24--27 show that the residual stream is the stable carrier. Two paths instead
attenuate it: layer-26 attention contributes -9.97% with its interval below
zero, and layer-27 MLP contributes -34.30% with its interval below zero.

At the terminal boundary, 76.23% of the original direction is attributable to
already-changed forward logits. The same-logits fused NLL/log-softmax VJP
contributes only 0.64% with an interval crossing zero; the actual `lm_head`
input-VJP MM contributes exactly 0%; final-RMSNorm backward contributes -0.58%
with an interval crossing zero. A separate complete final-RMSNorm F+B
materialization repair contributes 5.01%, also with an interval crossing zero.
All eager replays, restoration shams, and algebraic closures pass.

This is a strict natural complete F+B semantic-region directional-carrier case:
its actual F+B boundary, held-out direction, causal interventions, matched sham,
and paired live-weight accumulation are all closed. It is not a fully single-
kernel-attributed case. The downstream transport has been resolved, while the
upstream forward origins remain overlapping contributors inside the closed region.
The 51.5% key-forward repair and the terminal-logit repair are nested
interventions and cannot be added. Full evidence and claim boundary:
`docs/l23_qproj_tile.md`.

## Common conclusion and boundary

All four support

\[
\text{directional finite-arithmetic error}
\longrightarrow
\text{coherent gradient carrier}
\longrightarrow
\text{parameter/weight accumulation}.
\]

They do not support one shared kernel bug. All six strict project cases are
precision-mediated, although
shape, layout, support, fusion/materialization,
and saved-state policy decide whether fixed-precision arithmetic accumulates
along a trajectory. Six concrete cases remain too few for a defensible generalized
cross-operator property, and the two semantic regions are not eligible single-kernel property
label; only Liger and Phi also pass cross-state confirmation.

## Why the sweep has not produced more cases

The apparent paradox is that local differences are common while complete
Flash-style cases are rare. The legacy BF16 launch-site shape sweep observes all 506,
452, and 450 changed sites at seq64/128/256 on every natural checkpoint and
two repeats. The unified invocation atlas contains 949 changed, closed F+B
units after absorbing supplemental exact bindings and the direct embedding
scatter. Most
of those differences are materialization or reduction-rounding changes. They
produce finite local residuals, but the signed projection changes with the
state, is averaged by a later reduction, or reaches no shared parameter
coordinate. A local error therefore does not become a carrier bias.

The causal intervention pilot makes this distinction explicit: eight exact
mapped strict-FP32 regions did reach the tied-gradient carrier, but none kept a
fixed projection direction over the eight checkpoints. Liger `dW` and the
subsequently confirmed layer-23 `q_proj` tile pass the current full
directional-carrier and live-weight accumulation gates. The Qwen `lm_head`
also passes the Flash-style trajectory track but fails the separate cross-state
confirmation. This is a mechanism diagnosis, not evidence that the
unmeasured implementation matrix is safe.

The candidate-blind source census and GPU replay are complete for all six
FP32/TF32 shape cells: 670/670 invocations at seq64/128 and 724/724 at seq256,
including the fused loss and RMSNorm weight reductions. The current
implementation matrix is therefore exhausted for this frozen Qwen3-1.7B
scope, with unmatched topology/nonfinite rows retained as explicit boundaries.
The correct present statement is: six strict natural project cases pass the
Flash-style track—four root arithmetic and two closed semantic regions; two
root-arithmetic cases pass cross-state confirmation. Most other errors still lack causal repair, a real trajectory
carrier, or trajectory evidence.
The compact evidence split is recorded in
`results/final/missing_case_diagnosis.json`: it reports the 949 changed closed
units, the complete structured screen, one independently confirmed causal
candidate, zero previously tested persistent intervention arms, and the
completed six-cell replay.

The earlier short-bank exact key-RMSNorm bindings contained 28 complete forward/VJP chains whose
backward weight partial and final reductions end at a real parameter ABI. The
completed candidate-blind carrier replay found repeat-stable downstream deltas,
but no persistent direction: only `backward:852` was positive at steps 1/2/4/8
and it reversed at steps 16/32/64. The compact result is
`results/final/priority_carrier_replay.json`; that screen adds no case.

The later seq1024/4096-step campaign supersedes that short-bank negative result
for implementation-factor detection. The layer-19 k_norm factor still does not
count independently; the separately confirmed layer-23 `q_proj` carrier does.

## Why the current BF16 implementation sweep adds no new case

The evolving direct Triton layer now observes every changed site retained by
the three BF16 shape atlases on all eight natural checkpoints and two repeats:
506/506 sites at seq64, 452/452 at seq128, and 450/450 at seq256. The signed
endpoint mean changes sign over the bank for 498, 442, and 440 sites
respectively. Thus the sweep finds abundant local arithmetic differences, but
their directions are state-dependent and cancel before persistent parameter
accumulation. No additional complete F+B carrier case is supported by these
measurements. Shape128/256 transfers and non-BF16 generated inventories remain
explicit open boundaries; they are not silently treated as negative results.

The first exact-mapped region-level intervention confirms the remaining
causal gap. For one strict-FP32 seq128 RMSNorm/reduction region, replacing its
generated outputs with the candidate-blind reference changed the tied-
embedding gradient at both checkpoint 0 and checkpoint 1. A fixed 131072-
coordinate carrier sketch had step-1 cosine 0.0112, below the 0.05 directional
screen. It is therefore a causal local-to-carrier transmission pilot, not a
third natural bias case. See `results/final/region_intervention_pilot.json`.

That pilot is now extended to eight exact-mapped strict-FP32 seq128 regions
and checkpoints 0, 1, 2, 4, 8, 16, 32, and 64, with two repeats per arm. The
fixed step-0 carrier sketch is candidate-blind and all repeats match, but every
arm changes projection direction across the bank; no arm has a persistent
direction. This closes the mapped-region screening link without adding a
natural Flash-style case or authorizing a property claim. The compact record
is `results/final/region_intervention_batch.json`.

A candidate-blind seq256 probe operationally checked the first boundary: all
114 unresolved shape-transfer rows were observed at eight natural checkpoints
with two stable repeats. The only persistent positive residual was an internal
RMSNorm split-partial output; the final reduction output was exactly zero in
all checkpoints. Thus the residual cancels inside the region and is not a
complete F+B bias case. The formal shape-transfer rows remain unresolved for
coverage accounting.

## Screened complete F+B difference that is not a bias case

### Qwen seq128 layer-27 attention softmax

This non-MM unit is now independently closed from the actual forward program
through both Q/K backward output paths.  With

\[
S=QK^T,\qquad Z=\alpha S+M,\qquad P=\operatorname{softmax}(Z),
\]

and upstream cotangent \(G=\partial L/\partial P\), its analytic VJP is

\[
\frac{\partial L}{\partial Z}
=P\odot\left(G-\sum_j G_jP_j\right),\qquad
\frac{\partial L}{\partial S}=\alpha\frac{\partial L}{\partial Z},
\]

\[
\frac{\partial L}{\partial Q}=\frac{\partial L}{\partial S}K,
\qquad
\frac{\partial L}{\partial K}
=\left(\frac{\partial L}{\partial S}\right)^TQ.
\]

The AOT forward softmax and backward VJP share `seq_nr=14308`; the saved
probability crosses phases by exact Python-object identity, the upstream edge
is a real loss cotangent, and both Q/K paths reach backward outputs.  The exact
executed forward and backward Triton source hashes and call lines are bound in
`results/coverage/cases/qwen128_softmax_fb.json`.

The complete generated F+B derivation exposed an implementation detail hidden
by a backward-only comparison: Inductor does not read the AOT-saved FP32
probability in backward.  It stores BF16 logits plus FP32 row maximum/sum in
forward and reconstructs the probability from those auxiliaries.  The final
VJP error is therefore decomposed coordinatewise into five disjoint sources:
forward probability kernel arithmetic, forward BF16 probability rounding,
F-to-B saved-state reconstruction, backward kernel arithmetic, and backward
BF16 output rounding.

Over 32 frozen natural states and two exact repeats, the reconstruction is the
largest source: its probability difference reaches 0.06158 and its VJP
difference reaches \(3.52\times10^{-5}\).  It is not directional.  Its
cross-state U statistic is \(4.79\times10^{-14}\), with 95% distinct-cluster
bootstrap interval \([-2.64\times10^{-12},2.65\times10^{-12}]\).  The complete
semantic VJP error is also noncoherent: \(U=1.13\times10^{-13}\), interval
\([-2.65\times10^{-12},2.75\times10^{-12}]\).  All five source intervals fail
the positive-lower-bound gate.

Across unrelated states this remains noncoherent, but a matched saved-P repair
on the real 32-step training trajectory gives fixed-direction q/k master-weight
projections
\(0.004468,0.005082,0.005168,0.005250\) at steps 1/8/16/32 and 218,326 BF16
weight-coordinate divergences at step 32. It is therefore a strict Flash-style
semantic-region case and a cross-state negative. The causal boundary is the
saved-P-to-`dS` transformation, not one uniquely identified Triton instruction.

The Qwen3-VL AOTAutograd screen closed every concrete F+B unit in BF16 and
FP32. It also isolated one natural implementation difference: AOT decomposes
the 28 text SiLU backward invocations into graph-dtype elementary arithmetic,
whereas eager uses `aten.silu_backward`.

A backward-only intervention with unchanged SiLU forward reproduced the AOT
gradient bitwise for all 625 parameters in BF16 and FP32. The global gradient
delta is 21.3679 in BF16 and 0.0009945 in FP32. Nevertheless, over six frozen
natural states its global mean pairwise error inner product is negative, with
coherence ratio -0.01136; the three prespecified weight endpoints are also
negative. It is therefore a complete causal numerical-difference case, but it
fails the directional-carrier gate and is not counted in the table above.

See `round2.md` for the derivation boundary and full verdict.

## Evidence

- FlashAttention paper: <https://arxiv.org/pdf/2510.04212>
- `lm_head`: `results/final/precision.json.gz` (`mm_case`, `mm_arithmetic`,
  `mm_carrier`, and `mm_steps` entries in `results/final/manifest.json`)
- Liger fused CE: `archive/nonprecision_v1/runs/liger.fused_ce.mechanism.json`,
  `liger.fused_ce.certificate.json`, `liger.fused_ce.chunk.certificate.json`,
  and `results/final/trajectory.json.gz`
- Qwen3-VL negative directional case: `round2.md` and the compact round-2
  result package
- seq1024 carrier and mechanism controls:
  `results/final/long_horizon_trigger.json`,
  `results/final/backend_stage_diagnosis.json`, and
  `results/final/inductor_config_heldout.json`
