# TCMP experiment checkpoint

> Historical experiment checkpoint. Its 8-row roster and property-search
> decisions are provenance, not the current case count or paper method. Current
> definitions live in `docs/current_mainline.md` and `docs/method.md`; current
> long-run counts live in the machine audit under
> `results/property/declared_persistent_4096/`.

## Current decision (2026-08-21)

- The existing systematic ledger contains **8 complete paired F+B trajectory
  separation observations**. Of these, 7 pass the trajectory-local directional
  persistence gate and 4 pass the stricter same-contrast formation-plus-
  persistence chain. The number 8 is therefore an audit denominator, not a
  count of eight identical Flash-style mechanisms.
- The canonical source-persistent evidence remains mechanism-specific (the
  Liger accumulation case and the Phi lm-head transport case are the cleanest
  examples). The other retained trajectories include response-rectification,
  feedback, or mixed/partial mechanisms and must keep those labels.
- Gemma 4 adds one **new implementation, backward-visible,
  feedback-sustained** case. It is not a second Flash-style source carrier.
- The new Mamba state-space search has complete 980-invocation runtime
  coverage and a bound backward candidate, but strict deep F+B replay is
  `UNRESOLVED_COMPILE` after three bounded compiler attempts. OLMoE has
  complete screening and exact mathematical F+B witnesses for router/top-k/
  index-add semantics; the repaired eager probe is now a backward-visible,
  cross-state-canceling control. Neither is a new bias case. A separate
  DeepSeek layer-10 saved-softmax/backward-VJP region was fully measured and
  is an exact safe-under-protocol control.

The next search must not repeat already measured GEMM/lm-head
representatives. OLMoE is now closed for this lane; Mamba remains explicitly
blocked at compiler generation and should not be relabeled as a negative. Any
new semantic family selected from the existing census must first pass exact
F+B binding, repair/sham, and formation measurement; only a directional
formation result earns a 32-step consequence. No screen statistic is promoted
to a case without that chain.

## OLMoE complete screening cells

- `text128`: 8/8 screening states, 25,882 actual F+B implementation invocations.
- `text512`: 8/8 screening states, 11,080 actual F+B implementation invocations.
- Total retained OLMoE denominator: 36,962 actual invocations.
- New-metadata `text512` states 6–9 contain 5,419 invocations, 304 exact
  callsite/ABI identities, and 36 deduplicated implementation patterns.
- Repeated layers and states remain coverage/population evidence; they are not
  counted as new deep-measurement cases.

The sampled-coordinate pattern screen produced no BH-q=0.10 formal positive.
It did produce one preregisterable follow-up candidate not present in the prior
deep-measurement registry: a fused rotary/bmm forward region whose two outputs
were maximally aligned across all eight states (`A=sqrt(8)`, exact sign-flip
`p=0.0078125`). This is a screening candidate, not yet a complete F+B bias case.

## Gemma 3 text128 complete screening

- Real BF16 full F+B admitted at 18.85 GiB peak reserved memory; all 444
  parameter gradients were finite.
- Eight text128 states: 15,088 actual F+B implementation invocations, 989 exact
  identities, 56 deduplicated implementation patterns, zero unresolved.
- No sampled-coordinate local precision endpoint passed BH q=0.10. This is not
  a no-bias verdict: local-centered to gradient-biased transport remains a
  separate F+B deep-measurement target.
- Value-blind follow-up strata are the highest-ranked softmax region, backward
  reduction, and backward MM; repeated layers are excluded.
- `text512` is complete: eight states retain 16,432 actual F+B invocations,
  1,157 exact callsite/ABI identities and 59 implementation patterns, with
  zero unresolved identities. No endpoint passed BH q=0.10; the highest
  value-blind follow-ups are a new rotary/bmm fusion, RMSNorm reduction and
  backward sum patterns. These are candidates, not bias cases.

## Gemma 3 multimodal coverage

- The default full backward exceeds 47.4 GiB. Standard activation
  checkpointing preserves full-model F+B and all 883 observed parameter
  gradients at 21.30 GiB peak reserved memory; recomputation invocations are
  retained in the denominator as a distinct implementation configuration.
- The captured screening-state schedule contains 4,557 actual compute sites:
  2,484 Triton, 2,069 extern, two direct ATen and two direct torch-op calls.
- This graph adds a previously uncatalogued `masked_scatter_backward` F+B
  boundary. Its exact mask-order VJP is now recorded rather than ignored.
- Two-state engineering replay and all eight screening states completed with
  exact same-process wrapper binding. The retained denominator is 36,664
  actual F+B invocations, 1,675 exact identities, 149 implementation patterns
  and 99 semantic families, with zero unresolved identities.
- Relative to the text graph, 20 ATen semantics are genuinely new, including
  convolution/backward, average-pool/backward, native LayerNorm/backward,
  masked-scatter/backward, cumulative sum, maximum and index operations.
- No local endpoint passed BH q=0.10. Two new semantic bottlenecks are retained
  for exact F+B repair probes rather than declared cases: vision LayerNorm
  backward (`p=0.0234`, `A=1.866`) and the pooled-normalization backward path.
  `masked_scatter_backward` is an exact-zero safe control under this protocol.
- Exact repair of the vision LayerNorm-backward endpoint changes 413 complete
  parameter-gradient tensors, so the difference is backward-visible and not a
  local-only artifact. Across eight independent image states, however, the
  complete 401,614,160-coordinate parameter-gradient Gram gives `A=0.96994`
  and exact sign-flip `p=0.91406`. It is therefore a full-F+B
  variance-only/canceling control under this open-loop protocol, not a new
  directional-bias case. The pooled-normalization target changes zero parameter
  gradients and is closed without further deep measurement.
- The genuinely new external vision convolution was then isolated separately.
  Its FP32-storage output repair changes all 883 parameter-gradient tensors
  (`endpoint RMS=6.02e-4`), but the complete Gram on the convolution unit's own
  678,528 weight/bias coordinates gives `A=0.99612` and exact sign-flip
  `p=0.515625` over the same eight images. It is another backward-visible but
  cross-state-canceling F+B control, not a directional-bias case. This does not
  substitute for testing the distinct generated `convolution_backward`
  implementation.
- The generated `convolution_backward` weight-gradient implementation was
  therefore repaired independently. It changes exactly the patch-embedding
  weight gradient (`endpoint RMS=3.29e-5`), as required by its output mask. Its
  complete 677,376-coordinate eight-state Gram gives `A=0.99994` and exact
  sign-flip `p=0.515625`. Thus both mathematical sides of this new convolution
  F+B unit are backward-visible but canceling under the frozen population
  protocol; neither is counted as a persistent-bias case.

## Frozen held-out: Llama 3.2 and Ministral 3

> **Evidence amendment.** The first held-out run preserved the required
> prediction-before-consequence ordering, but its runner used eight total
> variants including the default and a plug-in orbit mean. The frozen protocol
> required one separately held default plus eight non-default variants with a
> deterministic 4+4 cross-fit. Its consequence also used a sign-flip test and
> a one-shot perturbation rather than the frozen repeated structure-matched
> empirical null. The numbers below are therefore a prospective pilot with a
> pre-measurement implementation deviation, not final confirmatory evidence.
> Corrected measurements are retrospective for these two models because their
> consequences have now been revealed.

- Both text128 cells completed eight-state all-implementation screening with
  zero unresolved identities: 10,208 actual invocations for Llama and 9,912
  for Ministral. Their seq512 schedules add no raw ATen semantics, so they are
  retained as coverage configurations without a duplicate deep screen.
- Before trajectory states were generated, the pilot transported-orbit
  predictor marked both exact `lm_head dX` implementations as risk:
  `A_m=2.131` (Llama) and `A_m=2.183` (Ministral), both exact sign-flip
  `p=0.00025`.
- On disjoint 32-step live-weight trajectories, both pilot predictions matched a
  persistent actual update drift: `A_D=1.221`, final norm `8.44e-6` for Llama;
  `A_D=1.229`, final norm `8.93e-6` for Ministral. Four-arm symmetric
  recurrence and telescoping residuals are exactly zero.
- The corrected, retrospective default-plus-eight 4+4 cross-fit retains the
  signal: `A_m=2.150`, mean/orbit-sigma `2.076` for Llama and `A_m=2.220`,
  mean/orbit-sigma `2.160` for Ministral (both sign-flip `p=0.00025`). Thus the
  pilot signal was not created by the plug-in estimator, although these
  corrected values are no longer prospective evidence.
- The dynamics are not a simple fixed source carrier. Realized local
  increments are near the diffusive boundary (`A_L=1.004/1.010`), while the
  feedback terms are persistent (`A_B=1.345/1.209`). A norm/support-matched
  one-shot random perturbation does not reproduce the natural direction
  (final cosine `-0.566/-0.015`) and does not grow after the first write. Thus
  generic Lyapunov amplification is not established by that one-shot control.
  A stronger retrospective control changes to a different real, semantics-
  preserving reduction orbit on every one of 32 steps for five frozen seeds.
  It retains `94.5%/93.8%` of natural drift norm and mean natural-drift cosine
  `0.816/0.738` (Llama/Ministral), while its local error norm is `88.2%/86.7%`
  of natural. The joint trajectory effective rank is `1.28/1.47`. Therefore a
  fixed reduction order is not the temporal anchor in these cases. The data
  instead support a tiling-conditional common orbit mean plus low-dimensional
  training feedback; schedule randomization weakens but does not remove drift.
- These two confirmations are `SEEN_IMPL / NEW_OPERANDS`: they validate
  cross-model operand generalization of one `lm_head dX` mathematical family,
  not two new operator mechanisms and not `NEW_IMPL` generalization.
- Two genuinely new Ministral semantic representatives were independently
  closed. YaRN `floor/log` is exact under the repair. The attention-mask /
  softmax fusion has local endpoint differences, but repairing one of its 26
  repeated regions changes neither loss nor any parameter gradient. All 26
  invocations remain in the denominator; only one representative is deeply
  measured.

## Counting rule

All runtime invocations and exact ABI identities remain in the denominator.
Orbit, transport, and trajectory experiments select one value-blind
representative per implementation pattern. Exact variants are reopened only
when ABI, tile/reduction topology, fusion semantics, or routing discontinuity
changes the mechanism.

## Frozen new-implementation audit: Gemma 4 E2B

This is the first `NEW_IMPL` audit after the development roster was frozen.
It is deliberately separate from the earlier Qwen/Phi/Liger cases.

- The model contributes 14,638 exact eager F+B events and 1,910 generated
  compute invocations (867 Triton, 1,041 extern, two direct Aten), with zero
  unresolved generated pointer-dataflow bindings. The eight-state screen kept
  115 distinct pattern endpoints; 83 had nonzero residuals and 72 patterns
  were absent from the previous implementation census. No sampled endpoint
  passed the BH-q=0.10 local screen. This is a screen result, not a safety
  certificate.
- A preregistered semantic-hotspot audit found that softcapped CE and the
  attention-softmax region had no parameter reach. They are `NOT_APPLICABLE`,
  not negative cases. PLE/RMSNorm (`forward:2`) changed exactly the embedding
  and per-layer projection gradients and was therefore the eligible hotspot.
- On 16 complete-coordinate confirmation states, that PLE/RMSNorm endpoint
  had large but cancelling source residuals: `A_L=0.99833`, odd/even cosine
  `0.00306`. Its source prediction was frozen as
  `NO_SOURCE_PERSISTENCE` before the separate trajectory was revealed.
- The 32-step four-arm trajectory nevertheless produced
  `A_L=1.00087`, `A_B=3.23535`, `A_D=3.22522`, final drift `0.10701`, and
  maximum recurrence residual below `2.5e-10`. This is a new implementation
  case of **feedback-sustained drift**, not a Flash-style source-persistent
  carrier. It supports the split between source persistence and feedback
  persistence; it does not establish a universal source property.
- The feedback attribution was then tested on the same frozen endpoint and
  trajectory with two optimizer interventions. Stateless SGD reduced the
  feedback/actual amplifications to `0.953/0.954` and final drift to
  `6.35e-4`. Resetting Adam moments at every step reduced them to
  `0.996/0.996` while preserving a comparable local residual scale. Thus the
  Gemma drift is specifically sustained by cross-step Adam state, not by a
  generic Lyapunov response to any repeated perturbation.
- The 12-family screen-negative recall audit found ten parameter-inaccessible
  endpoints (`NOT_APPLICABLE`) and two exact parameter-reachable candidates.
  The attention-softmax backward candidate (`backward:1880`) had only one
  nonzero formation state and its completed trajectory was numerical-floor
  zero (`A_D=1.0`, final drift `1.5e-8`), so it is a genuine canceling/control
  result under this protocol. The second candidate (`backward:1401`) is kept
  as `UNRESOLVED_PROVENANCE` because its rerun generated wrapper hashes that
  did not match the frozen release; it is not counted as a case or a negative.

The correct current claim is therefore: Gemma 4 adds one strictly new,
backward-visible, feedback-sustained case and one exact canceling control;
it does not add a second source-persistent Flash-style case. Screen-negative
counts are not used as the denominator for bias claims because most screened
endpoints never reached parameters.

## New-architecture search: Mamba 130M

Mamba was added to search a genuinely different state-space implementation
family rather than another transformer GEMM/lm-head copy. The frozen seq128
release contains 980 generated compute invocations (592 Triton, 363 extern,
24 direct torch operations, and one direct Aten call), with zero unresolved
runtime bindings. An eight-state forward/backward screen found 894 nonzero
endpoint patterns. These are local precision differences only; they are not
parameter-bias cases.

The most relevant new region is a Triton
`convolution_backward + silu_backward` state-space fusion (`backward:779`,
1536-coordinate `out_ptr1`). It was bound to its exact program, source digest,
and pointer ABI. Three bounded engineering replays were attempted, including
a 10-minute run reusing the existing Inductor cache; none reached runtime
measurement. The last interruption was in Inductor Triton code generation for
the huge slow-Mamba backward graph. The result is therefore
`UNRESOLVED_COMPILE`, not a negative and not a bias case. The exact binding and
screen evidence are retained in `heldout/mamba130m_seq128_newscan/` so the
search can resume without changing the denominator or selecting a result
after seeing it.

## Next source-search attempt: OLMoE router/expert path

The OLMoE full coverage census contains a genuinely new semantic family
(`topk`, `scatter`, `index_select`, and `index_add` routing), so it was selected
without repeating the existing GEMM/lm-head representatives. Its 2-state
preflight was within the 44 GiB budget (27.7 GiB peak allocated). Both an
allow-graph-break and a strict full-graph release attempt were then rejected
by the fail-closed direct-runtime-call inventory validator. No numerical screen
or trajectory verdict was issued. This remains `COVERAGE_ONLY/UNRESOLVED`, not
a negative case; the blocked artifact is recorded in
`heldout/olmoe_1b7b_text128/router_search_status.json`.

## OLMoE router accumulation formation probe (2026-08-21)

The direct runtime gate was repaired for a bounded, one-model-at-a-time eager
probe. At layer 0, the candidate keeps native BF16 `index_add_` accumulation
and the repair accumulates the identical BF16 expert summands in FP32 before
one final cast. Routing indices and routing weights were identical in all 26
states, so this is not an MoE routing-flip artifact. Each arm was a complete
forward/loss/backward step from the same frozen weights; no optimizer step was
taken.

The repair is backward-visible: the declared router-gate gradient slice had a
mean relative difference of about 9.1%, and the downstream layer-0 attention
slice about 9.3%. However, the 26-state complete Gram summaries are
cross-state canceling (`A=0.9805` for the router-gate slice and `A=0.9681` for
the layer-0 attention slice; both have negative mean off-diagonal inner
products). Therefore this new MoE `topk/index_add` implementation family is
currently a **backward-visible variance/canceling control**, not a directional
formation case and not a Flash-style persistent-bias case. It is not promoted
to the case count because a trajectory consequence was not run after the
formation screen failed to show a source carrier.

The complete probe summaries are retained under
`results/property/tcmp_allop_v1/heldout/olmoe_1b7b_text128/` with the explicit
claim boundary `ENGINEERING_ONLY`; the earlier partial two-batch files are
intermediate and are not part of the retained result set.

The same semantic family was then tested with a value-preserving expert-order
orbit: only the order of the 64 expert accumulations was permuted, while the
real-valued sum, routing decisions and routing weights were unchanged. On 16
states and eight total orders (default plus seven non-default variants), the
router-gate orbit-mean amplification was `1.0036` and the 4+4 cross-fit value
was `1.0037`. The downstream layer-0 attention slice was also non-directional
(`A=0.9832`, cross-fit `0.9635`). Thus the new MoE family does not currently
support either source-order or transport-order formation bias under this
protocol. These are stronger canceling controls than the BF16-vs-FP32 repair
screen, not additional cases.

The earlier official-fused Mamba screens were also checked before reopening
any new Mamba measurement. The independently confirmed seq64, seq256 and
seq1024 fused-scan releases all have `natural_bias_case_added=false` and fail
their frozen direction gate; they are not missing cases. The newer seq128
generated-convolution candidate remains a separate `UNRESOLVED_COMPILE`
binding. The search therefore neither duplicates an old Mamba screen nor
promotes a local residual without a directional F+B chain.

The final RMSNorm slice showed a small exploratory excess (`A=1.118`,
cross-fit `1.329`), but its frozen 4,000-draw sign-flip test gave `p=0.154`
and did not exceed the 95% null threshold. It is therefore not promoted as a
case. OLMoE is now closed for this search lane as a backward-visible,
non-directional control; further layers or repeated order variants would be
duplicate measurements rather than a new semantic family.

## DeepSeek layer-10 attention-softmax formation control (2026-08-21)

A new DeepSeek-R1-0528-Qwen3-8B seq128 semantic region was tested without
reusing the existing layer-35 `dV` candidate: layer-10 saved attention logits
(`forward:191`) together with their exact softmax backward VJP
(`backward:1529`). The experiment used 16 calibration and 16 disjoint
confirmation common states, with candidate, typed forward repair, typed
backward repair, joint repair, and matched sham arms. All arms were complete
forward/loss/backward executions from identical state components; no weights
were advanced.

The repair was ABI-safe and the sham was exact on the declared parameter
carrier (`model.layers.10.self_attn.q_norm.weight`). Forward local residuals
were zero and backward residuals stayed at numerical floor (the few
carrier-reachable events were isolated roundoff-level values); all three
formation layers were `CENTERED` in both state partitions and no parameter
carrier was reached. This is an **exact safe-under-protocol control**, not a
bias case. It contributes zero to the case count and no trajectory consequence
was run. The full certificate and compact status are retained in
`results/property/tcmp_allop_v1/heldout/deepseek8b_seq128_l10_softmax/`.

## DeepSeek layer-35 dV boundary audit (2026-08-21)

The existing DeepSeek seq64 layer-35 attention `dV` record was audited before
counting it as another case. Its strict 16+16 common-state F+B certificate
classifies the local endpoint, parameter-gradient, and effective-update
populations as `CENTERED`; the certificate did not include a live-weight
trajectory. A separate 32-state moving-frame analysis reports a conditional
gradient-scale alpha mean of `-0.003698` with 95% interval
`[-0.006637,-0.000592]`, but that is not a fixed-carrier or live-weight
persistence certificate. The record is therefore
`PARTIAL_CONDITIONAL_BIAS_NO_FLASH_STYLE_PERSISTENCE` and contributes zero to
the strict Flash-style case count. Its status is retained in
`results/property/persistence_v1/deepseek_l35_dv_status.json`; a future upgrade
would require a separately bound live-weight F+B consequence campaign, not a
reinterpretation of the existing alpha statistic.

## DeepSeek layer-0 normalization closure (2026-08-21)

The first genuinely new semantic-family candidate after the OLMoE and
layer-10 controls was the DeepSeek seq256 layer-0 fused normalization-backward
region (`backward:1952`). Its exact downstream closure (`backward:1957`) was
measured with the corresponding LayerNorm weight as the declared carrier.
The two-state reach preflight passed, and the formal campaign used 16
calibration plus 16 confirmation common states with complete candidate,
reference, sham, and repair F+B executions.

This endpoint is backward-visible: the local residual reaches the complete
LayerNorm gradient (mean state energy `0.07889` in calibration and `0.02459`
in confirmation). Nevertheless, the formation is centered in every layer.
The effective-update cross-state ratios are `0.001881` and `-0.002822`, far
inside the frozen centered margin. It is therefore a genuine
**backward-visible variance/canceling control**, not a positive bias case and
not a zero-residual or parameter-inaccessible negative. Since formation did
not produce a directional signal, no 32-step trajectory was run. The compact
status and complete certificate are retained in
`results/property/tcmp_allop_v1/heldout/deepseek8b_seq256_norm_l0/`.

## Mamba state-space closure feasibility boundary (2026-08-21)

To avoid treating the previous Mamba internal compile failure as a missing
negative, we attempted one exact downstream closure from the new
state-space-recurrent family: `backward:15107` at layer 4, with
`backbone.layers.4.mixer.in_proj.weight` as carrier. The frozen AOT binding is
valid, but the runner failed before any numerical observation during Inductor
joint-graph `bmm_to_mm` compilation. GPU memory was only about 570 MiB at the
bounded interruption, so this is a compiler/code-generation feasibility
failure, not an OOM and not a numerical verdict. It remains
`UNRESOLVED_COMPILE`, contributes zero to both positive and negative counts,
and is not sent to trajectory. Its resumable case plan and status are retained
under `results/property/tcmp_allop_v1/heldout/mamba_seq128_silu_state_closure/`.

## Semantic-family held-out continuation (2026-08-21)

The metadata-only held-out pool is frozen at 791 semantic cells and 493
deduplicated implementation-pattern representatives. It contains 485
pre-measurement candidates; existing deep measurements are excluded by exact
task identity, not by post-hoc verdict. The pool is a selection ledger, not a
bias label, and centered, not-applicable, and unresolved cells remain in its
denominator.

One new loss-path representative was selected mechanically without repeating
the prior GEMM/lm-head cases: DeepSeek seq256 cross-entropy backward MM
(`backward:659:output_0`). Its two-state reach preflight changed 8,276--8,409
endpoint coordinates and reached `model.norm.weight` on both states. The
formal 16+16 open-loop F+B measurement was nevertheless centered at all three
layers. Local residual energy was nonzero (`7.00e-10`/`7.72e-10`), while the
gradient/update cross-state ratios were `0.0485`/`0.0791` in both paired
partitions and stayed inside the frozen centered margin. This is a new,
backward-visible loss/CE variance-canceling control, not a persistent-bias
case; no trajectory was run.

A second nonduplicate semantic family was attempted: Mamba seq128
state-space-recurrent backward MM (`backward:15101:output_0`) with the
`x_proj` carrier. The bounded two-state engineering run produced no runtime
observation before Inductor Triton tiling/scheduling exceeded the compile
budget. GPU memory remained about 570 MiB, so this is `UNRESOLVED_COMPILE`, not
an OOM and not a numerical negative. Its status and resumable case plan are
retained under
`results/property/tcmp_allop_v1/heldout/mamba_seq128_state_mm/`.

The search therefore adds one genuine new canceling control and no new
Flash-style positive. Further candidates should be chosen from the frozen
pool by a deterministic family/implementation rule; repeated normalization,
softmax, or MM representatives are not additional cases.

The next nonduplicate Phi-4 normalization representative
(`backward:1125:in_out_ptr0`) was also preflighted. Its strict fullgraph warm-up
stopped in the Transformers LongRoPE frequency-update branch before any
runtime observation. It is retained as `UNRESOLVED_COMPILE`, not treated as a
negative or a case; see
`results/property/tcmp_allop_v1/heldout/phi4_seq64_norm_backward/status.json`.

A third nonduplicate representative, DeepSeek seq128 attention-projection
backward MM (`backward:1667:output_0`), passed the two-state reach preflight and
completed 16+16 formation. The endpoint reached
`model.layers.6.input_layernorm.weight` with nonzero local and gradient energy,
but all formation layers were centered: effective-update ratios were
`0.000811` (calibration) and `-0.008580` (confirmation). It is therefore a
second new backward-visible variance/canceling control, not a positive case;
no consequence trajectory was run.

## Repeated orbit nulls for held-out lm-head trajectories (2026-08-21)

The earlier one-step random perturbation control was insufficient to test the
null hypothesis that repeated noise injections could create the same drift. A
new five-seed, 32-step control injects one captured, semantics-preserving
reduction-orbit variant at every step while keeping the same paired initial
state and optimizer mapping.

For Llama 3.2, null final drift is `0.934--0.960` times natural drift and its
cosine with natural drift is `0.808--0.823`; the drift-subspace participation
ratio is `1.278`. For Ministral 3, the corresponding ranges are
`0.983--1.018`, `0.688--0.702`, and participation ratio `1.571`. Thus a single
random kick is not sufficient to explain the observed separation, but
per-step orbit randomization also does not remove the drift. These controls
support a low-dimensional closed-loop feedback interpretation and do not
by themselves prove a fixed operator source carrier. They remain retrospective
mechanism diagnostics, not held-out predictor confirmations.

The compact summary is
`results/property/tcmp_allop_v1/repeated_orbit_null_summary.json`; the full
per-seed traces remain beside each held-out model's consequence artifacts.

## Final decision checkpoint (2026-08-21)

The machine-readable decision matrix is
`results/property/tcmp_allop_v1/final_decision_matrix.json`, with the current
claim and remaining scope in `docs/current_mainline.md`. The historical result is
a conditional predictor for the exact LM-head reduction/VJP family: it
generalizes across two new operand distributions, but the frozen search found
zero new-implementation positive cases. New semantic families contribute
backward-visible canceling controls or explicit abstentions. The project must
therefore not claim a universal all-operator property. Gemma feedback drift is
reported as a separate consequence channel, and the repeated-orbit nulls are
feedback-compatible diagnostics rather than proof of source-specific causality.

## Post-checkpoint scope extension closeout (2026-08-22)

Five mechanically selected, previously unmeasured F+B endpoints were completed
after the final checkpoint.  Each endpoint had a nonzero implementation
residual, reached its declared parameter carrier, and was evaluated on disjoint
open-loop calibration and confirmation populations.  All three formation
layers were `CENTERED` in both populations:

| Model | Exact backward endpoint | Semantic role | States | Result |
| --- | --- | --- | ---: | --- |
| DeepSeek 8B | `backward:1665:in_out_ptr0` | attention softmax backward | 16+16 | complete centered control |
| Phi-4 | `backward:495:out_ptr1` | cross-entropy backward | 16+16 | complete centered control |
| Phi-4 | `backward:1031:in_out_ptr0` | attention softmax backward | 16+16 | complete centered control |
| Qwen3 1.7B | `backward:1293:in_out_ptr0` | attention softmax backward | 32+32 | complete centered control |
| Qwen3 1.7B | `backward:1308:output_0` | normalization backward | 16+16 | complete centered control |

The Qwen attention endpoint initially had insufficient power on a 128-coordinate
carrier at 16+16 states.  Its population was extended mechanically to 32+32;
both parameter-gradient partitions then resolved as `CENTERED`.  The Phi-4
measurements required the already declared graph-break execution boundary due
to the Transformers LongRoPE control-flow branch; this affects the execution
boundary, not the numerical verdict.

No consequence trajectory was run because no formation layer was biased.  The
extension therefore adds five genuine backward-visible canceling controls and
zero new implementation-class positives.  An unexecuted Mamba metadata plan is
not part of this result and is not counted as an abstention or a negative.

The compact machine-readable ledger is
`results/property/tcmp_allop_v1/scope_extension_20260822.json`.
