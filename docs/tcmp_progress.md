# TCMP experiment checkpoint

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

- Both text128 cells completed eight-state all-implementation screening with
  zero unresolved identities: 10,208 actual invocations for Llama and 9,912
  for Ministral. Their seq512 schedules add no raw ATen semantics, so they are
  retained as coverage configurations without a duplicate deep screen.
- Before trajectory states were generated, the frozen transported-orbit
  predictor marked both exact `lm_head dX` implementations as risk:
  `A_m=2.131` (Llama) and `A_m=2.183` (Ministral), both exact sign-flip
  `p=0.00025`.
- On disjoint 32-step live-weight trajectories, both predictions matched a
  persistent actual update drift: `A_D=1.221`, final norm `8.44e-6` for Llama;
  `A_D=1.229`, final norm `8.93e-6` for Ministral. Four-arm symmetric
  recurrence and telescoping residuals are exactly zero.
- The dynamics are not a simple fixed source carrier. Realized local
  increments are near the diffusive boundary (`A_L=1.004/1.010`), while the
  feedback terms are persistent (`A_B=1.345/1.209`). A norm/support-matched
  one-shot random perturbation does not reproduce the natural direction
  (final cosine `-0.566/-0.015`) and does not grow after the first write. Thus
  generic Lyapunov amplification is not an adequate explanation; repeated
  implementation injection and state feedback interact.
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
