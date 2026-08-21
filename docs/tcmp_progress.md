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
- Two-state engineering replay is running before the eight-state screen.

## Counting rule

All runtime invocations and exact ABI identities remain in the denominator.
Orbit, transport, and trajectory experiments select one value-blind
representative per implementation pattern. Exact variants are reopened only
when ABI, tile/reduction topology, fusion semantics, or routing discontinuity
changes the mechanism.
