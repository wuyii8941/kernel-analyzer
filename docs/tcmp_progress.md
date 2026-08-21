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

## Gemma 3 status

- Real BF16 full F+B admitted at 18.85 GiB peak reserved memory; all 444
  parameter gradients were finite.
- First text128 state: 1,886 actual compute sites, 989 exact identities, 56
  deduplicated implementation patterns, zero unresolved identities.
- Remaining seven text128 screening states are running on the same frozen
  static implementation graph; no static graph recapture is performed.

## Counting rule

All runtime invocations and exact ABI identities remain in the denominator.
Orbit, transport, and trajectory experiments select one value-blind
representative per implementation pattern. Exact variants are reopened only
when ABI, tile/reduction topology, fusion semantics, or routing discontinuity
changes the mechanism.
