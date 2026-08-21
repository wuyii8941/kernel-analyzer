# TCMP all-operator protocol v1.2

This is a pre-property and pre-heldout correction to `tcmp_allop_v1`. It does
not change any existing numerical verdict. It prevents repeated exact
implementations from consuming the deep-measurement budget and makes the
predictive claim falsifiable.

## Implementation identity and non-repetition

Every executed invocation remains in the coverage denominator. Deep orbit and
trajectory measurements select one representative per frozen exact
implementation identity:

- backend and implementation kind;
- generated program digest or declared external implementation;
- floating operand shape, stride, dtype and layout;
- accumulation/reduction contract when available;
- fusion boundary, phase and semantic endpoint;
- tile, chunk, split-K and scalar launch contract when available.

The report separates `SEEN_EXACT_IMPL_NEW_OPERANDS`,
`NEW_EXACT_IMPL_SEEN_PATTERN`, `NEW_IMPL_PATTERN`, and
`NEW_SEMANTIC_FAMILY`. Repeated invocations and repeated exact implementations
remain counted but are not presented as independent scientific cases.

## Falsifiable predictor boundary

The predictor may use only measurements obtained without advancing model
weights: split-orbit source statistics, local residual geometry, declared
schedule metadata, analytic or reference-only F+B transport, and static
semantic structure. It may not read live-trajectory quantities including
`A_L`, `A_B`, `A_D`, trajectory separation, T4/M6/SEUP verdicts, final drift,
historical case names or any statistic requiring candidate/repair weight
advancement.

The frozen prediction target is whether the independent 32-step trajectory has
drift amplification above the empirical null. This temporal separation is the
scientific hypothesis; persistence is not used to define its own predictor.

## Screen-negative audit

Screen-negative exact implementations are stratified by implementation
pattern, F/B phase, semantic bottleneck and error-magnitude bin. A committed
seed (`20260821`) selects 16 initial units before trajectory outcomes are
revealed. If all 16 are negative and resources permit, the audit expands to 32
using the same frozen ordering. The result is reported as the exact-binomial
upper bound on hidden-positive prevalence among screen negatives, not as an
unqualified recall guarantee.

## Feedback and discontinuous routing

The source predictor and a pre-trajectory feedback-susceptibility gate are
separate outputs. A unit declared feedback-susceptible before trajectory reveal
is outside the source-persistence claim and is reported separately; it is not
removed after a failed prediction.

For MoE models, every paired step records per-layer top-k expert decisions,
top-k margins, routing Hamming distance, first divergent token/layer, and expert
loads. Same-route steps are eligible for smooth F+B transport statistics.
Routing-divergent steps remain in the denominator and are classified as a
discrete routing regime; they are never silently dropped from the experiment.

Held-out Llama and Ministral results remain sealed until implementation
identities, the predictor, domain gates, thresholds and positive/negative
predictions are committed.
