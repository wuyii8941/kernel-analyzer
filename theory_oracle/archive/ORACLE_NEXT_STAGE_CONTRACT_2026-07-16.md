# Centered-logit and One-step Transition Contract — 2026-07-16

Frozen before reading formal outputs from this stage. Smoke runs may test only execution viability and schema validity; they are not part of the estimand and cannot be used to change a threshold in response to favorable numerical results.

## 1. Purpose

This stage closes two gaps left by the real-model inference survey:

1. determine how much of the confirmed Qwen raw-logit average shift is a decision-invariant common-mode translation rather than relative-logit distortion;
2. determine whether implementation-relative forward discrepancy reaches gradients and a controlled one-step parameter transition at matched states.

It remains a relative discrepancy study. Eager is a named baseline, not mathematical truth.

## 2. Implementation relation and execution validity

- reference: PyTorch eager CUDA;
- candidate: tracked `torch.compile`/Inductor execution;
- fixed local model artifacts and frozen state banks;
- deterministic algorithms, fixed shapes within a run, fixed RNG seed and no sampling;
- alternating eager/candidate execution order;
- every candidate measurement call must reach a tracked compiled callable;
- unexpected graph proliferation, fallback, non-finite output/gradient, token misalignment, unequal initial weights, or missing observation fails closed;
- same-path repetitions are measured separately from cross-path discrepancy.

## 3. Centered-logit diagnostic

### Population

- model: `data/phase0_policy_final`;
- discovery: rows `[0, 32)` of `data/phase0_grpo_samples.jsonl`;
- held-out confirmation: rows `[32, 64)`;
- same sequence construction and token nesting as the frozen multi-subject inference contract;
- three repetitions per state/path.

### Per token-position decomposition

Let `d = logits_compiled - logits_eager` over vocabulary coordinates.

- common-mode shift: `c = mean(d)`;
- centered residual: `r = d - c`;
- raw mean absolute discrepancy: `mean(abs(d))`;
- centered mean absolute discrepancy: `mean(abs(r))`;
- centered maximum absolute discrepancy: `max(abs(r))`;
- common-mode energy share: `||c 1||² / ||d||²`, defined as zero only when `||d||² = 0`;
- centered residual energy share: `||r||² / ||d||²`;
- exact numerical audit: the two energy shares must sum to one up to floating-point analysis tolerance.

The common-mode component is invariant for softmax probabilities, argmax and top-k ranking. The centered residual is still only a numerical discrepancy; it becomes semantically relevant only through an event-specific margin or probability functional.

### Confirmation rule

- compare discovery and confirmation common-mode and residual scales separately;
- a signed common-mode direction is persistent only if its direction and state-bootstrap interval reproduce;
- no semantic claim is made from common-mode shift;
- centered residual is compared with target-log-probability, top-1/top-5 margins and paired ranking events;
- this diagnostic is post-inference-survey but predeclared before its own outputs; it is not retroactively part of the earlier confirmation.

## 4. Controlled one-step transition populations

### T1. BERT supervised transition

- model: `data/external_models/bert-tiny-sst2`;
- state: fixed model weights, one frozen SST-2 example, label, deterministic execution protocol and controlled optimizer definition;
- discovery: `data/external_datasets/sst2_discovery_128`;
- confirmation: `data/external_datasets/sst2_confirmation_128`;
- loss: two-class cross entropy;
- numerical execution: FP32 master parameters with CUDA FP16 autocast, matching the controlled Qwen transition convention;
- controlled update: full-parameter SGD, no momentum, no weight decay, learning rate `1e-5`;
- fixed length 64, batch size 1, dropout disabled;
- three same-state repetitions without carrying the update into the next state.

### T2. Qwen teacher-forced transition

- model: `data/phase0_policy_final`;
- state: fixed model weights, a frozen minibatch of four rollout sequences, response-token targets, deterministic execution protocol and controlled optimizer definition;
- discovery: rows `[0, 32)` grouped consecutively into eight minibatch states;
- confirmation: rows `[32, 64)` grouped consecutively into eight minibatch states;
- loss: mean teacher-forced response-token negative log-probability;
- numerical execution: FP32 master parameters with CUDA FP16 autocast; logits are upcast to FP32 for log-softmax/loss;
- controlled update: full-parameter SGD, no momentum, no weight decay, learning rate `1e-6`;
- fixed sequence length 166, dropout disabled, no sampling;
- two same-state repetitions unless the viability smoke shows that a third is required to distinguish a nonzero runtime component; the repeat count is identical in discovery and confirmation;
- minibatches and checkpoints were not selected by eager/compiled discrepancy.

These controlled SGD states calibrate transition sensitivity. They are not claims about the missing historical Adam/AdamW optimizer state and are not replicas of production GRPO updates.

## 5. Transition endpoint profile

For each matched state, record:

### Validity

- graph count/hash and compiled runtime evidence;
- exact equality of initial parameter tensors;
- input/target identity and finite loss/gradient checks;
- same-path loss and gradient reproducibility;
- execution-order audit.

### Forward/loss

- signed and absolute loss delta;
- target-log-probability delta where defined;
- task event disagreement for BERT;
- response-token argmax/top-5 events for Qwen as contextual endpoints, not as the transition definition.

### Gradient geometry

- reference and candidate gradient L2 norms;
- exact gradient-difference L2 and relative L2;
- cosine similarity;
- maximum absolute coordinate discrepancy;
- per-parameter-block difference norms;
- zero/nonzero-gradient support disagreement;
- deterministic gradient sketches/hashes used only for same-path audit, not as substitutes for exact cross-path distance.

### Update/next state

For the declared SGD map, record:

- update-vector L2 difference and relative difference;
- next-parameter-state L2 difference;
- maximum parameter-coordinate difference;
- per-block update heterogeneity.

For no-momentum SGD, gradient and update discrepancies are linearly related. Both are reported to make the transition map explicit; they are not presented as independent evidence.

### Decomposition and inference

- average signed scalar shifts only for scalars with a declared orientation;
- between-state heterogeneity of loss/gradient/update effects;
- same-state runtime variability;
- state-bootstrap uncertainty, with Qwen minibatch as the resampling unit;
- discovery and confirmation reported separately.

## 6. Interpretation gates

1. A nonzero raw-logit mean without centered residual or semantic effect is common-mode numerical shift, not semantic bias.
2. A nonzero centered residual without event or transition effect remains a numerical measurement.
3. A nonzero gradient/update difference is local transition impact, not evidence of long-run harm.
4. A direction is called persistent only when it reproduces on the held-out state bank.
5. Zero disagreement does not establish operational equivalence without a predeclared tolerance or power bound.
6. BERT and Qwen endpoints are never pooled into a common event rate.
7. No correctness claim is permitted without an independent specification or higher-accuracy reference.

## 7. Entry condition for operator analysis

Operator repair/injection begins only for an endpoint satisfying all of:

- compiled execution and self-pair validity pass;
- the endpoint is precisely defined and not invariant-confounded;
- nonzero discrepancy or semantic/transition impact appears in discovery;
- its qualitative structure reproduces in confirmation;
- enough affected states exist to avoid attribution to a single anecdotal case.

The first intervention unit is a compiled region or graph boundary with an integrity check, not an assumed source-level operator. A repair that changes fusion, layout, graph partitioning or downstream compiler choices is reported as an intervention-dependent region effect, not an operator causal effect.

Repair and injection estimate separate counterfactuals. Neither is interpreted as unique root cause, necessity or sufficiency without explicit no-interference and no-alternative-cause assumptions.

## 8. Stage kill criteria

- centered residual adds no information beyond raw absolute error and event margins;
- transition effects are zero or fail held-out reproduction;
- observed transition differences are entirely explained by same-path runtime variability;
- compiled training execution cannot be verified without fallback or graph instability;
- the endpoint ranking is identical to a simple raw numerical baseline across all states;
- candidate repair/injection cannot preserve intervention integrity;
- conclusions require treating eager as truth or stress-selected states as a natural population.
