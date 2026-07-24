# Qwen3 Greedy-Impact Confirmation Contract v0.1 — 2026-07-17

## Status

Frozen after the 8-state mechanics discovery and before extracting or scoring
the confirmation bank. This contract confirms an application-impact endpoint;
it does not change numerical or compiler-correctness verdicts.

## 1. Question

For the fixed Qwen3 checkpoint and teacher-forced contexts, does eager versus
compiled execution change any greedy next-token choice on a separately selected
32-prompt bank?

Greedy argmax is not used by the cross-entropy training transition. Therefore
this is an inference-compatibility impact endpoint attached to a training-step
study, not a discrete training-control event.

## 2. Frozen source artifacts

```text
confirmation source:
data/phase0_grpo_samples.jsonl
SHA-256:
8822a63fa26b73605d37380c7fe792d97f606776b59ad7eba52d7be99e599862

discovery source used only for prompt exclusion:
data/phase6_step5_replay_samples.jsonl
SHA-256:
9dbaa3d5940e4aa29e89529ffc0fd68fd70ab22341ee9843d48f04560a5906bd

model checkpoint:
data/phase6_policy_step5_pre/model.safetensors
SHA-256:
f28ce3f7f7da92f7230438acae3f50f0adb83e13207558d03c1a93c3b9e31f11
```

## 3. Deterministic selection rule

Scan confirmation-source rows in file order. Represent a prompt by the exact
integer tuple `prompt_ids`.

Exclude a row if:

1. its prompt tuple occurs in the 8-state discovery source;
2. that prompt tuple was already selected;
3. `len(prompt_ids) >= 64`;
4. `response_ids` is empty.

Select the first remaining row for each unique prompt until 32 rows are
obtained. No outcome, logit, margin or compiled result participates in
selection.

The selected bank must contain 32 distinct prompt tuples and 32 distinct
`rollout_batch` values. Any violation makes the bank `INVALID`.

## 4. Implementation and transition protocol

Reuse `QWEN3_TRAINING_STEP_CONTRACT_V0_1_2026-07-17.md` unchanged:

- float32 Qwen3-0.6B weights;
- CUDA float16 autocast;
- SDPA math backend;
- eager versus full-graph Inductor core;
- new identical non-fused AdamW state;
- maximum sequence length 64;
- two deterministic repeats per state;
- correct external counter in both arms.

Candidate identity, exact transition and numerical abstention gates remain
authoritative. A state with invalid candidate identity cannot be scored for
impact.

## 5. Impact events and estimands

For every retained response target position, record:

- eager and compiled greedy token;
- eager and compiled top-1/top-2 logit margin;
- teacher-forced target token;
- token position.

Define token disagreement as unequal greedy tokens at the same matched
position. Define sequence disagreement as at least one token disagreement in a
state.

Report on this finite bank:

```text
total token disagreements
number of sequence-disagreement states
finite-bank sequence disagreement proportion
per-event direction as eager token -> compiled token
per-event eager and compiled margins
```

Argmax categories have no natural up/down order, so no scalar directional shift
is fabricated. Direction is a transition table over token pairs.

## 6. Strict compatibility verdict

Provided all 32 states have valid candidate identity, exact-core `ACCEPT` and
stable repeats:

- `ACCEPT`: zero sequence-disagreement states;
- `REJECT`: one or more stable sequence-disagreement states;
- `INDETERMINATE`: a disagreement is not stable across repeats or required
  decision detail is absent;
- `INVALID`: candidate/state identity gate fails.

`REJECT` means strict greedy compatibility failed on this finite bank. It does
not mean compiled violated mathematical semantics or that eager is correct.

## 7. Other ledgers

- exact transition: decided independently by the Training-Step contract;
- numerical transition: remains `UNINSTANTIATED`;
- runtime variability: reported from paired repeats;
- population sampling uncertainty: not inferred from this deterministic
  file-order bank;
- long-run training impact: outside this contract.

## 8. Anti-overclaim rules

- Do not combine the discovery 1/8 and confirmation count into one fork rate.
- Do not call the bank a random sample of Qwen or DL training.
- Do not infer model-quality improvement from agreement with the target token.
- Do not use observed margins to choose a numerical tolerance.
- Do not begin repair/injection attribution unless a stable confirmation event
  exists and candidate realization identity is preserved under intervention.

