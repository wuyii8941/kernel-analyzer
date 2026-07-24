# Multi-subject Matched-state Oracle Contract — 2026-07-15

Frozen before reading eager/compiled outputs from the real-model runs.

## Purpose

Test whether the discrepancy structure seen in the controlled CUDA pilot appears across different real model programs and non-discrepancy-selected input states. This is an exploratory discovery plus held-out confirmation study, not a compiler correctness test.

## Implementation relation

- reference: PyTorch eager CUDA execution;
- candidate: `torch.compile`, full graph, tracked Inductor backend;
- dtype: FP16 for model parameters and floating inputs;
- deterministic algorithms enabled, no dropout, no sampling RNG;
- fixed tensor shape within each run;
- batch size 1 so the execution state and case-level sampling unit are not conflated;
- every candidate call must increment a tracked compiled runtime counter;
- fallback, graph proliferation, missing output, NaN/Inf, or unverified path identity fails closed.

Eager is a baseline, not mathematical truth.

## Subject matrix

### S1. BERT-tiny SST-2 binary classification

- model: `M-FAC/bert-tiny-finetuned-sst2`;
- data: `stanfordnlp/sst2`, validation split;
- discovery: original rows `[0, 128)`;
- confirmation: original rows `[128, 256)`;
- fixed token length: 64;
- repeats per path/state: 3;
- decision map: `argmax(logits)` with PyTorch first-index tie behavior;
- signed decision variable: `positive_logit - negative_logit`;
- semantic endpoints: positive/negative direction, argmax disagreement, correctness-event disagreement relative to SST-2 label;
- numerical endpoints: both logits, signed margin, cross-entropy loss;
- transition endpoint: deferred to the training-step phase after inference calibration.

SST-2 labels provide a task outcome, not a specification for exact floating-point implementation correctness.

### S2. ResNet-18 ImageNet classification graph on CIFAR-10 images

- model: `microsoft/resnet-18` trained on ImageNet-1k;
- data: `uoft-cs/cifar10`, test split;
- discovery: original rows `[0, 128)`;
- confirmation: original rows `[128, 256)`;
- preprocessing: model's frozen image processor, fixed `3×224×224` tensor;
- repeats per path/state: 3;
- decision maps: argmax, ordered top-5, and top-5 set;
- decision variables: top-1 minus top-2 logit gap and fifth minus sixth logit gap;
- numerical endpoints: full 1,000-class logits and event-specific margins;
- transition endpoint: out of scope for this subject.

CIFAR-10 labels do not share the ImageNet-1k label space. This subject measures numerical/ranking reproducibility on real images but cannot report application accuracy or label correctness.

### S3. Qwen3-0.6B causal-LM teacher-forced decisions

- primary model: `data/phase0_policy_final`;
- data: `data/phase0_grpo_samples.jsonl`, original training rollout samples, not selected by eager/compiled discrepancy;
- discovery: rows `[0, 32)`;
- confirmation: rows `[32, 64)`;
- fixed sequence length: 166, consisting of prompt plus up to 128 response tokens;
- repeats per path/state: 3;
- decision maps at every response-token position: argmax, ordered top-5, top-5 set;
- decision variables: top-1/top-2 gap and fifth/sixth gap;
- numerical endpoints: response-position logits, teacher-forced target log-probability, cross-entropy, event-specific margins;
- transition endpoint: separate matched training-step run after inference calibration.

Checkpoint sensitivity uses `phase6_policy_step5_pre`, `phase8_policy_step11_pre`, and `phase9_policy_step14_pre` on the same first 16 discovery sequences. These checkpoints were retained around known effects and are therefore a **stress population**, not samples for estimating a checkpoint-population mean.

## Endpoint profile common to inference subjects

### M0. Measurement validity

- candidate runtime evidence on every call;
- graph count and graph code hash;
- eager/eager and compiled/compiled output hashes;
- finite-output, crash, missing and fallback accounting.

### M1. Numerical discrepancy

- mean signed, mean absolute and maximum absolute logit delta;
- signed and absolute event-margin delta;
- state-conditioned mean and heterogeneity;
- same-state repeat variability;
- state-bootstrap uncertainty.

### M2. Semantic discrepancy

- paired argmax disagreement;
- ordered top-5 disagreement;
- top-5 set disagreement;
- boundary-conditioned disagreement using the matching event margin;
- for BERT only, directional binary shift and task-correctness event change.

### M3. Outcome/transition impact

- teacher-forced loss/log-probability discrepancy for BERT and Qwen;
- gradient/update discrepancy in a later matched training-step phase;
- no long-run inference from one-step results.

## Confirmation rules

Confirmation is assessed structurally rather than by requiring the same sign:

1. measurement validity and self-pair controls must pass independently;
2. absolute discrepancy and its geometry are compared with discovery;
3. signed shift is called persistent only if direction and interval evidence reproduce;
4. semantic endpoints are compared separately and never pooled across different event spaces;
5. no-disagreement results are accompanied by their sample size and cannot establish equivalence without a declared bound/power argument;
6. Qwen checkpoint stress results are not pooled with the primary final-checkpoint distribution.

## Forbidden claims

- eager is truth;
- any nonzero discrepancy is a compiler bug;
- CIFAR-10 accuracy for the ImageNet ResNet model;
- natural application fork rate from boundary-enriched or stress populations;
- persistent bias from a discovery-only direction;
- long-run training harm from inference or one-step discrepancy;
- operator root cause before intervention-integrity checks.
