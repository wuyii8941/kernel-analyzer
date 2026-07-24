# R1 Held-Out Replication: Partial Status

Status date: 2026-07-14

This is an interruption record, not the final `reports/r1_heldout.md` result.

## Objective

Test whether eager/compile decision forks and the existing clipping risk model extend
to checkpoints and prompts that were not used to discover, calibrate, attribute, or
replay the canonical five forks.

## Eligibility And Controls

- Source checkpoints: optimizer step 240 and step 270.
- Source model SHA-256 values: `84f7e9965c8574c30a5db9ee1f895ca9a13f0c897430580d2494141bfca59b8f`
  and `5b863e96dd7eb8cbce8374bf003fc5ae3ab9bd548e5f5b4e1dcb067d8c5881b3`.
- Frozen target states: pre-minibatch step 242 and step 272, both policy iteration 2.
- Prompt sets: 64 prompts at offsets 64 and 128 in the deterministic offline
  arithmetic corpus; each set has zero overlap with the 64 prior prompt hashes and
  zero overlap with the other held-out set.
- Measured fixed responses: four per state, 128 response tokens each, 512 token rows.
- Path contract: same local checkpoint, tokens, training mode, FP32 master weights,
  FP16 autocast, SDPA MATH; only eager versus `torch.compile` differs.
- Self controls: four independent OS processes/CUDA contexts per state; compile A/B
  used independent cold Inductor caches and a discarded full warmup pass. Ref and alt
  self maxima are exactly zero for both states.
- All path model artifact fingerprints and tokenization hashes match.

Authoritative eligibility evidence: `results/r1/eligibility_manifest.json`.

## Blind Preregistration

No signed crossing or branch label was computed before the prediction records were
committed. The isolated preregistration commit is:

```text
c6085f6863c7ddffed6a1202cee74890f07e9167
```

The prediction uses only the unsigned clipping-margin distribution and absolute
eager/compile delta distribution.

| State | Applicable clipping tokens | Near margin < 1e-2 | Predicted forks | Poisson 95% CI |
|---|---:|---:|---:|---:|
| step 242 | 512 | 0 | 0 | [0, 0] |
| step 272 | 0 | 0 | 0 | [0, 0] |

At step 272 all four responses received identical rewards, so all 512 advantages are
zero. Clipping is algorithmically not applicable at that state; the 512 path deltas
remain valid for sampling and distribution-drift analysis.

Preregistration evidence: `results/r1/prereg_repo/` and
`results/r1/prereg_commit.json`.

## Clipping Result

| State | Applicable decisions | Observed forks | Prediction CI hit | Existence replicated |
|---|---:|---:|---|---|
| step 242 | 512 | 0 | Yes | No |
| step 272 | 0 | 0 | Not an existence test | Not applicable |

Coverage gates pass: no missing rollout rows, missing token IDs, token mismatches, or
wrong state labels. All step-242 regions remain `unknown`; step-272 rows are
`not_applicable`. No theoretical legal bound is introduced.

Interpretation: the registered clipping predictor correctly issued a zero-risk
prediction, but R1 has not replicated clipping-fork existence. Step 242 has no
near-boundary demand and step 272 has no nonzero-advantage clipping decisions. These
states are retained as selected rather than replaced after observing their margins.

Artifacts:

- `results/r1/from240_clipping_certificates.jsonl`
- `results/r1/from270_clipping_certificates.jsonl`
- `reports/r1_from240_clipping.md`
- `reports/r1_from270_clipping.md`

## Sampling Status

The sampling protocol uses top-k 50, top-p 0.9, temperature 1.0, 64 deterministic
common-random-number draws per token, and independent process self runs.

Completed path outputs for step 242:

- reference A: complete, 512 rows;
- reference B: complete, 512 rows;
- alternative A: complete, 512 rows;
- alternative B: not launched.

The four step-272 sampling path runs are also not launched. No cross-path sampling
count is reported until all eight path outputs exist and both self gates pass.

## Interruption

The fourth GPU service launch was rejected by the execution platform because its
approval quota was exhausted. The platform explicitly prohibited indirect execution
or workaround. This is an orchestration limitation, not a model, CUDA, or research
hypothesis failure. The three already-launched services completed successfully.

Remaining work for final R1 completion:

1. run step-242 alternative B;
2. run step-272 reference A/B and alternative A/B;
3. merge with `scripts/r1_merge_sampling.py` and enforce exact independent-process
   self controls;
4. generate final `reports/r1_heldout.md` with sampling counts and cluster intervals.

## External Validity

These are Qwen3-0.6B, Tesla T4, FP16 results. The clipping zero result cannot exclude
native-BF16 forks on Ampere-or-newer hardware.
