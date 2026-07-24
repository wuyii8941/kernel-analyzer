# Actual Sampling Forks With Common Random Numbers

## Objective

This batch addresses RQ1 and the second decision mechanism required by the research plan: determine whether eager/compile numerical differences change an actually sampled token, rather than only a top-k/top-p candidate set.

## Intervention / Comparison

Reference and alternative paths use the same Qwen3-0.6B step-5 checkpoint, fixed prefix token IDs, temperature, top-k/top-p settings and common uniform random number. The only execution-path difference is eager versus `torch.compile` under FP16 autocast and MATH-locked SDPA.

## Controls

- 8 fixed responses, 1,024 token contexts.
- 64 deterministic common draws per context, derived from case ID, token index, temperature and draw index.
- Two self runs per path; sampling and candidate-set self failures: 0.
- The first draw is reported separately as one coupled rollout realization. The remaining draws estimate boundary risk and are not treated as independent training tokens.
- Fixed token contexts are used for path comparison; free-running generations are not aligned post hoc.

## Result

For the first common draw per state:

| Mechanism | Forks / states | Rate | Prompt-cluster bootstrap 95% CI |
|---|---:|---:|---:|
| top-k | 17 / 1,024 | 1.660% | 0.391%-3.027% |
| top-p | 12 / 1,024 | 1.172% | 0.586%-1.758% |

Across 65,536 state-draw trials, top-k produced 550 sampled-token differences and top-p produced 554. These are repeated common-random-number probes of the same 1,024 contexts, so no token-i.i.d. confidence interval is attached.

All 29 first-draw events changed the rollout-prefix token hash immediately. Sixteen of 17 top-k events and 8 of 12 top-p events occurred without a candidate-set change. Therefore candidate-set equality is insufficient to establish sampling stability: path-dependent probability mass and CDF interval movement can change the sampled token while retaining the same set.

## Certificate / Artifacts

- Full CRN scan: `results/phase8_sampling_crn_full.jsonl`
- Actual sampling certificates: `results/phase8_sampling_actual_certificates.jsonl`
- Summary: `results/phase8_sampling_actual_summary.json`
- Scanner: `scripts/phase7_sampling_scan.py`
- Certificate builder: `scripts/phase8_sampling_certificates.py`

## Interpretation

This establishes a second natural training-semantic fork mechanism:

```text
eager/compile probability difference
-> shared u crosses different CDF intervals
-> sampled token differs
-> rollout training-data prefix differs
```

It does not yet show a reward or advantage change after free-running continuation, and it does not identify a unique compiled operator. The events are observed semantic forks, not certified implementation bugs.

## External Validity

The evidence is specific to Qwen3-0.6B, the frozen step-5 checkpoint, T4 FP16 and the measured eager/compile pair. It is not a BF16, FlashAttention, vLLM or newer-GPU stability result.

## Next Decision

**GO** for sampling consequence and attribution work. Preserve the 29 certificates, continue one selected prefix with common random numbers to quantify downstream response/reward divergence, and evaluate the signed-margin risk predictor on held-out rollout groups.
