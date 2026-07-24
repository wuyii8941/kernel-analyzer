# Phase 12 Clipping-surviving Mutation Sampling

## Objective

Test whether mutations with zero clipping fork at the discovery state cross top-k/top-p sampling boundaries under common random numbers.

## Controls

- Same checkpoint, prompt/response token IDs, temperature and common uniform draws.
- Restricted to rollout batch `1`, the exact batch used to establish zero clipping forks.
- Two complete deterministic runs for clean and every mutation path.
- Final-reduction mutations alter the probability normalization used by sampling rather than becoming no-op probes.

## Results

| Mutation | Canary max log-normalizer delta | top-k set forks | top-p set forks | top-k fork draws | top-p fork draws | top-k first draw | top-p first draw | Either first draw | Self failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rotary_phase_fp16 | 0.156021 | 59 | 33 | 482 | 504 | 5 | 7 | 10 | 0 |
| decoder0_output_bf16_round | 0.0304527 | 43 | 23 | 355 | 412 | 8 | 9 | 14 | 0 |
| logsoftmax_fp16 | 0.000673294 | 0 | 18 | 39 | 95 | 0 | 1 | 1 | 0 |
| logsumexp_chunked_reverse | 1.90735e-06 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Interpretation Boundary

Artificial mutations with zero discovery-state clipping fork. Candidate-set and common-random-number sampled-token forks are semantic events; they do not establish reward or task-quality harm.

## Artifacts

- `results/phase12_mutation_sampling_gated/summary.json`
- `results/phase12_mutation_sampling_gated/all_rows.jsonl`
- `scripts/phase12_mutation_sampling.py`
