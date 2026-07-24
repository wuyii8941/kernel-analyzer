# Phase 12 Clipping-surviving Mutation Sampling

## Objective

Test whether mutations with zero clipping fork at the discovery state cross top-k/top-p sampling boundaries under common random numbers.

## Controls

- Same checkpoint, prompt/response token IDs, temperature and common uniform draws.
- Two complete deterministic runs for clean and every mutation path.
- Final-reduction mutations alter the probability normalization used by sampling rather than becoming no-op probes.

## Results

| Mutation | Canary max log-normalizer delta | top-k set forks | top-p set forks | top-k fork draws | top-p fork draws | First-draw actual forks | Self failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| rotary_phase_fp16 | 0.156021 | 110 | 55 | 828 | 865 | 22 | 0 |
| decoder0_output_bf16_round | 0.03125 | 91 | 38 | 599 | 643 | 22 | 0 |
| logsoftmax_fp16 | 0.000675201 | 0 | 27 | 60 | 155 | 2 | 0 |
| logsumexp_chunked_reverse | 3.8147e-06 | 0 | 0 | 0 | 0 | 0 | 0 |

## Interpretation Boundary

Artificial mutations with zero discovery-state clipping fork. Candidate-set and common-random-number sampled-token forks are semantic events; they do not establish reward or task-quality harm.

## Artifacts

- `results/phase12_mutation_sampling/summary.json`
- `results/phase12_mutation_sampling/all_rows.jsonl`
- `scripts/phase12_mutation_sampling.py`
