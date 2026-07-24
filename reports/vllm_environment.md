# vLLM Environment Audit

## Scope

This environment is isolated from the canonical ForkCert conda environment and exists only for the HF-vLLM claim pair and generation-engine Phase 7 extensions.

## Current Environment

| item | value |
| --- | --- |
| path | `/data1/tzh/conda-envs/forkcert-vllm` |
| Python | 3.11 |
| vLLM | 0.9.2 |
| PyTorch | 2.7.0+cu126 |
| CUDA runtime | 12.6 |
| Transformers | 4.53.2 |
| canonical environment modified | no |

## Import Audit

The isolated environment now imports vLLM, initializes CUDA, loads the Qwen3-0.6B compatibility checkpoint and completes prompt-logprob scoring. Transformers is pinned to 4.53.2. The compatibility mirror changes only the Transformers-5-only tokenizer configuration representation; model and tokenizer vocabulary hashes remain identical to the source checkpoint.

Three independent vLLM processes scored 8 requests and 1,024 response tokens. Two original-order runs were bitwise identical, and a reversed-request-order run was also bitwise identical. Two independent HF processes were exact. The HF-vLLM cross delta remains large even after matching FP16 parameter storage (p50 about 0.113, p99 about 3.16), so this is reported as a composite external execution-path sensitivity result rather than a single-operator attribution.

## Hardware Semantics

Tesla T4 has compute capability 7.5. vLLM V1 requires compute capability 8.0 or newer and falls back to V0 on T4. Version 0.9.2 retains the V0 engine and Qwen3 support, but does not expose the later `logprobs_mode` raw/processed selector. Therefore:

- Phase 1 teacher-forcing through `prompt_logprobs` is complete on V0.
- V1 `raw_logprobs` versus `processed_logprobs` cannot be claimed on T4/v0.9.2.
- Processed-logit Phase 7 needs either a V0-compatible newer API proven on T4 or Ampere-or-newer hardware.

## Reproducibility

- Installer: `install_vllm_env.sh`
- One-process scorer: `scripts/phase1_vllm_score.py`
- HF/vLLM merger and self gate: `scripts/phase1_merge_hf_vllm.py`
- Environment audit: `scripts/audit_vllm_env.py`
- Matched result: `results/phase1_hf_vllm_fp16_matched.jsonl`
- Report: `reports/phase1_vllm_fp16_matched.md`
