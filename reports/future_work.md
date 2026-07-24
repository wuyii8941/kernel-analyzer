# ForkCert Future Work

## P1 Hardware Replication Gaps

The current HF-vLLM P1 experiment is scoped to Tesla T4, FP16, vLLM 0.9.2 V0, and XFormers. The following are hardware or engine capability gaps, not failed P1 controls:

- `FLASH_ATTN`: unavailable on T4 (compute capability 7.5); replicate on Ampere or newer hardware.
- `TORCH_SDPA` as a vLLM attention backend: unavailable in the audited V0/T4 stack; replicate on a supported newer stack.
- vLLM V1: disabled because this vLLM/T4 environment is restricted to the validated V0 execution path.
- native BF16: unsupported by T4 tensor cores; repeat the same frozen-state protocol on Ampere or newer hardware.

The future replication must preserve the P1 controls: identical checkpoint and token IDs, independent-process self runs, request-order invariance, processor-free raw/processed identity (or an explicit selector where supported), signed-delta analysis, and clipping plus common-random-number sampling scans.
