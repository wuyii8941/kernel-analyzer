# Phase 7 Remaining Decision Audit

## Claim Scope
Execution-path mismatch is relevant only when a strict training decision boundary amplifies it into a semantic fork. No item below is called fragile or bug because Phase 2 did not produce a usable analytic legal bound.

## Confound Checklist
- canonical_recipe_inspected: PASS
- model_architecture_inspected: PASS
- installed_generation_engines_inspected: PASS
- strict_boundary_required_before_experiment: PASS
- unavailable_or_absent_decisions_not_fabricated: PASS

## Delta Self Control
Sampling candidate sets and the paired step-5 gradient norms were each measured twice per path. Both controls produced zero self mismatch. Items that are not instantiated by the canonical recipe have no executable self-run.

## Summary

| Decision | Canonical status | Evidence | Next action |
| --- | --- | --- | --- |
| PPO/GRPO clipping | Complete | 5 natural forks in 39,936 applicable online decisions | Preserve as the core result |
| top-k/top-p truncation | Partial but positive | Raw-logit teacher-forced scan found top-k and top-p candidate-set forks; temperature sweep complete | Replicate through a generation engine's processed-logit API |
| gradient clipping | Partial, no natural fork in paired step-5 batch | Eager `7.82968`, compile `7.81473`, threshold `1.0`; both trigger; 300-step eager minimum margin `1.0` | A full paired 300-step scan is optional; current demand-side evidence predicts low yield |
| KL early stop | Not instantiated | Canonical GRPO explicitly uses `beta=0.0` and defines no KL early-stop threshold | Requires a new recipe with a strict documented stop boundary |
| MoE routing | Not applicable to current model | Qwen3-0.6B is dense (`Qwen3ForCausalLM`), with no expert-count or experts-per-token fields | Requires a Qwen3-MoE or other MoE checkpoint and gate-logit instrumentation |
| optimizer threshold | No valid target identified | Canonical AdamW epsilon is a continuous denominator stabilizer, not a discrete branch | Do not run until a strict optimizer comparison boundary is specified |

## Generation Engine Dependency

The current environment has no `vllm`, `sglang`, or `flash_attn` package. Existing top-k/top-p evidence applies explicit temperature and truncation transforms to teacher-forced HF logits; it does not establish parity or forks in raw-versus-processed engine APIs. This extension remains pending rather than failed.

## Gradient-Clipping Interpretation

The controlled midpoint threshold `7.822205` lies between the two measured step-5 norms and produces a fork, validating the trigger detector. It is calibration only. At the real threshold `1.0`, the paired result has no fork. The canonical eager history has 234 triggered and 66 zero-gradient steps; no logged norm lies within `0.2` of the boundary, so the demand-side margin distribution is unlike PPO clipping's near-boundary population.

## External Validity

All executable results here use FP16 autocast on Tesla T4. T4 has no native BF16 tensor-core support. FP16 forks are positive mechanism evidence; FP16 zero-fork observations cannot rule out production BF16 behavior. A native BF16 replication remains an external hardware requirement.
