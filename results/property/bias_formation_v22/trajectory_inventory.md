# Trajectory artifact completeness audit

The strict count is **8 complete paired trajectory artifacts**.  An additional
layer-23 key live-weight file exists, but it is **not** a complete case because
its same-weight sham is not exact; it is retained below as an excluded audit
candidate.

Included artifacts:

1. Liger fused CE live-weight repair;
2. Phi seq64 lm-head `dX`;
3. Qwen seq64 `v_proj` MM repair;
4. Qwen seq128 `v_proj` MM repair;
5. Qwen seq128 saved-P softmax semantic repair;
6. Qwen3-VL layer-0 SiLU invocation;
7. Mamba seq64 input-projection MM repair;
8. Qwen layer-23 attention-state repair;

The two Qwen `v_proj` artifacts are shape-specific instances of one
MM-accumulation family.  The alternate layer-23 key-materialization artifact
is not included because its same-weight sham fails exactness at every step.

## Excluded candidates

* `results/final/l23_key_live_weight_adamw.json`: incomplete common trajectory;
  each step has a nonzero same-weight `baseline_loss - repaired_loss`, so the
  sham gate fails closed.
* `results/property/seup_mainline/qwen_bmm_seq64_seup.json.gz`: registered
  negative control; endpoint repair is not nonzero at every step and its stable
  carrier gate fails.
* `results/property/seup_mainline/qwen_rsqrt_seq256_seup.json.gz`: registered
  negative control; no exact sham certificate and no stable carrier.
* The shorter Liger, Phi, and saved-P SEUP/consequence JSON files duplicate
  the complete artifacts listed above.

Excluded from the unique count:

* `results/property/seup_mainline/phi_seup.json` and the Phi consequence
  certificate duplicate the Phi seq64 trajectory;
* `results/property/seup_mainline/qwen_softmax_seup.json` is a shorter
  consequence/SEUP certificate for the saved-P semantic case;
* `*_geometry.json` files are carrier geometry analyses, not paired live-weight
  trajectories;
* `queue_complete.json` files are execution queues, not measurements;
* four-state or open-loop formation screens are not trajectory artifacts.

The eight included rows all pass the current artifact-level v2.2 observation
gate: complete paired F+B run, nonzero causal repair, exact sham/paired control,
closed parameter scope, and final basis-free live parameter separation greater
than the first-step separation. Seven now have an ordered-trajectory
persistence certificate. Qwen seq128 `v_proj` is the only aligned formation-
positive that resolves as diffusive/canceling; Qwen3-VL SiLU resolves as
feedback-sustained rather than local-source-persistent. This is an observation
ledger, not eight independent mechanisms and not a P1--P6 property result.
