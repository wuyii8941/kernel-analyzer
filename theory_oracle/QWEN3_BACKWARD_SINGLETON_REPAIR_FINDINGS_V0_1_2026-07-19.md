# Qwen3 backward singleton-repair findings v0.1

## Verdict

Five of the nine singleton generated backward families now have valid
selected-state repair evidence.  Three repairs are exactly null at all recorded
gradient and update endpoints; two SiLU-related repairs are non-null.  No family
has injection, population-transport, long-run, root-cause or correctness credit.

All valid runs preserve the frozen compiled scorer, GRPO loss construction,
candidate graph identity, and full backward/gradient-control/AdamW transition.
Each run resolves one generated module and replaces one singleton call once.

## Endpoint results

| Treatment | Clipped-gradient effect | Update effect | Direction relative to eager |
|---|---:|---:|---|
| tangent cast/view | exact null | exact null | no movement |
| embedding-gradient zero initialization | exact null | exact null | no movement |
| FP16-to-FP32 add | exact null | exact null | no movement |
| SiLU × multiply | non-null | non-null | gradient distance decreases about 0.10%; update distance increases about 0.12% |
| SiLU × multiply backward | non-null | non-null | gradient distance increases about 0.46%; update distance increases about 0.35% |

The two non-null repairs change all 310 parameter-gradient tensors and 282
parameter-update tensors.  The repair vectors have only weak positive alignment
with the compiled-to-eager vector.  They therefore do not explain most of the
global eager/compiled discrepancy at this state.

## Critical interpretation

The SiLU-multiply result is a counterexample to treating a single distance as
the operator Oracle: a local repair moves the clipped gradient slightly toward
eager but moves the post-AdamW update slightly away.  Propagation through global
gradient clipping and AdamW changes the endpoint geometry.

An exact-null repair means only that this exact replacement produces no
selected-state endpoint change.  It does not prove semantic equivalence across
states, nor does it clear adjacent accumulation or propagation families.

The two SiLU repairs are adjacent in one backward path.  Their separate
non-null effects do not establish additivity, necessity, sufficiency, or two
independent root causes.  A joint intervention would be required to measure
their interaction.

## Remaining singleton scope

Four singleton families remain uninstantiated because they fuse reductions,
slice-backward, normalization derivatives, attention safe-softmax, or embedding
accumulation.  Replacing them requires reproducing the complete source-graph
boundary; treating their constituent ATen names independently would change the
treatment.
